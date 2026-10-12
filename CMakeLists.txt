cmake_minimum_required(VERSION 3.25)
project(aurora LANGUAGES C CXX)
if (APPLE)
    enable_language(OBJC)
endif()
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(AURORA_ENABLE_GX "Enable GX implementation and WebGPU renderer" ON)
option(AURORA_ENABLE_DVD "Enable DVD implementation backed by nod" ON)
option(AURORA_ENABLE_CARD "Enable CARD implementation based on kabufuda" ON)
option(AURORA_ENABLE_THP "Enable THP decoder" ON)
option(AURORA_ENABLE_RMLUI "Enable HTML/CSS Based UI Library for use in end-user UI development." OFF)
option(AURORA_CACHE_USE_ZSTD "Compress WebGPU cache entries with zstd" ON)
# Timers and peaks a game's benchmark reads through extern "C" calls: the FIFO and render workers' busy time, each
# frame's path through its stages and, where the backend reports them, the GPU's frame time, the swapchain acquire
# and memory peaks.
option(AURORA_ENABLE_DIAGNOSTICS "Time Aurora's workers, the GPU and each frame's path, and keep memory peaks" OFF)
set(AURORA_GX_MAX_PNMTX 10 CACHE STRING
  "Position/normal matrix slots in GX matrix memory: 10 (the SDK's GX_PNMTX0-9) to 21, for a game that loads more")
set(AURORA_GFX_BACKEND "dawn" CACHE STRING
  "What draws GX: dawn (WebGPU), null (nothing: frames are recorded and discarded, with no window or GPU) or \
vulkan4xenon (the Xbox 360's GPU, through the vulkan4xenon layer)")
set_property(CACHE AURORA_GFX_BACKEND PROPERTY STRINGS dawn null vulkan4xenon)
if (NOT AURORA_GFX_BACKEND MATCHES "^(dawn|null|vulkan4xenon)$")
  message(FATAL_ERROR
    "AURORA_GFX_BACKEND is ${AURORA_GFX_BACKEND}; it must be dawn, null or vulkan4xenon")
endif ()
foreach (_aurora_layer vulkan4xenon)
  if (AURORA_ENABLE_GX AND AURORA_GFX_BACKEND STREQUAL _aurora_layer AND NOT TARGET ${_aurora_layer})
    message(FATAL_ERROR "AURORA_GFX_BACKEND is ${_aurora_layer}, but no ${_aurora_layer} target was added before Aurora")
  endif ()
endforeach ()
if (AURORA_ENABLE_GX AND AURORA_GFX_BACKEND STREQUAL "dawn")
  set(AURORA_USE_DAWN ON)
else ()
  set(AURORA_USE_DAWN OFF)
endif ()
# On the null backend, a directory to record into the GLSL each pipeline's shader config needs for the vulkan4xenon
# backend (lib/vulkan4xenon/glsl.hpp), for its offline compile. Empty records nothing.
set(AURORA_GLSL_RECORD_DIR "" CACHE STRING "Where the null backend records each pipeline's GLSL for vulkan4xenon")
if (NOT AURORA_GLSL_RECORD_DIR STREQUAL "" AND NOT AURORA_GFX_BACKEND STREQUAL "null")
  message(FATAL_ERROR "AURORA_GLSL_RECORD_DIR records on the null backend, not ${AURORA_GFX_BACKEND}")
endif ()
# RmlUi draws only through Dawn; on the null backend its documents run and nothing is drawn.
if (AURORA_ENABLE_RMLUI AND AURORA_USE_DAWN)
  set(AURORA_RMLUI_DRAWS ON)
else ()
  set(AURORA_RMLUI_DRAWS OFF)
endif ()
# The streaming pools a frame records into, in bytes; Dawn keeps several copies of them for staging. Overflowing the
# first four aborts, so a game sizes them from its peaks. The defaults are 24, 5, 2, 8 and 24 MiB.
set(AURORA_UNIFORM_POOL_SIZE 25165824 CACHE STRING "Bytes of uniform data a frame can record")
set(AURORA_VERTEX_POOL_SIZE 5242880 CACHE STRING "Bytes of vertex data a frame can record")
set(AURORA_INDEX_POOL_SIZE 2097152 CACHE STRING "Bytes of index data a frame can record")
set(AURORA_STORAGE_POOL_SIZE 8388608 CACHE STRING "Bytes of storage data a frame can record")
set(AURORA_TEXTURE_UPLOAD_POOL_SIZE 25165824 CACHE STRING
  "Bytes of texture uploads a frame stages in its pool; larger ones are staged apart")
# Frames recorded or drawing at once, each slot with its own copy of the pools.
set(AURORA_FRAME_SLOT_COUNT 2 CACHE STRING "Frames recorded or drawing at once")
# Quad, strip, fan, line and point draws of up to this many vertices take their indices from patterns built once, which
# every frame's draws share, instead of having them built into the frame's index pool; larger draws, and a draw another
# is merged into, still build theirs. 0, the default, builds every draw's.
set(AURORA_INDEX_PATTERN_VERTICES 0 CACHE STRING "Vertices up to which a draw takes its indices from prebuilt patterns")
# The GX command stream's memory, in 64 KiB chunks the game thread writes while the FIFO processor reads those before
# them; the game thread waits when every chunk holds commands not yet processed. The default is 2 MiB.
set(AURORA_FIFO_RING_BYTES 2097152 CACHE STRING "Bytes of memory for the GX command stream")
# The GX texture caches, in bytes: converted textures kept for reuse, and the background pre-conversion's results
# waiting to be drawn and sources queued for it. The defaults are 128, 128 and 64 MiB.
set(AURORA_TEXTURE_CONTENT_CACHE_BYTES 134217728 CACHE STRING "Bytes of converted textures kept for reuse")
set(AURORA_TEXTURE_PRECONVERT_READY_BYTES 134217728 CACHE STRING
  "Bytes of pre-converted textures waiting for their first draw")
set(AURORA_TEXTURE_PRECONVERT_QUEUE_BYTES 67108864 CACHE STRING "Bytes of textures queued for pre-conversion")
# The vulkan4xenon backend's decoded vertices, kept in the console's GPU memory from frame to frame for draws with the
# same inputs: one block for every frame slot, which must hold every vertex the frames in flight draw; draws past it
# are skipped. The default is 8 MiB.
set(AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES 8388608 CACHE STRING
  "Bytes of GPU memory for the vulkan4xenon backend's decoded vertices, kept across frames")

include(CMakeDependentOption)
cmake_dependent_option(AURORA_ENABLE_EXAMPLES "Enable examples" ON "CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR" OFF)
cmake_dependent_option(AURORA_ENABLE_TESTS "Enable tests" ON "NOT CMAKE_CROSSCOMPILING" OFF)

# Dependency versions
include(cmake/AuroraDependencyVersions.cmake)

# Platform-specific defaults
set(_default_provider "auto")
if (WIN32)
  set(_default_linkage "shared")
else ()
  set(_default_linkage "static")
endif ()

# Dawn/WebGPU (if AURORA_ENABLE_GX)
set(AURORA_DAWN_PROVIDER "${_default_provider}" CACHE STRING
  "How to provide Dawn: auto, vendor (build from source), system (find_package/imported), package (download prebuilt)")
set_property(CACHE AURORA_DAWN_PROVIDER PROPERTY STRINGS auto vendor system package)
set(AURORA_DAWN_LINKAGE "${_default_linkage}" CACHE STRING "Dawn linkage type preference")
set_property(CACHE AURORA_DAWN_LINKAGE PROPERTY STRINGS shared static)

# SDL3
set(AURORA_SDL3_PROVIDER "${_default_provider}" CACHE STRING
  "How to provide SDL3: auto, vendor (build from source), system (find_package/imported), or package (download prebuilt)")
set_property(CACHE AURORA_SDL3_PROVIDER PROPERTY STRINGS auto vendor system package)
set(AURORA_SDL3_LINKAGE "${_default_linkage}" CACHE STRING "SDL3 linkage type preference")
set_property(CACHE AURORA_SDL3_LINKAGE PROPERTY STRINGS shared static)

# nod (if AURORA_ENABLE_DVD)
set(AURORA_NOD_PROVIDER "${_default_provider}" CACHE STRING
  "How to provide nod: auto, vendor (build from source), system (find_package/imported), package (download prebuilt), \
iso (Aurora's reader of plain disc images, in C)")
set_property(CACHE AURORA_NOD_PROVIDER PROPERTY STRINGS auto vendor system package iso)
set(AURORA_NOD_LINKAGE "${_default_linkage}" CACHE STRING "nod linkage type preference")
set_property(CACHE AURORA_NOD_LINKAGE PROPERTY STRINGS shared static)

add_subdirectory(extern)

if (AURORA_ENABLE_DIAGNOSTICS)
  add_compile_definitions(AURORA_DIAGNOSTICS)
endif ()
# Sizes GX state, the uniform block and the shaders' matrix arrays, so every Aurora source and test sees the same value.
add_compile_definitions(AURORA_GX_MAX_PNMTX=${AURORA_GX_MAX_PNMTX})
# The same for the pools.
add_compile_definitions(
  AURORA_UNIFORM_POOL_SIZE=${AURORA_UNIFORM_POOL_SIZE}
  AURORA_VERTEX_POOL_SIZE=${AURORA_VERTEX_POOL_SIZE}
  AURORA_INDEX_POOL_SIZE=${AURORA_INDEX_POOL_SIZE}
  AURORA_STORAGE_POOL_SIZE=${AURORA_STORAGE_POOL_SIZE}
  AURORA_TEXTURE_UPLOAD_POOL_SIZE=${AURORA_TEXTURE_UPLOAD_POOL_SIZE}
  AURORA_FRAME_SLOT_COUNT=${AURORA_FRAME_SLOT_COUNT}
  AURORA_TEXTURE_CONTENT_CACHE_BYTES=${AURORA_TEXTURE_CONTENT_CACHE_BYTES}
  AURORA_TEXTURE_PRECONVERT_READY_BYTES=${AURORA_TEXTURE_PRECONVERT_READY_BYTES}
  AURORA_TEXTURE_PRECONVERT_QUEUE_BYTES=${AURORA_TEXTURE_PRECONVERT_QUEUE_BYTES}
)

include(cmake/aurora_core.cmake)
include(cmake/aurora_os.cmake)
if (AURORA_ENABLE_GX)
  include(cmake/aurora_gx.cmake)
  include(cmake/aurora_gd.cmake)
endif ()
include(cmake/aurora_pad.cmake)
include(cmake/aurora_ms.cmake)
include(cmake/aurora_si.cmake)
include(cmake/aurora_main.cmake)
include(cmake/aurora_mtx.cmake)
include(cmake/aurora_vi.cmake)
if (AURORA_ENABLE_THP)
  include(cmake/aurora_thp.cmake)
endif ()
if(AURORA_ENABLE_CARD)
  include(cmake/aurora_card.cmake)
endif()
if (AURORA_ENABLE_DVD)
  include(cmake/aurora_dvd.cmake)
endif ()

if (AURORA_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif ()

if (AURORA_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif ()
