add_library(aurora_gx STATIC
        lib/gfx/frame.cpp
        lib/gfx/frame_path.cpp
        lib/gfx/pipeline_cache.cpp
        lib/gfx/recording.cpp
        lib/gfx/render_worker.cpp
        lib/gfx/texture.cpp
        lib/gfx/texture_convert.cpp
        lib/gx/attr_fmt.cpp
        lib/gx/command_processor.cpp
        lib/gx/regs.cpp
        lib/gx/dl.cpp
        lib/gx/fifo.cpp
        lib/gx/gx.cpp
        lib/gx/texture.cpp
        lib/gx/shader_info.cpp
        lib/dolphin/gx/GXBump.cpp
        lib/dolphin/gx/GXCull.cpp
        lib/dolphin/gx/GXCpu2Efb.cpp
        lib/dolphin/gx/GXDispList.cpp
        lib/dolphin/gx/GXDraw.cpp
        lib/dolphin/gx/GXExtra.cpp
        lib/dolphin/gx/GXFifo.cpp
        lib/dolphin/gx/GXFrameBuffer.cpp
        lib/dolphin/gx/GXGeometry.cpp
        lib/dolphin/gx/GXGet.cpp
        lib/dolphin/gx/GXLighting.cpp
        lib/dolphin/gx/GXManage.cpp
        lib/dolphin/gx/GXPerf.cpp
        lib/dolphin/gx/GXPixel.cpp
        lib/dolphin/gx/GXTev.cpp
        lib/dolphin/gx/GXTexture.cpp
        lib/dolphin/gx/GXTransform.cpp
        lib/dolphin/gx/GXVert.cpp
        lib/dolphin/gx/GXAurora.cpp
)
add_library(aurora::gx ALIAS aurora_gx)
set_target_properties(aurora_gx PROPERTIES FOLDER "aurora")

target_link_libraries(aurora_gx PUBLIC aurora::core xxHash::xxhash)
target_link_libraries(aurora_gx PRIVATE absl::btree absl::flat_hash_map sqlite3 Tracy::TracyClient)
# The command processor and the backends read them, all in this library.
target_compile_definitions(aurora_gx PRIVATE AURORA_INDEX_PATTERN_VERTICES=${AURORA_INDEX_PATTERN_VERTICES}
                                             AURORA_FIFO_RING_BYTES=${AURORA_FIFO_RING_BYTES})
# The backend, and texture packs, which only Dawn loads.
if (AURORA_USE_DAWN)
    target_sources(aurora_gx PRIVATE
            lib/gfx/clear.cpp
            lib/gfx/depth_peek.cpp
            lib/gfx/encoding.cpp
            lib/gfx/resource_cache.cpp
            lib/gfx/dds_io.cpp
            lib/gfx/tex_copy_conv.cpp
            lib/gfx/tex_palette_conv.cpp
            lib/gfx/texture_format.cpp
            lib/gfx/texture_replacement.cpp
            lib/gx/pipeline.cpp
            lib/gx/shader.cpp
            lib/gfx/png_io.cpp
            lib/gfx/png_io.hpp
            lib/webgpu/backend.cpp
            lib/webgpu/extension.cpp
            lib/webgpu/present.cpp
    )
    target_link_libraries(aurora_gx PUBLIC dawn::webgpu_dawn)
    target_link_libraries(aurora_gx PRIVATE PNG::PNG)
    target_compile_definitions(aurora_gx PRIVATE WEBGPU_DAWN)
    # webgpu/backend.cpp starts the Dawn backends aurora_core's aurora.cpp used to.
    get_target_property(_aurora_dawn_backends aurora_core COMPILE_DEFINITIONS)
    list(FILTER _aurora_dawn_backends INCLUDE REGEX "^DAWN_ENABLE_BACKEND_")
    target_compile_definitions(aurora_gx PRIVATE ${_aurora_dawn_backends})
elseif (AURORA_GFX_BACKEND STREQUAL "vulkan4xenon")
    target_sources(aurora_gx PRIVATE
            lib/vulkan4xenon/backend.cpp
            lib/vulkan4xenon/decode.cpp
            lib/vulkan4xenon/device.cpp
            lib/vulkan4xenon/encoding.cpp
            lib/vulkan4xenon/glsl.cpp
            lib/vulkan4xenon/pipeline.cpp
            lib/vulkan4xenon/program.cpp
            lib/null/texture_replacement.cpp
    )
    target_compile_definitions(aurora_gx PRIVATE
            AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES=${AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES})
    # The layer, and the game's shader pack (aurora/glsl_pack.h) with its bundle, which the game links.
    target_link_libraries(aurora_gx PUBLIC vulkan4xenon)
    # glslang-xenon, which compiles the programs the pack doesn't hold on the console.
    target_link_libraries(aurora_gx PRIVATE glslang glslang-default-resource-limits)
else ()
    target_sources(aurora_gx PRIVATE
            lib/null/backend.cpp
            lib/null/texture_replacement.cpp
    )
    if (NOT AURORA_GLSL_RECORD_DIR STREQUAL "")
        target_sources(aurora_gx PRIVATE
                lib/vulkan4xenon/glsl.cpp
                lib/vulkan4xenon/record.cpp
        )
        target_compile_definitions(aurora_gx PRIVATE AURORA_GLSL_RECORD_DIR="${AURORA_GLSL_RECORD_DIR}")
    endif ()
endif ()

if (AURORA_RMLUI_DRAWS)
    target_sources(aurora_gx PRIVATE
        lib/rmlui/pipeline.cpp
        lib/rmlui/pipeline.hpp
    )
endif ()
