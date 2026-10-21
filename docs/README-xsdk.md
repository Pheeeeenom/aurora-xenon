# Xbox 360 (xsdk)

encounter's Aurora for the Xbox 360 through [xsdk](https://github.com/xenosworks/xsdk), a homebrew SDK that builds
titles for the console's retail kernel, drawing through [vulkan4xenon](https://github.com/xenosworks/vulkan4xenon), a
Vulkan 1.0 subset on xsdk's GPU layer. This is Aurora at `3227d76`, the commit Dusklight pinned when its port began,
with six of upstream's later commits picked in; the changes are on the `xenon` branch. It's built as
[dusklight-xenon](https://github.com/xenosworks/dusklight-xenon)'s `extern/aurora`.

## Building

Aurora builds inside a game's CMake build, with xsdk's toolchain file, as Dusklight's `CMakeLists.txt` does under
`if (XSDK)`:

-   `AURORA_GFX_BACKEND` set to `vulkan4xenon`, with a `vulkan4xenon` target, the layer's `libvulkan4xenon.a` built by
    its own make, and glslang-xenon's `glslang` target added before Aurora;
-   `AURORA_NOD_PROVIDER` set to `iso` and `AURORA_SDL3_PROVIDER` to `system`, SDL3 being
    [SDL-xenon](https://github.com/xenosworks/SDL-xenon)'s build;
-   the frame pools (`AURORA_*_POOL_SIZE`), the texture caches and the decoded vertices' room in GPU memory
    (`AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES`) sized for the game, as the CPU and GPU share the console's 512 MB.

## What the fork adds

-   **A backend interface,** `lib/gfx/backend.hpp`, so a port can bring its own graphics backend: `AURORA_GFX_BACKEND`
    is `dawn`, upstream's, `null`, which draws nothing, or `vulkan4xenon`.
-   **The vulkan4xenon backend,** `lib/vulkan4xenon/`. Upstream's shaders read the game's vertex arrays from storage
    buffers and unpack them with integer operations, and Xenos has neither. So the backend decodes vertices on the CPU
    into one stream (`decode.cpp`), and a GLSL generator of its own (`glsl.cpp`), modelled on upstream's WGSL in
    `lib/gx/shader.cpp`, writes each pipeline's programs.
-   **The shader pack.** On the null backend, `AURORA_GLSL_RECORD_DIR` records the GLSL each pipeline needs as the game
    runs. `tools/build_glsl_pack.py` compiles the recorded programs on the PC, glslang to SPIR-V and xost to Xenos
    microcode, into a C source the game links (`include/aurora/glsl_pack.h`). A program the pack doesn't hold is
    compiled on the console through glslang the first time it's asked for, and kept in the cache path's `shaders`
    folder (`program.cpp`).
-   **The disc:** nod's Rust doesn't build for xsdk, so `lib/nod_iso` gives nod's C API over plain GameCube images.
    Wii discs and compressed images, such as RVZ, aren't read there.
-   **A big-endian host:** the command stream is written and read in the host's order, arrays the game marks
    little-endian are read as the host wrote them, and textures' big-endian words are read as they are.
-   **Threads placed** on the console's six hardware threads by name (`lib/thread.cpp`), from a list a game changes
    with `aurora_set_hardware_threads`.
-   **Frame captures:** `aurora_capture_frame` writes a presented frame as a PPM, as Xenia shows none of vulkan4xenon's
    frames.
-   **Libraries built for xsdk:** abseil, sqlite (`lib/sqlite_os_xsdk.c`), Tracy (`lib/tracy_xsdk.cpp`) and ImGui; RmlUi
    runs on a backend that doesn't draw it, drawing nothing.
-   `AURORA_ENABLE_DIAGNOSTICS` times the workers, the GPU and each frame's path, and keeps memory's peaks.

## What it draws on the console

-   GX draws: TEV stages, texgens, lighting, fog with its range adjustment, matrices picked per vertex, indirect
    texturing, alpha compares and blending, and lines and points as upstream's WebGPU backend draws them.
-   The EFB at 1280x720 at 1, 2 or 4 samples, past eDRAM in vulkan4xenon's tiled passes; EFB copies, depth copies and
    `GXPeekZ`; palette conversions of EFB copies of 8-bit indices, such as Twilight Princess's minimap.
-   Textures as upstream converts them, to RGBA8, but CMPR as BC1 where it's opaque, and I8 as R8.

Not yet, each skipped where the backend meets it:

-   RmlUi's documents, Dusklight's menus among them, and Aurora's ImGui overlay (`encoding.cpp`). vulkan4xenon's
    milestone 16 is to draw Dusklight's menus.
-   The WebGPU extension's custom draws, texture copies and encoder tasks, which mods use.
-   Pipelines with destination alpha, a logic op other than clear, copy and noop, or polygon offset: their draws are
    skipped and logged (`pipeline.cpp`, `unsupported`).
-   Palette conversions of 16-bit indices, and depth copies into formats the backend doesn't convert.

A pipeline is built on Aurora's compile threads, as upstream builds it, and a draw whose pipeline isn't ready yet is
skipped. A pipeline whose programs the pack lacks takes longer, as they're compiled first.

## Checks

-   `vulkan4xenon_glsl_tests` (`tests/vulkan4xenon_glsl_test.cpp`) runs the GLSL generator on the PC, on configs made
    to hit what it writes.
-   `tools/check_glsl_math.py` runs the programs those tests write on xost's simulator, compiled as a pack is, and
    compares what they compute with its own reference of upstream's WGSL.
