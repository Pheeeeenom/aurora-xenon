#ifndef AURORA_GLSL_PACK_H
#define AURORA_GLSL_PACK_H
// The vulkan4xenon backend's programs, compiled offline: the SPIR-V of each GLSL program a run on the null backend
// recorded (lib/vulkan4xenon/glsl.hpp), which a game links in. vulkan4xenon's own bundle (v4x_shader_bundle) then finds
// each one's Xenos microcode by that SPIR-V.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  // The program's source's XXH3-64, seeded 0 for a vertex program and 1 for a fragment one (glsl::program_hash).
  uint64_t hash;
  const uint32_t* spirv;
  uint32_t bytes;
} AuroraGlslProgram;

typedef struct {
  // Sorted by hash.
  const AuroraGlslProgram* programs;
  uint32_t count;
} AuroraGlslPack;

// The game's pack; the backend's own weak, empty one when the game links none, so no program is found.
extern const AuroraGlslPack aurora_glsl_pack;

#ifdef __cplusplus
}
#endif

#endif
