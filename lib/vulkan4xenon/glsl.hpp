#pragma once
// The vulkan4xenon backend's GLSL programs for a GX shader config. There is no shader compiler on the Xbox 360, so the
// programs are compiled offline, glslang to SPIR-V and xost to Xenos microcode, and found by their source's hash.
//
// What the programs and the backend share:
// - Set 0 is a uniform buffer per uniform piece (gx::UniformPiece), binding i the piece as gx.hpp lays it out, at
//   DrawData::uniformOffsets[i]. A program declares only the pieces it reads, each as std140 vec4s: a matrix is its
//   rows, as Mat3x4 and Mat4x4 hold them.
// - Set 1 is the 8 GX texture maps, binding i GX_TEXMAPi, sampled by the fragment program alone.
// - The vertex stage's push constant is the current position matrix's three rows, 48 bytes: xost reads no uniform
//   array at an index known only at run time (its docs/xsc.md), so the backend pushes the matrix the draw selects.
// - One vertex stream, binding 0, in the layout below. Locations follow xost's usages: position at 0, colour 0 at 1,
//   texture coordinates from 2 (xost:xsc/resolve.py).
// - The vertex program writes the WGSL's position with y negated, Vulkan's clip space.
//
// The programs are written in the GLSL xost compiles (its docs/xsc.md): each local assigned once, no function calls,
// no unary minus, and at most one discard, on one comparison.
#include "../gx/pipeline.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::gfx::vulkan4xenon::glsl {
// vulkan4xenon's vertex formats, named here so the generator builds without Vulkan headers.
enum class AttrFormat : uint8_t {
  F32x1, // VK_FORMAT_R32_SFLOAT
  F32x2, // VK_FORMAT_R32G32_SFLOAT
  F32x3, // VK_FORMAT_R32G32B32_SFLOAT
  U8x4,  // VK_FORMAT_A8B8G8R8_UNORM_PACK32: one host-order word, red in its low byte
};

enum class NbtSlice : uint8_t {
  Normal,
  Binormal,
  Tangent,
};

struct Attribute {
  uint32_t location;
  AttrFormat format;
  uint32_t offset;                   // bytes into the vertex
  GXAttr attr;                       // the GX attribute it holds
  NbtSlice slice = NbtSlice::Normal; // which vector of an NBT normal, for GX_VA_NRM
};

struct VertexLayout {
  std::vector<Attribute> attributes;
  uint32_t stride = 0;
};

struct Programs {
  VertexLayout layout;  // exactly the attributes the vertex program reads
  std::string vertex;   // GLSL
  std::string fragment; // GLSL
};

// All empty for a config these programs don't express yet, with all it needs logged once, under the config's hash.
Programs build_programs(const gx::ShaderConfig& config);

// The backend's own programs. Clears: a full-screen triangle's position at location 0, and the colour in the fragment
// stage's 16-byte push constant.
std::string clear_vertex();
std::string clear_fragment();
// Copies and the present: the triangle's position at 0 and texture coordinates at 1, offset and scaled by the vertex
// stage's push constant; set 0's binding 0 the source.
std::string copy_vertex();
// A copy to format, as tex_copy_conv.cpp converts it, with alpha read as 1 when opaqueAlpha is set; empty for a format
// it doesn't convert. GX_TF_RGBA8 copies the texels as they are, as the present does.
std::string copy_fragment(GXTexFmt format, bool opaqueAlpha);
bool converts(GXTexFmt format) noexcept;
// Every one of those programs, and whether it's a fragment program, for the recorder.
std::vector<std::pair<std::string, bool>> fixed_programs();

// What finds a program's compiled form: its source's XXH3-64, seeded 0 for a vertex program and 1 for a fragment one.
uint64_t program_hash(std::string_view source, bool fragment) noexcept;

// Any thread: writes config's programs into dir as <hash>.vert and <hash>.frag, the hash in 16 hex digits, each once,
// for the offline step that compiles them. A config build_programs refuses writes nothing. The first call also writes
// the backend's own programs.
void record(const gx::ShaderConfig& config, const char* dir);
} // namespace aurora::gfx::vulkan4xenon::glsl
