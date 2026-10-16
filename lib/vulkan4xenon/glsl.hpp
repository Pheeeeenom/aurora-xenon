#pragma once
// The vulkan4xenon backend's GLSL programs for a GX shader config. There is no shader compiler on the Xbox 360, so the
// programs are compiled offline, glslang to SPIR-V and xost to Xenos microcode, and found by their source's hash.
//
// What the programs and the backend share:
// - Set 0 is a uniform buffer per uniform piece (gx::UniformPiece), binding i the piece as gx.hpp lays it out, at
//   DrawData::uniformOffsets[i]. A program declares only the pieces it reads, each as std140 vec4s: a matrix is its
//   rows, as Mat3x4 and Mat4x4 hold them.
// - Set 1 is the 8 GX texture maps, binding i GX_TEXMAPi, sampled by the fragment program alone.
// - Set 2, binding 0, for a fragment program with range fog, where Programs' fogRange says: the factor for each column
//   of the target, as gx/command_processor.cpp's build_fog_range_lut works it out, column x at texel
//   (x % FogRangeTableWidth, x / FogRangeTableWidth) of an R32_SFLOAT table, sampled nearest with u repeating and v
//   clamped (encoding.cpp).
// - The vertex stage's push constant is the current position matrix's three rows, 48 bytes, then, where Programs'
//   normalMatrix says, the current normal matrix's, 48 more: the matrices the draw selects. A matrix picked per vertex
//   is read from its piece at its first row, a vertex attribute (decode.cpp), as xost reads a block's array at an
//   index known only at run time (its docs/xsc.md). A line or point program's has, at byte LineMaskOffset, 1 or 0 for
//   each texture coordinate, whether the line's or point's texture offset moves it: the view piece's mask, which xost
//   can't read bit by bit.
// - A line or point draw is drawn as the WebGPU backend's (gx/shader.cpp): each segment or point a quad in screen
//   space. The stream holds six vertices for each, two triangles, written by write_corners: each one of its ends' GX
//   vertex, then its corner's own values (VertexLayout::corners).
// - One vertex stream, binding 0, in the layout below. Locations follow xost's usages: position at 0, colour 0 at 1,
//   texture coordinates from 2 (xost:xsc/resolve.py).
// - The vertex program writes the WGSL's position with y negated, Vulkan's clip space.
//
// The programs are written in the GLSL xost compiles (its docs/xsc.md): each local assigned once, no function calls,
// no unary minus but on a literal, and at most one discard, on one comparison.
#include "../gx/pipeline.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::gfx::vulkan4xenon::glsl {
// The range fog table's size: a 2048-column target's factors in two of Xenos's 32 by 32 tiles, 8 KiB, where one row
// would take 32 rows of tiles all the same.
inline constexpr uint32_t FogRangeSet = 2;
inline constexpr uint32_t FogRangeTableWidth = 64;
inline constexpr uint32_t FogRangeTableHeight = 32;
// Where a line or point program's texture coordinate mask sits in the push constant, after both matrices: two vec4s.
inline constexpr uint32_t LineMaskOffset = 96;
inline constexpr uint32_t LineMaskBytes = 32;

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

// A corner's own values, after its GX vertex's attributes.
enum class CornerValue : uint8_t {
  // vec2. A segment's: x, which side of it the corner is on, -1 or 1, as seen from its own end towards the other; y, 1
  // on the side the texture offset moves, else 0. A point's: which corner, x and y each -1 or 1.
  Corner,
  OtherPos,   // vec3, a segment's: the other end's position
  OtherPnmtx, // float, a segment's with a position matrix per vertex: the other end's matrix's first row
};

struct CornerAttribute {
  uint32_t location;
  AttrFormat format;
  uint32_t offset; // bytes into the vertex
  CornerValue value;
};

struct VertexLayout {
  std::vector<Attribute> attributes;    // a GX vertex's, as decode_vertices writes them
  std::vector<CornerAttribute> corners; // a line or point config's, after them; none for others
  uint32_t stride = 0;
};

// How many vertices a draw of count GX vertices takes in the stream: six for each segment or point of a line or point
// config, one for each vertex otherwise.
uint32_t stream_vertices(const gx::ShaderConfig& config, uint32_t count) noexcept;

// A line or point draw's stream from its count GX vertices, decoded at layout's stride: each segment's or point's six
// corners, written to out, in the CPU's cached memory: its stores aren't the vertex cache's word at a time.
void write_corners(const gx::ShaderConfig& config, const VertexLayout& layout, const uint8_t* decoded, uint32_t count,
                   uint8_t* out) noexcept;

struct Programs {
  VertexLayout layout;  // exactly the attributes the vertex program reads
  std::string vertex;   // GLSL
  std::string fragment; // GLSL
  // The vertex program reads the current normal matrix, pushed after the position matrix.
  bool normalMatrix = false;
  // The fragment program samples the range fog table at set 2.
  bool fogRange = false;
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
// it doesn't convert. GX_TF_RGBA8 copies the texels as they are, as the present does. GX_TF_Z8 and GX_TF_Z16 sample a
// depth image, whose depth the layer returns in red, and take GX's 24-bit z from it; opaqueAlpha means nothing there.
std::string copy_fragment(GXTexFmt format, bool opaqueAlpha);
bool converts(GXTexFmt format) noexcept;
// GXPeekZ's snapshot: GX's 24-bit z of the sampled depth in red, green and blue, its high byte first, and 0 in alpha.
std::string depth_peek_fragment();
// An EFB copy's 4- or 8-bit indices through the palette at set 1; the second push vector is (1, 0, 0.5, 0.5 N) / N.
std::string palette_fragment(bool fourBit);
// Every one of those programs, and whether it's a fragment program, for the recorder.
std::vector<std::pair<std::string, bool>> fixed_programs();

// What finds a program's compiled form: its source's XXH3-64, seeded 0 for a vertex program and 1 for a fragment one.
uint64_t program_hash(std::string_view source, bool fragment) noexcept;

// Any thread: writes config's programs into dir as <hash>.vert and <hash>.frag, the hash in 16 hex digits, each once,
// for the offline step that compiles them. A config build_programs refuses writes nothing. The first call also writes
// the backend's own programs.
void record(const gx::ShaderConfig& config, const char* dir);
} // namespace aurora::gfx::vulkan4xenon::glsl
