// The vulkan4xenon backend's vertex decode: GX vertex data into the layout's stream, value for value as the WGSL from
// gx/shader.cpp fetches it out of its storage buffers.
#include "decode.hpp"

#include "../internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <iterator>
#include <utility>
#include <vector>

#include <xxhash.h>

#include <semaphore>
#include <thread>

namespace aurora::gfx::vulkan4xenon::decode {
namespace {
constexpr Module Log{"aurora::gfx::vulkan4xenon::decode"};

using Pool = std::span<const uint8_t>;

struct Address {
  Pool pool;
  size_t offset;
  bool le;
};

// Zero past the end of the pool, as the WGSL's load_word returns past its buffer.
u32 load_u8(Pool pool, size_t offset) noexcept { return offset < pool.size() ? pool[offset] : 0; }

// A position matrix index's first row, the program's index into its rows (glsl.hpp): the WGSL's index, the byte over
// 3, held to the last matrix as Dawn holds an index past an array's end.
float pnmtx_row(u32 byte) noexcept { return static_cast<float>(std::min<u32>(byte / 3, gx::MaxPnMtx - 1) * 3); }

u32 load_u16(Pool pool, size_t offset, bool le) noexcept {
  const u32 b0 = load_u8(pool, offset);
  const u32 b1 = load_u8(pool, offset + 1);
  return le ? b0 | b1 << 8 : b0 << 8 | b1;
}

u32 load_u24(Pool pool, size_t offset, bool le) noexcept {
  const u32 b0 = load_u8(pool, offset);
  const u32 b1 = load_u8(pool, offset + 1);
  const u32 b2 = load_u8(pool, offset + 2);
  return le ? b0 | b1 << 8 | b2 << 16 : b0 << 16 | b1 << 8 | b2;
}

u32 load_u32(Pool pool, size_t offset, bool le) noexcept {
  const u32 lo = load_u16(pool, offset, le);
  const u32 hi = load_u16(pool, offset + 2, le);
  return le ? lo | hi << 16 : lo << 16 | hi;
}

// The WGSL's attr_address: indexed attributes read their index from the vertex, then their value from the array.
Address attr_address(const gx::AttrConfig& mapping, u32 attr, size_t vertex, u32 dlExtra, u32 within, Pool verts,
                     Pool storage, const gx::DrawImmediateData& immediates) noexcept {
  const size_t dlOffset = vertex + mapping.offset + dlExtra;
  if (mapping.attrType == GX_INDEX8 || mapping.attrType == GX_INDEX16) {
    const u32 index = mapping.attrType == GX_INDEX8 ? load_u8(verts, dlOffset) : load_u16(verts, dlOffset, false);
    return {storage, size_t{immediates.arrayStart[attr - GX_VA_POS]} + size_t{index} * mapping.stride + within,
            mapping.le};
  }
  return {verts, dlOffset + within, false};
}

u32 comp_size(u8 compType) noexcept {
  switch (compType) {
  case GX_U8:
  case GX_S8:
    return 1;
  case GX_U16:
  case GX_S16:
    return 2;
  default:
    return 4;
  }
}

// fetch_u8 to fetch_f32: the frac scale is the WGSL's division by f32(1u << frac).
float fetch_component(const Address& address, u32 component, u8 compType, u8 frac) noexcept {
  const size_t offset = address.offset + component * comp_size(compType);
  const float scale = static_cast<float>(1u << (frac & 31));
  switch (compType) {
  case GX_U8:
    return static_cast<float>(load_u8(address.pool, offset)) / scale;
  case GX_S8:
    return static_cast<float>(static_cast<int8_t>(load_u8(address.pool, offset))) / scale;
  case GX_U16:
    return static_cast<float>(load_u16(address.pool, offset, address.le)) / scale;
  case GX_S16:
    return static_cast<float>(static_cast<int16_t>(load_u16(address.pool, offset, address.le))) / scale;
  default:
    return std::bit_cast<float>(load_u32(address.pool, offset, address.le));
  }
}

// A channel of max levels as the nearest of 256; the WGSL keeps it as the float level / max.
uint8_t expand(u32 level, u32 max) noexcept { return static_cast<uint8_t>((level * 255 + max / 2) / max); }

// fetch_rgb565 to fetch_rgba8, as red, green, blue and alpha.
void fetch_color(const Address& address, u8 compType, uint8_t out[4]) noexcept {
  const auto byte = [&](u32 i) { return static_cast<uint8_t>(load_u8(address.pool, address.offset + i)); };
  switch (compType) {
  case GX_RGB565: {
    const u32 v = load_u16(address.pool, address.offset, address.le);
    out[0] = expand((v >> 11) & 0x1F, 0x1F);
    out[1] = expand((v >> 5) & 0x3F, 0x3F);
    out[2] = expand(v & 0x1F, 0x1F);
    out[3] = 255;
    break;
  }
  case GX_RGB8:
  case GX_RGBX8:
    out[0] = byte(0);
    out[1] = byte(1);
    out[2] = byte(2);
    out[3] = 255;
    break;
  case GX_RGBA4: {
    const u32 v = load_u16(address.pool, address.offset, address.le);
    out[0] = expand((v >> 12) & 0xF, 0xF);
    out[1] = expand((v >> 8) & 0xF, 0xF);
    out[2] = expand((v >> 4) & 0xF, 0xF);
    out[3] = expand(v & 0xF, 0xF);
    break;
  }
  case GX_RGBA6: {
    const u32 v = load_u24(address.pool, address.offset, address.le);
    out[0] = expand((v >> 18) & 0x3F, 0x3F);
    out[1] = expand((v >> 12) & 0x3F, 0x3F);
    out[2] = expand((v >> 6) & 0x3F, 0x3F);
    out[3] = expand(v & 0x3F, 0x3F);
    break;
  }
  default:
    for (u32 i = 0; i < 4; ++i) {
      out[i] = byte(i);
    }
    break;
  }
}

// The output is the vertex cache, write-combined memory the GPU reads, so a word at a time and volatile: GCC merges
// plain stores into 8-byte ones, which raise an alignment exception there unless they're 8-byte aligned (xsdk's
// CLAUDE.md; vulkan4xenon's docs/SUPPORT.md, Memory and buffers). Every value starts at a multiple of 4.
void write_floats(uint8_t* out, const float* values, size_t count) noexcept {
  auto* words = reinterpret_cast<volatile uint32_t*>(out);
  for (size_t i = 0; i < count; ++i) {
    words[i] = std::bit_cast<uint32_t>(values[i]);
  }
}

// A colour as VK_FORMAT_A8B8G8R8_UNORM_PACK32 holds it: one word in the host's order, red in its low byte, which is
// the bytes red to alpha on a little-endian host and alpha to red on the Xbox 360.
void store_color(uint8_t* out, const uint8_t rgba[4]) noexcept {
  *reinterpret_cast<volatile uint32_t*>(out) =
      u32{rgba[0]} | u32{rgba[1]} << 8 | u32{rgba[2]} << 16 | u32{rgba[3]} << 24;
}

// The bytes fetch_color reads.
u32 color_size(u8 compType) noexcept {
  switch (compType) {
  case GX_RGB565:
  case GX_RGBA4:
    return 2;
  case GX_RGB8:
  case GX_RGBX8:
  case GX_RGBA6:
    return 3;
  default:
    return 4;
  }
}

// Where each indexed attribute's array ends, and whether every element read so far lay wholly before its end.
struct Reach {
  const ArrayEnds& ends;
  bool inside = true;
};

void note_reach(Reach& reach, const gx::AttrConfig& mapping, u32 attr, const Address& address, size_t size) noexcept {
  if (attr >= GX_VA_POS && (mapping.attrType == GX_INDEX8 || mapping.attrType == GX_INDEX16) &&
      address.offset + size > reach.ends[attr - GX_VA_POS]) {
    reach.inside = false;
  }
}

// One attribute of the vertex at byte vertex of verts, written to value: the decode for any layout, and the plan's for
// array elements past the end of their array or of the storage pool.
void decode_attribute(const gx::ShaderConfig& config, const Attribute& attribute,
                      const gx::DrawImmediateData& immediates, size_t vertex, Pool verts, Pool storage, Reach& reach,
                      uint8_t* value) noexcept {
  const auto& mapping = config.attrs[attribute.attr];
  if (attribute.attr == GX_VA_PNMTXIDX) {
    const auto address = attr_address(mapping, attribute.attr, vertex, 0, 0, verts, storage, immediates);
    const float row = pnmtx_row(load_u8(address.pool, address.offset));
    write_floats(value, &row, 1);
  } else if (attribute.attr >= GX_VA_TEX0MTXIDX && attribute.attr <= GX_VA_TEX7MTXIDX) {
    // The matrix's slot, a position matrix's index or MaxPnMtx plus a texture matrix's, as the WGSL works them out;
    // as its first row, the slot times 3.
    const auto address = attr_address(mapping, attribute.attr, vertex, 0, 0, verts, storage, immediates);
    const u32 row = load_u8(address.pool, address.offset);
    const float slot = static_cast<float>(row >= GX_TEXMTX0 ? gx::MaxPnMtx + (row - GX_TEXMTX0) / 3 : row / 3);
    const float first = slot * 3.f;
    write_floats(value, &first, 1);
  } else if (attribute.attr == GX_VA_CLR0 || attribute.attr == GX_VA_CLR1) {
    const auto address = attr_address(mapping, attribute.attr, vertex, 0, 0, verts, storage, immediates);
    note_reach(reach, mapping, attribute.attr, address, color_size(mapping.compType));
    uint8_t rgba[4];
    fetch_color(address, mapping.compType, rgba);
    store_color(value, rgba);
  } else if (attribute.attr == GX_VA_NRM) {
    // attr_load_nbt_slice: an NBT3 vertex has an index for each vector.
    const u32 slice = static_cast<u32>(attribute.slice);
    u32 dlExtra = 0;
    if (mapping.nbt3) {
      dlExtra = mapping.attrType == GX_INDEX8 ? slice : mapping.attrType == GX_INDEX16 ? slice * 2 : 0;
    }
    const u32 within = slice * 3 * comp_size(mapping.compType);
    const auto address = attr_address(mapping, attribute.attr, vertex, dlExtra, within, verts, storage, immediates);
    note_reach(reach, mapping, attribute.attr, address, 3 * comp_size(mapping.compType));
    float values[3];
    for (u32 c = 0; c < 3; ++c) {
      values[c] = fetch_component(address, c, mapping.compType, mapping.frac);
    }
    write_floats(value, values, 3);
  } else {
    // Positions are three components and texture coordinates two; the WGSL fills a missing one with zero.
    const auto address = attr_address(mapping, attribute.attr, vertex, 0, 0, verts, storage, immediates);
    const u32 count = attribute.attr == GX_VA_POS ? 3 : 2;
    note_reach(reach, mapping, attribute.attr, address,
               std::min<u32>(mapping.cnt, count) * comp_size(mapping.compType));
    float values[3]{};
    for (u32 c = 0; c < mapping.cnt && c < count; ++c) {
      values[c] = fetch_component(address, c, mapping.compType, mapping.frac);
    }
    write_floats(value, values, count);
  }
}

void decode_any(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
                Pool storage, Reach& reach, uint8_t* out) noexcept {
  const auto& immediates = draw.immediateData;
  for (u32 v = 0; v < draw.vtxCount; ++v) {
    const size_t vertex = size_t{immediates.vtxStart} + size_t{v} * config.vtxStride;
    uint8_t* dst = out + size_t{v} * layout.stride;
    for (const auto& attribute : layout.attributes) {
      decode_attribute(config, attribute, immediates, vertex, verts, storage, reach, dst + attribute.offset);
    }
  }
}

// decode_any for the listed vertices only, packed.
void decode_any_firsts(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
                       Pool storage, std::span<const uint16_t> firsts, Reach& reach, uint8_t* out) noexcept {
  const auto& immediates = draw.immediateData;
  for (const uint16_t first : firsts) {
    const size_t vertex = size_t{immediates.vtxStart} + size_t{first} * config.vtxStride;
    for (const auto& attribute : layout.attributes) {
      decode_attribute(config, attribute, immediates, vertex, verts, storage, reach, out + attribute.offset);
    }
    out += layout.stride;
  }
}

// The attribute formats games use most, each decoded without branching on the format. Others take decode_any, which
// gives the same bytes.
enum class Op : uint8_t {
  PnmtxDirect,
  S16x3Be,
  S16x3Le,
  F32x3Be,
  F32x3Le,
  S8x3,
  S16x2Be,
  S16x2Le,
  Rgba8,
};

struct Step {
  Op op;
  uint8_t inOffset;   // the attribute's byte, or its index, in the GX vertex
  uint16_t outOffset; // in the decoded vertex
  uint32_t stride;    // between array elements
  size_t arrayStart;  // in the storage pool
  // Indices below it reach elements wholly inside both the attribute's array and the storage span; the others take
  // decode_attribute, which reads zero past the span byte by byte, as the WGSL does, and notes a read past the array.
  uint32_t indexLimit;
  float scale; // 2^-frac
  const Attribute* attribute;
};

// Each position in the vertex gets its own copy of the step code (run_steps), so the plan stops at 8 attributes, and
// longer layouts take decode_any.
constexpr size_t MaxSteps = 8;

bool op_for(const gx::AttrConfig& mapping, const Attribute& attribute, Op& op) noexcept {
  if (attribute.attr == GX_VA_PNMTXIDX) {
    op = Op::PnmtxDirect;
    return mapping.attrType == GX_DIRECT;
  }
  if (mapping.attrType != GX_INDEX16 || attribute.slice != NbtSlice::Normal) {
    return false;
  }
  if (attribute.attr == GX_VA_CLR0 || attribute.attr == GX_VA_CLR1) {
    op = Op::Rgba8;
    return mapping.compType == GX_RGBA8;
  }
  if (attribute.attr >= GX_VA_TEX0 && attribute.attr <= GX_VA_TEX7) {
    op = mapping.le ? Op::S16x2Le : Op::S16x2Be;
    return mapping.cnt == 2 && mapping.compType == GX_S16;
  }
  if ((attribute.attr != GX_VA_POS && attribute.attr != GX_VA_NRM) || mapping.cnt != 3) {
    return false;
  }
  switch (mapping.compType) {
  case GX_S16:
    op = mapping.le ? Op::S16x3Le : Op::S16x3Be;
    return true;
  case GX_F32:
    op = mapping.le ? Op::F32x3Le : Op::F32x3Be;
    return true;
  case GX_S8:
    op = Op::S8x3;
    return true;
  default:
    return false;
  }
}

// The bytes an operation reads from an array element.
size_t element_size(Op op) noexcept {
  switch (op) {
  case Op::S16x3Be:
  case Op::S16x3Le:
    return 6;
  case Op::F32x3Be:
  case Op::F32x3Le:
    return 12;
  case Op::S8x3:
    return 3;
  default:
    return 4;
  }
}

// One step per attribute, or false for a draw the plan can't decode exactly.
bool make_plan(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
               Pool storage, const ArrayEnds& ends, std::array<Step, MaxSteps>& steps, size_t& count) noexcept {
  // The vertices' own bytes are read with no bounds checks, so they must all lie inside the span.
  if (size_t{draw.immediateData.vtxStart} + size_t{draw.vtxCount} * config.vtxStride > verts.size() ||
      layout.attributes.size() > MaxSteps) {
    return false;
  }
  count = 0;
  for (const auto& attribute : layout.attributes) {
    const auto& mapping = config.attrs[attribute.attr];
    Step& step = steps[count++];
    // A byte or a u16 index, inside the vertex.
    if (!op_for(mapping, attribute, step.op) ||
        mapping.offset + (step.op == Op::PnmtxDirect ? 1u : 2u) > config.vtxStride) {
      return false;
    }
    step.inOffset = mapping.offset;
    step.outOffset = static_cast<uint16_t>(attribute.offset);
    step.stride = mapping.stride;
    step.scale = std::bit_cast<float>((127u - (mapping.frac & 31u)) << 23);
    step.attribute = &attribute;
    step.arrayStart = 0;
    step.indexLimit = 0;
    if (step.op == Op::PnmtxDirect) {
      continue;
    }
    step.arrayStart = draw.immediateData.arrayStart[attribute.attr - GX_VA_POS];
    const size_t size = element_size(step.op);
    const size_t end = std::min<size_t>(storage.size(), ends[attribute.attr - GX_VA_POS]);
    if (step.arrayStart + size <= end) {
      // An element is inside when arrayStart + index * stride + size <= end; u16 indices stop at 0xFFFF.
      const size_t inside = step.stride == 0 ? 0x10000 : (end - step.arrayStart - size) / step.stride + 1;
      step.indexLimit = static_cast<uint32_t>(std::min<size_t>(inside, 0x10000));
    }
  }
  return true;
}

inline u32 be16(const uint8_t* p) noexcept { return u32{p[0]} << 8 | p[1]; }
inline u32 le16(const uint8_t* p) noexcept { return u32{p[1]} << 8 | p[0]; }
inline u32 be32(const uint8_t* p) noexcept { return be16(p) << 16 | be16(p + 2); }
inline u32 le32(const uint8_t* p) noexcept { return le16(p + 2) << 16 | le16(p); }

// Multiplying by 2^-frac gives the same floats as the WGSL's division by 2^frac: both are exact, as the scaled
// integers stay normal.
inline float s16(u32 raw, float scale) noexcept { return static_cast<float>(static_cast<int16_t>(raw)) * scale; }

// One attribute of the vertex at byte vertex of verts.
ALWAYS_INLINE void run_step(const Step& step, const gx::ShaderConfig& config, const gx::DrawImmediateData& immediates,
                            size_t vertex, Pool verts, Pool storage, Reach& reach, uint8_t* out) noexcept {
  const uint8_t* in = verts.data() + vertex;
  uint8_t* value = out + step.outOffset;
  float f[3];
  if (step.op == Op::PnmtxDirect) {
    f[0] = pnmtx_row(in[step.inOffset]);
    write_floats(value, f, 1);
    return;
  }
  const u32 index = be16(in + step.inOffset);
  if (index >= step.indexLimit) [[unlikely]] {
    decode_attribute(config, *step.attribute, immediates, vertex, verts, storage, reach, value);
    return;
  }
  const uint8_t* src = storage.data() + step.arrayStart + size_t{index} * step.stride;
  switch (step.op) {
  case Op::S16x3Be:
    f[0] = s16(be16(src), step.scale);
    f[1] = s16(be16(src + 2), step.scale);
    f[2] = s16(be16(src + 4), step.scale);
    write_floats(value, f, 3);
    break;
  case Op::S16x3Le:
    f[0] = s16(le16(src), step.scale);
    f[1] = s16(le16(src + 2), step.scale);
    f[2] = s16(le16(src + 4), step.scale);
    write_floats(value, f, 3);
    break;
  case Op::F32x3Be:
    f[0] = std::bit_cast<float>(be32(src));
    f[1] = std::bit_cast<float>(be32(src + 4));
    f[2] = std::bit_cast<float>(be32(src + 8));
    write_floats(value, f, 3);
    break;
  case Op::F32x3Le:
    f[0] = std::bit_cast<float>(le32(src));
    f[1] = std::bit_cast<float>(le32(src + 4));
    f[2] = std::bit_cast<float>(le32(src + 8));
    write_floats(value, f, 3);
    break;
  case Op::S8x3:
    f[0] = static_cast<float>(static_cast<int8_t>(src[0])) * step.scale;
    f[1] = static_cast<float>(static_cast<int8_t>(src[1])) * step.scale;
    f[2] = static_cast<float>(static_cast<int8_t>(src[2])) * step.scale;
    write_floats(value, f, 3);
    break;
  case Op::S16x2Be:
    f[0] = s16(be16(src), step.scale);
    f[1] = s16(be16(src + 2), step.scale);
    write_floats(value, f, 2);
    break;
  case Op::S16x2Le:
    f[0] = s16(le16(src), step.scale);
    f[1] = s16(le16(src + 2), step.scale);
    write_floats(value, f, 2);
    break;
  case Op::Rgba8:
    store_color(value, src);
    break;
  case Op::PnmtxDirect: // decoded above, from the vertex itself
    break;
  }
}

// Steps I onwards, each position with its own copy of run_step, so the jump into its switch goes to the same case for
// every vertex of the draw, where a single switch for every position would jump to a different case at almost every
// attribute.
template <size_t I>
ALWAYS_INLINE void run_steps(const std::array<Step, MaxSteps>& steps, size_t count, const gx::ShaderConfig& config,
                             const gx::DrawImmediateData& immediates, size_t vertex, Pool verts, Pool storage,
                             Reach& reach, uint8_t* out) noexcept {
  if constexpr (I < MaxSteps) {
    if (I < count) {
      run_step(steps[I], config, immediates, vertex, verts, storage, reach, out);
      run_steps<I + 1>(steps, count, config, immediates, vertex, verts, storage, reach, out);
    }
  }
}

void run_plan(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
              Pool storage, const std::array<Step, MaxSteps>& steps, size_t count, Reach& reach,
              uint8_t* out) noexcept {
  size_t vertex = draw.immediateData.vtxStart;
  for (u32 v = 0; v < draw.vtxCount; ++v, vertex += config.vtxStride, out += layout.stride) {
    run_steps<0>(steps, count, config, draw.immediateData, vertex, verts, storage, reach, out);
  }
}

// run_plan for the listed vertices only, packed.
void run_plan_firsts(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
                     Pool storage, const std::array<Step, MaxSteps>& steps, size_t count,
                     std::span<const uint16_t> firsts, Reach& reach, uint8_t* out) noexcept {
  const size_t start = draw.immediateData.vtxStart;
  for (const uint16_t first : firsts) {
    run_steps<0>(steps, count, config, draw.immediateData, start + size_t{first} * config.vtxStride, verts, storage,
                 reach, out);
    out += layout.stride;
  }
}
} // namespace

// Layouts of the planned formats take a plan of one operation per attribute, chosen once for the draw, rather than
// branching on each attribute's format for every vertex. Other layouts keep that per-vertex decode.
bool decode_vertices(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
                     Pool storage, const ArrayEnds& ends, uint8_t* out) noexcept {
  std::array<Step, MaxSteps> steps;
  size_t count = 0;
  Reach reach{ends};
  if (make_plan(config, layout, draw, verts, storage, ends, steps, count)) {
    run_plan(config, layout, draw, verts, storage, steps, count, reach, out);
  } else {
    decode_any(config, layout, draw, verts, storage, reach, out);
  }
  return reach.inside;
}

void decode_vertices(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
                     Pool storage, uint8_t* out) noexcept {
  decode_vertices(config, layout, draw, verts, storage, UnknownArrayEnds, out);
}

// A vertex decodes from nothing but its own bytes and what the whole draw shares (the arrays, the pools, the config and
// the layout), so a repeat would decode to its first vertex's bytes, and read what it reads.
bool decode_firsts(const gx::ShaderConfig& config, const VertexLayout& layout, const gx::DrawData& draw, Pool verts,
                   Pool storage, const ArrayEnds& ends, std::span<const uint16_t> firsts, uint8_t* out) noexcept {
  std::array<Step, MaxSteps> steps;
  size_t count = 0;
  Reach reach{ends};
  if (make_plan(config, layout, draw, verts, storage, ends, steps, count)) {
    run_plan_firsts(config, layout, draw, verts, storage, steps, count, firsts, reach, out);
  } else {
    decode_any_firsts(config, layout, draw, verts, storage, firsts, reach, out);
  }
  return reach.inside;
}

IndexList build_index_list(std::span<const uint8_t> vertices, uint32_t stride, uint32_t count) {
  IndexList list;
  list.places.resize(count);
  // Open addressing on the vertices' bytes, each slot a place plus one, or zero; at most half full.
  size_t slots = 16;
  while (slots < size_t{count} * 2) {
    slots *= 2;
  }
  std::vector<uint32_t> table(slots, 0);
  for (uint32_t v = 0; v < count; ++v) {
    const uint8_t* bytes = vertices.data() + size_t{v} * stride;
    size_t slot = XXH3_64bits(bytes, stride) & (slots - 1);
    while (table[slot] != 0 &&
           std::memcmp(vertices.data() + size_t{list.firsts[table[slot] - 1]} * stride, bytes, stride) != 0) {
      slot = (slot + 1) & (slots - 1);
    }
    if (table[slot] == 0) {
      table[slot] = static_cast<uint32_t>(list.firsts.size()) + 1;
      list.firsts.push_back(static_cast<uint16_t>(v));
    }
    list.places[v] = static_cast<uint16_t>(table[slot] - 1);
  }
  return list;
}

const IndexList& IndexLists::get(uint64_t hash, std::span<const uint8_t> vertices, uint32_t stride, uint32_t count) {
  const DecodeKey key{hash, uint64_t{count} | uint64_t{stride} << 32};
  auto& kept = m_lists[key];
  if (!kept) {
    kept = std::make_unique<Kept>(Kept{build_index_list(vertices, stride, count), 0});
    m_bytes += kept_bytes(kept->list);
    ++m_built;
  }
  kept->lastFrame = m_frame;
  return kept->list;
}

void IndexLists::begin_frame() noexcept {
  ++m_frame;
  if (m_bytes <= m_capacity) {
    return;
  }
  // The lists used longest ago go, down to three quarters of the capacity, so this runs now and then, not every frame.
  std::vector<std::pair<uint64_t, DecodeKey>> byAge;
  byAge.reserve(m_lists.size());
  for (const auto& [key, kept] : m_lists) {
    byAge.emplace_back(kept->lastFrame, key);
  }
  std::ranges::sort(byAge, {}, &std::pair<uint64_t, DecodeKey>::first);
  for (const auto& [lastFrame, key] : byAge) {
    if (m_bytes <= m_capacity / 4 * 3) {
      break;
    }
    const auto it = m_lists.find(key);
    m_bytes -= kept_bytes(it->second->list);
    m_lists.erase(it);
  }
}

uint64_t IndexLists::kept_bytes(const IndexList& list) noexcept {
  return (list.places.size() + list.firsts.size()) * sizeof(uint16_t) + sizeof(Kept);
}

namespace {
// A pass too short to pay for waking the helper and waiting for it isn't shared, and neither is a short range of
// hashing.
constexpr uint64_t MinSharedVertices = 256;
constexpr uint64_t MinSharedHashBytes = 12 * 1024;

// Items the caller and the helper take in turn: one that starts late, or whose core is busy, takes fewer. take takes
// items until none are left and returns what it did, for the stats.
struct SharedWork {
  uint64_t (*take)(void* work) noexcept = nullptr;
  void* work = nullptr;
};

// A pass's draws to decode.
struct PassWork {
  std::span<DecodeJob> jobs;
  Pool verts;
  Pool storage;
  std::atomic<size_t> next{0};
};

// Decodes draws until none are left to take, and returns their vertices.
uint64_t decode_taken(void* work) noexcept {
  auto& pass = *static_cast<PassWork*>(work);
  uint64_t vertices = 0;
  for (size_t i = pass.next.fetch_add(1, std::memory_order_relaxed); i < pass.jobs.size();
       i = pass.next.fetch_add(1, std::memory_order_relaxed)) {
    DecodeJob& job = pass.jobs[i];
    if (job.list != nullptr) {
      // The GPU's copy of the list, before the vertices it names.
      const auto& places = job.list->places;
      std::memcpy(job.out, places.data(), places.size() * sizeof(uint16_t));
      job.inside = decode_firsts(*job.config, *job.layout, *job.draw, pass.verts, pass.storage, job.ends,
                                 job.list->firsts, job.out + listed_indices_size(job.draw->vtxCount));
      vertices += job.list->firsts.size();
    } else {
      job.inside = decode_vertices(*job.config, *job.layout, *job.draw, pass.verts, pass.storage, job.ends, job.out);
      vertices += job.draw->vtxCount;
    }
  }
  return vertices;
}

// Ranges to hash.
struct HashWork {
  std::span<HashRange> ranges;
  std::atomic<size_t> next{0};
};

uint64_t hash_taken(void* work) noexcept {
  auto& hashes = *static_cast<HashWork*>(work);
  uint64_t bytes = 0;
  for (size_t i = hashes.next.fetch_add(1, std::memory_order_relaxed); i < hashes.ranges.size();
       i = hashes.next.fetch_add(1, std::memory_order_relaxed)) {
    HashRange& range = hashes.ranges[i];
    range.hash = XXH3_64bits(range.data, range.size);
    bytes += range.size;
  }
  return bytes;
}

// A wake-up between the two threads. The semaphore orders each thread's memory before the other's.
class Event {
public:
  bool valid() const noexcept { return true; }
  void post() noexcept { m_sema.release(); }
  void wait() noexcept { m_sema.acquire(); }

private:
  std::binary_semaphore m_sema{0};
};

// The helper thread: woken with shared work, it takes items until none are left, then says so.
class Helper {
public:
  Helper() noexcept : m_thread([this] { run(); }) {}

  ~Helper() {
    if (!running()) {
      return;
    }
    m_work = {};
    m_start.post();
    m_thread.join();
  }

  bool running() const noexcept { return m_thread.joinable(); }

  void begin(SharedWork work) noexcept {
    m_work = work;
    m_start.post();
  }

  // Waits for the helper to finish its share, and returns what its take returned.
  uint64_t end() noexcept {
    m_done.wait();
    return m_taken;
  }

private:
  void run() noexcept {
    while (true) {
      m_start.wait();
      if (m_work.take == nullptr) {
        return;
      }
      m_taken = m_work.take(m_work.work);
      m_done.post();
    }
  }

  Event m_start;
  Event m_done;
  SharedWork m_work;
  uint64_t m_taken = 0;
  std::thread m_thread;
};

HelperStats g_stats;

// Started by the first pass, on the render worker.
Helper& helper() noexcept {
  static Helper helper;
  return helper;
}

// Runs the work on the caller and, when it's worth waking, the helper; returns the helper's share.
uint64_t run_shared(SharedWork work, bool share) noexcept {
  if (!share || !helper().running()) {
    work.take(work.work);
    return 0;
  }
  helper().begin(work);
  work.take(work.work);
  return helper().end();
}
} // namespace

void decode_pass(std::span<DecodeJob> jobs, Pool verts, Pool storage) noexcept {
  PassWork work{.jobs = jobs, .verts = verts, .storage = storage};
  uint64_t vertices = 0;
  for (const DecodeJob& job : jobs) {
    vertices += job.list != nullptr ? job.list->firsts.size() : job.draw->vtxCount;
  }
  const bool share = jobs.size() >= 2 && vertices >= MinSharedVertices;
  const uint64_t helperVertices = run_shared({decode_taken, &work}, share);
  if (share && helper().running()) {
    ++g_stats.passes;
    g_stats.vertices += vertices;
    g_stats.helperVertices += helperVertices;
  }
}

HelperStats take_helper_stats() noexcept { return std::exchange(g_stats, HelperStats{}); }

void hash_ranges(std::span<HashRange> ranges) noexcept {
  HashWork work{.ranges = ranges};
  uint64_t bytes = 0;
  for (const HashRange& range : ranges) {
    bytes += range.size;
  }
  run_shared({hash_taken, &work}, ranges.size() >= 2 && bytes >= MinSharedHashBytes);
}

namespace {
template <typename T>
void put(std::vector<uint8_t>& out, const T& value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

// Writes value's bytes at out, and returns where they end.
template <typename T>
uint8_t* write_value(uint8_t* out, const T& value) noexcept {
  std::memcpy(out, &value, sizeof(T));
  return out + sizeof(T);
}

bool indexed(const gx::AttrConfig& mapping, u32 attr) noexcept {
  return attr >= GX_VA_POS && (mapping.attrType == GX_INDEX8 || mapping.attrType == GX_INDEX16);
}
} // namespace

uint64_t decode_format(const gx::ShaderConfig& config, const VertexLayout& layout) noexcept {
  std::vector<uint8_t> bytes;
  put(bytes, config.vtxStride);
  put(bytes, layout.stride);
  for (const auto& attribute : layout.attributes) {
    put(bytes, static_cast<uint32_t>(attribute.attr));
    put(bytes, attribute.slice);
    put(bytes, attribute.offset);
    put(bytes, config.attrs[attribute.attr]);
  }
  return XXH3_64bits(bytes.data(), bytes.size());
}

void VertexKeys::compute(std::span<const KeyedDraw> draws, Pool verts, Pool storage,
                         std::span<std::optional<DecodeKey>> keys, std::span<ArrayEnds> ends, bool poolHashes) {
  m_ranges.clear();
  m_vertexRanges.clear();
  m_vertexHashes.assign(draws.size(), 0);
  m_drawArrays.clear();
  m_drawArraysStart.clear();
  // What to hash: each draw's vertex bytes, and each array the frame hasn't hashed yet, whole. The size push_array put
  // before an array says where it ends; one that says it passes the pool gets no hash, nor do its draws keys.
  for (const KeyedDraw& keyed : draws) {
    m_drawArraysStart.push_back(static_cast<uint32_t>(m_drawArrays.size()));
    const auto& draw = *keyed.draw;
    const size_t start = draw.immediateData.vtxStart;
    const size_t size = size_t{draw.vtxCount} * keyed.config->vtxStride;
    if (start + size > verts.size()) {
      m_vertexRanges.push_back(NoRange);
    } else if (poolHashes && draw.vertHash != 0 && draw.vertRange.offset == start && draw.vertRange.size == size) {
      m_vertexHashes[m_vertexRanges.size()] = draw.vertHash;
      m_vertexRanges.push_back(PoolHash);
    } else {
      m_vertexRanges.push_back(static_cast<uint32_t>(m_ranges.size()));
      m_ranges.push_back({verts.data() + start, size, 0});
    }
    for (const auto& attribute : keyed.layout->attributes) {
      if (!indexed(keyed.config->attrs[attribute.attr], attribute.attr)) {
        continue;
      }
      const uint32_t arrayStart = draw.immediateData.arrayStart[attribute.attr - GX_VA_POS];
      const auto [it, added] = m_arrayPlaces.try_emplace(arrayStart, static_cast<uint32_t>(m_arrays.size()));
      m_drawArrays.push_back(it->second);
      if (!added) {
        continue;
      }
      Array& array = m_arrays.emplace_back(Array{0, NoRange, 0});
      uint32_t arraySize = 0;
      if (arrayStart >= sizeof(arraySize) && arrayStart <= storage.size()) {
        std::memcpy(&arraySize, storage.data() + arrayStart - sizeof(arraySize), sizeof(arraySize));
        if (arraySize <= storage.size() - arrayStart && poolHashes && arrayStart >= sizeof(arraySize) + sizeof(u64)) {
          array = {arrayStart + arraySize, Hashed, 0};
          std::memcpy(&array.hash, storage.data() + arrayStart - sizeof(arraySize) - sizeof(u64), sizeof(u64));
        } else if (arraySize <= storage.size() - arrayStart) {
          array = {arrayStart + arraySize, static_cast<uint32_t>(m_ranges.size()), 0};
          m_ranges.push_back({storage.data() + arrayStart, arraySize, 0});
        }
      }
    }
  }
  hash_ranges(m_ranges);
  for (size_t i = 0; i < draws.size(); ++i) {
    if (m_vertexRanges[i] != NoRange && m_vertexRanges[i] != PoolHash) {
      m_vertexHashes[i] = m_ranges[m_vertexRanges[i]].hash;
    }
  }
  for (Array& array : m_arrays) {
    if (array.range != NoRange && array.range != Hashed) {
      array.hash = m_ranges[array.range].hash;
      array.range = Hashed;
    }
  }
  // Each draw's key: its format, vertex count and those hashes, by attribute. The bytes are written in place, where a
  // vector's inserts took 15 calls a draw.
  for (size_t i = 0; i < draws.size(); ++i) {
    const KeyedDraw& keyed = draws[i];
    keys[i].reset();
    ends[i] = UnknownArrayEnds;
    if (m_vertexRanges[i] == NoRange) {
      continue;
    }
    const size_t most = sizeof(keyed.format) + sizeof(keyed.draw->vtxCount) + sizeof(uint64_t) +
                        keyed.layout->attributes.size() * (2 * sizeof(uint32_t) + sizeof(uint64_t)) + 1;
    if (m_keyBytes.size() < most) {
      m_keyBytes.resize(most);
    }
    uint8_t* out = write_value(m_keyBytes.data(), keyed.format);
    out = write_value(out, keyed.draw->vtxCount);
    out = write_value(out, m_vertexHashes[i]);
    const uint32_t* places = m_drawArrays.data() + m_drawArraysStart[i];
    bool known = true;
    for (const auto& attribute : keyed.layout->attributes) {
      if (!indexed(keyed.config->attrs[attribute.attr], attribute.attr)) {
        continue;
      }
      const uint32_t arrayStart = keyed.draw->immediateData.arrayStart[attribute.attr - GX_VA_POS];
      const Array& array = m_arrays[*places++];
      known = known && array.range == Hashed;
      ends[i][attribute.attr - GX_VA_POS] = array.end;
      out = write_value(out, static_cast<uint32_t>(attribute.attr));
      out = write_value(out, array.end - arrayStart);
      out = write_value(out, array.hash);
    }
    // A listed draw's entry holds its list and its first vertices, not every vertex. A draw that isn't keeps the key
    // bytes it always had.
    if (keyed.listed) {
      out = write_value(out, uint8_t{1});
    }
    if (known) {
      const XXH128_hash_t hash = XXH3_128bits(m_keyBytes.data(), out - m_keyBytes.data());
      keys[i] = DecodeKey{hash.low64, hash.high64};
    }
  }
}

VertexCache::VertexCache(uint64_t capacity, uint64_t framesInFlight, FrameWaiter waitForFrame)
: m_framesInFlight(framesInFlight), m_waitForFrame(std::move(waitForFrame)) {
  release(0, capacity);
}

void VertexCache::begin_frame(uint64_t frame) noexcept {
  m_frame = frame;
  // The frame's slot was taken once its last frame's fence signalled (backend::begin_frame).
  if (frame > m_framesInFlight) {
    m_done = std::max(m_done, frame - m_framesInFlight);
  }
  m_stats.peakFrameBytes = std::max(m_stats.peakFrameBytes, m_frameBytes);
  m_frameBytes = 0;
}

std::optional<VertexCache::Place> VertexCache::find(const DecodeKey& key) noexcept {
  const auto it = m_byKey.find(key);
  if (it == m_byKey.end()) {
    return std::nullopt;
  }
  return use(it->second, m_entries[it->second].size);
}

std::optional<VertexCache::Place> VertexCache::place(const std::optional<DecodeKey>& key, uint64_t size) {
  if (key) {
    if (const auto it = m_byKey.find(*key); it != m_byKey.end()) {
      return use(it->second, size);
    }
  }
  const uint64_t rounded = (size + 15) & ~uint64_t{15};
  std::optional<uint64_t> offset;
  while (!(offset = allocate(rounded))) {
    if (!evict_oldest() && !wait_for_oldest_frame()) {
      ++m_stats.failures;
      return std::nullopt;
    }
  }
  uint32_t index;
  if (!m_freeEntries.empty()) {
    index = m_freeEntries.back();
    m_freeEntries.pop_back();
  } else {
    index = static_cast<uint32_t>(m_entries.size());
    m_entries.emplace_back();
  }
  m_entries[index] = Entry{.key = key.value_or(DecodeKey{}),
                           .keyed = key.has_value(),
                           .offset = *offset,
                           .size = rounded,
                           .lastFrame = m_frame,
                           .newer = None,
                           .older = None};
  if (key) {
    m_byKey[*key] = index;
  }
  link_newest(index);
  m_frameBytes += rounded;
  ++m_stats.misses;
  m_stats.missBytes += size;
  m_used += rounded;
  m_stats.peakUsedBytes = std::max(m_stats.peakUsedBytes, m_used);
  return Place{*offset, false};
}

void VertexCache::forget(const DecodeKey& key) noexcept {
  if (const auto it = m_byKey.find(key); it != m_byKey.end()) {
    m_entries[it->second].keyed = false;
    m_byKey.erase(it);
    ++m_stats.forgotten;
  }
}

// An entry a draw found: in use by this frame from now.
VertexCache::Place VertexCache::use(uint32_t index, uint64_t size) noexcept {
  if (m_entries[index].lastFrame != m_frame) {
    m_frameBytes += m_entries[index].size;
  }
  touch(index);
  ++m_stats.hits;
  m_stats.hitBytes += size;
  return Place{m_entries[index].offset, true};
}

// Used by this frame: it becomes the newest, so the list runs from this frame's entries back to the oldest's.
void VertexCache::touch(uint32_t index) noexcept {
  if (index != m_newest) {
    unlink(index);
    link_newest(index);
  }
  m_entries[index].lastFrame = m_frame;
}

void VertexCache::link_newest(uint32_t index) noexcept {
  Entry& entry = m_entries[index];
  entry.newer = None;
  entry.older = m_newest;
  if (m_newest != None) {
    m_entries[m_newest].newer = index;
  } else {
    m_oldest = index;
  }
  m_newest = index;
}

void VertexCache::unlink(uint32_t index) noexcept {
  Entry& entry = m_entries[index];
  if (entry.newer != None) {
    m_entries[entry.newer].older = entry.older;
  } else {
    m_newest = entry.older;
  }
  if (entry.older != None) {
    m_entries[entry.older].newer = entry.newer;
  } else {
    m_oldest = entry.newer;
  }
}

// The least recently used entry's space, unless a frame in flight may still draw it: then so may every newer one.
bool VertexCache::evict_oldest() {
  if (m_oldest == None || m_entries[m_oldest].lastFrame > m_done) {
    return false;
  }
  const uint32_t index = m_oldest;
  const Entry& entry = m_entries[index];
  unlink(index);
  if (entry.keyed) {
    m_byKey.erase(entry.key);
  }
  release(entry.offset, entry.size);
  m_used -= entry.size;
  m_freeEntries.push_back(index);
  ++m_stats.evictions;
  return true;
}

// With the least recently used entry pinned: waits for the frame that used it last, the oldest frame in flight that
// holds any, to be done, so it and what older frames held may be evicted. False when there is no waiter, or that frame
// is this one: then every entry left is this frame's own, and waiting can free none of them.
bool VertexCache::wait_for_oldest_frame() {
  if (!m_waitForFrame || m_oldest == None || m_entries[m_oldest].lastFrame >= m_frame) {
    return false;
  }
  const uint64_t frame = m_entries[m_oldest].lastFrame;
  const auto start = std::chrono::steady_clock::now();
  if (!m_waitForFrame(frame)) {
    return false;
  }
  const auto ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
  m_done = frame;
  ++m_stats.waits;
  m_stats.waitNs += ns;
  m_stats.longestWaitNs = std::max(m_stats.longestWaitNs, ns);
  return true;
}

// The smallest free run that fits.
std::optional<uint64_t> VertexCache::allocate(uint64_t size) {
  const auto bySize = m_freeBySize.lower_bound(size);
  if (bySize == m_freeBySize.end()) {
    return std::nullopt;
  }
  const uint64_t runSize = bySize->first;
  const uint64_t offset = bySize->second;
  m_freeBySize.erase(bySize);
  m_freeByOffset.erase(offset);
  if (runSize > size) {
    m_freeByOffset.emplace(offset + size, runSize - size);
    m_freeBySize.emplace(runSize - size, offset + size);
  }
  return offset;
}

// Back into the free runs, joined with its neighbours.
void VertexCache::release(uint64_t offset, uint64_t size) {
  const auto erase_by_size = [&](uint64_t runOffset, uint64_t runSize) {
    const auto [first, last] = m_freeBySize.equal_range(runSize);
    for (auto it = first; it != last; ++it) {
      if (it->second == runOffset) {
        m_freeBySize.erase(it);
        return;
      }
    }
  };
  auto next = m_freeByOffset.lower_bound(offset);
  if (next != m_freeByOffset.end() && next->first == offset + size) {
    size += next->second;
    erase_by_size(next->first, next->second);
    next = m_freeByOffset.erase(next);
  }
  if (next != m_freeByOffset.begin()) {
    const auto previous = std::prev(next);
    if (previous->first + previous->second == offset) {
      offset = previous->first;
      size += previous->second;
      erase_by_size(previous->first, previous->second);
      m_freeByOffset.erase(previous);
    }
  }
  m_freeByOffset.emplace(offset, size);
  m_freeBySize.emplace(size, offset);
}
} // namespace aurora::gfx::vulkan4xenon::decode
