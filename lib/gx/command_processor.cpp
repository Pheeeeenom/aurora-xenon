#include "command_processor.hpp"

#include "../gfx/backend.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../internal.hpp"
#include "dolphin/gd/GDGeometry.h"
#include "dolphin/gx/GXAurora.h"
#include "fifo.hpp"
#include "gx.hpp"
#include "pipeline.hpp"
#include "regs.hpp"
#include "shader_info.hpp"
#include "texture.hpp"

#include <absl/container/flat_hash_map.h>
#include <tracy/Tracy.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

// Quad, strip, fan, line and point draws of up to this many vertices take their indices from index_patterns(), not from
// the frame's index pool; 0 builds every draw's into the pool.
#ifndef AURORA_INDEX_PATTERN_VERTICES
#define AURORA_INDEX_PATTERN_VERTICES 0
#endif

namespace aurora::gx::fifo {
namespace {
constexpr Module Log{"aurora::gx::fifo"};
} // namespace

u32 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) noexcept {
  u32 numIndices = 0;
  // Written in place: ByteBuffer::append costs a resize check and a memcpy per index.
  if (prim == GX_QUADS) {
    const u32 quads = vtxCount / 4;
    numIndices = quads * 6;
    u16* out = reinterpret_cast<u16*>(buf.append_uninitialized(numIndices * sizeof(u16)));
    for (u32 q = 0; q < quads; ++q) {
      const u16 idx0 = static_cast<u16>(vtxStart + q * 4);
      out[0] = idx0;
      out[1] = static_cast<u16>(idx0 + 1);
      out[2] = static_cast<u16>(idx0 + 2);
      out[3] = static_cast<u16>(idx0 + 2);
      out[4] = static_cast<u16>(idx0 + 3);
      out[5] = idx0;
      out += 6;
    }
  } else if (prim == GX_TRIANGLES) {
    numIndices = vtxCount;
    u16* out = reinterpret_cast<u16*>(buf.append_uninitialized(numIndices * sizeof(u16)));
    for (u16 v = 0; v < vtxCount; ++v) {
      out[v] = static_cast<u16>(vtxStart + v);
    }
  } else if (prim == GX_TRIANGLEFAN) {
    if (vtxCount < 3) {
      numIndices = vtxCount;
      u16* out = reinterpret_cast<u16*>(buf.append_uninitialized(numIndices * sizeof(u16)));
      for (u16 v = 0; v < vtxCount; ++v) {
        out[v] = static_cast<u16>(vtxStart + v);
      }
    } else {
      numIndices = (u32(vtxCount) - 2) * 3;
      u16* out = reinterpret_cast<u16*>(buf.append_uninitialized(numIndices * sizeof(u16)));
      for (u16 v = 2; v < vtxCount; ++v) {
        const u16 idx = static_cast<u16>(vtxStart + v);
        out[0] = vtxStart;
        out[1] = static_cast<u16>(idx - 1);
        out[2] = idx;
        out += 3;
      }
    }
  } else if (prim == GX_TRIANGLESTRIP) {
    if (vtxCount < 3) {
      numIndices = vtxCount;
      u16* out = reinterpret_cast<u16*>(buf.append_uninitialized(numIndices * sizeof(u16)));
      for (u16 v = 0; v < vtxCount; ++v) {
        out[v] = static_cast<u16>(vtxStart + v);
      }
    } else {
      numIndices = (u32(vtxCount) - 2) * 3;
      u16* out = reinterpret_cast<u16*>(buf.append_uninitialized(numIndices * sizeof(u16)));
      for (u16 v = 2; v < vtxCount; ++v) {
        const u16 idx = static_cast<u16>(vtxStart + v);
        if ((v & 1) == 0) {
          out[0] = static_cast<u16>(idx - 2);
          out[1] = static_cast<u16>(idx - 1);
        } else {
          out[0] = static_cast<u16>(idx - 1);
          out[1] = static_cast<u16>(idx - 2);
        }
        out[2] = idx;
        out += 3;
      }
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

IndexPatterns build_index_patterns(u32 maxVertices) noexcept {
  IndexPatterns patterns;
  patterns.maxVertices = std::min<u32>(maxVertices, UINT16_MAX);
  if (patterns.maxVertices == 0) {
    return patterns;
  }
  constexpr std::array kinds{GX_QUADS, GX_TRIANGLESTRIP, GX_TRIANGLEFAN, GX_LINES};
  for (size_t i = 0; i < kinds.size(); ++i) {
    // Aligned as push_indices aligns a draw's.
    if (patterns.indices.size() % 4 != 0) {
      patterns.indices.append<u16>(0);
    }
    patterns.offsets[i] = patterns.indices.size();
    prepare_idx_buffer(patterns.indices, kinds[i], 0, patterns.maxVertices);
  }
  if (patterns.indices.size() % 4 != 0) {
    patterns.indices.append<u16>(0);
  }
  return patterns;
}

const IndexPatterns& index_patterns() noexcept {
  static const IndexPatterns patterns = build_index_patterns(AURORA_INDEX_PATTERN_VERTICES);
  return patterns;
}

// The index counts are prepare_idx_buffer's.
std::optional<gfx::Range> index_pattern_range(const IndexPatterns& patterns, GXPrimitive prim, u32 vtxCount) noexcept {
  if (vtxCount > patterns.maxVertices || patterns.maxVertices == 0) {
    return std::nullopt;
  }
  u32 pattern = 0;
  u32 numIndices = 0;
  switch (prim) {
  case GX_QUADS:
    numIndices = vtxCount / 4 * 6;
    break;
  case GX_TRIANGLESTRIP:
  case GX_TRIANGLEFAN:
    pattern = prim == GX_TRIANGLESTRIP ? 1 : 2;
    numIndices = vtxCount < 3 ? vtxCount : (vtxCount - 2) * 3;
    break;
  case GX_LINES:
  case GX_LINESTRIP:
  case GX_POINTS:
    pattern = 3;
    numIndices = 6;
    break;
  default:
    return std::nullopt;
  }
  return gfx::Range{patterns.offsets[pattern], static_cast<u32>(numIndices * sizeof(u16))};
}

namespace {
// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
constexpr u8 CP_CMD_NOP = GX_NOP;
constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

struct FogRangeLutKey {
  std::array<u16, 10> rangeK;
  f32 rangeCenter;
  f32 renderWidth;
  u32 targetWidth;

  bool operator==(const FogRangeLutKey&) const = default;
};

struct FogRangeLutEntry {
  FogRangeLutKey key;
  std::vector<f32> factors;
};

constexpr size_t MaxFogRangeLuts = 32;
std::vector<FogRangeLutEntry> sFogRangeLuts;

struct DrawCache {
  uint64_t targetLayoutKey = 0;
  PipelineConfig config{};
  ShaderInfo shaderInfo{};
  gfx::PipelineRef pipelineRef{};
  GXBindGroups bindGroups{};
  uint64_t bindGeneration = 0;
  GXVtxFmt fmt = GX_MAX_VTXFMT;
  u8 lineMode = 0;
  bool hasPipeline = false;
  // Where each uniform piece was last pushed this frame, and the pieces whose state changed since.
  std::array<u32, UniformPieceCount> uniformOffsets{};
  u8 staleUniforms = AllUniformPieces;
  gfx::Range fogRange{};
  FogRangeLutKey fogRangeKey{};
  bool hasFogRange = false;
  GXVtxFmt lastDrawFmt = GX_MAX_VTXFMT;
};
DrawCache sDrawCache;

// A config's shader info depends on the config alone, and its pipeline on the config and the render target layout,
// so each pair is analysed and looked up once, by pipeline_memo_key.
struct PipelineMemo {
  ShaderInfo shaderInfo;
  gfx::PipelineRef pipelineRef;
};
absl::flat_hash_map<uint64_t, PipelineMemo> sPipelineMemo;

// The state populate_pipeline_config (gx.cpp) reads, and the render target layout the pipeline is looked up for, as
// the last rebuild read them. A config built from the same state comes out the same, with the same shader info and
// pipeline.
struct PipelineInputs {
  uint64_t targetLayoutKey = 0;
  u8 lineMode = 0;
  u8 numTevStages = 0;
  u8 numIndStages = 0;
  u8 numTexGens = 0;
  GXFogType fogType = GX_FOG_NONE;
  bool fogRangeEnabled = false;
  GXPixelFmt pixelFmt = GX_PF_RGB8_Z24;
  std::array<GXAttrType, MaxVtxAttr> vtxDesc{};
  VtxFmt vtxFmt{};
  std::array<u16, MaxVtxAttr> indexedArrays{}; // stride, and byte order in bit 8
  std::array<TevSwap, MaxTevSwap> tevSwapTable{};
  std::array<TevStage, MaxTevStages> tevStages{};
  std::array<IndStage, MaxIndStages> indStages{};
  std::array<ColorChannelConfig, MaxColorChannels> colorChannels{};
  std::array<u8, GX::MaxLights> unitLightTerms{};
  std::array<TcgConfig, MaxTexCoord> tcgs{};
  AlphaCompare alphaCompare{};
  GXCullMode cullMode = GX_CULL_NONE;
  std::array<f32, 5> polygonOffset{};
  GXCompare depthFunc = GX_NEVER;
  GXBlendMode blendMode = GX_BM_NONE;
  GXBlendFactor blendFacSrc = GX_BL_ZERO;
  GXBlendFactor blendFacDst = GX_BL_ZERO;
  GXLogicOp blendOp = GX_LO_CLEAR;
  u32 dstAlpha = 0;
  std::array<bool, 4> writes{}; // depth compare, depth, alpha and color updates
};
PipelineInputs sPipelineInputs;

// Whether size bytes at a and b differ: the answer memcmp gives, without its branch on every word. The pipeline's
// inputs below are about a kilobyte, compared for every draw. With GCC's vector extensions the loop compares a vector
// at a time however the call is inlined; GCC's own vectoriser did so only where the size was a constant, and kept a
// word loop as slow as memcmp's in the LTO build.
ALWAYS_INLINE bool bytes_differ(const void* a, const void* b, size_t size) noexcept {
  const auto* x = static_cast<const u8*>(a);
  const auto* y = static_cast<const u8*>(b);
  size_t i = 0;
#ifdef USE_GCC_VECTOR_EXTENSIONS
  typedef u32 Words __attribute__((vector_size(16)));
  Words wide{};
  for (; i + sizeof(Words) <= size; i += sizeof(Words)) {
    Words u;
    Words v;
    std::memcpy(&u, x + i, sizeof(u));
    std::memcpy(&v, y + i, sizeof(v));
    wide |= u ^ v;
  }
  u32 diff = wide[0] | wide[1] | wide[2] | wide[3];
#else
  u32 diff = 0;
#endif
  for (; i + sizeof(u32) <= size; i += sizeof(u32)) {
    u32 u;
    u32 v;
    std::memcpy(&u, x + i, sizeof(u));
    std::memcpy(&v, y + i, sizeof(v));
    diff |= u ^ v;
  }
  for (; i < size; ++i) {
    diff |= x[i] ^ y[i];
  }
  return diff != 0;
}

// Puts now's bytes in kept, and says whether they differed. Bytes, as the config's hash sees them: 0.f and -0.f differ.
// Inlined, so the compares of small fields stay a load and a compare.
ALWAYS_INLINE bool keep_bytes(void* kept, const void* now, size_t size) noexcept {
  if (!bytes_differ(kept, now, size)) {
    return false;
  }
  std::memcpy(kept, now, size);
  return true;
}

template <typename T>
bool keep(T& kept, const T& now) noexcept {
  return keep_bytes(&kept, &now, sizeof(T));
}

// The first count of an array, as populate_pipeline_config copies them.
template <typename T, size_t N>
bool keep_first(std::array<T, N>& kept, const std::array<T, N>& now, size_t count) noexcept {
  return keep_bytes(kept.data(), now.data(), std::min(count, N) * sizeof(T));
}

// Whether the state populate_pipeline_config reads, or the layout, differs from inputs, which then hold them.
bool update_pipeline_inputs(PipelineInputs& inputs, GXVtxFmt fmt, u8 lineMode, uint64_t targetLayoutKey) noexcept {
  const auto& state = g_gxState;
  std::array<u16, MaxVtxAttr> indexedArrays{};
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    if (state.vtxDesc[i] == GX_INDEX8 || state.vtxDesc[i] == GX_INDEX16) {
      indexedArrays[i] = state.arrays[i].stride | (state.arrays[i].le ? 0x100 : 0);
    }
  }
  bool changed = keep(inputs.targetLayoutKey, targetLayoutKey);
  changed |= keep(inputs.lineMode, lineMode);
  changed |= keep(inputs.numTevStages, state.numTevStages);
  changed |= keep(inputs.numIndStages, state.numIndStages);
  changed |= keep(inputs.numTexGens, state.numTexGens);
  changed |= keep(inputs.fogType, state.fog.type);
  changed |= keep(inputs.fogRangeEnabled, state.fog.rangeEnabled);
  changed |= keep(inputs.pixelFmt, state.pixelFmt);
  changed |= keep(inputs.vtxDesc, state.vtxDesc);
  changed |= keep(inputs.vtxFmt, state.vtxFmts[fmt]);
  changed |= keep(inputs.indexedArrays, indexedArrays);
  changed |= keep(inputs.tevSwapTable, state.tevSwapTable);
  changed |= keep_first(inputs.tevStages, state.tevStages, state.numTevStages);
  changed |= keep_first(inputs.indStages, state.indStages, state.numIndStages);
  changed |= keep(inputs.colorChannels, state.colorChannelConfig);
  changed |= keep(inputs.unitLightTerms, state.unitLightTerms);
  changed |= keep_first(inputs.tcgs, state.tcgs, state.numTexGens);
  changed |= keep(inputs.alphaCompare, state.alphaCompare);
  changed |= keep(inputs.cullMode, state.cullMode);
  changed |=
      keep(inputs.polygonOffset, {state.frontOffset, state.frontScale, state.backOffset, state.backScale, state.clamp});
  changed |= keep(inputs.depthFunc, state.depthFunc);
  changed |= keep(inputs.blendMode, state.blendMode);
  changed |= keep(inputs.blendFacSrc, state.blendFacSrc);
  changed |= keep(inputs.blendFacDst, state.blendFacDst);
  changed |= keep(inputs.blendOp, state.blendOp);
  changed |= keep(inputs.dstAlpha, state.dstAlpha);
  changed |= keep(inputs.writes, {state.depthCompare, state.depthUpdate, state.alphaUpdate, state.colorUpdate});
  return changed;
}

// Where each array this frame copied into the storage pool landed, so an array that a later draw points at again isn't
// copied again.
absl::flat_hash_map<std::pair<const void*, u32>, gfx::Range> sArrayRanges;

FogRangeLutKey fog_range_lut_key() noexcept {
  const auto& state = g_gxState.fog;
  const f32 logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const f32 renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  return {
      .rangeK = state.rangeK,
      .rangeCenter = ((static_cast<f32>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
      .renderWidth = renderWidth,
      .targetWidth = gfx::get_render_target_size().x,
  };
}

std::vector<f32> build_fog_range_lut(const FogRangeLutKey& key) {
  // A backend whose GPU reads the table builds it as close as it can to that GPU's own arithmetic.
  if (const auto build = gfx::backend::limits().fogRangeTable) {
    std::vector<f32> lut(key.targetWidth);
    build(key.rangeK, key.rangeCenter, key.renderWidth, lut);
    return lut;
  }
  std::array<f32, 10> rangeK;
  for (u32 i = 0; i < rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    rangeK[i] = static_cast<f32>(key.rangeK[source]) / 64.f;
  }

  std::vector<f32> lut(key.targetWidth);
  for (u32 x = 0; x < key.targetWidth; ++x) {
    const f32 screenX = ((static_cast<f32>(x) + 0.5f) / key.renderWidth) * 2.f - 1.f;
    const f32 offset = screenX - key.rangeCenter;
    const f32 rangeIndex = std::clamp(9.f - std::abs(offset) * 9.f, 0.f, 9.f);
    const u32 lower = static_cast<u32>(rangeIndex);
    const u32 upper = std::min(lower + 1, 9u);
    const f32 fraction = rangeIndex - static_cast<f32>(lower);
    const f32 k = std::max(rangeK[lower] * (1.f - fraction) + rangeK[upper] * fraction, 0.000001f);
    lut[x] = std::sqrt(offset * offset + k * k) / k;
  }
  return lut;
}

const std::vector<f32>& resolve_fog_range_lut(const FogRangeLutKey& key) {
  for (const auto& entry : sFogRangeLuts) {
    if (entry.key == key) {
      return entry.factors;
    }
  }
  if (sFogRangeLuts.size() == MaxFogRangeLuts) {
    sFogRangeLuts.erase(sFogRangeLuts.begin());
  }
  sFogRangeLuts.emplace_back(FogRangeLutEntry{key, build_fog_range_lut(key)});
  return sFogRangeLuts.back().factors;
}

gfx::Range push_fog_range_lut(const FogRangeLutKey& key) {
  const auto& lut = resolve_fog_range_lut(key);
  // The backend's own table goes where its GPU reads it; fogRangeBase then counts words of that pool.
  if (gfx::backend::limits().fogRangeTable != nullptr) {
    return gfx::push_uniform(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
  }
  return gfx::push_storage(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
}

u8 line_mode_for_prim(GXPrimitive prim) noexcept {
  switch (prim) {
  case GX_LINES:
    return 1;
  case GX_LINESTRIP:
    return 2;
  case GX_POINTS:
    return 3;
  default:
    return 0;
  }
}
} // namespace

// The TEV stages past tevStageCount are always the reset's (populate_pipeline_config), so the key leaves them out:
// most of the config's 2.7 KB. XXH64 read 1 KB 1.4 to 1.6 times as fast as XXH3 on the Xbox 360 (milestone 34).
uint64_t pipeline_memo_key(const PipelineConfig& config, uint64_t targetLayoutKey) noexcept {
  constexpr size_t StagesAt = offsetof(PipelineConfig, shaderConfig) + offsetof(ShaderConfig, tevStages);
  constexpr size_t CountAt = offsetof(PipelineConfig, shaderConfig) + offsetof(ShaderConfig, tevStageCount);
  const auto* bytes = reinterpret_cast<const u8*>(&config);
  const size_t stages = std::min<size_t>(config.shaderConfig.tevStageCount, MaxTevStages);
  const uint64_t head = XXH64(bytes, StagesAt + stages * sizeof(TevStage), targetLayoutKey);
  return XXH64(bytes + CountAt, sizeof(PipelineConfig) - CountAt, head);
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept;
// Returns the display list a GX_AURORA_CALL_DISPLAY_LIST names, or no bytes.
static std::span<const u8> handle_aurora(ByteReader& reader) noexcept;

ProcessResult process(const u8* data, u32 size) noexcept {
  ZoneScoped;
  ByteReader reader{{data, size}};

  while (!reader.empty()) {
    const u8 cmd = reader.read<u8>();
    u8 opcode = cmd & CP_OPCODE_MASK;

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      const u32 value = reader.read<u32>();
      handle_bp(value);
      const u32 reg = reg_get(value, 8, 24);
      if (reg == GX_BP_REG_DRAWDONE) {
        return {static_cast<u32>(reader.offset()), true};
      }
      if (reg == GX_BP_REG_PETOKEN || reg == GX_BP_REG_PETOKENINT) {
        pass_draw_sync(static_cast<u16>(value), reg == GX_BP_REG_PETOKENINT);
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      const u8 addr = reader.read<u8>();
      handle_cp(addr, reader.read<u32>());
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const u32 header = reader.read<u32>();
      const u32 count = ((header >> 16) & 0xFFFF) + 1;
      const u16 addr = header & 0xFFFF;
      handle_xf(addr, reader.take(count * sizeof(u32)));
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const u32 arrayType = GX_POS_MTX_ARRAY + (opcode - CP_CMD_LOAD_INDX_A) / 0x08;
      const u16 srcArrayIdx = reader.read<u16>();
      const u16 addrLen = reader.read<u16>();

      const u16 len = (addrLen >> 12) + 1;
      const u16 dstAddr = addrLen & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      const u32 srcOffset = static_cast<u32>(srcArrayIdx) * array.stride;
      const u32 srcSize = static_cast<u32>(len) * sizeof(u32);
      AURORA_ASSERT(array.data != nullptr, "indexed XF load from unmapped array {}", arrayType);
      AURORA_ASSERT(srcOffset <= array.size && srcSize <= array.size - srcOffset,
                    "indexed XF load outside array {}: offset={}, size={}, array size={}", arrayType, srcOffset,
                    srcSize, array.size);
      auto const* srcData = static_cast<const u8*>(array.data) + srcOffset;
      if (!copy_xf_data(dstAddr, srcData, len, array.le ? std::endian::little : std::endian::big)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      reader.skip(8);
      break;
    }

    case CP_CMD_INVAL_VTX: {
      // The game changed vertex data.
      clear_array_ranges();
      g_gxState.dirty |= DirtyImmediates;
      break;
    }

    case GX_AURORA: {
      // A frame's end is a position the game thread waits for, so processing stops there to say it got there.
      const bool frameEnd =
          reader.remaining() >= sizeof(u16) && read_bits<u16>(reader.data() + reader.offset()) == GX_AURORA_END_FRAME;
      if (const auto list = handle_aurora(reader); list.data() != nullptr) {
        return {static_cast<u32>(reader.offset()), false, list.data(), static_cast<u32>(list.size())};
      }
      if (frameEnd) {
        return {static_cast<u32>(reader.offset()), false};
      }
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, reader);
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        handle_draw(cmd, reader);
      } else {
        // Hex dump surrounding bytes for debugging
        {
          const size_t pos = reader.offset();
          size_t dumpStart = (pos > 17) ? pos - 17 : 0;
          size_t dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (size_t i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, reader.offset() - 1);
      }
      break;
    }
  }
  return {size, false};
}

[[noreturn]] static void handle_draw_overrun(size_t totalVtxBytes, const ByteReader& reader) noexcept {
  // Hex dump around the draw command for debugging
  const size_t pos = reader.offset();
  const size_t size = reader.size();
  const u8* data = reader.data();
  size_t cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  size_t dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  size_t dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (size_t i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, reader.remaining());
}

static u32 calc_vtx_size(GXVtxFmt fmt) noexcept {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attrFmt = vtxFmt.attrs[i];
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 6 : 2;
      break;
    }
  }
  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;
  return vtxSize;
}

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, u64 vertHash,
                         gfx::Range idxRange, u32 numIndices, bool patternIndices) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;

  DrawImmediateData immediates{.vtxStart = vertRange.offset, .currentPnMtx = state.currentPnMtx};
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (state.vtxDesc[i] != GX_INDEX8 && state.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = state.arrays[i];
    if (array.cachedRange.size == 0) {
      // One copy a frame of each array, however many times an attribute is pointed back at it.
      const auto [it, inserted] = sArrayRanges.try_emplace({array.data, array.size});
      if (inserted) {
        it->second = gfx::push_array(static_cast<const uint8_t*>(array.data), array.size);
      }
      array.cachedRange = it->second;
    }
    immediates.arrayStart[i - GX_VA_POS] = array.cachedRange.offset;
  }

  const u8 lineMode = line_mode_for_prim(prim);
  const auto targetLayoutKey = gfx::get_render_target_layout_key();
  const bool pipelineValid = cache.hasPipeline && (state.dirty & DirtyPipeline) == 0 && cache.fmt == fmt &&
                             cache.lineMode == lineMode && cache.targetLayoutKey == targetLayoutKey;
  if (!pipelineValid) {
    const bool hadPipeline = cache.hasPipeline;
    const auto prevSampledTextures = cache.shaderInfo.sampledTextures;
    const auto prevSampledIndTextures = cache.shaderInfo.sampledIndTextures;
    // A register write marks the pipeline dirty, but the state often changes and changes back between two draws; see
    // PipelineInputs.
    if (update_pipeline_inputs(sPipelineInputs, fmt, lineMode, targetLayoutKey) || !hadPipeline) {
      populate_pipeline_config(cache.config, prim, fmt);
      const uint64_t memoKey = pipeline_memo_key(cache.config, targetLayoutKey);
      if (const auto it = sPipelineMemo.find(memoKey); it != sPipelineMemo.end()) {
        cache.shaderInfo = it->second.shaderInfo;
        cache.pipelineRef = it->second.pipelineRef;
      } else {
        cache.shaderInfo = build_shader_info(cache.config.shaderConfig);
        cache.pipelineRef = gfx::pipeline_ref(cache.config);
        sPipelineMemo.emplace(memoKey, PipelineMemo{cache.shaderInfo, cache.pipelineRef});
      }
    }
    cache.targetLayoutKey = targetLayoutKey;
    cache.fmt = fmt;
    // The uniform pieces hold the same state for every pipeline, but for the material's sizes of the
    // textures sampled.
    if (prevSampledTextures != cache.shaderInfo.sampledTextures) {
      state.dirty |= DirtyMaterial;
    }
    cache.lineMode = lineMode;
    cache.hasPipeline = true;
    state.dirty &= ~DirtyPipeline;
    if (!hadPipeline || prevSampledTextures != cache.shaderInfo.sampledTextures ||
        prevSampledIndTextures != cache.shaderInfo.sampledIndTextures) {
      cache.bindGeneration = 0;
    }
  }

  const bool bindGroupsValid =
      (state.dirty & DirtyTextures) == 0 && cache.bindGeneration == texture::current_bind_generation();
  if (!bindGroupsValid) {
    const auto prevBindGroup = cache.bindGroups.textureBindGroup;
    resolve_sampled_textures(cache.shaderInfo);
    cache.bindGroups = build_bind_groups(cache.shaderInfo);
    cache.bindGeneration = texture::current_bind_generation();
    state.dirty &= ~DirtyTextures;
    // For texture_size_bias uniform
    if (cache.bindGroups.textureBindGroup != prevBindGroup) {
      state.dirty |= DirtyMaterial;
    }
  }

  // A piece the shaders read is pushed again only if its state changed since its last push this frame; the draw binds
  // the earlier copies of the rest. Line and point draws push the view each time, as its line state isn't tracked.
  if (lineMode != 0) {
    state.dirty |= DirtyView;
  }
  cache.staleUniforms |= (state.dirty & DirtyUniform) >> DirtyUniformShift;
  state.dirty &= ~DirtyUniform;
  // A shader that writes the render target's normal attachment reads the normal matrices, lit or not.
  u8 readPieces = cache.shaderInfo.uniformPieces;
  if (cache.config.shaderConfig.attrs[GX_VA_NRM].attrType != GX_NONE && gfx::has_normal_attachment()) {
    readPieces |= 1 << UniformNrmMtx;
  }
  if (const u8 stale = cache.staleUniforms & readPieces; stale != 0) {
    push_uniforms(cache.shaderInfo, stale, cache.uniformOffsets);
    cache.staleUniforms &= ~stale;
  }
  // Only fogged programs read the table (gx/shader.cpp), and a draw can have range fog on but fog off.
  if (cache.config.shaderConfig.fogRangeEnabled && cache.shaderInfo.usesFog) {
    const auto key = fog_range_lut_key();
    if (!cache.hasFogRange || cache.fogRangeKey != key) {
      cache.fogRange = push_fog_range_lut(key);
      cache.fogRangeKey = key;
      cache.hasFogRange = true;
    }
  }
  immediates.fogRangeBase = cache.fogRange.offset / sizeof(u32);

  state.dirty &= ~DirtyImmediates;

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  cache.lastDrawFmt = fmt;
  gfx::push_draw_command(DrawData{
      .pipeline = cache.pipelineRef,
      .bindGroups = cache.bindGroups,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformOffsets = cache.uniformOffsets,
      .immediateData = immediates,
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .dstAlpha = state.dstAlpha,
      .patternIndices = patternIndices,
      .vertHash = vertHash,
  });
}

// The draw's vertex bytes into the frame's pool, hashed as they're copied when the backend keys a cache by them.
static gfx::Range push_draw_verts(std::span<const u8> bytes, size_t alignment, u64& hash) noexcept {
  if (!gfx::backend::limits().contentHashes) {
    hash = 0;
    return gfx::push_verts(bytes.data(), bytes.size(), alignment);
  }
  return gfx::push_verts_hashed(bytes.data(), bytes.size(), alignment, hash);
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange,
                                 u64 vertHash) noexcept {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;
  bool patternIndices = false;

  if (const auto pattern = index_pattern_range(index_patterns(), prim, vtxCount)) {
    idxRange = *pattern;
    numIndices = pattern->size / sizeof(u16);
    patternIndices = true;
  } else if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    static ByteBuffer idxBuf;
    numIndices = prepare_idx_buffer(idxBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), 4);
    idxBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, vertHash, idxRange, numIndices, patternIndices);
}

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, ByteReader& reader) noexcept {
  ZoneScoped;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calc_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (totalVtxBytes > reader.remaining())
    UNLIKELY { handle_draw_overrun(totalVtxBytes, reader); }

  const bool cleanState = g_gxState.dirty == 0 && fmt == sDrawCache.lastDrawFmt && sDrawCache.lineMode == 0 &&
                          prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS;
  auto* lastDraw = cleanState ? gfx::get_last_draw_command<DrawData>() : nullptr;
  const bool canMerge = lastDraw != nullptr && lastDraw->instanceCount == 1;

  // Push raw vertex data to buffer. Merged draws must remain contiguous with the previous range.
  const auto vertexData = reader.take(totalVtxBytes);
  u64 vertHash = 0;
  gfx::Range vertRange =
      canMerge ? gfx::push_verts(vertexData.data(), vertexData.size(), 0) : push_draw_verts(vertexData, 4, vertHash);

  // Try to merge with previous draw call
  if (canMerge) {
    if (lastDraw->patternIndices) {
      // Merged indices go on from the previous draw's in the frame's pool, so its pattern's are copied there first.
      lastDraw->idxRange =
          gfx::push_indices(index_patterns().indices.data() + lastDraw->idxRange.offset, lastDraw->idxRange.size, 4);
      lastDraw->patternIndices = false;
    }
    u32 numIndices = 0;
    gfx::Range idxRange;
    static ByteBuffer idxBuf;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(idxBuf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(idxBuf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), hadIndexRange ? 0 : 4);
      idxBuf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    // Its bytes are two pushes' now; the backend hashes them itself.
    lastDraw->vertHash = 0;
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    gfx::detail::increment_merged_draw_count();
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange, vertHash);
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept {
  const auto fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const auto prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  draw_prim(prim, fmt, reader.read<u16>(), reader);
}

std::span<const u8> handle_aurora(ByteReader& reader) noexcept {
  ZoneScoped;
  const u16 subCmd = reader.read<u16>();

  if (subCmd == GX_AURORA_CALL_DISPLAY_LIST) {
    const auto* list = reinterpret_cast<const u8*>(reader.read<u64>());
    const u32 size = reader.read<u32>();
    return {list, size};
  } else if (subCmd == GX_AURORA_BEGIN_FRAME) {
    const u8 slot = reader.read<u8>();
    g_gxState.viewportPolicy = static_cast<AuroraViewportPolicy>(reader.read<u8>());
    const u32 viWidth = reader.read<u32>();
    const u32 viHeight = reader.read<u32>();
    g_gxState.viSize = {viWidth, viHeight};
    gfx::detail::begin_frame_recording(slot);
  } else if (subCmd == GX_AURORA_END_FRAME) {
    const u8 slot = reader.read<u8>();
    const bool close = reader.read<u8>() != 0;
    gfx::detail::note_fifo_done(slot);
    // The draw cache holds offsets into this frame's pools.
    clear_draw_cache();
    texture::end_frame();
    gfx::detail::end_frame_recording(slot, close);
  } else if (subCmd == GX_AURORA_SET_VI_SIZE) {
    const u32 viWidth = reader.read<u32>();
    const u32 viHeight = reader.read<u32>();
    g_gxState.viSize = {viWidth, viHeight};
  } else if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    const f32 left = reader.read<f32>();
    const f32 top = reader.read<f32>();
    const f32 width = reader.read<f32>();
    const f32 height = reader.read<f32>();
    const f32 nearZ = reader.read<f32>();
    const f32 farZ = reader.read<f32>();
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = reader.read<f32>();
      }
    }
    // Invalidate projection XF regs
    for (u32 reg = 0x20; reg <= 0x26; ++reg) {
      g_gxState.xfRegValid.reset(reg);
    }
    g_gxState.dirty |= DirtyView;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    const u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;
    const u64 arrayAddr = reader.read<u64>();
    const u32 arraySize = reader.read<u32>();
    // Games mark an array le when the host wrote it, in the host's byte order (GXSetArray; J3D's skinned vertices and
    // matrices): little-endian on the PCs they were ported to, but GX's own, big-endian, on a big-endian host such as
    // the Xbox 360.
    const bool le = reader.read<u8>() == 1 && std::endian::native == std::endian::little;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      if (array.le != le) {
        // Endianness is baked into the shader
        g_gxState.dirty |= DirtyPipeline;
      }
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.cachedRange = {};
      g_gxState.dirty |= DirtyImmediates;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 newWidth = reader.read<u32>();
    const u32 newHeight = reader.read<u32>();
    const auto newFormat = static_cast<GXTexFmt>(reader.read<u32>());
    const auto newTlut = static_cast<GXTlut>(reader.read<u32>());
    u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (reader.read<u8>() != 0) {
      newFlags |= 1u;
    } else {
      newFlags &= ~1u;
    }
    const u32 newTexObjId = reader.read<u32>();
    const u32 newTexDataVersion = reader.read<u32>();
    if (slot.data != newData || slot.mWidth != newWidth || slot.mHeight != newHeight ||
        slot.mFormat != static_cast<u32>(newFormat) || slot.tlut != newTlut || slot.flags != newFlags ||
        slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion) {
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      slot.flags = newFlags;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    const auto idx = reader.read<u8>();
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const auto newFormat = static_cast<GXTlutFmt>(reader.read<u32>());
    const u16 newNumEntries = reader.read<u16>();
    const u32 newTlutObjId = reader.read<u32>();
    const u32 newTlutDataVersion = reader.read<u32>();
    const u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (slot.data != newData || slot.format != newFormat || slot.numEntries != newNumEntries ||
        slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion || slot.flags != newFlags) {
      if (slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion) {
        texture::invalidate_bindings();
      }
      slot.data = newData;
      slot.format = newFormat;
      slot.numEntries = newNumEntries;
      slot.tlutObjId = newTlutObjId;
      slot.tlutDataVersion = newTlutDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    const f32 frontOffset = reader.read<f32>();
    const f32 frontScale = reader.read<f32>();
    const f32 backOffset = reader.read<f32>();
    const f32 backScale = reader.read<f32>();
    const f32 clamp = reader.read<f32>();
    if (g_gxState.frontOffset != frontOffset || g_gxState.frontScale != frontScale ||
        g_gxState.backOffset != backOffset || g_gxState.backScale != backScale || g_gxState.clamp != clamp) {
      g_gxState.frontOffset = frontOffset;
      g_gxState.frontScale = frontScale;
      g_gxState.backOffset = backOffset;
      g_gxState.backScale = backScale;
      g_gxState.clamp = clamp;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    g_gxState.texCopyDstWidth = reader.read<u32>();
    g_gxState.texCopyDstHeight = reader.read<u32>();
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(reader.read<u32>());
    reader.skip(1); // mipmap is not implemented, but remains part of the command payload
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    g_gxState.texCopyDest = reinterpret_cast<const void*>(reader.read<u64>());
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    gfx::backend::request_depth_snapshot();
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_CLEAR_COPY_SRC) {
    clear_copy_src();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    evict_texture_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    evict_tlut_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    evict_copy_texture(reinterpret_cast<const void*>(reader.read<u64>()));
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    const u8 cmd = reader.read<u8>();
    const u32 byteLen = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calc_vtx_size(fmt);
      }
      AURORA_ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
                    "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      AURORA_ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), reader);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    const u8 cmd = reader.read<u8>();
    const u16 vtxCount = reader.read<u16>();
    const u32 indexCount = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    AURORA_ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
                  static_cast<u32>(prim));
    const size_t idxBytes = static_cast<size_t>(indexCount) * sizeof(u16);
    // Index data is always host-endian; push it to the GPU buffer as-is
    const auto indexData = reader.take(idxBytes);
    const gfx::Range idxRange = gfx::push_indices(indexData.data(), indexData.size(), 4);
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calc_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    const auto vertexData = reader.take(totalVtxBytes);
    u64 vertHash = 0;
    const gfx::Range vertRange = push_draw_verts(vertexData, 4, vertHash);
    if (indexCount != 0) {
      push_gx_draw(prim, fmt, vtxCount, vertRange, vertHash, idxRange, indexCount, false);
    }
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = reader.read_string();
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = reader.read_string();
    gfx::insert_debug_marker(std::move(label));
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
  return {};
}

// erase_if keeps the table, where clear() frees one of 128 slots or more (absl raw_hash_set.h:2892).
void clear_array_ranges() noexcept {
  absl::erase_if(sArrayRanges, [](const auto&) { return true; });
  for (auto& array : g_gxState.arrays) {
    array.cachedRange = {};
  }
}

void clear_draw_cache() noexcept {
  // Bounded, as nothing else evicts from the memo.
  if (sPipelineMemo.size() > 8192) {
    sPipelineMemo.clear();
  }
  sDrawCache.bindGeneration = 0;
  sDrawCache.staleUniforms = AllUniformPieces; // The next frame has a new uniform pool
  sDrawCache.fogRange = {};
  sDrawCache.hasFogRange = false;
}

} // namespace aurora::gx::fifo
