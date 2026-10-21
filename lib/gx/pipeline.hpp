#pragma once

#include "../gfx/pipeline_cache.hpp"
#include "gx.hpp"

namespace aurora::gx {
struct DrawData {
  gfx::PipelineRef pipeline;
  GXBindGroups bindGroups;
  gfx::Range vertRange;
  gfx::Range idxRange;
  std::array<uint32_t, UniformPieceCount> uniformOffsets; // In the frame's pool, by UniformPiece
  DrawImmediateData immediateData;
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t dstAlpha;
  // idxRange is in the index patterns (fifo::index_patterns), which the backend keeps apart, not in the frame's pool.
  bool patternIndices = false;
  // With backend::Limits::contentHashes, run_hash of vertRange's bytes as the frame's pool holds them; zero when not
  // hashed (draws merged into one, or the limit off).
  uint64_t vertHash = 0;
};

// 14 for ColorChannelConfig's lightMask; a cached 13 would read as a lit channel with no lights.
constexpr uint32_t GXPipelineConfigVersion = 14;
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1; // deprecated
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  uint32_t polygonOffsetBits;
  uint32_t polygonOffsetScaleBits;
  uint32_t polygonOffsetClampBits;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

struct PipelineOptions {
  DstAlphaMode dstAlphaMode = DstAlphaMode::None;
  bool colorUpdate = false;
  bool alphaUpdate = false;
  bool depthUpdate = false;
};

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
