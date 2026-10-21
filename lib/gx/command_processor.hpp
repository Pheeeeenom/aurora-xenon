#pragma once

#include "../internal.hpp"

#include <aurora/gfx.hpp>
#include <dolphin/gx/GXEnum.h>

#include <array>
#include <cstdint>
#include <optional>

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::gx::fifo {

// Writes the u16 indices that draw prim's vtxCount vertices from vtxStart as triangles, returning how many. u32, since
// a strip, fan or quad draw of a u16 vertex count can need more than 65,535 indices.
uint32_t prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, uint16_t vtxStart, uint16_t vtxCount) noexcept;

// The indices of a quad, strip or fan draw from vertex 0 depend on its vertex count alone, and each count's are the
// first of a longer draw's; line and point draws all take the same six. So one pattern a kind, built once, holds every
// such draw's indices up to maxVertices vertices, and they never go stale, as they never come from the game.
struct IndexPatterns {
  ByteBuffer indices;                // u16
  std::array<uint32_t, 4> offsets{}; // in bytes, each a multiple of 4: quads, strips, fans, lines and points
  uint32_t maxVertices = 0;
};
IndexPatterns build_index_patterns(uint32_t maxVertices) noexcept;
// AURORA_INDEX_PATTERN_VERTICES's patterns, which draws take their indices from and backends keep a copy of: built on
// first use, then never changed. None when it's 0.
const IndexPatterns& index_patterns() noexcept;
// Where prim's draw of vtxCount vertices from vertex 0 finds its indices in patterns, if it does.
std::optional<gfx::Range> index_pattern_range(const IndexPatterns& patterns, GXPrimitive prim,
                                              uint32_t vtxCount) noexcept;

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
  // A display list called by reference (GX_AURORA_CALL_DISPLAY_LIST), which the caller processes next.
  const uint8_t* list = nullptr;
  uint32_t listSize = 0;
};

// Process GX FIFO commands until the next draw done event, display list call, frame end, or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// The pipeline memo's key for a config populate_pipeline_config filled, looked up for a render target layout: equal for
// equal configs and layouts, and 64 bits on every host.
uint64_t pipeline_memo_key(const PipelineConfig& config, uint64_t targetLayoutKey) noexcept;
// Forgets the arrays the frame's storage pool holds, at the frame's end and when vertex data changes.
void clear_array_ranges() noexcept;

} // namespace aurora::gx::fifo
