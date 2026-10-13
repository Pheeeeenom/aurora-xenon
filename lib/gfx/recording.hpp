#pragma once

#include "backend.hpp"
#include "types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace aurora::gfx::detail {

struct FramePacket;
struct RenderPass;
struct CustomDrawCommand;
struct EncoderTask;

struct RecordedFrame {
  FramePacket* packet;
  size_t frameSlot;
};

void begin_recording(FramePacket& packet, size_t frameSlot);
RecordedFrame end_recording();
void shutdown_recording();
void increment_merged_draw_count() noexcept;
FramePacket& current_frame_packet();
// A pass is open to record into.
bool in_render_pass() noexcept;

// For a backend's own API (the WebGPU extension API and RmlUi on Dawn), which checks its arguments first.
// A draw it replays in the open pass.
void push_custom_draw_command(const CustomDrawCommand& draw);
// Seals the open EFB pass, records the task after it and continues on an EFB pass that loads its contents.
void push_encoder_task(const EncoderTask& task);
// Copies the open pass's targets into the given textures (any may be null), then seals it: an offscreen pass ends and
// the EFB is restored; an EFB pass continues on one that loads its contents.
void resolve_current_pass(TextureHandle color, TextureHandle depth, TextureHandle normal);
// Seals the open pass and opens this one, which a backend's own targets back, with a full-target viewport and scissor.
void begin_backend_pass(RenderPass pass);
// Seals the open pass, then records the copy.
void push_texture_copy(std::shared_ptr<const backend::TextureCopy> copy);

namespace testing {
struct OffscreenCacheStats {
  uint64_t bytes;
  size_t entries;
};
void suppress_render_worker(bool suppress) noexcept;
void seed_offscreen_cache(uint32_t width, uint32_t height, TextureFormat colorFormat, TextureFormat depthFormat);
OffscreenCacheStats offscreen_cache_stats() noexcept;
bool is_offscreen_cached(uint32_t width, uint32_t height, TextureFormat colorFormat, TextureFormat depthFormat);
}

} // namespace aurora::gfx::detail

namespace aurora::gfx {
void finish();
void end_color_pass();
void begin_offscreen(uint32_t width, uint32_t height);
void end_offscreen();
bool has_normal_attachment() noexcept;
RenderTargetLayout get_render_target_layout() noexcept;
// get_render_target_layout().key, without building the layout: worked out once for each pass.
uint64_t get_render_target_layout_key() noexcept;

namespace tex_palette_conv {
struct ConvRequest;
}
void queue_palette_conv(tex_palette_conv::ConvRequest req);

Range push_verts(const uint8_t* data, size_t length, size_t alignment);
// As push_verts, and hash is XXH3-64 of the bytes as the pool holds them, hashed while they're warm.
Range push_verts_hashed(const uint8_t* data, size_t length, size_t alignment, uint64_t& hash);
template <typename T>
Range push_verts(ArrayRef<T> data, size_t alignment) {
  return push_verts(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_indices(const uint8_t* data, size_t length, size_t alignment);
template <typename T>
Range push_indices(ArrayRef<T> data, size_t alignment) {
  return push_indices(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T), alignment);
}
Range push_uniform(const uint8_t* data, size_t length);
template <typename T>
Range push_uniform(const T& data) {
  return push_uniform(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
// Room for length bytes in the frame's uniform pool where push_uniform would put them, in range, for the caller to
// write in place before anything else is pushed; null outside a frame. The room holds whatever the pool held there.
uint8_t* map_uniform(size_t length, Range& range);
Range push_storage(const uint8_t* data, size_t length);
template <typename T>
Range push_storage(ArrayRef<T> data) {
  return push_storage(reinterpret_cast<const uint8_t*>(data.data()), data.size() * sizeof(T));
}
template <typename T>
Range push_storage(const T& data) {
  return push_storage(reinterpret_cast<const uint8_t*>(&data), sizeof(T));
}
// A GX vertex array, into the storage pool as push_storage puts it; with backend::Limits::arraySizes, its size in bytes
// goes first, a u32 in the four bytes before the returned offset, so a backend can tell where the array ends, and with
// backend::Limits::contentHashes its XXH3-64 in the eight bytes before that.
Range push_array(const uint8_t* data, size_t length);
Range push_texture_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage, uint32_t alignment);

template <typename DrawData>
void push_draw_command(DrawData data);
template <typename DrawData>
DrawData* get_last_draw_command();
template <typename PipelineConfig>
PipelineRef pipeline_ref(const PipelineConfig& config);

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat);
uint32_t align_uniform(uint32_t value);
Vec2<uint32_t> get_render_target_size() noexcept;
void set_viewport(const Viewport& viewport) noexcept;
void set_scissor(const ClipRect& scissor) noexcept;
void push_debug_group(std::string label);
void insert_debug_marker(std::string label);
} // namespace aurora::gfx
