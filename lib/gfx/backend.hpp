#pragma once
// What Aurora's GX core asks of the graphics API that draws; one backend is built in.

#include "types.hpp"

#include <aurora/aurora.h>
#include <dolphin/gx/GXEnum.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace aurora::imgui {
class DrawData;
} // namespace aurora::imgui

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gfx::detail {
struct FramePacket;
struct FrameOp;
} // namespace aurora::gfx::detail

namespace aurora::gfx {
struct TextureSet;
} // namespace aurora::gfx

namespace aurora::gfx::ui {
struct Frame;
} // namespace aurora::gfx::ui

// A backend also defines imgui.hpp's initialize, shutdown, process_event and new_frame, and the C calls left to it:
// aurora_enable_vsync, aurora_present_waits_for_vblank, aurora_set_resampler, aurora_get_available_backends,
// aurora_gpu_frame_time and aurora_imgui_add_texture.
namespace aurora::gfx::backend {
// Defined by the backend. The game thread, the FIFO worker and the render worker (clearing a frame packet) all release
// them; a backend keeps their memory until the GPU is past every frame that used it.
struct Pipeline;
struct Texture;
struct UploadBuffer;
// Recorded through a backend's own API, and read only by that backend: the targets of a pass it opened, and a copy
// between its textures (RmlUi's, on Dawn).
struct PassTargets;
struct TextureCopy;

// The window, device and surface, and the backend's own resources. In: the backend the app asked for; out: the one
// started. False when none would start.
bool initialize(AuroraBackend& backend) noexcept;
void shutdown() noexcept;
// Whether the backend draws to a window SDL makes; without one Aurora starts no SDL video, which a port's SDL may lack.
bool uses_sdl_window() noexcept;
// Main thread, before a frame: releases or recreates the surface as the window needs; false when the frame can't be
// shown.
bool can_present() noexcept;
void resize(const AuroraWindowSize& size) noexcept;
// The surface must be reconfigured, as when vsync changes.
void refresh_surface() noexcept;

enum class TextureUsage : uint8_t {
  Sampled, // GX textures and TLUTs, filled by uploads
  Copy,    // EFB copy destinations
  Conv,    // destinations of a conversion drawn by the backend
  Render,  // offscreen colour targets
  Depth,   // offscreen depth targets
};

struct TextureDesc {
  uint32_t width;
  uint32_t height;
  uint32_t mips;
  TextureFormat format;
  u32 gxFormat;
  TextureUsage usage;
  const char* label;
  // The UI's, which a backend may keep apart from the game's memory, so that what the UI releases goes back at once.
  bool ui = false;
};

std::shared_ptr<Texture> create_texture(const TextureDesc& desc) noexcept;
// Game thread or FIFO worker: an upload too big for the frame's pool, its rows copied into memory of its own at the
// pool's row pitch, or one of the UI's (ui), as TextureDesc::ui. Null when there is no memory for it.
std::shared_ptr<UploadBuffer> stage_upload(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                                           bool ui = false) noexcept;
#ifdef AURORA_DIAGNOSTICS
// The memory the texture takes, padding included, on the backends RmlUi's recorder draws through.
size_t texture_bytes(const Texture& texture) noexcept;
#endif
} // namespace aurora::gfx::backend

namespace aurora::gfx {
// The pipelines a draw binds: the prepass, when the backend splits the draw in two, then the main one.
struct CompiledPipeline {
  std::shared_ptr<backend::Pipeline> main;
  std::shared_ptr<backend::Pipeline> prepass;

  [[nodiscard]] uint32_t pipeline_count() const {
    return static_cast<uint32_t>(static_cast<bool>(main)) + static_cast<uint32_t>(static_cast<bool>(prepass));
  }
};
} // namespace aurora::gfx

namespace aurora::gfx::backend {
// The pipelines a config draws with in a pass of the given layout; a null main when they can't be made. On the
// pipeline cache's compile threads; without them, inline on the thread that asks for the pipeline: the FIFO worker as
// it records, or the main thread as the cache loads.
CompiledPipeline create_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout) noexcept;
std::shared_ptr<Pipeline> create_pipeline(const clear::PipelineConfig& config,
                                          const RenderTargetLayout& layout) noexcept;
// Every pipeline the cache had saved has been created.
void pipelines_loaded() noexcept;
// FIFO worker: the textures and samplers a draw binds, under the ref the draw carries.
void prepare_textures(BindGroupRef ref, const TextureSet& set) noexcept;
// Whether an EFB copy to this format is made by a conversion pass rather than texel for texel.
bool converts_copy(GXTexFmt format) noexcept;

// GXPeekZ: a depth snapshot is wanted at the end of this frame; read_depth reads the latest one.
void request_depth_snapshot() noexcept;
bool read_depth(uint16_t x, uint16_t y, uint32_t& z) noexcept;

struct Limits {
  uint32_t uniformAlignment;
  uint32_t storageAlignment;
  // Row pitch in the frame's texture upload pool.
  uint32_t textureRowAlignment;
  bool bcTextures;
  // CMPR uploads as BC1 when it has no transparent texels.
  bool cmprAsBc1;
  // BC textures need power-of-two sides.
  bool bcPowerOfTwo;
  // R8 and RG8 PC textures sample as intensity without being expanded.
  bool componentSwizzle;
  // Pipelines are created on threads of the pipeline cache's own.
  bool pipelineThreads;
  // Each vertex array in the storage pool has its size in bytes, a u32, in the four bytes before it (push_array).
  bool arraySizes = false;
  // With arraySizes, for a backend that keys a cache by the content of the pools: the FIFO worker hashes each GX draw's
  // vertex bytes and each array as it copies them, run_hash of the bytes as the pool holds them. A draw's is its
  // gx::DrawData::vertHash; an array's is the eight bytes before its size. The backend then needn't read them again.
  bool contentHashes = false;
  // Range fog's factor for each target column, as close as the backend comes to what its GPU works out, from the fog's
  // range_k registers, its range centre and the render viewport's width; the GPU reads it from the uniform pool.
  // Without it, Aurora's own table goes to the storage pool (gx/command_processor.cpp).
  void (*fogRangeTable)(const std::array<uint16_t, 10>& rangeK, float rangeCenter, float renderWidth,
                        std::span<float> factors) = nullptr;
  // A lit channel's config marks the lights whose spot or distance coefficients are exactly (1, 0, 0), for a backend
  // whose programs read such a term as the coefficient (gx::ColorChannelConfig).
  bool unitLightTerms = false;
};
const Limits& limits() noexcept;
// The EFB's size, formats and samples, with the normal attachment once it is on.
RenderTargetLayout efb_layout() noexcept;

// Game thread: points the frame's streaming pools at memory; false when none is free.
bool begin_frame(detail::FramePacket& frame) noexcept;
// FIFO processor, as the frame's recording begins: queues what the render worker does first for the frame, ahead of
// the frame's passes.
void start_frame(detail::FramePacket& frame) noexcept;
// Game thread, at the frame's end: anything the backend records after the frame's last GX pass (Dawn: RmlUi's passes,
// with the command stream drained).
void finish_frame(detail::FramePacket& frame) noexcept;
// Render worker: one recorded op of the frame, in order.
void encode(detail::FramePacket& frame, const detail::FrameOp& op) noexcept;
// Render worker: submits the frame and presents it with ImGui's overlay and, on a backend that draws it, RmlUi's list
// (ui.hpp).
void end_frame(detail::FramePacket& frame, const imgui::DrawData& overlay, const ui::Frame& ui) noexcept;
} // namespace aurora::gfx::backend
