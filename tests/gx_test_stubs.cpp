// Stub implementations for renderer symbols that the GX/FIFO/command_processor code
// references but that live in the full renderer (gfx, gx.cpp, model/shader.cpp, etc.).
// These allow the test binary to link without pulling in WebGPU runtime.

#include "gx/gx.hpp"
#include "dolphin/vi/vi_internal.hpp"
#include "gfx/clear.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/resources.hpp"
#include "gfx/depth_peek.hpp"
#include "gfx/backend.hpp"
#include "gfx/frame.hpp"
#include "gfx/recording.hpp"
#include "gfx/tex_palette_conv.hpp"
#include "gfx/texture.hpp"
#include "gfx/texture_replacement.hpp"
#include "gx/pipeline.hpp"
#include "gx/shader_info.hpp"
#include "internal.hpp"
#include "webgpu/gpu.hpp"
#include "gx_test_common.hpp"

#include <atomic>
#include <cstdio>
#include <fmt/format.h>

// --- aurora::g_config ---
namespace aurora {
AuroraConfig g_config{};
} // namespace aurora

// --- aurora::log_internal ---
namespace aurora {
void log_internal(AuroraLogLevel level, const char* module, const char* message, unsigned int len) noexcept {
  fprintf(stderr, "[%d] %s: %.*s\n", static_cast<int>(level), module, len, message);
}
} // namespace aurora

// --- fmt::formatter<AuroraLogLevel> ---
auto fmt::formatter<AuroraLogLevel>::format(AuroraLogLevel level, format_context& ctx) const
    -> format_context::iterator {
  return fmt::format_to(ctx.out(), "{}", static_cast<int>(level));
}

// --- Merged draw counter ---
namespace aurora::gfx::detail {
void increment_merged_draw_count() noexcept {}
} // namespace aurora::gfx::detail

namespace aurora::webgpu {
GraphicsConfig g_graphicsConfig{};
bool g_dualSourceBlendingSupported = false;
} // namespace aurora::webgpu

// --- GXState ---
namespace aurora::gx {
GXState g_gxState{};
void set_viewport_policy(AuroraViewportPolicy policy) noexcept {}
} // namespace aurora::gx

namespace aurora::vi {
Vec2<uint32_t> configured_fb_size() noexcept { return {640, 480}; }
void configure(const GXRenderModeObj*) noexcept {}
// What the FIFO registers, for the VI size test.
ConfigureHook g_testConfigureHook = nullptr;
void set_configure_hook(ConfigureHook hook) noexcept { g_testConfigureHook = hook; }
} // namespace aurora::vi

// --- get_texture ---
namespace aurora::gx {
const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[id]; }
namespace texture {
void invalidate_bindings() noexcept {}
uint64_t current_bind_generation() noexcept { return 1; }
void end_frame() noexcept {}
} // namespace texture
void evict_texture_object(u32 texObjId) noexcept {
  for (auto& obj : g_gxState.loadedTextures) {
    if (obj.texObjId == texObjId) {
      obj.set_no_cache(true);
    }
  }
}
void evict_tlut_object(u32 tlutObjId) noexcept {
  for (auto& obj : g_gxState.loadedTluts) {
    if (obj.tlutObjId == tlutObjId) {
      obj.set_no_cache(true);
    }
  }
}
void evict_copy_texture(const void* dest) noexcept {
  g_gxState.copyTextures.erase(dest);
  for (auto it = g_gxState.copyTextureCache.begin(); it != g_gxState.copyTextureCache.end();) {
    if (it->first.dest == dest) {
      g_gxState.copyTextureCache.erase(it++);
    } else {
      ++it;
    }
  }
}
void shutdown() noexcept {}
Vec2<uint32_t> logical_fb_size() noexcept { return {640, 480}; }
gfx::Viewport map_logical_viewport(const gfx::Viewport& logicalViewport) noexcept { return logicalViewport; }
gfx::ClipRect map_logical_scissor(const gfx::ClipRect& logicalScissor) noexcept { return logicalScissor; }
void set_logical_viewport(const gfx::Viewport& viewport) noexcept {
  g_gxState.logicalViewport = viewport;
  set_render_viewport(map_logical_viewport(viewport));
}
void set_render_viewport(const gfx::Viewport& viewport) noexcept { g_gxState.renderViewport = viewport; }
void set_logical_scissor(const gfx::ClipRect& scissor) noexcept {
  g_gxState.logicalScissor = scissor;
  set_render_scissor(map_logical_scissor(scissor));
}
void set_render_scissor(const gfx::ClipRect& scissor) noexcept { g_gxState.renderScissor = scissor; }
} // namespace aurora::gx

// --- Shader/pipeline stubs ---
namespace aurora::gx {
uint32_t g_testPipelineBuildCount = 0;
void populate_pipeline_config(PipelineConfig& config, GXPrimitive primitive, GXVtxFmt fmt) noexcept {
  ++g_testPipelineBuildCount;
  // Whether the draw has normals, which decides whether a shader writes the normal attachment.
  config.shaderConfig.attrs[GX_VA_NRM].attrType = static_cast<u8>(g_gxState.vtxDesc[GX_VA_NRM]);
}
GXBindGroups build_bind_groups(const ShaderInfo& info) noexcept { return {}; }
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept { return {}; }
// The pieces pushed, as UniformPiece bits, for the normal target test.
u8 g_testPushedPieces = 0;
void push_uniforms(const ShaderInfo& info, u8 pieces, std::array<u32, UniformPieceCount>& offsets) noexcept {
  g_testPushedPieces |= pieces;
}
void resolve_sampled_textures(const ShaderInfo& info) noexcept {}
} // namespace aurora::gx

// --- Buffer push stubs ---
namespace aurora::gfx {
// The last draw's vertex bytes, for the display list tests.
std::vector<uint8_t> g_testLastVerts;
Range push_verts(const uint8_t* data, size_t length, size_t alignment) {
  g_testLastVerts.assign(data, data + length);
  return {};
}
Range push_verts_hashed(const uint8_t* data, size_t length, size_t alignment, uint64_t& hash) {
  hash = 0;
  return push_verts(data, length, alignment);
}
// The frame's index pool, for the index pattern tests.
std::vector<uint8_t> g_testIndices;
Range push_indices(const uint8_t* data, size_t length, size_t alignment) {
  if (alignment != 0) {
    g_testIndices.resize((g_testIndices.size() + alignment - 1) / alignment * alignment);
  }
  const Range range{static_cast<uint32_t>(g_testIndices.size()), static_cast<uint32_t>(length)};
  g_testIndices.insert(g_testIndices.end(), data, data + length);
  return range;
}
Range push_uniform(const uint8_t* data, size_t length) { return {}; }
// Counted, each at its own offset, for the array copy test.
uint32_t g_testStoragePushes = 0;
Range push_storage(const uint8_t* data, size_t length) {
  return {.offset = 256 * g_testStoragePushes++, .size = static_cast<uint32_t>(length)};
}
Range push_array(const uint8_t* data, size_t length) { return push_storage(data, length); }

Vec2<uint32_t> get_render_target_size() noexcept { return {640, 480}; }
void set_viewport(const Viewport& viewport) noexcept {}
void set_scissor(uint32_t x, uint32_t y, uint32_t w, uint32_t h) noexcept {}
bool g_testNormalAttachment = false;
bool has_normal_attachment() noexcept { return g_testNormalAttachment; }
uint64_t get_render_target_layout_key() noexcept { return get_render_target_layout().key; }
RenderTargetLayout get_render_target_layout() noexcept {
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{ColorAttachmentSemantic::SceneColor, TextureFormat::RGBA8Unorm}}},
      .depthStencilFormat = TextureFormat::Depth32Float,
      .sampleCount = 1,
  };
  if (g_testNormalAttachment) {
    layout.colorAttachmentCount = 2;
    layout.colorAttachments[1] = {ColorAttachmentSemantic::Normal, TextureFormat::RGB10A2Unorm};
  }
  detail::finalize_render_target_layout(layout);
  return layout;
}
} // namespace aurora::gfx

// --- Pipeline/draw command stubs ---
namespace aurora::gfx {
template <>
PipelineRef pipeline_ref<clear::PipelineConfig>(const clear::PipelineConfig& config) {
  return 0;
}
template <>
void push_draw_command<clear::DrawData>(clear::DrawData data) {
  // No-op
}
template <>
PipelineRef pipeline_ref<gx::PipelineConfig>(const gx::PipelineConfig& config) {
  return 0;
}
gx::DrawData g_testLastDraw{};
uint32_t g_testDrawCount = 0;
std::atomic<uint32_t> g_testProcessedDrawCount{0};

template <>
void push_draw_command<gx::DrawData>(gx::DrawData data) {
  g_testLastDraw = data;
  ++g_testDrawCount;
  g_testProcessedDrawCount.fetch_add(1, std::memory_order_release);
}
// Whether a draw may merge into the last one, for the index pattern tests.
bool g_testMergeDraws = false;
template <>
gx::DrawData* get_last_draw_command() {
  return g_testMergeDraws ? &g_testLastDraw : nullptr;
}

std::vector<TestFrameEvent> g_testFrameEvents;
namespace {
void record_frame_event(TestFrameEvent::Kind kind, size_t slot) {
  const auto& state = gx::g_gxState;
  g_testFrameEvents.push_back(
      {kind, slot, g_testDrawCount, state.viewportPolicy, state.viSize, state.bpRegCache[0x41]});
}
} // namespace
namespace detail {
void begin_frame_recording(size_t slot) noexcept { record_frame_event(TestFrameEvent::Begin, slot); }
void note_fifo_done(size_t slot) noexcept {}
void end_frame_recording(size_t slot, bool close) noexcept {
  record_frame_event(close ? TestFrameEvent::Close : TestFrameEvent::End, slot);
}
} // namespace detail
} // namespace aurora::gfx

// --- TextureBind::sampler_config ---
namespace aurora::gfx {
SamplerConfig TextureBind::sampler_config() const noexcept { return {}; }
} // namespace aurora::gfx

// --- Texture creation/write/replacement stubs ---
namespace aurora::gfx {
TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                    ArrayRef<uint8_t> data, bool tlut, const char* label) noexcept {
  return {};
}
TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                     const char* label) noexcept {
  return {};
}
TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  return {};
}
TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept { return {}; }
void queue_texture_upload(TextureUpload upload) {}
bool queue_texture_upload_data(const uint8_t* data, TextureUpload upload, uint32_t blockSize) { return true; }
void queue_palette_conv(tex_palette_conv::ConvRequest req) {}
namespace testing {
std::atomic<uint32_t> beginOffscreenCount{0};
std::atomic<uint32_t> endOffscreenCount{0};
std::atomic<uint32_t> resolvePassCount{0};
ResolveCall lastResolve;
std::atomic<uint32_t> offscreenWidth{0};
std::atomic<uint32_t> offscreenHeight{0};
} // namespace testing

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat) {
  testing::lastResolve = {
      .target = static_cast<bool>(texture),
      .rect = rect,
      .clearColor = clearColor,
      .clearAlpha = clearAlpha,
      .clearDepth = clearDepth,
      .clearColorValue = clearColorValue,
      .clearDepthValue = clearDepthValue,
  };
  testing::resolvePassCount.fetch_add(1, std::memory_order_release);
}
void begin_offscreen(uint32_t width, uint32_t height) {
  testing::offscreenWidth.store(width, std::memory_order_relaxed);
  testing::offscreenHeight.store(height, std::memory_order_relaxed);
  testing::beginOffscreenCount.fetch_add(1, std::memory_order_release);
}
void end_offscreen() { testing::endOffscreenCount.fetch_add(1, std::memory_order_release); }
bool is_offscreen() noexcept { return false; }
} // namespace aurora::gfx

namespace aurora::gfx::depth_peek {
namespace {
bool s_snapshotRequested = false;
uint32_t s_width = 0;
uint32_t s_height = 0;
std::vector<uint32_t> s_data;
} // namespace

void initialize() {}
void shutdown() {}
void request_snapshot() noexcept { s_snapshotRequested = true; }
void poll() noexcept {}
void encode_frame_snapshot(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView,
                           wgpu::Extent3D sourceSize, uint32_t msaaSamples) noexcept {}
void after_submit() noexcept {}

bool read_latest(uint16_t x, uint16_t y, uint32_t& z) noexcept {
  if (x >= s_width || y >= s_height || s_data.empty()) {
    return false;
  }
  z = s_data[static_cast<size_t>(y) * s_width + x] & 0x00ffffffu;
  return true;
}

namespace testing {
void reset() noexcept {
  s_snapshotRequested = false;
  s_width = 0;
  s_height = 0;
  s_data.clear();
}

bool snapshot_requested() noexcept { return s_snapshotRequested; }

void set_latest(uint32_t width, uint32_t height, const std::vector<uint32_t>& data) {
  s_width = width;
  s_height = height;
  s_data = data;
}
} // namespace testing
} // namespace aurora::gfx::depth_peek

namespace aurora::gfx::backend {
// No backend's fog range table, so Aurora's, as under Dawn and the null backend; a test may set what it needs.
Limits g_testLimits{};
const Limits& limits() noexcept { return g_testLimits; }
void request_depth_snapshot() noexcept { depth_peek::request_snapshot(); }
bool read_depth(uint16_t x, uint16_t y, uint32_t& z) noexcept { return depth_peek::read_latest(x, y, z); }
bool converts_copy(GXTexFmt format) noexcept { return false; }
} // namespace aurora::gfx::backend

namespace aurora::gfx::tex_palette_conv {
void queue(ConvRequest req) {}
} // namespace aurora::gfx::tex_palette_conv

namespace aurora::gfx::texture_replacement {
u32 compute_texture_upload_size(const GXTexObj_& obj) noexcept { return 0; }
bool has_replacement(const GXTexObj_&) noexcept { return false; }
bool has_replacement(const GXTexObj_&, const GXTlutObj_&) noexcept { return false; }
} // namespace aurora::gfx::texture_replacement

// --- Window stub ---
#include "../lib/window.hpp"
namespace aurora::window {
AuroraWindowSize get_window_size() { return {640, 480, 640, 480, 640, 480, 1.0f}; }
void set_frame_buffer_aspect_fit(bool) {}
} // namespace aurora::window

// --- WebGPU C API stubs (prevent linker errors from wgpu:: destructors) ---
extern "C" {
void wgpuDeviceRelease(WGPUDevice) {}
void wgpuQueueRelease(WGPUQueue) {}
void wgpuSurfaceRelease(WGPUSurface) {}
void wgpuBufferRelease(WGPUBuffer) {}
void wgpuTextureRelease(WGPUTexture) {}
void wgpuTextureViewRelease(WGPUTextureView) {}
void wgpuSamplerRelease(WGPUSampler) {}
void wgpuShaderModuleRelease(WGPUShaderModule) {}
void wgpuRenderPipelineRelease(WGPURenderPipeline) {}
void wgpuBindGroupRelease(WGPUBindGroup) {}
void wgpuBindGroupLayoutRelease(WGPUBindGroupLayout) {}
void wgpuPipelineLayoutRelease(WGPUPipelineLayout) {}
void wgpuInstanceRelease(WGPUInstance) {}
void wgpuDeviceAddRef(WGPUDevice) {}
void wgpuQueueAddRef(WGPUQueue) {}
void wgpuSurfaceAddRef(WGPUSurface) {}
void wgpuBufferAddRef(WGPUBuffer) {}
void wgpuTextureAddRef(WGPUTexture) {}
void wgpuTextureViewAddRef(WGPUTextureView) {}
void wgpuInstanceAddRef(WGPUInstance) {}
}

void aurora::gfx::push_debug_group(std::string) {}
void push_debug_group(const char*) {}
void pop_debug_group() {}
void aurora::gfx::insert_debug_marker(std::string) {}

// GXInitTexObjLOD calls this, and the FIFO tests don't link the texture cache.
namespace aurora::gx::texture {
void preconvert_texture(const GXTexObj_&) noexcept {}
} // namespace aurora::gx::texture
