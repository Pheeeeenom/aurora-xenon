#pragma once
// The Dawn backend's own declarations, for the Dawn-only files.

#include "../gfx/backend.hpp"
#include "../gfx/clear.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/tex_copy_conv.hpp"
#include "../gfx/tex_palette_conv.hpp"
#include "../gx/pipeline.hpp"
#include "gpu.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace aurora::imgui {
class DrawData;

void render(const wgpu::RenderPassEncoder& pass, const DrawData& drawData) noexcept;
} // namespace aurora::imgui

namespace aurora::webgpu {
// gfx::TextureFormat's values are WebGPU's (checked in backend.cpp).
constexpr wgpu::TextureFormat to_wgpu(gfx::TextureFormat format) noexcept {
  return static_cast<wgpu::TextureFormat>(format);
}
// Undefined for a format the GX path has no name for.
gfx::TextureFormat to_gfx(wgpu::TextureFormat format) noexcept;

// Null for a null pipeline, which Dawn returns when it can't make one.
std::shared_ptr<gfx::backend::Pipeline> make_pipeline(wgpu::RenderPipeline pipeline) noexcept;

using AfterSubmitCallback = std::function<void()>;
// Render worker: draws the frame to the surface with the overlays, then submits and presents it, capturing it to
// capturePath when that isn't empty.
void present(wgpu::CommandEncoder& encoder, std::vector<AfterSubmitCallback> afterSubmitCallbacks,
             const gfx::Viewport& viewport, const imgui::DrawData& imguiDrawData, const wgpu::BindGroup& rmlBindGroup,
             bool rmlOverlay, const std::string& capturePath);

// The buffers, staging and bind groups the GX core's frames use.
void initialize_resources();
void shutdown_resources();

// What Dawn keeps for the frame in a frame slot.
struct FrameState {
  size_t stagingBuffer = 0;
  gfx::detail::StagingHighWater copied;
  wgpu::CommandEncoder encoder;
  // The frame's encoder tasks' completions, run after its submit.
  std::vector<AfterSubmitCallback> afterSubmitCallbacks;
  // Taken on the game thread as the frame's recording ends, for its present.
  gfx::Viewport viewport{};
  wgpu::BindGroup rmlBindGroup;
  bool rmlOverlay = false;
};
FrameState& frame_state(size_t slot) noexcept;

// The WebGPU extension API's state (webgpu/extension.cpp). Game thread, before a frame records.
void begin_extension_frame(size_t slot) noexcept;
void shutdown_extension() noexcept;
} // namespace aurora::webgpu

namespace aurora::gfx::detail {
inline constexpr size_t StagingBufferCount = FrameSlotCount + 3;
inline constexpr uint64_t StagingBufferSize = UniformBufferSize + VertexBufferSize + IndexBufferSize +
                                              StorageBufferSize + (UseTextureBuffer ? TextureUploadSize : 0);

const wgpu::Buffer& staging_buffer(size_t slot);

struct Resources {
  wgpu::Buffer vertexBuffer;
  wgpu::Buffer uniformBuffer;
  wgpu::Buffer indexBuffer;
  wgpu::Buffer storageBuffer;
  wgpu::BindGroupLayout staticBindGroupLayout;
  wgpu::BindGroup staticBindGroup;
  wgpu::BindGroupLayout uniformBindGroupLayout;
  wgpu::BindGroup uniformBindGroup;
  wgpu::Limits limits;
};

Resources& resources() noexcept;

struct RegisteredDrawType {
  DrawCallback draw = nullptr;
  void* userdata = nullptr;
};

struct RegisteredEncoderTaskType {
  EncoderTaskCallback callback = nullptr;
  void* userdata = nullptr;
  EncoderTaskCompletionCallback afterSubmit = nullptr;
};

std::optional<RegisteredDrawType> find_runtime_draw_type(DrawTypeId id);
std::optional<RegisteredEncoderTaskType> find_runtime_encoder_task_type(EncoderTaskId id);
} // namespace aurora::gfx::detail

namespace aurora::gfx {
wgpu::SamplerDescriptor to_wgpu(const SamplerConfig& config) noexcept;
} // namespace aurora::gfx

namespace aurora::gfx::tex_copy_conv {
enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct ConvRequest {
  GXTexFmt fmt;
  GXPixelFmt srcFmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // Uniforms
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);

bool snapshot_depth_supported() noexcept;
void snapshot_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& srcDepth, uint32_t msaaSamples,
                    const wgpu::TextureView& dst);
} // namespace aurora::gfx::tex_copy_conv

namespace aurora::gfx::tex_palette_conv {
void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
} // namespace aurora::gfx::tex_palette_conv

namespace aurora::gfx::clear {
wgpu::RenderPipeline create_pipeline(const PipelineConfig& config, const RenderTargetLayout& layout);
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass, const TextureExtent& targetSize);
} // namespace aurora::gfx::clear

namespace aurora::gx {
extern wgpu::BindGroup g_emptyTextureBindGroup;
// The index patterns (fifo::index_patterns), for draws whose patternIndices is set; none when they're off.
extern wgpu::Buffer g_indexPatternBuffer;

// The bind group layouts, the empty texture binding, the pipeline layout GX pipelines use and the index patterns.
void initialize_bindings() noexcept;
void release_bindings() noexcept;
wgpu::RenderPipeline build_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout,
                                    const PipelineOptions& options, ArrayRef<wgpu::VertexBufferLayout> vtxBuffers,
                                    wgpu::ShaderModule shader, const char* label) noexcept;
std::string build_shader_source(const ShaderConfig& config, DstAlphaMode dstAlphaMode,
                                uint32_t normalAttachment = UINT32_MAX) noexcept;
wgpu::ShaderModule build_shader(const ShaderConfig& config, const gfx::RenderTargetLayout& layout,
                                DstAlphaMode dstAlphaMode) noexcept;
gfx::CompiledPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout);
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);
wgpu::BindGroup create_texture_bind_group(const gfx::TextureSet& set) noexcept;
} // namespace aurora::gx

namespace aurora::gfx::backend {
struct Pipeline {
  wgpu::RenderPipeline pipeline;
};

struct Texture {
  wgpu::Texture texture;
  wgpu::TextureView sampleTextureView;
  wgpu::TextureView attachmentTextureView;
  // Can be a texture pack's format, which gfx::TextureFormat has no name for.
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
};

struct UploadBuffer {
  wgpu::Buffer buffer;
};

// A pass begin_color_pass opened, on views and with load and store operations of RmlUi's own.
struct PassTargets {
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::TextureView depthStencilView;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Store;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
};

struct TextureCopy {
  wgpu::TexelCopyTextureInfo src;
  wgpu::TexelCopyTextureInfo dst;
  wgpu::Extent3D size;
};
} // namespace aurora::gfx::backend

namespace aurora::gfx {
// queue_texture_upload_data for a texture the caller made with WebGPU itself, as RmlUi's renderer does.
bool queue_texture_upload_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               const wgpu::TexelCopyTextureInfo& tex, wgpu::Extent3D size);

// A pass on targets of RmlUi's own.
struct ColorPassDescriptor {
  const char* label = nullptr;
  wgpu::TextureView colorView;
  wgpu::TextureView resolveView;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  wgpu::TextureView depthStencilView;
  wgpu::TextureFormat depthStencilFormat = wgpu::TextureFormat::Undefined;
  wgpu::Extent3D targetSize;
  uint32_t sampleCount = 1;
  wgpu::LoadOp colorLoadOp = wgpu::LoadOp::Clear;
  wgpu::StoreOp colorStoreOp = wgpu::StoreOp::Store;
  wgpu::Color clearColor{0.f, 0.f, 0.f, 0.f};
  bool hasDepth = false;
  wgpu::LoadOp depthLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp depthStoreOp = wgpu::StoreOp::Undefined;
  float depthClearValue = 0.f;
  bool hasStencil = false;
  wgpu::LoadOp stencilLoadOp = wgpu::LoadOp::Undefined;
  wgpu::StoreOp stencilStoreOp = wgpu::StoreOp::Undefined;
  uint32_t stencilClearValue = 0;
};

void begin_color_pass(const ColorPassDescriptor& desc);
void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size);
} // namespace aurora::gfx
