// The WebGPU extension API (<aurora/gfx.hpp>) and RmlUi's passes, which only the Dawn backend has: what they record
// goes through the GX recorder, and Dawn's encoder replays it.
#include "backend.hpp"

#include "gpu.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/tex_copy_conv.hpp"
#include "../gfx/texture.hpp"
#include "../gx/fifo.hpp"
#include "../internal.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>

namespace aurora::gfx {
using namespace detail;
using webgpu::g_device;
using webgpu::g_queue;

namespace {
constexpr Module Log{"aurora::gfx"};

struct RuntimeDrawType {
  std::string label;
  DrawCallback draw = nullptr;
  void* userdata = nullptr;
  uint32_t generation = 1;
};

struct RuntimeEncoderTaskType {
  std::string label;
  EncoderTaskCallback callback = nullptr;
  void* userdata = nullptr;
  EncoderTaskCompletionCallback afterSubmit = nullptr;
  uint32_t generation = 1;
};

constexpr uint32_t draw_type_index(DrawTypeId id) { return static_cast<uint32_t>(id & 0xFFFFFFFFu); }
constexpr uint32_t draw_type_generation(DrawTypeId id) { return static_cast<uint32_t>(id >> 32); }
constexpr DrawTypeId make_draw_type_id(uint32_t index, uint32_t generation) {
  return static_cast<DrawTypeId>(generation) << 32 | index;
}

std::vector<RuntimeDrawType> g_runtimeDrawTypes;
std::vector<uint32_t> g_freeDrawTypeSlots;
std::vector<RuntimeEncoderTaskType> g_runtimeEncoderTaskTypes;
std::vector<uint32_t> g_freeEncoderTaskTypeSlots;
std::mutex g_runtimeTypeMutex;

// Requires g_runtimeTypeMutex held.
const RuntimeDrawType* find_runtime_draw_type_locked(DrawTypeId id) {
  const auto idx = draw_type_index(id);
  if (id == InvalidDrawType || idx >= g_runtimeDrawTypes.size()) {
    return nullptr;
  }
  const auto& slot = g_runtimeDrawTypes[idx];
  if (slot.generation != draw_type_generation(id) || slot.draw == nullptr) {
    return nullptr;
  }
  return &slot;
}

// Requires g_runtimeTypeMutex held.
const RuntimeEncoderTaskType* find_runtime_encoder_task_type_locked(EncoderTaskId id) {
  const auto idx = draw_type_index(id);
  if (id == InvalidEncoderTask || idx >= g_runtimeEncoderTaskTypes.size()) {
    return nullptr;
  }
  const auto& slot = g_runtimeEncoderTaskTypes[idx];
  if (slot.generation != draw_type_generation(id) || slot.callback == nullptr) {
    return nullptr;
  }
  return &slot;
}

// Set by resolve_pass; the normal buffer is turned on before the next frame records.
bool g_normalRequested = false;

// Pooled destinations for the public resolve_pass API. Entries are recycled
// per frame slot: a slot is only re-acquired after the render worker has
// submitted its previous frame, and queue serialization orders the new frame's
// copies after the old frame's reads.
struct PassSnapshotEntry {
  webgpu::TextureWithSampler color;
  webgpu::TextureWithSampler depth; // R32Float raw depth
  webgpu::TextureWithSampler normal;
};
struct PassSnapshotPool {
  std::vector<PassSnapshotEntry> entries;
  size_t used = 0;
};
std::array<PassSnapshotPool, FrameSlotCount> g_passSnapshotPools;

PassSnapshotEntry& acquire_pass_snapshot(uint32_t width, uint32_t height, bool wantColor, bool wantDepth,
                                         bool wantNormal) {
  auto& pool = g_passSnapshotPools[current_frame_packet().slot];
  if (pool.used == pool.entries.size()) {
    pool.entries.emplace_back();
  }
  auto& entry = pool.entries[pool.used++];
  const wgpu::Extent3D size{width, height, 1};
  if (wantColor && (!entry.color.texture || entry.color.size.width != width || entry.color.size.height != height ||
                    entry.color.format != webgpu::g_graphicsConfig.surfaceConfiguration.format)) {
    const auto format = webgpu::g_graphicsConfig.surfaceConfiguration.format;
    const wgpu::TextureDescriptor desc{
        .label = "Pass Snapshot Color",
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = size,
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    auto texture = webgpu::g_device.CreateTexture(&desc);
    auto view = texture.CreateView();
    entry.color = webgpu::TextureWithSampler{
        .texture = std::move(texture),
        .view = std::move(view),
        .size = size,
        .format = format,
    };
  }
  if (wantDepth && (!entry.depth.texture || entry.depth.size.width != width || entry.depth.size.height != height)) {
    const wgpu::TextureDescriptor desc{
        .label = "Pass Snapshot Depth",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = size,
        .format = wgpu::TextureFormat::R32Float,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    auto texture = webgpu::g_device.CreateTexture(&desc);
    auto view = texture.CreateView();
    entry.depth = webgpu::TextureWithSampler{
        .texture = std::move(texture),
        .view = std::move(view),
        .size = size,
        .format = wgpu::TextureFormat::R32Float,
    };
  }
  if (wantNormal && (!entry.normal.texture || entry.normal.size.width != width || entry.normal.size.height != height)) {
    const wgpu::TextureDescriptor desc{
        .label = "Pass Snapshot Normal",
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc,
        .dimension = wgpu::TextureDimension::e2D,
        .size = size,
        .format = webgpu::NormalBufferFormat,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    auto texture = webgpu::g_device.CreateTexture(&desc);
    auto view = texture.CreateView();
    entry.normal = webgpu::TextureWithSampler{
        .texture = std::move(texture),
        .view = std::move(view),
        .size = size,
        .format = webgpu::NormalBufferFormat,
    };
  }
  return entry;
}

// A pass's snapshot destination, as the recorder holds textures.
TextureHandle snapshot_target(const webgpu::TextureWithSampler& texture) {
  return std::make_shared<TextureRef>(std::make_shared<backend::Texture>(backend::Texture{
                                          .texture = texture.texture,
                                          .sampleTextureView = texture.view,
                                          .attachmentTextureView = texture.view,
                                          .format = texture.format,
                                      }),
                                      TextureExtent{texture.size.width, texture.size.height},
                                      webgpu::to_gfx(texture.format), 1, InvalidTextureFormat);
}
} // namespace

namespace detail {
std::optional<RegisteredDrawType> find_runtime_draw_type(DrawTypeId id) {
  std::lock_guard lock{g_runtimeTypeMutex};
  const auto* slot = find_runtime_draw_type_locked(id);
  if (slot == nullptr) {
    return std::nullopt;
  }
  return RegisteredDrawType{.draw = slot->draw, .userdata = slot->userdata};
}

std::optional<RegisteredEncoderTaskType> find_runtime_encoder_task_type(EncoderTaskId id) {
  std::lock_guard lock{g_runtimeTypeMutex};
  const auto* slot = find_runtime_encoder_task_type_locked(id);
  if (slot == nullptr) {
    return std::nullopt;
  }
  return RegisteredEncoderTaskType{
      .callback = slot->callback,
      .userdata = slot->userdata,
      .afterSubmit = slot->afterSubmit,
  };
}
} // namespace detail

wgpu::Device device() noexcept { return g_device; }

wgpu::Queue queue() noexcept { return g_queue; }

wgpu::TextureFormat color_format() noexcept { return webgpu::g_graphicsConfig.surfaceConfiguration.format; }

wgpu::TextureFormat depth_format() noexcept { return webgpu::g_graphicsConfig.depthFormat; }

uint32_t sample_count() noexcept { return webgpu::g_graphicsConfig.msaaSamples; }

DrawTypeId register_draw_type(const DrawTypeDescriptor& desc) {
  if (desc.draw == nullptr) {
    Log.warn("register_draw_type: draw callback is null");
    return InvalidDrawType;
  }

  std::lock_guard lock{g_runtimeTypeMutex};
  uint32_t idx;
  if (!g_freeDrawTypeSlots.empty()) {
    idx = g_freeDrawTypeSlots.back();
    g_freeDrawTypeSlots.pop_back();
  } else {
    idx = static_cast<uint32_t>(g_runtimeDrawTypes.size());
    g_runtimeDrawTypes.emplace_back();
  }
  auto& slot = g_runtimeDrawTypes[idx];
  slot.label = desc.label != nullptr ? desc.label : "";
  slot.draw = desc.draw;
  slot.userdata = desc.userdata;
  return make_draw_type_id(idx, slot.generation);
}

void unregister_draw_type(DrawTypeId type) noexcept {
  std::lock_guard lock{g_runtimeTypeMutex};
  if (find_runtime_draw_type_locked(type) == nullptr) {
    return;
  }
  const auto idx = draw_type_index(type);
  auto& slot = g_runtimeDrawTypes[idx];
  slot.label.clear();
  slot.draw = nullptr;
  slot.userdata = nullptr;
  ++slot.generation;
  g_freeDrawTypeSlots.push_back(idx);
}

EncoderTaskId register_encoder_task_type(const EncoderTaskDescriptor& desc) {
  if (desc.callback == nullptr) {
    Log.warn("register_encoder_task_type: callback is null");
    return InvalidEncoderTask;
  }

  std::lock_guard lock{g_runtimeTypeMutex};
  uint32_t idx;
  if (!g_freeEncoderTaskTypeSlots.empty()) {
    idx = g_freeEncoderTaskTypeSlots.back();
    g_freeEncoderTaskTypeSlots.pop_back();
  } else {
    idx = static_cast<uint32_t>(g_runtimeEncoderTaskTypes.size());
    g_runtimeEncoderTaskTypes.emplace_back();
  }
  auto& slot = g_runtimeEncoderTaskTypes[idx];
  slot.label = desc.label != nullptr ? desc.label : "";
  slot.callback = desc.callback;
  slot.userdata = desc.userdata;
  slot.afterSubmit = desc.afterSubmit;
  return make_draw_type_id(idx, slot.generation);
}

void unregister_encoder_task_type(EncoderTaskId type) noexcept {
  std::lock_guard lock{g_runtimeTypeMutex};
  if (find_runtime_encoder_task_type_locked(type) == nullptr) {
    return;
  }
  const auto idx = draw_type_index(type);
  auto& slot = g_runtimeEncoderTaskTypes[idx];
  slot.label.clear();
  slot.callback = nullptr;
  slot.userdata = nullptr;
  slot.afterSubmit = nullptr;
  ++slot.generation;
  g_freeEncoderTaskTypeSlots.push_back(idx);
}

bool push_custom_draw(DrawTypeId type, const void* payload, size_t payloadSize) {
  if (type == InvalidDrawType) {
    Log.warn("push_custom_draw: invalid draw type");
    return false;
  }
  if (payloadSize > InlineDrawPayloadSize) {
    Log.warn("push_custom_draw: payload size {} exceeds inline payload size {}", payloadSize, InlineDrawPayloadSize);
    return false;
  }
  if (payloadSize > 0 && payload == nullptr) {
    Log.warn("push_custom_draw: non-zero payload size with null payload");
    return false;
  }
  if (!find_runtime_draw_type(type)) {
    Log.warn("push_custom_draw: unregistered draw type {:#x}", type);
    return false;
  }

  gx::fifo::drain();

  if (!in_render_pass()) {
    Log.warn("push_custom_draw: called outside an active render pass");
    return false;
  }

  CustomDrawCommand draw{};
  draw.type = type;
  draw.payloadSize = static_cast<uint32_t>(payloadSize);
  if (payloadSize > 0) {
    std::memcpy(draw.payload.data(), payload, payloadSize);
  }
  push_custom_draw_command(draw);
  return true;
}

bool resolve_pass(const ResolveDesc& desc, ResolvedTargets& out) {
  out = {};
  gx::fifo::drain();

  if (!in_render_pass()) {
    Log.warn("resolve_pass: called outside an active render pass");
    return false;
  }

  bool wantDepth = desc.depth;
  if (wantDepth && !tex_copy_conv::snapshot_depth_supported()) {
    Log.warn("resolve_pass: depth snapshots are unsupported on this device");
    wantDepth = false;
  }

  const auto [width, height] = get_render_target_size();
  // Only the EFB has a normal attachment, and only once the normal buffer is on.
  const bool wantNormal = desc.normal && has_normal_attachment();
  if (desc.normal && !is_offscreen() && !webgpu::g_graphicsConfig.normalBuffer && webgpu::g_hasCoreFeatures &&
      webgpu::g_graphicsConfig.msaaSamples == 1) {
    g_normalRequested = true;
  }
  // Requesting no snapshots is a plain pass break (or offscreen close, discarding its output).
  TextureHandle color;
  TextureHandle depth;
  TextureHandle normal;
  if (desc.color || wantDepth || wantNormal) {
    auto& entry = acquire_pass_snapshot(width, height, desc.color, wantDepth, wantNormal);
    if (desc.color) {
      color = snapshot_target(entry.color);
      out.colorTexture = entry.color.texture;
      out.color = entry.color.view;
      out.colorFormat = entry.color.format;
    }
    if (wantDepth) {
      depth = snapshot_target(entry.depth);
      out.depthTexture = entry.depth.texture;
      out.depth = entry.depth.view;
    }
    if (wantNormal) {
      normal = snapshot_target(entry.normal);
      out.normalTexture = entry.normal.texture;
      out.normal = entry.normal.view;
    }
  }
  out.width = width;
  out.height = height;
  resolve_current_pass(std::move(color), std::move(depth), std::move(normal));
  return true;
}

bool push_encoder_task(EncoderTaskId type, const void* payload, size_t payloadSize) {
  if (type == InvalidEncoderTask) {
    Log.warn("push_encoder_task: invalid encoder task type");
    return false;
  }
  if (payloadSize > InlineDrawPayloadSize) {
    Log.warn("push_encoder_task: payload size {} exceeds inline payload size {}", payloadSize, InlineDrawPayloadSize);
    return false;
  }
  if (payloadSize > 0 && payload == nullptr) {
    Log.warn("push_encoder_task: non-zero payload size with null payload");
    return false;
  }
  if (!find_runtime_encoder_task_type(type)) {
    Log.warn("push_encoder_task: unregistered encoder task type {:#x}", type);
    return false;
  }

  gx::fifo::drain();

  if (!in_render_pass()) {
    Log.warn("push_encoder_task: called outside an active render pass");
    return false;
  }
  if (is_offscreen()) {
    Log.warn("push_encoder_task: unsupported while an offscreen pass is active");
    return false;
  }

  EncoderTask task{.type = type};
  task.payloadSize = static_cast<uint32_t>(payloadSize);
  if (payloadSize > 0) {
    std::memcpy(task.payload.data(), payload, payloadSize);
  }
  detail::push_encoder_task(task);
  return true;
}

void queue_texture_copy(wgpu::TexelCopyTextureInfo src, wgpu::TexelCopyTextureInfo dst, wgpu::Extent3D size) {
  push_texture_copy(std::make_shared<const backend::TextureCopy>(backend::TextureCopy{
      .src = std::move(src),
      .dst = std::move(dst),
      .size = size,
  }));
}

void begin_color_pass(const ColorPassDescriptor& desc) {
  RenderPass pass{
      .label = desc.label != nullptr ? desc.label : "",
      .colorAttachmentCount = 1,
      .depthStencilFormat = webgpu::to_gfx(desc.depthStencilFormat),
      .backendTargets = std::make_shared<const backend::PassTargets>(backend::PassTargets{
          .colorView = desc.colorView,
          .resolveView = desc.resolveView,
          .colorLoadOp = desc.colorLoadOp,
          .colorStoreOp = desc.colorStoreOp,
          .depthStencilView = desc.depthStencilView,
          .depthLoadOp = desc.depthLoadOp,
          .depthStoreOp = desc.depthStoreOp,
          .stencilLoadOp = desc.stencilLoadOp,
          .stencilStoreOp = desc.stencilStoreOp,
          .stencilClearValue = desc.stencilClearValue,
      }),
      .msaaSamples = desc.sampleCount,
      .clearDepthValue = desc.depthClearValue,
      .clearDepth = desc.depthLoadOp == wgpu::LoadOp::Clear,
      .hasDepth = desc.hasDepth,
      .hasStencil = desc.hasStencil,
  };
  auto& color = pass.colorAttachments[SceneColorAttachmentIndex];
  color.semantic = ColorAttachmentSemantic::SceneColor;
  color.format = webgpu::to_gfx(desc.colorFormat);
  color.size = {desc.targetSize.width, desc.targetSize.height};
  color.clearValue = {
      static_cast<float>(desc.clearColor.r),
      static_cast<float>(desc.clearColor.g),
      static_cast<float>(desc.clearColor.b),
      static_cast<float>(desc.clearColor.a),
  };
  color.clear = desc.colorLoadOp == wgpu::LoadOp::Clear;
  begin_backend_pass(std::move(pass));
}
} // namespace aurora::gfx

namespace aurora::webgpu {
void begin_extension_frame(size_t slot) noexcept {
  // The normal attachment changes the EFB's layout, which the FIFO processor reads for the frames before.
  if (gfx::g_normalRequested) {
    gx::fifo::finish_published();
  }
  if (gfx::g_normalRequested && enable_normal_buffer()) {
    gfx::g_normalRequested = false;
  }
  gfx::g_passSnapshotPools[slot].used = 0;
}

void shutdown_extension() noexcept {
  {
    std::lock_guard lock{gfx::g_runtimeTypeMutex};
    gfx::g_runtimeDrawTypes.clear();
    gfx::g_freeDrawTypeSlots.clear();
    gfx::g_runtimeEncoderTaskTypes.clear();
    gfx::g_freeEncoderTaskTypeSlots.clear();
  }
  for (auto& pool : gfx::g_passSnapshotPools) {
    pool = {};
  }
  gfx::g_normalRequested = false;
}
} // namespace aurora::webgpu
