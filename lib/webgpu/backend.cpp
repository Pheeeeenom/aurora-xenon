// The Dawn implementation of Aurora's GX backend interface.
#include "backend.hpp"

#include "gpu.hpp"
#include "gpu_prof.hpp"
#include "../gfx/depth_peek.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/render_worker.hpp"
#include "../gfx/resource_cache.hpp"
#include "../gfx/tex_copy_conv.hpp"
#include "../gfx/tex_palette_conv.hpp"
#include "../gfx/texture_convert.hpp"
#include "../gx/command_processor.hpp"
#include "../gx/gx.hpp"
#include "../gx/pipeline.hpp"
#include "../gx/shader_info.hpp"
#include "../internal.hpp"
#include "../window.hpp"
#ifdef AURORA_ENABLE_RMLUI
#include "../rmlui.hpp"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include <SDL3/SDL_error.h>
#include <absl/container/flat_hash_map.h>
#include <fmt/format.h>
#include <magic_enum.hpp>
#include <tracy/Tracy.hpp>

namespace aurora::webgpu {
namespace {
#define CHECK_FORMAT(name)                                                                                             \
  static_assert(static_cast<uint32_t>(gfx::TextureFormat::name) == static_cast<uint32_t>(wgpu::TextureFormat::name))
CHECK_FORMAT(Undefined);
CHECK_FORMAT(R8Unorm);
CHECK_FORMAT(R16Sint);
CHECK_FORMAT(RG8Unorm);
CHECK_FORMAT(R32Float);
CHECK_FORMAT(RGBA8Unorm);
CHECK_FORMAT(BGRA8Unorm);
CHECK_FORMAT(RGB10A2Unorm);
CHECK_FORMAT(Stencil8);
CHECK_FORMAT(Depth32Float);
CHECK_FORMAT(BC1RGBAUnorm);
#undef CHECK_FORMAT
} // namespace

gfx::TextureFormat to_gfx(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::R8Unorm:
  case wgpu::TextureFormat::R16Sint:
  case wgpu::TextureFormat::RG8Unorm:
  case wgpu::TextureFormat::R32Float:
  case wgpu::TextureFormat::RGBA8Unorm:
  case wgpu::TextureFormat::BGRA8Unorm:
  case wgpu::TextureFormat::RGB10A2Unorm:
  case wgpu::TextureFormat::Stencil8:
  case wgpu::TextureFormat::Depth32Float:
  case wgpu::TextureFormat::BC1RGBAUnorm:
    return static_cast<gfx::TextureFormat>(format);
  default:
    return gfx::TextureFormat::Undefined;
  }
}
} // namespace aurora::webgpu

namespace aurora::gfx {
using webgpu::g_device;

namespace {
bool setup_swizzle(wgpu::TextureComponentSwizzleDescriptor& swizzle, u32 format, TextureFormat uploadFormat) noexcept {
  if (!webgpu::g_textureComponentSwizzleSupported) {
    return false;
  }

  switch (format) {
  case GX_TF_I8:
    // Uploaded as R8 by convert_texture; an EFB copy's I8 texture is RGBA8 and needs nothing.
    if (uploadFormat != TextureFormat::R8Unorm) {
      return false;
    }
    [[fallthrough]];
  case GX_TF_R8_PC:
    swizzle.swizzle.r = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.g = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.b = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.a = wgpu::ComponentSwizzle::R;
    return true;
  case GX_TF_RG8_PC:
    swizzle.swizzle.r = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.g = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.b = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.a = wgpu::ComponentSwizzle::G;
    return true;
  default:
    return false;
  }
}
} // namespace

namespace backend {
std::shared_ptr<Texture> create_texture(const TextureDesc& desc) noexcept {
  const auto format = webgpu::to_wgpu(desc.format);
  auto usage = wgpu::TextureUsage::TextureBinding;
  switch (desc.usage) {
  case TextureUsage::Sampled:
    usage |= wgpu::TextureUsage::CopyDst;
    break;
  case TextureUsage::Copy:
    usage |= wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment;
    break;
  case TextureUsage::Conv:
  case TextureUsage::Depth:
    usage |= wgpu::TextureUsage::RenderAttachment;
    break;
  case TextureUsage::Render:
    usage |= wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
    break;
  }
  const wgpu::TextureDescriptor textureDescriptor{
      .label = desc.label,
      .usage = usage,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {desc.width, desc.height, 1},
      .format = format,
      .mipLevelCount = desc.mips,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);
  const auto viewLabel = fmt::format("{} view", desc.label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = format,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = desc.mips,
  };
  wgpu::TextureComponentSwizzleDescriptor swizzle;
  if (desc.usage == TextureUsage::Sampled && setup_swizzle(swizzle, desc.gxFormat, desc.format)) {
    textureViewDescriptor.nextInChain = &swizzle;
  }
  auto view = texture.CreateView(&textureViewDescriptor);
  // Targets are sampled and drawn to through the same view.
  auto attachmentView = desc.usage == TextureUsage::Sampled ? wgpu::TextureView{} : view;
  return std::make_shared<Texture>(Texture{
      .texture = std::move(texture),
      .sampleTextureView = std::move(view),
      .attachmentTextureView = std::move(attachmentView),
      .format = format,
  });
}

std::shared_ptr<UploadBuffer> stage_upload(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage) noexcept {
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, limits().textureRowAlignment);
  const uint64_t uploadSize = copyBytesPerRow * rowsPerImage;
  const wgpu::BufferDescriptor descriptor{
      .label = "Overflow Texture Upload Buffer",
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = uploadSize,
      .mappedAtCreation = true,
  };
  auto buffer = g_device.CreateBuffer(&descriptor);
  auto* dst = static_cast<uint8_t*>(buffer.GetMappedRange(0, uploadSize));
  for (uint32_t row = 0; row < rowsPerImage; ++row) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  buffer.Unmap();
  return std::make_shared<UploadBuffer>(UploadBuffer{std::move(buffer)});
}
} // namespace backend

bool queue_texture_upload_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               const wgpu::TexelCopyTextureInfo& tex, wgpu::Extent3D size) {
  // Rows here are multiples of 256 bytes, so every offset is on a block.
  return queue_texture_upload_data(data,
                                   {
                                       .texture = std::make_shared<backend::Texture>(backend::Texture{
                                           .texture = tex.texture,
                                       }),
                                       .mipLevel = tex.mipLevel,
                                       .bytesPerRow = bytesPerRow,
                                       .rowsPerImage = rowsPerImage,
                                       .width = size.width,
                                       .height = size.height,
                                       .layers = size.depthOrArrayLayers,
                                   },
                                   1);
}
} // namespace aurora::gfx

namespace aurora::gfx {
using namespace detail;
using webgpu::g_instance;

namespace {
constexpr Module Log{"aurora::gfx"};

Resources g_resources;

enum class BufferMapState {
  Unmapped,
  Mapping,
  Mapped,
};

std::array<wgpu::Buffer, StagingBufferCount> g_stagingBuffers;
std::array<std::atomic<BufferMapState>, StagingBufferCount> g_mappingStates;
render_worker::FrameSlotPool g_stagingSlots{StagingBufferCount};
std::atomic_bool g_processEventsQueued = false;
std::array<webgpu::FrameState, FrameSlotCount> g_frames;
backend::Limits g_limits{};

void process_events() {
  ZoneScopedN("ProcessEvents");
  if (g_instance) {
    g_instance.ProcessEvents();
  }
}

void enqueue_process_events() {
  if (render_worker::is_worker_thread()) {
    process_events();
    return;
  }

  bool expected = false;
  if (!g_processEventsQueued.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  render_worker::enqueue_work([] {
    process_events();
    g_processEventsQueued.store(false, std::memory_order_release);
  });
}

void wait_for_gpu_progress(std::chrono::nanoseconds sleepDuration) {
  if (render_worker::is_idle()) {
    enqueue_process_events();
  }
  std::this_thread::sleep_for(sleepDuration);
}

void map_staging_buffer(size_t slot, bool releaseSlotOnCompletion = false) {
  auto expected = BufferMapState::Unmapped;
  if (!g_mappingStates[slot].compare_exchange_strong(expected, BufferMapState::Mapping, std::memory_order_acq_rel,
                                                     std::memory_order_acquire)) {
    return;
  }

  g_stagingBuffers[slot].MapAsync(
      wgpu::MapMode::Write, 0, StagingBufferSize, wgpu::CallbackMode::AllowSpontaneous,
      [slot, releaseSlotOnCompletion](wgpu::MapAsyncStatus status, wgpu::StringView message) {
        if (status == wgpu::MapAsyncStatus::CallbackCancelled || status == wgpu::MapAsyncStatus::Aborted) {
          Log.warn("Buffer mapping {}: {}", magic_enum::enum_name(status), message);
          g_mappingStates[slot].store(BufferMapState::Unmapped, std::memory_order_release);
          if (releaseSlotOnCompletion) {
            g_stagingSlots.release(slot);
          }
          return;
        }
        AURORA_ASSERT(status == wgpu::MapAsyncStatus::Success, "Buffer mapping failed: {} {}",
                      magic_enum::enum_name(status), message);
        g_mappingStates[slot].store(BufferMapState::Mapped, std::memory_order_release);
        if (releaseSlotOnCompletion) {
          g_stagingSlots.release(slot);
        }
      });
}

bool wait_for_staging_buffer(size_t slot) {
  ZoneScopedN("Wait for buffer map");
  map_staging_buffer(slot);
  while (true) {
    const auto mappingState = g_mappingStates[slot].load(std::memory_order_acquire);
    if (mappingState == BufferMapState::Mapped) {
      return true;
    }
    if (mappingState == BufferMapState::Unmapped) {
      return false;
    }
    wait_for_gpu_progress(std::chrono::milliseconds{1});
  }
}

std::optional<size_t> acquire_mapped_staging_buffer() {
  ZoneScopedN("Acquire mapped staging buffer");
  while (true) {
    // Waits on the pool, woken by a release, rather than polling; Dawn's events are pumped between waits.
    if (auto slot = g_stagingSlots.acquire_for(std::chrono::milliseconds{1})) {
      if (wait_for_staging_buffer(*slot)) {
        return *slot;
      }
      g_stagingSlots.release(*slot);
      return std::nullopt;
    }
    if (render_worker::is_idle()) {
      enqueue_process_events();
    }
  }
}
} // namespace

namespace detail {
Resources& resources() noexcept { return g_resources; }

const wgpu::Buffer& staging_buffer(size_t slot) { return g_stagingBuffers[slot]; }
} // namespace detail
} // namespace aurora::gfx

namespace aurora::webgpu {
using namespace gfx;

FrameState& frame_state(size_t slot) noexcept { return g_frames[slot]; }

void initialize_resources() {
  g_processEventsQueued.store(false, std::memory_order_release);
  // For uniform & storage buffer offset alignments
  g_device.GetLimits(&g_resources.limits);
  g_limits = {
      .uniformAlignment = g_resources.limits.minUniformBufferOffsetAlignment,
      .storageAlignment = g_resources.limits.minStorageBufferOffsetAlignment,
      // CopyBufferToTexture's row pitch (Dawn's kTextureBytesPerRowAlignment).
      .textureRowAlignment = 256,
      .bcTextures = g_bcTexturesSupported,
      .cmprAsBc1 = g_cmprAsBc1,
      .bcPowerOfTwo = false,
      .componentSwizzle = g_textureComponentSwizzleSupported,
      .pipelineThreads = g_backendType != wgpu::BackendType::WebGPU,
  };
  depth_peek::initialize();
  tex_copy_conv::initialize();
  tex_palette_conv::initialize();

  const auto createBuffer = [](wgpu::Buffer& out, wgpu::BufferUsage usage, uint64_t size, const char* label) {
    if (size <= 0) {
      return;
    }
    const wgpu::BufferDescriptor descriptor{
        .label = label,
        .usage = usage,
        .size = size,
    };
    out = g_device.CreateBuffer(&descriptor);
  };
  createBuffer(g_resources.uniformBuffer, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, UniformBufferSize,
               "Shared Uniform Buffer");
  createBuffer(g_resources.vertexBuffer,
               wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst, VertexBufferSize,
               "Shared Vertex Buffer");
  createBuffer(g_resources.indexBuffer, wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst, IndexBufferSize,
               "Shared Index Buffer");
  createBuffer(g_resources.storageBuffer, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, StorageBufferSize,
               "Shared Storage Buffer");
  for (size_t i = 0; i < g_stagingBuffers.size(); ++i) {
    const auto label = fmt::format("Staging Buffer {}", i);
    createBuffer(g_stagingBuffers[i], wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc, StagingBufferSize,
                 label.c_str());
  }
  for (auto& state : g_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
  for (size_t slot = 0; slot < g_stagingBuffers.size(); ++slot) {
    map_staging_buffer(slot);
  }

  {
    constexpr std::array layoutEntries{
        // Vertex data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Vertex,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
        // Storage data buffer
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
            .buffer =
                wgpu::BufferBindingLayout{
                    .type = wgpu::BufferBindingType::ReadOnlyStorage,
                },
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Static bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_resources.staticBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const std::array entries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .buffer = g_resources.vertexBuffer,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .buffer = g_resources.storageBuffer,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Static bind group",
        .layout = g_resources.staticBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_resources.staticBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  }

  {
    // One uniform buffer per uniform piece (gx::UniformPiece), each with a dynamic offset.
    std::array<wgpu::BindGroupLayoutEntry, gx::UniformPieceCount> layoutEntries;
    std::array<wgpu::BindGroupEntry, gx::UniformPieceCount> entries;
    for (uint32_t i = 0; i < gx::UniformPieceCount; ++i) {
      layoutEntries[i] = {
          .binding = i,
          .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
          .buffer =
              wgpu::BufferBindingLayout{
                  .type = wgpu::BufferBindingType::Uniform,
                  .hasDynamicOffset = true,
              },
      };
      entries[i] = {
          .binding = i,
          .buffer = g_resources.uniformBuffer,
          .size = gx::UniformPieceSizes[i],
      };
    }
    const wgpu::BindGroupLayoutDescriptor layoutDesc{
        .label = "Uniform bind group layout",
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data(),
    };
    g_resources.uniformBindGroupLayout = g_device.CreateBindGroupLayout(&layoutDesc);
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .label = "Uniform bind group",
        .layout = g_resources.uniformBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_resources.uniformBindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  }

  gx::initialize_bindings();
}

void shutdown_resources() {
  g_processEventsQueued.store(false, std::memory_order_release);
  gx::release_bindings();
  depth_peek::shutdown();
  tex_copy_conv::shutdown();
  tex_palette_conv::shutdown();
  shutdown_resource_cache();
  shutdown_extension();
  g_resources.vertexBuffer = {};
  g_resources.uniformBuffer = {};
  g_resources.indexBuffer = {};
  g_resources.storageBuffer = {};
  g_stagingBuffers.fill({});
  g_frames.fill({});
  g_resources.staticBindGroup = {};
  g_resources.staticBindGroupLayout = {};
  g_resources.uniformBindGroup = {};
  g_resources.uniformBindGroupLayout = {};
  g_stagingSlots.reset();
  for (auto& state : g_mappingStates) {
    state.store(BufferMapState::Unmapped, std::memory_order_release);
  }
}
} // namespace aurora::webgpu

namespace aurora::gfx::backend {
const Limits& limits() noexcept { return g_limits; }

RenderTargetLayout efb_layout() noexcept {
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{
          .semantic = ColorAttachmentSemantic::SceneColor,
          .format = webgpu::to_gfx(webgpu::g_frameBuffer.format),
          .width = webgpu::g_frameBuffer.size.width,
          .height = webgpu::g_frameBuffer.size.height,
      }}},
      .depthStencilFormat = webgpu::to_gfx(webgpu::g_graphicsConfig.depthFormat),
      .sampleCount = webgpu::g_graphicsConfig.msaaSamples,
  };
  if (webgpu::g_graphicsConfig.normalBuffer) {
    layout.colorAttachments[layout.colorAttachmentCount++] = {
        .semantic = ColorAttachmentSemantic::Normal,
        .format = webgpu::to_gfx(webgpu::g_normalBuffer.format),
        .width = webgpu::g_normalBuffer.size.width,
        .height = webgpu::g_normalBuffer.size.height,
    };
  }
  finalize_render_target_layout(layout);
  return layout;
}

bool begin_frame(FramePacket& frame) noexcept {
  const auto stagingSlot = acquire_mapped_staging_buffer();
  if (!stagingSlot) {
    return false;
  }
  auto& state = g_frames[frame.slot];
  state = {.stagingBuffer = *stagingSlot};
  size_t bufferOffset = 0;
  const auto& stagingBuf = g_stagingBuffers[*stagingSlot];
  const auto mapBuffer = [&](ByteBuffer& buf, uint64_t size) {
    if (size <= 0) {
      return;
    }
    buf = ByteBuffer{static_cast<u8*>(stagingBuf.GetMappedRange(bufferOffset, size)), static_cast<size_t>(size)};
    bufferOffset += size;
  };
  mapBuffer(frame.verts, VertexBufferSize);
  mapBuffer(frame.uniforms, UniformBufferSize);
  mapBuffer(frame.indices, IndexBufferSize);
  mapBuffer(frame.storage, StorageBufferSize);
  if constexpr (UseTextureBuffer) {
    mapBuffer(frame.textureUpload, TextureUploadSize);
  }
  webgpu::begin_extension_frame(frame.slot);
  return true;
}

void start_frame(FramePacket& frame) noexcept {
  render_worker::enqueue_begin_frame(frame.frameId, [slot = frame.slot] {
    constexpr wgpu::CommandEncoderDescriptor EncoderDescriptor{.label = "Redraw encoder"};
    g_frames[slot].encoder = g_device.CreateCommandEncoder(&EncoderDescriptor);
    webgpu::gpu_prof::frame_begin(g_frames[slot].encoder);
  });
}

void finish_frame(FramePacket& frame) noexcept {
  auto& state = g_frames[frame.slot];
  const auto& presentSource = webgpu::present_source();
  state.viewport = webgpu::calculate_present_viewport(webgpu::g_graphicsConfig.surfaceConfiguration.width,
                                                      webgpu::g_graphicsConfig.surfaceConfiguration.height,
                                                      presentSource.size.width, presentSource.size.height);
#ifdef AURORA_ENABLE_RMLUI
  if (rmlui::is_initialized()) {
    auto rmlFrame = rmlui::record_frame(state.viewport);
    state.rmlBindGroup = std::move(rmlFrame.bindGroup);
    state.rmlOverlay = rmlFrame.overlay;
  }
#endif
}

// RmlUi records its own passes here (finish_frame), so its list is empty.
void end_frame(FramePacket& frame, const imgui::DrawData& overlay, const ui::Frame& ui) noexcept {
  (void)ui;
  auto& state = g_frames[frame.slot];
  const size_t stagingSlot = state.stagingBuffer;
  g_stagingBuffers[stagingSlot].Unmap();
  g_mappingStates[stagingSlot].store(BufferMapState::Unmapped, std::memory_order_release);
  auto encoder = std::move(state.encoder);
  auto afterSubmitCallbacks = std::move(state.afterSubmitCallbacks);
  auto rmlBindGroup = std::move(state.rmlBindGroup);
  webgpu::present(encoder, std::move(afterSubmitCallbacks), state.viewport, overlay, rmlBindGroup, state.rmlOverlay,
                  frame.capturePath);
  expire_cached_bind_groups();
  map_staging_buffer(stagingSlot, true);
  process_events();
}
} // namespace aurora::gfx::backend

namespace aurora::gx {
using webgpu::g_device;

namespace {
// "aurora::gx", the module this code logged under before it moved here.
aurora::Module Log("aurora::gx");
} // namespace

wgpu::BindGroup g_emptyTextureBindGroup;
wgpu::Buffer g_indexPatternBuffer;

namespace {
wgpu::Sampler sEmptySampler;
wgpu::Texture sEmptyTexture;
wgpu::TextureView sEmptyTextureView;
std::mutex sBindGroupLayoutMutex;
absl::flat_hash_map<u32, wgpu::BindGroupLayout> sUniformBindGroupLayouts;
absl::flat_hash_map<u32, std::pair<wgpu::BindGroupLayout, wgpu::BindGroupLayout>> sTextureBindGroupLayouts;
wgpu::BindGroupLayout sTextureBindGroupLayout;
wgpu::BindGroupLayout sSamplerBindGroupLayout;
wgpu::PipelineLayout sPipelineLayout;

template <typename T>
T round_away_from_zero(float value) noexcept {
  return static_cast<T>(value < 0.0f ? std::floor(value) : std::ceil(value));
}

wgpu::BlendFactor to_blend_factor(GXBlendFactor fac, bool isDst) {
  switch (fac) {
    DEFAULT_FATAL("invalid blend factor {}", underlying(fac));
  case GX_BL_ZERO:
    return wgpu::BlendFactor::Zero;
  case GX_BL_ONE:
    return wgpu::BlendFactor::One;
  case GX_BL_SRCCLR: // + GX_BL_DSTCLR
    if (isDst) {
      return wgpu::BlendFactor::Src;
    } else {
      return wgpu::BlendFactor::Dst;
    }
  case GX_BL_INVSRCCLR: // + GX_BL_INVDSTCLR
    if (isDst) {
      return wgpu::BlendFactor::OneMinusSrc;
    } else {
      return wgpu::BlendFactor::OneMinusDst;
    }
  case GX_BL_SRCALPHA:
    return wgpu::BlendFactor::SrcAlpha;
  case GX_BL_INVSRCALPHA:
    return wgpu::BlendFactor::OneMinusSrcAlpha;
  case GX_BL_DSTALPHA:
    return wgpu::BlendFactor::DstAlpha;
  case GX_BL_INVDSTALPHA:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

wgpu::CompareFunction to_compare_function(GXCompare func) {
  switch (func) {
    DEFAULT_FATAL("invalid depth fn {}", underlying(func));
  case GX_NEVER:
    return wgpu::CompareFunction::Never;
  case GX_LESS:
    return UseReversedZ ? wgpu::CompareFunction::Greater : wgpu::CompareFunction::Less;
  case GX_EQUAL:
    return wgpu::CompareFunction::Equal;
  case GX_LEQUAL:
    return UseReversedZ ? wgpu::CompareFunction::GreaterEqual : wgpu::CompareFunction::LessEqual;
  case GX_GREATER:
    return UseReversedZ ? wgpu::CompareFunction::Less : wgpu::CompareFunction::Greater;
  case GX_NEQUAL:
    return wgpu::CompareFunction::NotEqual;
  case GX_GEQUAL:
    return UseReversedZ ? wgpu::CompareFunction::LessEqual : wgpu::CompareFunction::GreaterEqual;
  case GX_ALWAYS:
    return wgpu::CompareFunction::Always;
  }
}

wgpu::BlendState to_blend_state(GXBlendMode mode, GXBlendFactor srcFac, GXBlendFactor dstFac, GXLogicOp op,
                                u32 dstAlpha, bool dualSource) {
  wgpu::BlendComponent colorBlendComponent;
  switch (mode) {
    DEFAULT_FATAL("unsupported blend mode {}", underlying(mode));
  case GX_BM_NONE:
    colorBlendComponent = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::Zero,
    };
    break;
  case GX_BM_BLEND:
    colorBlendComponent = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = to_blend_factor(srcFac, false),
        .dstFactor = to_blend_factor(dstFac, true),
    };
    break;
  case GX_BM_SUBTRACT:
    colorBlendComponent = {
        .operation = wgpu::BlendOperation::ReverseSubtract,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::One,
    };
    break;
  case GX_BM_LOGIC:
    switch (op) {
      DEFAULT_FATAL("unsupported logic op {}", underlying(op));
    case GX_LO_CLEAR:
      colorBlendComponent = {
          .operation = wgpu::BlendOperation::Add,
          .srcFactor = wgpu::BlendFactor::Zero,
          .dstFactor = wgpu::BlendFactor::Zero,
      };
      break;
    case GX_LO_COPY:
      colorBlendComponent = {
          .operation = wgpu::BlendOperation::Add,
          .srcFactor = wgpu::BlendFactor::One,
          .dstFactor = wgpu::BlendFactor::Zero,
      };
      break;
    case GX_LO_NOOP:
      colorBlendComponent = {
          .operation = wgpu::BlendOperation::Add,
          .srcFactor = wgpu::BlendFactor::Zero,
          .dstFactor = wgpu::BlendFactor::One,
      };
      break;
    }
    break;
  }
  if (dualSource) {
    const auto remap = [](wgpu::BlendFactor factor) {
      if (factor == wgpu::BlendFactor::SrcAlpha) {
        return wgpu::BlendFactor::Src1Alpha;
      }
      if (factor == wgpu::BlendFactor::OneMinusSrcAlpha) {
        return wgpu::BlendFactor::OneMinusSrc1Alpha;
      }
      return factor;
    };
    colorBlendComponent.srcFactor = remap(colorBlendComponent.srcFactor);
    colorBlendComponent.dstFactor = remap(colorBlendComponent.dstFactor);
  }
  wgpu::BlendComponent alphaBlendComponent;
  if (dstAlpha != UINT32_MAX) {
    alphaBlendComponent = wgpu::BlendComponent{
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = wgpu::BlendFactor::Constant,
        .dstFactor = wgpu::BlendFactor::Zero,
    };
  } else {
    alphaBlendComponent = colorBlendComponent;
  }
  return {
      .color = colorBlendComponent,
      .alpha = alphaBlendComponent,
  };
}

wgpu::ColorWriteMask to_write_mask(bool colorUpdate, bool alphaUpdate) {
  wgpu::ColorWriteMask writeMask = wgpu::ColorWriteMask::None;
  if (colorUpdate) {
    writeMask |= wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue;
  }
  if (alphaUpdate) {
    writeMask |= wgpu::ColorWriteMask::Alpha;
  }
  return writeMask;
}

wgpu::PrimitiveState to_primitive_state(GXCullMode gx_cullMode) {
  auto cullMode = wgpu::CullMode::None;
  switch (gx_cullMode) {
    DEFAULT_FATAL("unsupported cull mode {}", underlying(gx_cullMode));
  case GX_CULL_FRONT:
    cullMode = wgpu::CullMode::Front;
    break;
  case GX_CULL_BACK:
    cullMode = wgpu::CullMode::Back;
    break;
  case GX_CULL_ALL:
    // WebGPU has no mode that culls both faces. populate_pipeline_config turns every write off for GX_CULL_ALL
    // instead, so the cull mode chosen here doesn't matter.
    cullMode = wgpu::CullMode::Back;
    break;
  case GX_CULL_NONE:
    break;
  }
  return {
      .topology = wgpu::PrimitiveTopology::TriangleList,
      .stripIndexFormat = wgpu::IndexFormat::Undefined,
      .frontFace = wgpu::FrontFace::CW,
      .cullMode = cullMode,
  };
}
} // namespace

wgpu::RenderPipeline build_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout,
                                    const PipelineOptions& options, ArrayRef<wgpu::VertexBufferLayout> vtxBuffers,
                                    wgpu::ShaderModule shader, const char* label) noexcept {
  ZoneScoped;
  const float depthBias = (UseReversedZ ? -1.0f : 1.0f) * std::bit_cast<float>(config.polygonOffsetBits);
  const float depthBiasSlopeScale = (UseReversedZ ? -1.0f : 1.0f) * std::bit_cast<float>(config.polygonOffsetScaleBits);
  const float depthBiasClamp = webgpu::g_hasCoreFeatures ? std::bit_cast<float>(config.polygonOffsetClampBits) : 0.0f;
  const bool writesDepth = config.depthCompare && options.depthUpdate;
  const wgpu::DepthStencilState depthStencil{
      .format = webgpu::to_wgpu(layout.depthStencilFormat),
      .depthWriteEnabled = writesDepth,
      .depthCompare = config.depthCompare ? to_compare_function(config.depthFunc) : wgpu::CompareFunction::Always,
      .depthBias = round_away_from_zero<int32_t>(depthBias),
      .depthBiasSlopeScale = depthBiasSlopeScale,
      .depthBiasClamp = depthBiasClamp,
  };
  const auto blendState = to_blend_state(config.blendMode, config.blendFacSrc, config.blendFacDst, config.blendOp,
                                         config.dstAlpha, options.dstAlphaMode == DstAlphaMode::DualSource);
  std::array<wgpu::ColorTargetState, gfx::MaxColorAttachments> colorTargets{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    colorTargets[i] = {
        .format = webgpu::to_wgpu(layout.colorAttachments[i].format),
        .writeMask = layout.colorAttachments[i].semantic == gfx::ColorAttachmentSemantic::Normal && writesDepth
                         ? wgpu::ColorWriteMask::All
                         : wgpu::ColorWriteMask::None,
    };
  }
  colorTargets[gfx::SceneColorAttachmentIndex].blend = &blendState;
  colorTargets[gfx::SceneColorAttachmentIndex].writeMask = to_write_mask(options.colorUpdate, options.alphaUpdate);
  const wgpu::FragmentState fragmentState{
      .module = shader,
      .entryPoint = "fs_main",
      .targetCount = layout.colorAttachmentCount,
      .targets = colorTargets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = sPipelineLayout,
      .vertex =
          {
              .module = shader,
              .entryPoint = "vs_main",
              .bufferCount = static_cast<uint32_t>(vtxBuffers.size()),
              .buffers = vtxBuffers.data(),
          },
      .primitive = to_primitive_state(config.cullMode),
      .depthStencil = layout.depthStencilFormat != gfx::TextureFormat::Undefined ? &depthStencil : nullptr,
      .multisample = wgpu::MultisampleState{.count = layout.sampleCount},
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::BindGroup create_texture_bind_group(const gfx::TextureSet& set) noexcept {
  // Using C WGPU types instead of C++ wrappers to avoid destructor overhead
  std::array<WGPUBindGroupEntry, MaxTextures * 2> textureEntries{};
  for (u32 i = 0; i < MaxTextures; ++i) {
    const auto& tex = set.textures[i];
    WGPUBindGroupEntry& textureEntry = textureEntries[i * 2];
    WGPUBindGroupEntry& samplerEntry = textureEntries[i * 2 + 1];
    textureEntry.binding = i * 2;
    samplerEntry.binding = i * 2 + 1;
    if (tex) {
      textureEntry.textureView = tex->gpu->sampleTextureView.Get();
      samplerEntry.sampler = gfx::sampler_ref(gfx::to_wgpu(set.samplers[i])).Get();
    } else {
      textureEntry.textureView = sEmptyTextureView.Get();
      samplerEntry.sampler = sEmptySampler.Get();
    }
  }
  const WGPUBindGroupDescriptor textureBindGroupDescriptor{
      .label = {"GX Texture Bind Group", WGPU_STRLEN},
      .layout = sTextureBindGroupLayout.Get(),
      .entryCount = textureEntries.size(),
      .entries = textureEntries.data(),
  };
  return wgpu::BindGroup::Acquire(wgpuDeviceCreateBindGroup(g_device.Get(), &textureBindGroupDescriptor));
}

void initialize_bindings() noexcept {
  {
    std::array<wgpu::BindGroupLayoutEntry, MaxTextures * 2> textureEntries;
    for (u32 i = 0; i < MaxTextures; ++i) {
      textureEntries[i * 2] = {
          .binding = i * 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              {
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      };
      textureEntries[i * 2 + 1] = {
          .binding = i * 2 + 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = {.type = wgpu::SamplerBindingType::Filtering},
      };
    }
    const wgpu::BindGroupLayoutDescriptor descriptor{
        .label = "GX Texture Bind Group Layout",
        .entryCount = textureEntries.size(),
        .entries = textureEntries.data(),
    };
    sTextureBindGroupLayout = g_device.CreateBindGroupLayout(&descriptor);
  }
  {
    constexpr wgpu::SamplerDescriptor descriptor{.label = "Empty sampler"};
    sEmptySampler = gfx::sampler_ref(descriptor);
  }
  {
    constexpr wgpu::TextureDescriptor descriptor{
        .label = "Empty texture",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {1, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    sEmptyTexture = g_device.CreateTexture(&descriptor);
    sEmptyTextureView = sEmptyTexture.CreateView();
  }
  {
    std::array<wgpu::BindGroupEntry, MaxTextures * 2> entries;
    for (u32 i = 0; i < MaxTextures; ++i) {
      entries[i * 2] = {
          .binding = i * 2,
          .textureView = sEmptyTextureView,
      };
      entries[i * 2 + 1] = {
          .binding = i * 2 + 1,
          .sampler = sEmptySampler,
      };
    }
    const wgpu::BindGroupDescriptor desc{
        .label = "GX Empty Texture Bind Group",
        .layout = sTextureBindGroupLayout,
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_emptyTextureBindGroup = g_device.CreateBindGroup(&desc);
  }
  {
    const std::array layouts{
        gfx::detail::resources().staticBindGroupLayout,
        gfx::detail::resources().uniformBindGroupLayout,
        sTextureBindGroupLayout,
    };
    const wgpu::PipelineLayoutDescriptor desc{
        .label = "GX Pipeline Layout",
        .bindGroupLayoutCount = layouts.size(),
        .bindGroupLayouts = layouts.data(),
        .immediateSize = sizeof(DrawImmediateData),
    };
    sPipelineLayout = g_device.CreatePipelineLayout(&desc);
  }
  if (const auto& patterns = fifo::index_patterns(); !patterns.indices.empty()) {
    const wgpu::BufferDescriptor descriptor{
        .label = "GX Index Patterns",
        .usage = wgpu::BufferUsage::Index,
        .size = patterns.indices.size(),
        .mappedAtCreation = true,
    };
    g_indexPatternBuffer = g_device.CreateBuffer(&descriptor);
    std::memcpy(g_indexPatternBuffer.GetMappedRange(), patterns.indices.data(), patterns.indices.size());
    g_indexPatternBuffer.Unmap();
  }
}

void release_bindings() noexcept {
  // TODO we should probably store this all in g_state.gx instead
  // Released while the device lives: by static destruction a Vulkan layer may already be gone.
  g_emptyTextureBindGroup = {};
  g_indexPatternBuffer = {};
  sEmptyTextureView = {};
  sEmptyTexture = {};
  sEmptySampler = {};
  sPipelineLayout = {};
  sSamplerBindGroupLayout = {};
  sTextureBindGroupLayout = {};
  {
    std::lock_guard lock{sBindGroupLayoutMutex};
    sUniformBindGroupLayouts.clear();
    sTextureBindGroupLayouts.clear();
  }
}
} // namespace aurora::gx

namespace aurora::gfx {
namespace {
wgpu::AddressMode wgpu_address_mode(GXTexWrapMode mode) {
  switch (mode) {
    DEFAULT_FATAL("invalid wrap mode {}", underlying(mode));
  case GX_CLAMP:
    return wgpu::AddressMode::ClampToEdge;
  case GX_REPEAT:
    return wgpu::AddressMode::Repeat;
  case GX_MIRROR:
    return wgpu::AddressMode::MirrorRepeat;
  }
}

wgpu::FilterMode wgpu_filter_mode(SamplerFilter filter) {
  return filter == SamplerFilter::Linear ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
}

wgpu::MipmapFilterMode wgpu_mip_filter_mode(SamplerMipFilter filter) {
  switch (filter) {
    DEFAULT_FATAL("invalid mip filter {}", underlying(filter));
  case SamplerMipFilter::None:
    return wgpu::MipmapFilterMode::Undefined;
  case SamplerMipFilter::Nearest:
    return wgpu::MipmapFilterMode::Nearest;
  case SamplerMipFilter::Linear:
    return wgpu::MipmapFilterMode::Linear;
  }
}
} // namespace

wgpu::SamplerDescriptor to_wgpu(const SamplerConfig& config) noexcept {
  return {
      .label = "Generated Filtering Sampler",
      .addressModeU = wgpu_address_mode(config.wrapS),
      .addressModeV = wgpu_address_mode(config.wrapT),
      .addressModeW = wgpu::AddressMode::Repeat,
      .magFilter = wgpu_filter_mode(config.magFilter),
      .minFilter = wgpu_filter_mode(config.minFilter),
      .mipmapFilter = wgpu_mip_filter_mode(config.mipFilter),
      .lodMinClamp = config.minLod,
      .lodMaxClamp = config.maxLod,
      .maxAnisotropy = config.maxAnisotropy,
  };
}
} // namespace aurora::gfx

namespace aurora::webgpu {
std::shared_ptr<gfx::backend::Pipeline> make_pipeline(wgpu::RenderPipeline pipeline) noexcept {
  if (!pipeline) {
    return {};
  }
  return std::make_shared<gfx::backend::Pipeline>(gfx::backend::Pipeline{std::move(pipeline)});
}
} // namespace aurora::webgpu

namespace aurora::gfx::backend {
CompiledPipeline create_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout) noexcept {
  return gx::create_pipeline(config, layout);
}

std::shared_ptr<Pipeline> create_pipeline(const clear::PipelineConfig& config,
                                          const RenderTargetLayout& layout) noexcept {
  return webgpu::make_pipeline(clear::create_pipeline(config, layout));
}

void pipelines_loaded() noexcept { webgpu::cache_prune(); }
} // namespace aurora::gfx::backend

namespace aurora::gfx::backend {
using webgpu::g_surface;

namespace {
constexpr std::array PreferredBackendOrder{
#ifdef ENABLE_BACKEND_WEBGPU
    BACKEND_WEBGPU,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D12
    BACKEND_D3D12,
#endif
#ifdef DAWN_ENABLE_BACKEND_METAL
    BACKEND_METAL,
#endif
#ifdef DAWN_ENABLE_BACKEND_VULKAN
    BACKEND_VULKAN,
#endif
#ifdef DAWN_ENABLE_BACKEND_D3D11
    BACKEND_D3D11,
#endif
// #ifdef DAWN_ENABLE_BACKEND_DESKTOP_GL
//     BACKEND_OPENGL,
// #endif
#ifdef DAWN_ENABLE_BACKEND_OPENGLES
    BACKEND_OPENGLES,
#endif
#ifdef DAWN_ENABLE_BACKEND_NULL
    BACKEND_NULL,
#endif
};
} // namespace

bool initialize(AuroraBackend& backend) noexcept {
  /* Attempt to create a window using the calling application's desired backend */
  AuroraBackend selectedBackend = backend;
  bool windowCreated = false;
  if (selectedBackend != BACKEND_AUTO && window::create_window(selectedBackend)) {
    if (webgpu::initialize(selectedBackend, g_config.allowCpuAdapter)) {
      windowCreated = true;
    } else {
      window::destroy_window();
    }
  }

  if (!windowCreated) {
    for (const auto backendType : PreferredBackendOrder) {
      selectedBackend = backendType;
      if (!window::create_window(selectedBackend)) {
        continue;
      }
      if (webgpu::initialize(selectedBackend, g_config.allowCpuAdapter)) {
        windowCreated = true;
        break;
      } else {
        window::destroy_window();
      }
    }
  }

  if (!windowCreated) {
    return false;
  }

  // Initialize SDL_Renderer for ImGui when we can't use a Dawn backend
  if (webgpu::g_backendType == wgpu::BackendType::Null) {
    AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
  }
  backend = selectedBackend;
  webgpu::initialize_resources();
  return true;
}

void shutdown() noexcept {
  webgpu::shutdown_resources();
  webgpu::shutdown();
}

bool uses_sdl_window() noexcept { return true; }

bool can_present() noexcept {
  if (!window::is_presentable()) {
    webgpu::release_surface();
    return false;
  }
  if (window::is_paused()) {
    return false;
  }
  if (!g_surface) {
    webgpu::refresh_surface(true);
    if (!g_surface) {
      return false;
    }
  }
  return true;
}

void resize(const AuroraWindowSize& size) noexcept {
  webgpu::resize_swapchain(size.fb_width, size.fb_height, size.native_fb_width, size.native_fb_height);
}

void refresh_surface() noexcept { webgpu::refresh_surface(false); }

bool converts_copy(GXTexFmt format) noexcept { return tex_copy_conv::needs_conversion(format); }

void request_depth_snapshot() noexcept { depth_peek::request_snapshot(); }

bool read_depth(uint16_t x, uint16_t y, uint32_t& z) noexcept { return depth_peek::read_latest(x, y, z); }
} // namespace aurora::gfx::backend

const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::gfx::backend::PreferredBackendOrder.size();
  }
  return aurora::gfx::backend::PreferredBackendOrder.data();
}
void aurora_set_resampler(AuroraSampler sampler) { aurora::webgpu::set_resampler(sampler); }
