#include "recording.hpp"

#include "backend.hpp"
#include "frame.hpp"

#include "clear.hpp"
#include "pipeline_cache.hpp"
#include "render_worker.hpp"
#include "tex_copy_conv.hpp"
#include "tex_palette_conv.hpp"
#include "texture.hpp"
#include "../gx/command_processor.hpp"
#include "../gx/fifo.hpp"
#include "../gx/gx.hpp"
#include "../gx/pipeline.hpp"
#ifdef AURORA_RMLUI_DRAWS
#include "../rmlui/pipeline.hpp"
#endif
#include "../window.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <magic_enum.hpp>
#include <tracy/Tracy.hpp>

namespace aurora::gfx {
using namespace detail;

namespace {
constexpr Module Log{"aurora::gfx"};

struct FrameRecorder {
  FramePacket* packet = nullptr;
  size_t frameSlot = 0;
  uint32_t currentRenderPass = UINT32_MAX;
  uint32_t drawCallCount = 0;
  uint32_t mergedDrawCallCount = 0;
  bool inOffscreen = false;
  std::optional<RenderPass> suspendedEfbPass;
  Viewport suspendedEfbViewport;
  ClipRect suspendedEfbScissor;
  TextureHandle offscreenColor;
  TextureHandle offscreenDepth;
  Viewport cachedViewport;
  ClipRect cachedScissor;
  // The target layout's key and whether it has a normal attachment, which every GX draw asks for, worked out for the
  // pass at layoutPass. A pass keeps its targets once draws are recorded into it, so it is worked out once a pass.
  uint32_t layoutPass = UINT32_MAX;
  uint64_t layoutKey = 0;
  bool layoutHasNormal = false;
  bool suppressRenderWorker = false;
#ifdef AURORA_GFX_DEBUG_GROUPS
  std::vector<std::string> debugGroupStack;
#endif

  [[nodiscard]] bool active() const noexcept { return packet != nullptr; }

  FramePacket& frame() const {
    CHECK(packet != nullptr, "No active recording session");
    return *packet;
  }

  RenderPassList& passes() const { return frame().renderPasses; }
};

FrameRecorder g_recorder;

std::string pass_label(std::string_view kind) {
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (!g_recorder.debugGroupStack.empty()) {
    return fmt::format("{} ({})", kind, g_recorder.debugGroupStack.back());
  }
#endif
  return std::string{kind};
}

// The EFB's targets are the backend's: a pass leaves them null.
void set_efb_targets(RenderPass& pass) {
  g_recorder.layoutPass = UINT32_MAX;
  const auto layout = backend::efb_layout();
  pass.colorAttachmentCount = layout.colorAttachmentCount;
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    auto& color = pass.colorAttachments[i];
    color.semantic = layout.colorAttachments[i].semantic;
    color.format = layout.colorAttachments[i].format;
    color.size = {layout.colorAttachments[i].width, layout.colorAttachments[i].height};
    color.target = {};
    AURORA_ASSERT(i == SceneColorAttachmentIndex || color.semantic == ColorAttachmentSemantic::Normal,
                  "Scene render-target attachment {} has no backing texture", i);
  }
  pass.depthTarget = {};
  pass.depthStencilFormat = layout.depthStencilFormat;
  pass.msaaSamples = layout.sampleCount;
  pass.hasDepth = true;
  pass.hasStencil = false;
}

void set_single_color_target(RenderPass& pass, TextureFormat format, const TextureExtent size, TextureHandle target) {
  g_recorder.layoutPass = UINT32_MAX;
  auto& color = pass.colorAttachments[SceneColorAttachmentIndex];
  color.semantic = ColorAttachmentSemantic::SceneColor;
  color.format = format;
  color.size = size;
  color.target = std::move(target);
  pass.colorAttachmentCount = 1;
}

struct OffscreenCacheKey {
  uint32_t width;
  uint32_t height;

  bool operator==(const OffscreenCacheKey& rhs) const { return width == rhs.width && height == rhs.height; }
  template <typename H>
  friend H AbslHashValue(H h, const OffscreenCacheKey& key) {
    return H::combine(std::move(h), key.width, key.height);
  }
};
struct OffscreenCacheEntry {
  TextureHandle color;
  TextureHandle depth;
};
absl::flat_hash_map<OffscreenCacheKey, OffscreenCacheEntry> g_offscreenCache;

RenderPassList& current_render_passes() { return g_recorder.passes(); }

StagingHighWater current_high_water(const FramePacket& frame) noexcept {
  return {
      .verts = static_cast<uint32_t>(frame.verts.size()),
      .uniforms = static_cast<uint32_t>(frame.uniforms.size()),
      .indices = static_cast<uint32_t>(frame.indices.size()),
      .storage = static_cast<uint32_t>(frame.storage.size()),
      .textureUpload = static_cast<uint32_t>(frame.textureUpload.size()),
      .textureUploadCount = frame.textureUploads.size(),
  };
}

FrameOp capture_frame_op(FramePacket& frame, FrameOpType type, uint32_t index) {
  FrameOp op{
      .type = type,
      .index = index,
      .renderPass =
          type == FrameOpType::RenderPass && index < frame.renderPasses.size() ? &frame.renderPasses[index] : nullptr,
      .textureCopy = type == FrameOpType::TextureCopy && index < frame.textureCopies.size()
                         ? frame.textureCopies[index].get()
                         : nullptr,
      .encoderTask =
          type == FrameOpType::EncoderTask && index < frame.encoderTasks.size() ? &frame.encoderTasks[index] : nullptr,
      .highWater = current_high_water(frame),
  };
  op.textureUploads.reserve(op.highWater.textureUploadCount);
  for (size_t i = 0; i < op.highWater.textureUploadCount; ++i) {
    op.textureUploads.push_back(&frame.textureUploads[i]);
  }
  return op;
}

void seal_pass(FramePacket& frame, uint32_t passIndex) {
  if (passIndex >= frame.renderPasses.size()) {
    return;
  }
  auto& pass = frame.renderPasses[passIndex];
  if (pass.sealed) {
    return;
  }
  pass.sealed = true;
}

Range push(ByteBuffer& target, const uint8_t* data, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  const auto begin = target.size();
  if (length > 0) {
    target.append(data, length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

Range map(ByteBuffer& target, size_t length, size_t alignment) {
  if (alignment != 0) {
    const size_t begin = target.size();
    const size_t alignedBegin = AURORA_ALIGN(begin, alignment);
    if (alignedBegin > begin) {
      target.append_zeroes(alignedBegin - begin);
    }
  }
  auto begin = target.size();
  if (length > 0) {
    target.append_zeroes(length);
  }
  return {static_cast<uint32_t>(begin), static_cast<uint32_t>(length)};
}

// For our public API, warn instead of fatal-ing when called outside an active recording frame.
bool check_recording(const char* name) {
  if (!g_recorder.active())
    UNLIKELY {
      Log.warn("{}: called outside an active frame", name);
      return false;
    }
  return true;
}

void push_command(CommandType type, const Command::Data& data) {
  if (g_recorder.currentRenderPass == UINT32_MAX)
    UNLIKELY {
      Log.warn("Dropping command {}", magic_enum::enum_name(type));
      return;
    }
  auto& renderPass = current_render_passes()[g_recorder.currentRenderPass];
  AURORA_ASSERT(!renderPass.sealed, "Attempted to append command {} to sealed render pass {}",
                magic_enum::enum_name(type), g_recorder.currentRenderPass);
  if (type == CommandType::Draw || type == CommandType::CustomDraw) {
    renderPass.hasDraws = true;
  }
  renderPass.commands.push_back({
      .type = type,
#ifdef AURORA_GFX_DEBUG_GROUPS
      .debugGroupStack = g_recorder.debugGroupStack,
#endif
      .data = data,
  });
}

template <InlinePayload T>
DrawCommand make_draw_command(DrawKind kind, const T& data) {
  static_assert(sizeof(T) <= InlineDrawPayloadSize);
  static_assert(alignof(T) <= alignof(std::max_align_t));
  DrawCommand command{.kind = kind};
  std::memcpy(command.payload.data(), &data, sizeof(data));
  return command;
}

void push_draw_command(DrawCommand data) {
  push_command(CommandType::Draw, Command::Data{.draw = data});
  ++g_recorder.drawCallCount;
}

OffscreenCacheEntry get_offscreen_textures(uint32_t width, uint32_t height) {
  OffscreenCacheKey key{width, height};
  if (const auto it = g_offscreenCache.find(key); it != g_offscreenCache.end()) {
    return it->second;
  }
  const auto layout = backend::efb_layout();
  OffscreenCacheEntry entry{
      .color = new_texture(width, height, 1, layout.colorAttachments[SceneColorAttachmentIndex].format,
                           InvalidTextureFormat, backend::TextureUsage::Render, "Offscreen Color"),
      .depth = new_texture(width, height, 1, layout.depthStencilFormat, InvalidTextureFormat,
                           backend::TextureUsage::Depth, "Offscreen Depth"),
  };
  auto [insertIt, _] = g_offscreenCache.emplace(key, std::move(entry));
  return insertIt->second;
}

void enqueue_pass(FramePacket& frame, uint32_t passIndex);

void resume_efb_pass_loading(const RenderPass& prevPass) {
  RenderPass newPass{
      .label = pass_label("EFB"),
      .colorAttachments = prevPass.colorAttachments,
      .colorAttachmentCount = prevPass.colorAttachmentCount,
      .depthTarget = prevPass.depthTarget,
      .depthStencilFormat = prevPass.depthStencilFormat,
      .msaaSamples = prevPass.msaaSamples,
      .clearDepth = false,
      .hasDepth = prevPass.hasDepth,
      .hasStencil = prevPass.hasStencil,
  };
  for (uint32_t i = 0; i < newPass.colorAttachmentCount; ++i) {
    newPass.colorAttachments[i].clear = false;
  }
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_recorder.currentRenderPass;
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void suspend_efb() {
  AURORA_ASSERT(g_recorder.active() && g_recorder.currentRenderPass != UINT32_MAX,
                "suspend_efb called outside of an active recording frame");
  AURORA_ASSERT(!g_recorder.inOffscreen, "suspend_efb called while offscreen rendering is active");
  AURORA_ASSERT(!g_recorder.suspendedEfbPass, "suspend_efb called with an EFB pass already suspended");

  auto& currentPass = current_render_passes()[g_recorder.currentRenderPass];
  if (!currentPass.has_consumer()) {
    g_recorder.suspendedEfbPass = std::move(currentPass);
    current_render_passes().pop_back();
    --g_recorder.currentRenderPass;
    // The next pass takes the suspended pass's index.
    g_recorder.layoutPass = UINT32_MAX;
  } else {
    enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  }
  g_recorder.suspendedEfbViewport = g_recorder.cachedViewport;
  g_recorder.suspendedEfbScissor = g_recorder.cachedScissor;
}

void finish_current_offscreen() {
  AURORA_ASSERT(g_recorder.active() && g_recorder.currentRenderPass != UINT32_MAX,
                "finish_current_offscreen called outside of an active recording frame");
  AURORA_ASSERT(g_recorder.inOffscreen, "finish_current_offscreen called without an active offscreen pass");

  auto& offscreenPass = current_render_passes()[g_recorder.currentRenderPass];
  offscreenPass.discardable = !offscreenPass.has_consumer();
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  g_recorder.offscreenColor = {};
  g_recorder.offscreenDepth = {};
}

void start_offscreen(uint32_t width, uint32_t height) {
  AURORA_ASSERT(width != 0 && height != 0, "start_offscreen requires nonzero dimensions ({}x{})", width, height);
  AURORA_ASSERT(g_recorder.active(), "start_offscreen called outside of an active recording frame");

  auto offscreenEntry = get_offscreen_textures(width, height);
  g_recorder.offscreenColor = std::move(offscreenEntry.color);
  g_recorder.offscreenDepth = std::move(offscreenEntry.depth);

  RenderPass newPass{
      .label = pass_label("Offscreen"),
      .depthTarget = g_recorder.offscreenDepth,
      .depthStencilFormat = g_recorder.offscreenDepth->format,
      .msaaSamples = 1,
      .clearDepthValue = gx::UseReversedZ ? 0.f : 1.f,
      .clearDepth = true,
      .hasDepth = true,
      .hasStencil = false,
  };
  set_single_color_target(newPass, g_recorder.offscreenColor->format, {width, height}, g_recorder.offscreenColor);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_recorder.currentRenderPass;
  g_recorder.inOffscreen = true;

  g_recorder.cachedViewport = {0.f, 0.f, static_cast<float>(width), static_cast<float>(height), 0.f, 1.f};
  g_recorder.cachedScissor = {0, 0, static_cast<int32_t>(width), static_cast<int32_t>(height)};
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void restore_efb() {
  AURORA_ASSERT(g_recorder.active(), "restore_efb called outside of an active recording frame");
  AURORA_ASSERT(g_recorder.inOffscreen, "restore_efb called without a suspended EFB pass");

  g_recorder.inOffscreen = false;
  if (g_recorder.suspendedEfbPass) {
    current_render_passes().emplace_back(std::move(*g_recorder.suspendedEfbPass));
    g_recorder.suspendedEfbPass.reset();
  } else {
    auto& pass = current_render_passes().emplace_back();
    pass.label = pass_label("EFB");
    set_efb_targets(pass);
    for (uint32_t i = 0; i < pass.colorAttachmentCount; ++i) {
      pass.colorAttachments[i].clear = false;
    }
    pass.clearDepth = false;
  }
  ++g_recorder.currentRenderPass;
  set_efb_targets(current_render_passes()[g_recorder.currentRenderPass]);

  g_recorder.cachedViewport = g_recorder.suspendedEfbViewport;
  g_recorder.cachedScissor = g_recorder.suspendedEfbScissor;
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void enqueue_op(FramePacket& frame, uint32_t opIndex) {
  if (opIndex >= frame.ops.size() || g_recorder.suppressRenderWorker) {
    return;
  }
  auto op = frame.ops[opIndex];
  render_worker::enqueue_encode_pass(frame.frameId, opIndex, [packet = &frame, op = std::move(op)] {
    if (op.renderPass == nullptr && op.textureCopy == nullptr && op.encoderTask == nullptr) {
      return;
    }
    backend::encode(*packet, op);
  });
}

void enqueue_pass(FramePacket& frame, uint32_t passIndex) {
  seal_pass(frame, passIndex);
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, FrameOpType::RenderPass, passIndex));
  enqueue_op(frame, opIndex);
}
} // namespace

namespace detail {

void begin_recording(FramePacket& packet, size_t frameSlot) {
  CHECK(!g_recorder.active(), "A recording session is already active");
  g_recorder.packet = &packet;
  g_recorder.frameSlot = frameSlot;
  g_recorder.drawCallCount = 0;
  g_recorder.mergedDrawCallCount = 0;
  g_recorder.suspendedEfbPass.reset();

  current_render_passes().emplace_back();
  auto& pass = current_render_passes()[0];
  pass.label = pass_label("EFB");
  set_efb_targets(pass);
  pass.colorAttachments[SceneColorAttachmentIndex].clearValue = gx::g_gxState.clearColor;
  pass.clearDepthValue = gx::clear_depth_value();
  g_recorder.currentRenderPass = 0;
  g_recorder.cachedViewport = gx::map_logical_viewport(gx::g_gxState.logicalViewport);
  g_recorder.cachedScissor = gx::map_logical_scissor(gx::g_gxState.logicalScissor);
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

RecordedFrame end_recording() {
  CHECK(g_recorder.active(), "No active recording session");
  AURORA_ASSERT(!g_recorder.inOffscreen, "end_frame called while offscreen rendering is active");
  AURORA_ASSERT(g_recorder.currentRenderPass == UINT32_MAX,
                "end_frame called before finish finalized the current render pass");
  auto& frame = g_recorder.frame();
  frame.stats.drawCallCount = g_recorder.drawCallCount;
  frame.stats.mergedDrawCallCount = g_recorder.mergedDrawCallCount;
  frame.stats.lastVertSize = frame.verts.size();
  frame.stats.lastUniformSize = frame.uniforms.size();
  frame.stats.lastIndexSize = frame.indices.size();
  frame.stats.lastStorageSize = frame.storage.size();
  frame.stats.lastTextureUploadSize = frame.textureUpload.size();

  // The next frame has a new storage pool.
  gx::fifo::clear_array_ranges();
#if defined(AURORA_GFX_DEBUG_GROUPS)
  if (!g_recorder.debugGroupStack.empty()) {
    for (auto& item : std::ranges::reverse_view(g_recorder.debugGroupStack)) {
      Log.warn("Debug group was not popped at end of frame: {}", item);
    }
    g_recorder.debugGroupStack.clear();
  }
#endif
  const RecordedFrame recorded{.packet = g_recorder.packet, .frameSlot = g_recorder.frameSlot};
  g_recorder.packet = nullptr;
  g_recorder.frameSlot = 0;
  return recorded;
}

void shutdown_recording() {
  g_recorder.currentRenderPass = UINT32_MAX;
  g_offscreenCache.clear();
  g_recorder.offscreenColor = {};
  g_recorder.offscreenDepth = {};
  g_recorder.suspendedEfbPass.reset();
  g_recorder.inOffscreen = false;
  g_recorder.packet = nullptr;
  g_recorder.frameSlot = 0;
  g_recorder.suppressRenderWorker = false;
}

namespace testing {

void suppress_render_worker(bool suppress) noexcept { g_recorder.suppressRenderWorker = suppress; }

void seed_offscreen_cache(uint32_t width, uint32_t height, TextureFormat colorFormat, TextureFormat depthFormat) {
  const TextureExtent size{width, height};
  g_offscreenCache.insert_or_assign(
      OffscreenCacheKey{width, height},
      OffscreenCacheEntry{
          .color = std::make_shared<TextureRef>(nullptr, size, colorFormat, 1, InvalidTextureFormat),
          .depth = std::make_shared<TextureRef>(nullptr, size, depthFormat, 1, InvalidTextureFormat),
      });
}

} // namespace testing

void increment_merged_draw_count() noexcept {
  if (g_recorder.active()) {
    ++g_recorder.mergedDrawCallCount;
  }
}

FramePacket& current_frame_packet() { return g_recorder.frame(); }

bool in_render_pass() noexcept { return g_recorder.active() && g_recorder.currentRenderPass != UINT32_MAX; }

void push_custom_draw_command(const CustomDrawCommand& draw) {
  push_command(CommandType::CustomDraw, Command::Data{.customDraw = draw});
  ++g_recorder.drawCallCount;
}

void push_encoder_task(const EncoderTask& task) {
  // Seal the current EFB pass, record the task between it and a continuation
  // pass that loads the existing contents. EFB writes persist into the continuation, so
  // content keeps the sealed pass alive even without a consumer.
  auto& frame = current_frame_packet();
  auto& prevPass = current_render_passes()[g_recorder.currentRenderPass];
  prevPass.discardable = !prevPass.has_consumer() && !prevPass.has_content();
  enqueue_pass(frame, g_recorder.currentRenderPass);

  const auto taskIndex = static_cast<uint32_t>(frame.encoderTasks.size());
  frame.encoderTasks.push_back(task);
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, FrameOpType::EncoderTask, taskIndex));
  enqueue_op(frame, opIndex);

  resume_efb_pass_loading(prevPass);
}

void resolve_current_pass(TextureHandle color, TextureHandle depth, TextureHandle normal) {
  auto& prevPass = current_render_passes()[g_recorder.currentRenderPass];
  prevPass.snapshotColor = std::move(color);
  prevPass.snapshotDepth = std::move(depth);
  prevPass.snapshotNormal = std::move(normal);

  if (g_recorder.inOffscreen) {
    // Seal the offscreen pass and resume the EFB.
    end_offscreen();
    return;
  }

  // Seal the current EFB pass and continue on a new one that loads the existing contents.
  // EFB writes persist into the loading continuation, so content keeps the pass alive even
  // without a consumer.
  prevPass.discardable = !prevPass.has_consumer() && !prevPass.has_content();
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  resume_efb_pass_loading(prevPass);
}

void begin_backend_pass(RenderPass pass) {
  ZoneScoped;
  auto& frame = current_frame_packet();
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    enqueue_pass(frame, g_recorder.currentRenderPass);
  }

  const auto size = pass.colorAttachments[SceneColorAttachmentIndex].size;
  pass.commands.reserve(128);
  frame.renderPasses.emplace_back(std::move(pass));
  g_recorder.currentRenderPass = static_cast<uint32_t>(frame.renderPasses.size() - 1);

  g_recorder.cachedViewport = {0.f, 0.f, static_cast<float>(size.width), static_cast<float>(size.height), 0.f, 1.f};
  g_recorder.cachedScissor = {0, 0, static_cast<int32_t>(size.width), static_cast<int32_t>(size.height)};
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void push_texture_copy(std::shared_ptr<const backend::TextureCopy> copy) {
  ZoneScoped;
  auto& frame = current_frame_packet();
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    enqueue_pass(frame, g_recorder.currentRenderPass);
    g_recorder.currentRenderPass = UINT32_MAX;
  }

  const auto copyIndex = static_cast<uint32_t>(frame.textureCopies.size());
  frame.textureCopies.emplace_back(std::move(copy));
  const auto opIndex = static_cast<uint32_t>(frame.ops.size());
  frame.ops.emplace_back(capture_frame_op(frame, FrameOpType::TextureCopy, copyIndex));
  enqueue_op(frame, opIndex);
}

} // namespace detail

void queue_texture_upload(TextureUpload upload) {
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    AURORA_ASSERT(!current_render_passes()[g_recorder.currentRenderPass].sealed,
                  "Attempted to append texture upload to sealed render pass {}", g_recorder.currentRenderPass);
  }
  current_frame_packet().textureUploads.emplace_back(std::move(upload));
}

bool queue_texture_upload_data(const uint8_t* data, TextureUpload upload, uint32_t blockSize) {
  const auto bytesPerRow = upload.bytesPerRow;
  const auto rowsPerImage = upload.rowsPerImage;
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, backend::limits().textureRowAlignment);
  auto& frame = current_frame_packet();
  if (AURORA_ALIGN(frame.textureUpload.size(), blockSize) + copyBytesPerRow * rowsPerImage <= TextureUploadSize) {
    upload.offset = push_texture_data(data, bytesPerRow, rowsPerImage, blockSize).offset;
    queue_texture_upload(std::move(upload));
    return true;
  }

  // The backend stages what overflows the pool; without the memory nothing is queued.
  upload.overflow = backend::stage_upload(data, bytesPerRow, rowsPerImage);
  if (!upload.overflow) {
    return false;
  }
  queue_texture_upload(std::move(upload));
  return true;
}

void end_color_pass() {
  ZoneScoped;
  if (g_recorder.currentRenderPass == UINT32_MAX) {
    return;
  }
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);
  g_recorder.currentRenderPass = UINT32_MAX;
}

template <>
gx::DrawData* get_last_draw_command() {
  if (g_recorder.currentRenderPass >= current_render_passes().size()) {
    return nullptr;
  }
  auto& last = current_render_passes()[g_recorder.currentRenderPass].commands.back();
  if (last.type != CommandType::Draw || last.data.draw.kind != DrawKind::Gx) {
    return nullptr;
  }
  return &inline_payload<gx::DrawData>(last.data.draw.payload.data());
}

Vec2<uint32_t> get_render_target_size() noexcept {
  if (g_recorder.currentRenderPass < current_render_passes().size()) {
    const auto& size =
        current_render_passes()[g_recorder.currentRenderPass].colorAttachments[SceneColorAttachmentIndex].size;
    return {size.width, size.height};
  }
  const auto windowSize = window::get_window_size();
  return {windowSize.fb_width, windowSize.fb_height};
}

void set_viewport(const Viewport& cmd) noexcept {
  if (cmd != g_recorder.cachedViewport) {
    push_command(CommandType::SetViewport, Command::Data{.setViewport = cmd});
    g_recorder.cachedViewport = cmd;
  }
}

void set_scissor(const ClipRect& cmd) noexcept {
  if (cmd != g_recorder.cachedScissor) {
    push_command(CommandType::SetScissor, Command::Data{.setScissor = cmd});
    g_recorder.cachedScissor = cmd;
  }
}

template <>
void push_draw_command(clear::DrawData data) {
  push_draw_command(make_draw_command(DrawKind::Clear, data));
}

template <>
PipelineRef pipeline_ref(const clear::PipelineConfig& config) {
  return find_pipeline(config, get_render_target_layout());
}

void resolve_pass_into(TextureHandle texture, ClipRect rect, bool clearColor, bool clearAlpha, bool clearDepth,
                       Vec4<float> clearColorValue, float clearDepthValue, GXTexFmt resolveFormat,
                       GXPixelFmt sourceFormat) {
  // Resolve current render pass
  auto& prevPass = current_render_passes()[g_recorder.currentRenderPass];
  prevPass.resolveTarget = std::move(texture);
  prevPass.resolveRect = rect;
  prevPass.resolveFormat = resolveFormat;
  prevPass.resolveSourceFormat = sourceFormat;
  if (prevPass.resolveTarget) {
    const auto srcW = static_cast<float>(prevPass.colorAttachments[SceneColorAttachmentIndex].size.width);
    const auto srcH = static_cast<float>(prevPass.colorAttachments[SceneColorAttachmentIndex].size.height);
    const tex_copy_conv::Uniforms uniforms{
        .offset = {static_cast<float>(rect.x) / srcW, static_cast<float>(rect.y) / srcH},
        .scale = {static_cast<float>(rect.width) / srcW, static_cast<float>(rect.height) / srcH},
        .opaqueAlpha = !gx::efb_has_alpha(sourceFormat),
    };
    prevPass.resolveUniformRange = push_uniform(uniforms);
  } else {
    // A clear without a copy (GX_AURORA_CLEAR_COPY_SRC): a pass that draws nothing and that nothing reads goes.
    prevPass.discardable = !prevPass.has_consumer() && !prevPass.has_content();
  }
  enqueue_pass(current_frame_packet(), g_recorder.currentRenderPass);

  // Populate new render pass from previous
  const auto msaaSamples = prevPass.msaaSamples;
  const auto& targetSize = prevPass.colorAttachments[SceneColorAttachmentIndex].size;
  const auto left = std::clamp<int32_t>(rect.x, 0, static_cast<int32_t>(targetSize.width));
  const auto top = std::clamp<int32_t>(rect.y, 0, static_cast<int32_t>(targetSize.height));
  const auto right = std::clamp<int32_t>(rect.x + rect.width, left, static_cast<int32_t>(targetSize.width));
  const auto bottom = std::clamp<int32_t>(rect.y + rect.height, top, static_cast<int32_t>(targetSize.height));
  const ClipRect clearRect{.x = left, .y = top, .width = right - left, .height = bottom - top};
  const bool fullTarget = left == 0 && top == 0 && right == targetSize.width && bottom == targetSize.height;
  RenderPass newPass{
      .label = pass_label(g_recorder.inOffscreen ? "Offscreen" : "EFB"),
      .colorAttachments = prevPass.colorAttachments,
      .colorAttachmentCount = prevPass.colorAttachmentCount,
      .depthTarget = prevPass.depthTarget,
      .depthStencilFormat = prevPass.depthStencilFormat,
      .msaaSamples = msaaSamples,
      .clearDepthValue = clearDepthValue,
      .clearDepth = clearDepth && fullTarget,
      .hasDepth = prevPass.hasDepth,
      .hasStencil = prevPass.hasStencil,
  };
  const bool fullColorClear = clearColor && clearAlpha && fullTarget;
  for (uint32_t i = 0; i < newPass.colorAttachmentCount; ++i) {
    auto& color = newPass.colorAttachments[i];
    if (color.semantic == ColorAttachmentSemantic::Normal && clearDepth && fullTarget) {
      color.clear = true;
    } else {
      color.clear = false;
    }
  }
  if (fullColorClear) {
    auto& sceneColor = newPass.colorAttachments[SceneColorAttachmentIndex];
    sceneColor.clear = true;
    sceneColor.clearValue = clearColorValue;
  }
  newPass.commands.reserve(2048);
  current_render_passes().emplace_back(std::move(newPass));
  ++g_recorder.currentRenderPass;

  const bool drawClearColor = clearColor && !fullColorClear;
  const bool drawClearAlpha = clearAlpha && !fullColorClear;
  const bool drawClearDepth = clearDepth && !fullTarget;
  if (clearRect.width > 0 && clearRect.height > 0 && (drawClearColor || drawClearAlpha || drawClearDepth)) {
    // If we're only clearing a portion of the render target (or only one of color/alpha), perform a clear draw
    push_draw_command(clear::DrawData{
        .pipeline = pipeline_ref(clear::PipelineConfig{
            .clearColor = drawClearColor,
            .clearAlpha = drawClearAlpha,
            .clearDepth = drawClearDepth,
        }),
        .color = {clearColorValue.x(), clearColorValue.y(), clearColorValue.z(), clearColorValue.w()},
        .depth = clearDepthValue,
        .rect = clearRect,
    });
  }
  push_command(CommandType::SetViewport, Command::Data{.setViewport = g_recorder.cachedViewport});
  push_command(CommandType::SetScissor, Command::Data{.setScissor = g_recorder.cachedScissor});
}

void queue_palette_conv(tex_palette_conv::ConvRequest req) {
  auto& renderPass = current_render_passes()[g_recorder.currentRenderPass];
  AURORA_ASSERT(!renderPass.sealed, "Attempted to append palette conversion to sealed render pass {}",
                g_recorder.currentRenderPass);
  renderPass.paletteConvs.push_back(std::move(req));
}

bool is_offscreen() noexcept { return g_recorder.inOffscreen; }

// Works out the current pass's layout facts, if they are another pass's.
static void update_layout_facts() noexcept {
  if (g_recorder.layoutPass == g_recorder.currentRenderPass) {
    return;
  }
  const auto& pass = current_render_passes()[g_recorder.currentRenderPass];
  g_recorder.layoutKey = pass.target_layout().key;
  g_recorder.layoutHasNormal = false;
  for (uint32_t i = 0; i < pass.colorAttachmentCount; ++i) {
    if (pass.colorAttachments[i].semantic == ColorAttachmentSemantic::Normal) {
      g_recorder.layoutHasNormal = true;
    }
  }
  g_recorder.layoutPass = g_recorder.currentRenderPass;
}

bool has_normal_attachment() noexcept {
  CHECK(g_recorder.currentRenderPass != UINT32_MAX, "has_normal_attachment called outside of a frame");
  update_layout_facts();
  return g_recorder.layoutHasNormal;
}

uint64_t get_render_target_layout_key() noexcept {
  CHECK(g_recorder.currentRenderPass != UINT32_MAX, "get_render_target_layout_key called outside of a frame");
  update_layout_facts();
  return g_recorder.layoutKey;
}

RenderTargetLayout get_render_target_layout() noexcept {
  CHECK(g_recorder.currentRenderPass != UINT32_MAX, "get_render_target_layout called outside of a frame");
  return current_render_passes()[g_recorder.currentRenderPass].target_layout();
}

// The offscreen targets only; Dawn's resize empties its own bind groups.
void clear_caches() noexcept { g_offscreenCache.clear(); }

void begin_offscreen(uint32_t width, uint32_t height) {
  ZoneScoped;
  AURORA_ASSERT(width != 0 && height != 0, "begin_offscreen requires nonzero dimensions ({}x{})", width, height);
  AURORA_ASSERT(g_recorder.active() && g_recorder.currentRenderPass != UINT32_MAX,
                "begin_offscreen called outside of an active recording frame");

  if (g_recorder.inOffscreen) {
    finish_current_offscreen();
  } else {
    suspend_efb();
  }
  start_offscreen(width, height);
}

void end_offscreen() {
  ZoneScoped;
  finish_current_offscreen();
  restore_efb();
}

bool create_pass(uint32_t width, uint32_t height) {
  if (width == 0 || height == 0) {
    Log.warn("create_pass: invalid size {}x{}", width, height);
    return false;
  }

  gx::fifo::drain();

  if (!g_recorder.active() || g_recorder.currentRenderPass == UINT32_MAX) {
    Log.warn("create_pass: called outside an active render pass");
    return false;
  }
  if (g_recorder.inOffscreen) {
    Log.warn("create_pass: an offscreen pass is already active (nesting is unsupported)");
    return false;
  }

  begin_offscreen(width, height);
  return true;
}

template <>
void push_draw_command(gx::DrawData data) {
  push_draw_command(make_draw_command(DrawKind::Gx, data));
}

#ifdef AURORA_RMLUI_DRAWS
template <>
void push_draw_command(rmlui::DrawData data) {
  push_draw_command(make_draw_command(DrawKind::RmlUi, data));
}
#endif

template <>
PipelineRef pipeline_ref(const gx::PipelineConfig& config) {
  return find_pipeline(config, get_render_target_layout());
}

#ifdef AURORA_RMLUI_DRAWS
template <>
PipelineRef pipeline_ref(const rmlui::PipelineConfig& config) {
  return find_pipeline(config);
}
#endif

void finish() {
  ZoneScoped;
  if (!g_recorder.active()) {
    return;
  }
  AURORA_ASSERT(!g_recorder.inOffscreen, "finish called while offscreen rendering is active");
  if (g_recorder.currentRenderPass != UINT32_MAX) {
    auto& frame = current_frame_packet();
    frame.uniforms.append_zeroes(gx::MaxUniformSize);
    auto& pass = frame.renderPasses[g_recorder.currentRenderPass];
    pass.captureDepthSnapshot = true;
    enqueue_pass(frame, g_recorder.currentRenderPass);
    g_recorder.currentRenderPass = UINT32_MAX;
  }
}

Range push_verts(const uint8_t* data, size_t length, size_t alignment) {
  ZoneScoped;
  if (!check_recording("push_verts")) {
    return {};
  }
  return push(current_frame_packet().verts, data, length, alignment);
}

Range push_verts_hashed(const uint8_t* data, size_t length, size_t alignment, uint64_t& hash) {
  ZoneScoped;
  if (!check_recording("push_verts")) {
    hash = 0;
    return {};
  }
  auto& verts = current_frame_packet().verts;
  const Range range = push(verts, data, length, alignment);
  hash = XXH3_64bits(verts.data() + range.offset, range.size);
  return range;
}

Range push_indices(const uint8_t* data, size_t length, size_t alignment) {
  ZoneScoped;
  if (!check_recording("push_indices")) {
    return {};
  }
  return push(current_frame_packet().indices, data, length, alignment);
}

Range push_uniform(const uint8_t* data, size_t length) {
  ZoneScoped;
  if (!check_recording("push_uniform")) {
    return {};
  }
  return push(current_frame_packet().uniforms, data, length, backend::limits().uniformAlignment);
}

uint8_t* map_uniform(size_t length, Range& range) {
  ZoneScoped;
  if (!check_recording("map_uniform")) {
    range = {};
    return nullptr;
  }
  auto& uniforms = current_frame_packet().uniforms;
  range = map(uniforms, length, backend::limits().uniformAlignment);
  return uniforms.data() + range.offset;
}

Range push_storage(const uint8_t* data, size_t length) {
  ZoneScoped;
  if (!check_recording("push_storage")) {
    return {};
  }
  return push(current_frame_packet().storage, data, length, backend::limits().storageAlignment);
}

Range push_array(const uint8_t* data, size_t length) {
  ZoneScoped;
  if (!check_recording("push_array")) {
    return {};
  }
  auto& storage = current_frame_packet().storage;
  const auto& limits = backend::limits();
  const size_t alignment = limits.storageAlignment;
  if (!limits.arraySizes) {
    return push(storage, data, length, alignment);
  }
  // A whole alignment's worth before the array, so the array starts aligned and the size sits right before it, and its
  // hash before that.
  const size_t headerSize = limits.contentHashes ? sizeof(u64) + sizeof(u32) : sizeof(u32);
  const auto header = map(storage, std::max(size_t{alignment}, headerSize), alignment);
  const auto size = static_cast<u32>(length);
  std::memcpy(storage.data() + header.offset + header.size - sizeof(size), &size, sizeof(size));
  const Range range = push(storage, data, length, alignment);
  if (limits.contentHashes) {
    const u64 hash = XXH3_64bits(storage.data() + range.offset, range.size);
    std::memcpy(storage.data() + range.offset - sizeof(size) - sizeof(hash), &hash, sizeof(hash));
  }
  return range;
}

Range push_texture_data(const uint8_t* data, u32 bytesPerRow, u32 rowsPerImage, u32 alignment) {
  // Rows at the backend's pitch; Dawn's CopyBufferToTexture needs 256 (kTextureBytesPerRowAlignment).
  const auto copyBytesPerRow = AURORA_ALIGN(bytesPerRow, backend::limits().textureRowAlignment);
  const auto range = map(current_frame_packet().textureUpload, copyBytesPerRow * rowsPerImage, alignment);
  u8* dst = current_frame_packet().textureUpload.data() + range.offset;
  for (u32 i = 0; i < rowsPerImage; ++i) {
    memcpy(dst, data, bytesPerRow);
    data += bytesPerRow;
    dst += copyBytesPerRow;
  }
  return range;
}

uint32_t align_uniform(uint32_t value) { return AURORA_ALIGN(value, backend::limits().uniformAlignment); }

void insert_debug_marker(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  auto& markers = current_frame_packet().debugMarkers;
  const auto idx = markers.size();
  markers.emplace_back(std::move(label));
  push_command(CommandType::DebugMarker, {.debugMarkerIndex = idx});
#endif
}

void push_debug_group(std::string label) {
#if defined(AURORA_GFX_DEBUG_GROUPS)
  g_recorder.debugGroupStack.push_back(std::move(label));
#endif
}
} // namespace aurora::gfx

void push_debug_group(const char* label) {
#ifdef AURORA_GFX_DEBUG_GROUPS
  aurora::gfx::g_recorder.debugGroupStack.emplace_back(label);
#endif
}
void pop_debug_group() {
#ifdef AURORA_GFX_DEBUG_GROUPS
  if (aurora::gfx::g_recorder.debugGroupStack.empty()) {
    aurora::gfx::Log.error("Debug group stack underflowed!");
    return;
  }

  aurora::gfx::g_recorder.debugGroupStack.pop_back();
#endif
}
