// The vulkan4xenon backend's UI: RmlUi's list (gfx/ui.hpp) and ImGui's draw lists drawn into the UI image, which the
// present composites over the EFB (vulkan4xenon's ROADMAP.md, milestone 16).
#include "backend.hpp"

#include "../imgui.hpp"
#include "ui_layout.hpp"

#include <aurora/imgui.h>
#include <imgui.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace aurora::gfx::vulkan4xenon {
namespace {
// A frame slot's room for the UI's vertices and indices: Dusklight's menu took at most 73 KB a frame on the console
// (milestone 16's first step).
constexpr VkDeviceSize RingBytes = 256 * 1024;

// A slot's vertices and indices, write-combined; gone once the GPU is done with every frame that read it.
struct Ring {
  Buffer buffer;

  Ring() noexcept = default;
  Ring(const Ring&) = delete;
  Ring& operator=(const Ring&) = delete;
  ~Ring() {
    if (g_device != VK_NULL_HANDLE) {
      retire([buffer = buffer]() mutable { destroy_buffer(buffer); });
    }
  }
};

// Made at the first frame anything shows and released at the first that shows nothing, as Dusklight runs with 42 to 70
// MB free: the UI image the present composites, a D24S8 for RmlUi's clip mask, which takes no memory as it's never
// sampled, a white texel drawn with where RmlUi samples nothing, as Dawn's null texture is
// (rmlui/WebGPURenderInterface.cpp, CreateNullTexture), RmlUi's layers from 1, and a ring for each slot.
struct Targets {
  uint32_t width = 0;
  uint32_t height = 0;
  std::shared_ptr<backend::Texture> image;
  std::shared_ptr<backend::Texture> stencil;
  std::shared_ptr<backend::Texture> white;
  std::vector<std::shared_ptr<backend::Texture>> layers;
  std::array<std::shared_ptr<Ring>, detail::FrameSlotCount> rings;
};
Targets g_targets;

// ImGui's textures: its atlas, uploaded once ImGui draws, and aurora_imgui_add_texture's.
struct ImGuiTexture {
  std::shared_ptr<backend::Texture> gpu;
};
ImGuiTexture g_atlas;
// Staged on the game thread, copied in by the render worker before the frame's UI.
std::mutex g_imguiMutex;
std::vector<ui::Upload> g_imguiUploads;
std::vector<ui::Upload> g_imguiTaken;
std::chrono::steady_clock::time_point g_lastNewFrame{};
// Render worker only.
ui_layout::ImGuiLayout g_imguiLayout;

#ifdef AURORA_DIAGNOSTICS
// Frames ended, frames whose UI drew, UI passes, composites at the present and the render worker's time in encode_ui,
// since the last take.
std::array<std::atomic<uint64_t>, 5> g_counts{};
// The next frame's EFB, UI image and their composite, read back once it's done, as bytes red to alpha a texel, row by
// row.
std::mutex g_readbackMutex;
bool g_readbackWanted = false;
bool g_readbackReady = false;
uint32_t g_readbackWidth = 0;
uint32_t g_readbackHeight = 0;
std::vector<uint8_t> g_readbackEfb;
std::vector<uint8_t> g_readbackUi;
std::vector<uint8_t> g_readbackComposite;
// Render worker only: the composite's target while a readback is pending.
std::shared_ptr<backend::Texture> g_composite;
#endif

std::shared_ptr<backend::Texture> make_target(uint32_t width, uint32_t height, bool depth, const char* label) noexcept {
  return make_texture(
      {
          .width = width,
          .height = height,
          .mips = 1,
          .format = depth ? TextureFormat::Depth32Float : TextureFormat::RGBA8Unorm,
          .gxFormat = InvalidTextureFormat,
          .usage = depth ? backend::TextureUsage::Depth : backend::TextureUsage::Render,
          .label = label,
          .ui = true,
      },
      VK_SAMPLE_COUNT_1_BIT, 0, false);
}

bool make_targets(uint32_t width, uint32_t height) noexcept {
  if (g_targets.image && g_targets.width == width && g_targets.height == height) {
    return true;
  }
  g_targets = {};
  auto image = make_target(width, height, false, "UI image");
  auto stencil = make_target(width, height, true, "UI stencil");
  auto white = make_texture({
      .width = 1,
      .height = 1,
      .mips = 1,
      .format = TextureFormat::RGBA8Unorm,
      .gxFormat = InvalidTextureFormat,
      .usage = backend::TextureUsage::Sampled,
      .label = "UI white",
      .ui = true,
  });
  if (!image || !stencil || !white) {
    Log.error("No memory for the UI's {}x{} image; it isn't drawn", width, height);
    return false;
  }
  // Its whole memory, which is the texel whatever the tiling.
  std::memset(white->memory.mapped, 0xFF, white->memory.size);
  for (auto& ring : g_targets.rings) {
    ring = std::make_shared<Ring>();
    if (!create_buffer(RingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "UI ring",
                       ring->buffer)) {
      g_targets = {};
      return false;
    }
  }
  g_targets.width = width;
  g_targets.height = height;
  g_targets.image = std::move(image);
  g_targets.stencil = std::move(stencil);
  g_targets.white = std::move(white);
  return true;
}

// The frame's UI as it's recorded: which layer's pass is open, the layer RmlUi draws into, and the slot's ring.
struct Replay {
  VkCommandBuffer cmd;
  const ui::Frame& rml;
  Ring& ring;
  VkDeviceSize used = 0;
  uint32_t skipped = 0;
  int open = -1;
  uint8_t top = 0;
  // The first pass clears the stencil, and the UI image's first pass its colour.
  bool stencilCleared = false;
  bool imageCleared = false;
  VkPipeline bound = VK_NULL_HANDLE;
  uint32_t passes = 0;
};

backend::Texture* layer_image(uint8_t layer) noexcept {
  if (layer == 0) {
    return g_targets.image.get();
  }
  while (g_targets.layers.size() < layer) {
    auto image = make_target(g_targets.width, g_targets.height, false, "UI layer");
    if (!image) {
      Log.error("No memory for RmlUi's layer {}; it isn't drawn", g_targets.layers.size() + 1);
      return nullptr;
    }
    g_targets.layers.push_back(std::move(image));
  }
  return g_targets.layers[layer - 1].get();
}

void end_pass(Replay& replay) noexcept {
  if (replay.open >= 0) {
    vkCmdEndRenderPass(replay.cmd);
    replay.open = -1;
  }
}

// A pass on the layer, its colour cleared or loaded, the stencil cleared by the frame's first pass and loaded after.
bool begin_pass(Replay& replay, uint8_t layer, bool clearColor) noexcept {
  end_pass(replay);
  backend::Texture* const image = layer_image(layer);
  const VkFramebuffer framebuffer =
      image != nullptr ? target_framebuffer(*image, g_targets.stencil.get()) : VK_NULL_HANDLE;
  if (framebuffer == VK_NULL_HANDLE) {
    return false;
  }
  const std::array clearValues{VkClearValue{.color = {}}, VkClearValue{.depthStencil = {0.f, 0}}};
  const VkRenderPassBeginInfo begin{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = ui_pass(clearColor ? Load::Clear : Load::Load, replay.stencilCleared ? Load::Load : Load::Clear),
      .framebuffer = framebuffer,
      .renderArea = {{0, 0}, {image->width, image->height}},
      .clearValueCount = 2,
      .pClearValues = clearValues.data(),
  };
  vkCmdBeginRenderPass(replay.cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
  const VkViewport viewport{0.f, 0.f, static_cast<float>(image->width), static_cast<float>(image->height), 0.f, 1.f};
  vkCmdSetViewport(replay.cmd, 0, 1, &viewport);
  replay.stencilCleared = true;
  replay.open = layer;
  replay.bound = VK_NULL_HANDLE;
  ++replay.passes;
  return true;
}

// The pass open on the layer; the UI image's first clears it.
bool open_pass(Replay& replay, uint8_t layer) noexcept {
  if (replay.open == layer) {
    return true;
  }
  const bool clear = layer == 0 && !replay.imageCleared;
  replay.imageCleared = replay.imageCleared || layer == 0;
  return begin_pass(replay, layer, clear);
}

// Bytes into the ring, on 4 bytes; false, counted, when the slot's ring has no room.
bool place(Replay& replay, const void* data, size_t size, VkDeviceSize& offset) noexcept {
  const VkDeviceSize at = (replay.used + 3) & ~VkDeviceSize{3};
  if (at + size > RingBytes) {
    return false;
  }
  std::memcpy(replay.ring.buffer.mapped + at, data, size);
  offset = at;
  replay.used = at + size;
  return true;
}

void bind(Replay& replay, VkPipeline pipeline) noexcept {
  if (pipeline != replay.bound) {
    vkCmdBindPipeline(replay.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    replay.bound = pipeline;
  }
}

void push(Replay& replay, const std::array<float, 16>& matrix, std::array<float, 2> translation) noexcept {
  std::array<float, UiPushBytes / sizeof(float)> values{};
  std::copy(matrix.begin(), matrix.end(), values.begin());
  values[16] = translation[0];
  values[17] = translation[1];
  vkCmdPushConstants(replay.cmd, ui_pipeline_layout(), VK_SHADER_STAGE_VERTEX_BIT, 0, UiPushBytes, values.data());
}

void set_scissor(Replay& replay, const ui::Rect& rect) noexcept {
  const VkRect2D scissor{{rect.x, rect.y}, {static_cast<uint32_t>(rect.width), static_cast<uint32_t>(rect.height)}};
  vkCmdSetScissor(replay.cmd, 0, 1, &scissor);
}

bool stenciled(ui::Pipeline pipeline) noexcept {
  return pipeline != ui::Pipeline::Blend && pipeline != ui::Pipeline::Replace;
}

// Vertices and 32-bit indices through one of RmlUi's pipelines, sampling texture.
void draw(Replay& replay, const ui::Op& op, const ui::Vertex* vertices, size_t vertexCount, const uint32_t* indices,
          size_t indexCount, backend::Texture& texture, bool linear, const std::array<float, 16>& matrix) noexcept {
  const VkPipeline pipeline = ui_pipeline(op.pipeline);
  const VkDescriptorSet set = source_set(texture, linear);
  if (pipeline == VK_NULL_HANDLE || set == VK_NULL_HANDLE || op.scissor.width <= 0 || op.scissor.height <= 0 ||
      !open_pass(replay, op.type == ui::OpType::Composite ? op.destination : replay.top)) {
    return;
  }
  VkDeviceSize vertexOffset = 0;
  VkDeviceSize indexOffset = 0;
  if (!place(replay, vertices, vertexCount * sizeof(ui::Vertex), vertexOffset) ||
      !place(replay, indices, indexCount * sizeof(uint32_t), indexOffset)) {
    ++replay.skipped;
    return;
  }
  bind(replay, pipeline);
  vkCmdBindDescriptorSets(replay.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ui_pipeline_layout(), 0, 1, &set, 0, nullptr);
  push(replay, matrix, op.translation);
  set_scissor(replay, op.scissor);
  if (stenciled(op.pipeline)) {
    vkCmdSetStencilReference(replay.cmd, VK_STENCIL_FACE_FRONT_AND_BACK, op.stencilRef);
  }
  vkCmdBindVertexBuffers(replay.cmd, 0, 1, &replay.ring.buffer.buffer, &vertexOffset);
  vkCmdBindIndexBuffer(replay.cmd, replay.ring.buffer.buffer, indexOffset, VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(replay.cmd, static_cast<uint32_t>(indexCount), 1, 0, 0, 0);
}

// A layer drawn over another at the opacity, as Dawn's full-screen texture (rmlui/WebGPURenderInterface.cpp,
// CompositeLayers): a triangle over the target, its texels sampled where they land.
void composite(Replay& replay, const ui::Op& op) noexcept {
  backend::Texture* const source = layer_image(op.source);
  if (source == nullptr) {
    return;
  }
  if (replay.open == op.source) {
    end_pass(replay);
  }
  const auto width = static_cast<float>(source->width);
  const auto height = static_cast<float>(source->height);
  const auto alpha = static_cast<uint8_t>(std::lround(op.opacity * 255.f));
  const std::array<ui::Vertex, 3> vertices{{
      {0.f, 0.f, {alpha, alpha, alpha, alpha}, 0.f, 0.f},
      {2.f * width, 0.f, {alpha, alpha, alpha, alpha}, 2.f, 0.f},
      {0.f, 2.f * height, {alpha, alpha, alpha, alpha}, 0.f, 2.f},
  }};
  constexpr std::array<uint32_t, 3> indices{0, 1, 2};
  draw(replay, op, vertices.data(), vertices.size(), indices.data(), indices.size(), *source, false,
       replay.rml.matrices[0]);
}

// The layer's scissor region copied into the texture with the copy draw, as Dawn copies it
// (rmlui/WebGPURenderInterface.cpp, SaveLayerAsTexture).
void save(Replay& replay, const ui::Op& op) noexcept {
  backend::Texture* const source = layer_image(op.source);
  if (source == nullptr || !op.texture) {
    return;
  }
  end_pass(replay);
  backend::Texture& target = *op.texture;
  const VkPipeline pipeline = copy_pipeline(GX_TF_RGBA8, false);
  const VkFramebuffer framebuffer = target_framebuffer(target, nullptr);
  const VkDescriptorSet set = source_set(*source, false);
  if (pipeline == VK_NULL_HANDLE || framebuffer == VK_NULL_HANDLE || set == VK_NULL_HANDLE) {
    return;
  }
  const auto width = static_cast<float>(source->width);
  const auto height = static_cast<float>(source->height);
  const std::array<float, 4> region{op.scissor.x / width, op.scissor.y / height, op.scissor.width / width,
                                    op.scissor.height / height};
  const VkExtent2D extent{target.width, target.height};
  const VkViewport viewport{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
  draw_texture(replay.cmd, color_pass(Load::DontCare), framebuffer, extent, viewport, pipeline, set, region);
}

void replay_rml(Replay& replay) noexcept {
  for (const ui::Op& op : replay.rml.ops) {
    switch (op.type) {
    case ui::OpType::Draw:
      draw(replay, op, op.geometry->vertices.data(), op.geometry->vertices.size(), op.geometry->indices.data(),
           op.geometry->indices.size(), op.texture ? *op.texture : *g_targets.white, true, replay.rml.matrices[op.matrix]);
      break;
    case ui::OpType::PushLayer:
      replay.top = op.destination;
      begin_pass(replay, replay.top, true);
      break;
    case ui::OpType::Composite:
      composite(replay, op);
      break;
    case ui::OpType::PopLayer:
      if (replay.open == replay.top) {
        end_pass(replay);
      }
      replay.top = op.destination;
      break;
    case ui::OpType::Save:
      save(replay, op);
      break;
    }
  }
}

// ImGui's lists into the UI image after RmlUi, as imgui_impl_vulkan.cpp draws them: the display through a projection,
// each command scissored to its clip rectangle.
void replay_imgui(Replay& replay, const ImDrawData& data) noexcept {
  ui_layout::lay_out_imgui(data, g_targets.width, g_targets.height, g_imguiLayout);
  const VkPipeline pipeline = g_imguiLayout.draws.empty() ? VK_NULL_HANDLE : imgui_pipeline();
  if (pipeline == VK_NULL_HANDLE || !open_pass(replay, 0)) {
    return;
  }
  const VkDeviceSize start = (replay.used + 3) & ~VkDeviceSize{3};
  if (start + g_imguiLayout.bytes > RingBytes) {
    replay.skipped += static_cast<uint32_t>(g_imguiLayout.draws.size());
    return;
  }
  uint8_t* const base = replay.ring.buffer.mapped + start;
  for (size_t i = 0; i < g_imguiLayout.lists.size(); ++i) {
    const ImDrawList& list = *g_imguiLayout.lists[i];
    std::memcpy(base + g_imguiLayout.listVertices[i], list.VtxBuffer.Data, list.VtxBuffer.Size * sizeof(ImDrawVert));
  }
  for (const auto& command : g_imguiLayout.draws) {
    std::memcpy(base + command.indices, command.source, command.indexCount * sizeof(ImDrawIdx));
  }
  replay.used = start + g_imguiLayout.bytes;
  const float left = data.DisplayPos.x;
  const float top = data.DisplayPos.y;
  const float right = left + data.DisplaySize.x;
  const float bottom = top + data.DisplaySize.y;
  const std::array<float, 16> projection{
      2.f / (right - left),
      0.f,
      0.f,
      0.f,
      0.f,
      2.f / (bottom - top),
      0.f,
      0.f,
      0.f,
      0.f,
      1.f,
      0.f,
      (right + left) / (left - right),
      (bottom + top) / (top - bottom),
      0.f,
      1.f,
  };
  bind(replay, pipeline);
  push(replay, projection, {0.f, 0.f});
  for (const auto& command : g_imguiLayout.draws) {
    auto* const texture = reinterpret_cast<ImGuiTexture*>(static_cast<uintptr_t>(command.texture));
    const VkDescriptorSet set = texture->gpu ? source_set(*texture->gpu, true) : VK_NULL_HANDLE;
    if (set == VK_NULL_HANDLE) {
      continue;
    }
    vkCmdBindDescriptorSets(replay.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ui_pipeline_layout(), 0, 1, &set, 0, nullptr);
    set_scissor(replay, {command.x, command.y, command.width, command.height});
    const VkDeviceSize vertices = start + command.vertices;
    vkCmdBindVertexBuffers(replay.cmd, 0, 1, &replay.ring.buffer.buffer, &vertices);
    vkCmdBindIndexBuffer(replay.cmd, replay.ring.buffer.buffer, start + command.indices, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(replay.cmd, command.indexCount, 1, 0, static_cast<int32_t>(command.vertexOffset), 0);
  }
}

// The textures staged since the last frame, copied in before anything samples them.
void upload(VkCommandBuffer cmd, const std::vector<ui::Upload>& uploads) noexcept {
  for (const auto& texture : uploads) {
    if (!texture.texture || !texture.data) {
      continue;
    }
    const VkBufferImageCopy region{
        .bufferRowLength = texture.width,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {texture.width, texture.height, 1},
    };
    vkCmdCopyBufferToImage(cmd, texture.data->buffer, texture.texture->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  }
}

// ImGui's texture of RGBA8 texels, staged for the render worker.
ImTextureID make_imgui_texture(ImGuiTexture& texture, uint32_t width, uint32_t height, const uint8_t* rgba8) noexcept {
  auto gpu = make_texture({
      .width = width,
      .height = height,
      .mips = 1,
      .format = TextureFormat::RGBA8Unorm,
      .gxFormat = InvalidTextureFormat,
      .usage = backend::TextureUsage::Sampled,
      .label = "ImGui texture",
      .ui = true,
  });
  auto data = gpu ? backend::stage_upload(rgba8, width * 4, height, true) : nullptr;
  if (!data) {
    Log.error("No memory for ImGui's {}x{} texture", width, height);
    return ImTextureID{};
  }
  {
    std::lock_guard lock{g_imguiMutex};
    g_imguiUploads.push_back({.texture = gpu, .data = std::move(data), .width = width, .height = height});
  }
  texture.gpu = std::move(gpu);
  return static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(&texture));
}
} // namespace

void shutdown_ui() noexcept {
  g_targets = {};
  g_atlas.gpu.reset();
  std::lock_guard lock{g_imguiMutex};
  g_imguiUploads.clear();
  g_imguiTaken.clear();
}

backend::Texture* encode_ui(size_t slot, const ui::Frame& rml, const imgui::DrawData& imgui,
                            VkCommandBuffer cmd) noexcept {
#ifdef AURORA_DIAGNOSTICS
  const auto start = std::chrono::steady_clock::now();
  struct Timer {
    std::chrono::steady_clock::time_point start;
    ~Timer() {
      g_counts[4] +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    }
  } timer{start};
#endif
  {
    std::lock_guard lock{g_imguiMutex};
    std::swap(g_imguiTaken, g_imguiUploads);
  }
  upload(cmd, g_imguiTaken);
  g_imguiTaken.clear();
  upload(cmd, rml.uploads);
  const ImDrawData* const lists = imgui.data();
  const bool imguiDraws = lists != nullptr && lists->CmdListsCount > 0 && lists->TotalIdxCount > 0;
#ifdef AURORA_DIAGNOSTICS
  ++g_counts[0];
#endif
  const bool draws = !rml.ops.empty() || imguiDraws;
  if (!draws) {
    // Nothing shows: no pass, and the targets go once the frames before are done with them.
    g_targets = {};
    return nullptr;
  }
  // RmlUi's context, or ImGui's display, or the EFB's.
  TextureExtent size = efb_size();
  if (!rml.ops.empty()) {
    size = {rml.width, rml.height};
  } else if (imguiDraws) {
    size = {static_cast<uint32_t>(lists->DisplaySize.x * lists->FramebufferScale.x),
            static_cast<uint32_t>(lists->DisplaySize.y * lists->FramebufferScale.y)};
  }
  const uint32_t width = size.width;
  const uint32_t height = size.height;
  if (width == 0 || height == 0 || !make_targets(width, height)) {
    return nullptr;
  }
  Replay replay{.cmd = cmd, .rml = rml, .ring = *g_targets.rings[slot]};
  replay_rml(replay);
  if (imguiDraws) {
    replay_imgui(replay, *lists);
  }
  // A frame whose draws were all refused still clears the image, so the composite shows nothing.
  open_pass(replay, 0);
  end_pass(replay);
  if (replay.skipped != 0) {
    Log.error("No room for {} of the UI's draws in the frame's {} bytes; they were skipped", replay.skipped, RingBytes);
  }
#ifdef AURORA_DIAGNOSTICS
  ++g_counts[1];
  g_counts[2] += replay.passes;
  ++g_counts[3];
#endif
  return g_targets.image.get();
}

#ifdef AURORA_DIAGNOSTICS
void compose_for_readback(VkCommandBuffer cmd, backend::Texture& efb, backend::Texture* ui) noexcept {
  {
    std::lock_guard lock{g_readbackMutex};
    if (!g_readbackWanted) {
      return;
    }
  }
  if (!g_composite || g_composite->width != efb.width || g_composite->height != efb.height) {
    g_composite = make_target(efb.width, efb.height, false, "UI check composite");
  }
  const VkPipeline copy = copy_pipeline(GX_TF_RGBA8, false);
  const VkPipeline composite = ui_check_pipeline();
  const VkFramebuffer framebuffer = g_composite ? target_framebuffer(*g_composite, nullptr) : VK_NULL_HANDLE;
  const VkDescriptorSet efbSet = source_set(efb, false);
  if (copy == VK_NULL_HANDLE || composite == VK_NULL_HANDLE || framebuffer == VK_NULL_HANDLE ||
      efbSet == VK_NULL_HANDLE) {
    return;
  }
  const VkExtent2D extent{efb.width, efb.height};
  const VkViewport viewport{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
  constexpr std::array<float, 4> whole{0.f, 0.f, 1.f, 1.f};
  draw_texture(cmd, color_pass(Load::DontCare), framebuffer, extent, viewport, copy, efbSet, whole);
  const VkDescriptorSet uiSet = ui != nullptr ? source_set(*ui, false) : VK_NULL_HANDLE;
  if (uiSet != VK_NULL_HANDLE && ui->width == efb.width && ui->height == efb.height) {
    draw_texture(cmd, color_pass(Load::Load), framebuffer, extent, viewport, composite, uiSet, whole);
  }
}

void read_back_ui(const backend::Texture& efb, const backend::Texture* ui, uint64_t serial) noexcept {
  {
    std::lock_guard lock{g_readbackMutex};
    if (!g_readbackWanted) {
      return;
    }
    g_readbackWanted = false;
  }
  if (!wait_for_frame(serial)) {
    return;
  }
  const auto read = [](const backend::Texture& image, std::vector<uint8_t>& out) {
    out.resize(size_t{image.width} * image.height * 4);
    for (uint32_t y = 0; y < image.height; ++y) {
      for (uint32_t x = 0; x < image.width; ++x) {
        const auto texel = read_texel(image, x, y);
        std::memcpy(&out[(size_t{y} * image.width + x) * 4], texel.data(), texel.size());
      }
    }
  };
  std::lock_guard lock{g_readbackMutex};
  read(efb, g_readbackEfb);
  if (ui != nullptr && ui->width == efb.width && ui->height == efb.height && g_composite) {
    read(*ui, g_readbackUi);
    read(*g_composite, g_readbackComposite);
  } else {
    g_readbackUi.clear();
    g_readbackComposite.clear();
  }
  g_composite.reset();
  g_readbackWidth = efb.width;
  g_readbackHeight = efb.height;
  g_readbackReady = true;
}
#endif
} // namespace aurora::gfx::vulkan4xenon

namespace aurora::imgui {
using namespace gfx::vulkan4xenon;

// The atlas goes to the GPU once ImGui has drawn something with it, and its CPU copies go: Dusklight builds its fonts
// at start, and its tools are mostly off.
void new_frame(const AuroraWindowSize& size) noexcept {
  ImGuiIO& io = ImGui::GetIO();
  // A renderer would build the font atlas, and NewFrame needs it built.
  if (!io.Fonts->IsBuilt()) {
    io.Fonts->Build();
  }
  const ImDrawData* const last = ImGui::GetDrawData();
  if (io.Fonts->TexPixelsAlpha8 != nullptr && last != nullptr && last->CmdListsCount > 0) {
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    std::vector<uint8_t> bytes(static_cast<size_t>(width) * height * 4);
    ui_layout::atlas_bytes({reinterpret_cast<const uint32_t*>(pixels), static_cast<size_t>(width) * height},
                           bytes.data());
    io.Fonts->SetTexID(make_imgui_texture(g_atlas, width, height, bytes.data()));
    io.Fonts->ClearTexData();
  }
  const auto now = std::chrono::steady_clock::now();
  if (g_lastNewFrame != std::chrono::steady_clock::time_point{}) {
    io.DeltaTime = std::max(std::chrono::duration<float>(now - g_lastNewFrame).count(), 1e-6f);
  }
  g_lastNewFrame = now;
  io.DisplaySize = {static_cast<float>(size.width), static_cast<float>(size.height)};
  ImGui::NewFrame();
}
} // namespace aurora::imgui

// Kept for the life of the title, as nothing releases them.
ImTextureID aurora_imgui_add_texture(uint32_t width, uint32_t height, const void* rgba8) {
  using namespace aurora::gfx::vulkan4xenon;
  return make_imgui_texture(*new ImGuiTexture{}, width, height, static_cast<const uint8_t*>(rgba8));
}

#ifdef AURORA_DIAGNOSTICS
// Frames ended, frames whose UI drew, the UI's passes, the present's composites and the render worker's nanoseconds
// in the UI, since the last call.
extern "C" void aurora_vulkan4xenon_take_ui_counts(unsigned long long* counts) {
  for (size_t i = 0; i < aurora::gfx::vulkan4xenon::g_counts.size(); ++i) {
    counts[i] = aurora::gfx::vulkan4xenon::g_counts[i].exchange(0, std::memory_order_relaxed);
  }
}

// The UI's memory by kind, in bytes: its image, its D24S8, its layers and its rings.
extern "C" void aurora_vulkan4xenon_ui_memory(unsigned long long* bytes) {
  using namespace aurora::gfx::vulkan4xenon;
  const auto size = [](const std::shared_ptr<aurora::gfx::backend::Texture>& texture) -> unsigned long long {
    return texture ? texture->memory.size : 0;
  };
  bytes[0] = size(g_targets.image);
  bytes[1] = size(g_targets.stencil);
  bytes[2] = 0;
  for (const auto& layer : g_targets.layers) {
    bytes[2] += size(layer);
  }
  bytes[3] = 0;
  for (const auto& ring : g_targets.rings) {
    bytes[3] += ring ? ring->buffer.size : 0;
  }
}

// Asks for the next frame's EFB and UI image to be read back once it's done.
extern "C" void aurora_vulkan4xenon_request_ui_readback() {
  std::lock_guard lock{aurora::gfx::vulkan4xenon::g_readbackMutex};
  aurora::gfx::vulkan4xenon::g_readbackWanted = true;
  aurora::gfx::vulkan4xenon::g_readbackReady = false;
}

// Frees the frame read back, 11 MB at 1280x720.
extern "C" void aurora_vulkan4xenon_release_ui_readback() {
  using namespace aurora::gfx::vulkan4xenon;
  std::lock_guard lock{g_readbackMutex};
  g_readbackReady = false;
  g_readbackEfb = {};
  g_readbackUi = {};
  g_readbackComposite = {};
}

// The frame read back, its bytes red to alpha a texel, row by row, valid till the next request or release: the EFB,
// the UI image and their composite on the GPU, ui and composite null when the UI drew nothing. False until it's ready.
extern "C" bool aurora_vulkan4xenon_take_ui_readback(const unsigned char** efb, const unsigned char** ui,
                                                     const unsigned char** composite, unsigned int* width,
                                                     unsigned int* height) {
  using namespace aurora::gfx::vulkan4xenon;
  std::lock_guard lock{g_readbackMutex};
  if (!g_readbackReady) {
    return false;
  }
  *efb = g_readbackEfb.data();
  *ui = g_readbackUi.empty() ? nullptr : g_readbackUi.data();
  *composite = g_readbackComposite.empty() ? nullptr : g_readbackComposite.data();
  *width = g_readbackWidth;
  *height = g_readbackHeight;
  return true;
}
#endif
