// The vulkan4xenon backend's frames: Aurora's recorded passes as Vulkan commands, its EFB copies, and the present.
#include "backend.hpp"

#include "../dolphin/vi/vi_internal.hpp"
#include "../gfx/frame_packet.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/render_worker.hpp"
#include "../gx/command_processor.hpp"
#include "../gx/gx.hpp"
#include "../io.hpp"
#include "../window.hpp"
#include "glsl.hpp"
#include "tiled_index.hpp"
#include "ui_layout.hpp"

#include <xxhash.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#ifdef AURORA_DIAGNOSTICS
#include <chrono>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Room for decoded vertices as decode_vertices writes them, in GPU memory, kept from frame to frame
// (decode::VertexCache). When the frames in flight hold it all, a draw waits for the oldest of them to be done; sized
// to hold what they draw, the wait stays rare. One frame's own vertices must fit, or its draws past it are skipped and
// logged.
#ifndef AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES
#define AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES 8388608
#endif

namespace aurora::gfx::vulkan4xenon {
namespace {
constexpr VkDeviceSize VertexCacheSize = AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES;
// An index list's places are 16-bit, so a listed draw has at most this many vertices.
constexpr uint32_t MaxListedVertices = 65536;

// The CPU pools each start on a page, so where the vertex decode's inputs fall in the CPU's caches doesn't depend on
// what newlib's heap held before them. From newlib's memalign.
constexpr size_t CpuPoolAlignment = 4096;

struct CpuPoolDeleter {
  void operator()(uint8_t* pool) const noexcept { std::free(pool); }
};
using CpuPool = std::unique_ptr<uint8_t[], CpuPoolDeleter>;

CpuPool new_cpu_pool(size_t size) noexcept { return CpuPool{static_cast<uint8_t*>(memalign(CpuPoolAlignment, size))}; }

struct FrameState {
  // What the GPU reads, in GPU memory. It's write-combined, where GCC's merged 8-byte stores raise an alignment
  // exception unless they're 8-byte aligned (xsdk's CLAUDE.md), so the game records into the CPU pools below and
  // encode copies each op's new bytes over with memcpy, which stores bytes or aligned words (vulkan4xenon's
  // docs/SUPPORT.md, Memory and buffers).
  Buffer uniforms;
  Buffer indices;
  Buffer uploads;
  CpuPool uniformData;
  CpuPool indexData;
  CpuPool uploadData;
  uint32_t uniformsCopied = 0;
  uint32_t indicesCopied = 0;
  uint32_t uploadsCopied = 0;
  // What only the CPU reads, in newlib's heap. One of each per slot: the render worker reads a frame's while the
  // game records the next.
  CpuPool verts;
  CpuPool storage;
  VkDescriptorSet uniformSet = VK_NULL_HANDLE;
  std::shared_ptr<backend::Texture> efbColor;
  std::shared_ptr<backend::Texture> efbDepth;
  // At 2 or 4 samples (update_efb), else null.
  std::shared_ptr<backend::Texture> efbMultisampled;
  std::shared_ptr<backend::Texture> efbDepthRead;
  // Render worker only.
  bool recording = false;
  size_t uploadsDone = 0;
  // Draws the vertex cache had no room for even with every frame before this one done, logged once a frame.
  uint32_t verticesSkipped = 0;
};

std::array<FrameState, detail::FrameSlotCount> g_frames;
// Render worker: decoded vertices, kept for later frames' draws with the same inputs, in g_vertexCache's video RAM.
// Hashing the inputs costs far less than the decoding it saves, and runs where the decode did, on the render worker
// and its helper, off the game thread and FIFO worker that set the frame. Its frames are numbered as submit_frame
// numbers them, so a draw that finds no room can wait for the oldest frame in flight to be done (wait_for_frame).
Buffer g_vertexCache;
decode::VertexCache g_cache{VertexCacheSize, detail::FrameSlotCount, wait_for_frame};
decode::VertexKeys g_keys;
// Render worker: the index lists of non-indexed draws, by their vertex bytes. A list of a 64,000-vertex draw is 250 KB
// with its first vertices, so this keeps a few frames' lists and drops those a game stops drawing.
constexpr uint64_t IndexListBytes = 4u << 20;
decode::IndexLists g_lists{IndexListBytes};
// The index patterns (gx::fifo::index_patterns), for every frame's draws; none when they're off.
Buffer g_indexPatterns;
// Game thread, between frames: the EFB the next frame draws into. At 2 or 4 samples its passes draw into the
// multisampled colour and depth, the colour's average going into g_efbColor and the depth's sample 0, when a copy or
// GXPeekZ reads it, into g_efbDepthRead (encode_pass).
TextureExtent g_efbSize{};
std::shared_ptr<backend::Texture> g_efbColor;
std::shared_ptr<backend::Texture> g_efbDepth;
std::shared_ptr<backend::Texture> g_efbMultisampled;
std::shared_ptr<backend::Texture> g_efbDepthRead;

// Render worker: what the pass being recorded has bound.
PipelineRef g_currentPipeline = UINTPTR_MAX;
std::shared_ptr<backend::Pipeline> g_currentPipelineObject;
VkPipelineLayout g_currentLayout = VK_NULL_HANDLE;
VkDescriptorSet g_boundTextures = VK_NULL_HANDLE;
// The position and normal matrices last pushed, by their places in the frame's uniform data, and a line or point
// draw's texture coordinate mask (glsl.hpp), NoLineMask for none.
const uint8_t* g_pushedMatrix = nullptr;
const uint8_t* g_pushedNormalMatrix = nullptr;
constexpr uint32_t NoLineMask = UINT32_MAX;
uint32_t g_pushedLineMask = NoLineMask;
// The texture set last looked up, by its ref: consecutive draws often share one, and texture_set takes a lock and a
// lookup. A ref's set stays the same while a pass is recorded; expire_texture_sets runs after the frame's last.
std::optional<BindGroupRef> g_lookedUpRef;
VkDescriptorSet g_lookedUpSet = VK_NULL_HANDLE;
VkDescriptorSet g_boundFogRange = VK_NULL_HANDLE;

// Render worker: the range fog tables (glsl.hpp), the most recently used last. The FIFO worker pushes Aurora's table
// into the frame's storage pool when a draw's fog changes it (gx/command_processor.cpp, push_gx_draw), so a frame has
// one per fog setting it draws with and the next frame pushes the same again; a table is found by its factors.
struct FogRangeTable {
  uint64_t hash = 0;
  std::vector<float> factors;
  std::shared_ptr<backend::Texture> texture;
  VkDescriptorSet set = VK_NULL_HANDLE;
};
// 8 KiB each.
constexpr size_t MaxFogRangeTables = 8;
constexpr uint32_t FogRangeColumns = glsl::FogRangeTableWidth * glsl::FogRangeTableHeight;
std::vector<FogRangeTable> g_fogRangeTables;
// The pass's tables, by where the FIFO worker pushed them, in words into the storage pool.
std::vector<std::pair<uint32_t, VkDescriptorSet>> g_passFogRanges;
constexpr VkDeviceSize NoIndices = UINT64_MAX;
// Render worker: the pass being recorded's GX draws, in order, as lay_out_pass placed them.
struct PassDraw {
  std::shared_ptr<backend::Pipeline> pipeline; // none for a draw that records nothing
  VkDeviceSize offset = 0;                     // its vertices, in the vertex cache
  VkDeviceSize indices = NoIndices;            // a listed draw's index list, in the vertex cache
  bool decoded = false;                        // false when the cache had no room: its pipeline is bound, no more
  VkDescriptorSet fogRange = VK_NULL_HANDLE;   // a range-fog program's table
};
std::vector<PassDraw> g_passDraws;
// The pass's draws with a pipeline, their place in g_passDraws, keys and arrays' ends; and those to decode.
std::vector<decode::KeyedDraw> g_keyedDraws;
std::vector<size_t> g_keyedPlaces;
std::vector<std::optional<decode::DecodeKey>> g_drawKeys;
std::vector<decode::ArrayEnds> g_drawEnds;
std::vector<bool> g_drawHits;
std::vector<decode::DecodeJob> g_decodeJobs;
std::vector<size_t> g_decodeDraws; // each job's draw in g_keyedDraws

constexpr std::array<float, 4> WholeSource{0.f, 0.f, 1.f, 1.f};

// GXPeekZ, as depth_peek.cpp takes it on the WebGPU backend: while the game asks, at most 30 times a second, the
// frame's last pass leaves its depth to be read, which the layer copies out of eDRAM into the depth image, and a draw
// packs GX's 24-bit z from it, at the GX framebuffer's size, into an RGBA8 slot (glsl.hpp, depth_peek_fragment).
// read_depth reads a word of the newest slot whose frame is done, on the game thread. Of the two slots, one is drawn
// into only once the other's frame is done, so the slot read_depth reads is never one the GPU is drawing.
constexpr auto DepthPeekInterval = std::chrono::nanoseconds{1'000'000'000 / 30};
struct DepthPeekSlot {
  std::shared_ptr<backend::Texture> texture;
  // The frame that drew it; 0 for none yet.
  uint64_t serial = 0;
};
std::mutex g_depthPeekMutex;
std::array<DepthPeekSlot, 2> g_depthPeekSlots;
bool g_depthPeekRequested = false;
std::chrono::steady_clock::time_point g_nextDepthPeek{};

#ifdef AURORA_DIAGNOSTICS
// The most a frame's decoded vertices took of the cache; only the render worker writes it.
std::atomic<uint64_t> g_decodedPeak = 0;
// Frames since the vertex cache's stats were last logged.
uint32_t g_cacheFrames = 0;
// The render worker's time in vkAcquireNextImageKHR, which waits for the display, since the benchmark last asked.
std::atomic<uint64_t> g_acquireNs = 0;
#endif

} // namespace

// A tiled R8G8B8A8 image's pitch is a multiple of 32 texels (xsdk:include/xsdk/texture.h), and its texels' bytes run
// red to alpha in memory (xsdk:docs/HARDWARE.md, "Byte-order resolves"). Write-combined memory, so a word at a time:
// each read goes to memory.
std::array<uint8_t, 4> read_texel(const backend::Texture& image, uint32_t x, uint32_t y) noexcept {
  const auto* texels = reinterpret_cast<const volatile uint32_t*>(image.memory.mapped);
  const uint32_t texel = texels[tiled_index(x, y, (image.width + 31) & ~31u)];
  std::array<uint8_t, 4> bytes;
  std::memcpy(bytes.data(), &texel, bytes.size());
  return bytes;
}

namespace {
// A capture (aurora_capture_frame): the frame's EFB as a binary PPM, as the WebGPU backend writes its present source,
// once the GPU has drawn it, with the UI image over it as the present composites them where the UI drew. Xenia puts a
// resolve in memory only with --readback_resolve=full.
void write_capture(const backend::Texture& efb, const backend::Texture* ui, uint64_t serial,
                   const std::string& path) noexcept {
  if (!wait_for_frame(serial) || efb.memory.mapped == nullptr) {
    Log.warn("Frame capture: could not write {}", path);
    return;
  }
  if (ui != nullptr && (ui->width != efb.width || ui->height != efb.height)) {
    Log.warn("Frame capture: the UI is {}x{} and the EFB {}x{}, so {} shows the EFB alone", ui->width, ui->height,
             efb.width, efb.height, path);
    ui = nullptr;
  }
  const std::string header = fmt::format("P6\n{} {}\n255\n", efb.width, efb.height);
  std::vector<uint8_t> ppm(header.begin(), header.end());
  ppm.reserve(header.size() + size_t{efb.width} * efb.height * 3);
  for (uint32_t y = 0; y < efb.height; ++y) {
    for (uint32_t x = 0; x < efb.width; ++x) {
      auto bytes = read_texel(efb, x, y);
      if (ui != nullptr) {
        const auto over = read_texel(*ui, x, y);
        for (size_t channel = 0; channel < 3; ++channel) {
          bytes[channel] = ui_layout::composite(bytes[channel], over[channel], over[3]);
        }
      }
      ppm.insert(ppm.end(), bytes.begin(), bytes.begin() + 3);
    }
  }
  if (io::write_file(io::fs_path_from_string(path), ppm)) {
    Log.info("Wrote frame capture to {}", path);
  } else {
    Log.warn("Frame capture: could not write {}", path);
  }
}

// The EFB follows the window between frames, as under the null backend; a framebuffer is at most the screen.
bool update_efb() noexcept {
  const auto window = window::get_window_size();
  const auto screen = swapchain_extent();
  const TextureExtent size{std::clamp(window.fb_width, 1u, screen.width),
                           std::clamp(window.fb_height, 1u, screen.height)};
  if (g_efbColor && size.width == g_efbSize.width && size.height == g_efbSize.height) {
    return true;
  }
  if (size.width != window.fb_width || size.height != window.fb_height) {
    Log.warn("The EFB is {}x{}, not the window's {}x{}: vulkan4xenon draws at most {}x{}", size.width, size.height,
             window.fb_width, window.fb_height, screen.width, screen.height);
  }
  const backend::TextureDesc colorDesc{
      .width = size.width,
      .height = size.height,
      .mips = 1,
      .format = TextureFormat::RGBA8Unorm,
      .gxFormat = InvalidTextureFormat,
      .usage = backend::TextureUsage::Render,
      .label = "EFB colour",
  };
  const backend::TextureDesc depthDesc{
      .width = size.width,
      .height = size.height,
      .mips = 1,
      .format = TextureFormat::Depth32Float,
      .gxFormat = InvalidTextureFormat,
      .usage = backend::TextureUsage::Depth,
      .label = "EFB depth",
  };
  const VkSampleCountFlagBits samples = efb_samples();
  const bool multisampled = samples != VK_SAMPLE_COUNT_1_BIT;
  auto color = make_texture(colorDesc);
  auto depth = make_texture(depthDesc, samples, multisampled ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
  auto drawn = multisampled ? make_texture(colorDesc, samples) : nullptr;
  auto depthRead =
      multisampled ? make_texture(depthDesc, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_TRANSFER_DST_BIT) : nullptr;
  if (!color || !depth || (multisampled && (!drawn || !depthRead))) {
    return false;
  }
  g_efbColor = std::move(color);
  g_efbDepth = std::move(depth);
  g_efbMultisampled = std::move(drawn);
  g_efbDepthRead = std::move(depthRead);
  g_efbSize = size;
  return true;
}

// webgpu/gpu.cpp's calculate_present_viewport: the content's aspect, centred.
VkViewport present_viewport(VkExtent2D surface, uint32_t contentWidth, uint32_t contentHeight) noexcept {
  uint32_t width = surface.width;
  uint32_t height = std::min<uint32_t>(
      surface.height, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(width) *
                                                                               contentHeight / contentWidth))));
  if (height == surface.height) {
    width = std::min<uint32_t>(
        surface.width, std::max<uint32_t>(1u, static_cast<uint32_t>(std::lround(static_cast<double>(height) *
                                                                                contentWidth / contentHeight))));
  }
  return {
      .x = static_cast<float>((surface.width - width) / 2),
      .y = static_cast<float>((surface.height - height) / 2),
      .width = static_cast<float>(width),
      .height = static_cast<float>(height),
      .minDepth = 0.f,
      .maxDepth = 1.f,
  };
}

VkViewport to_viewport(const Viewport& viewport) noexcept {
  // Aurora's reversed depth (gfx/encoding.cpp, apply_viewport).
  return {
      .x = viewport.left,
      .y = viewport.top,
      .width = viewport.width,
      .height = viewport.height,
      .minDepth = gx::UseReversedZ ? 1.f - viewport.zfar : viewport.znear,
      .maxDepth = gx::UseReversedZ ? 1.f - viewport.znear : viewport.zfar,
  };
}

VkRect2D to_scissor(const ClipRect& scissor, const TextureExtent& size) noexcept {
  // As gfx/encoding.cpp's apply_scissor clamps it.
  const auto x = std::clamp(static_cast<uint32_t>(scissor.x), 0u, size.width);
  const auto y = std::clamp(static_cast<uint32_t>(scissor.y), 0u, size.height);
  return {
      .offset = {static_cast<int32_t>(x), static_cast<int32_t>(y)},
      .extent = {std::clamp(static_cast<uint32_t>(scissor.width), 0u, size.width - x),
                 std::clamp(static_cast<uint32_t>(scissor.height), 0u, size.height - y)},
  };
}

} // namespace

// A colour target's framebuffer, alone or with a depth target, or the multisampled EFB's with its depth and the EFB
// its colour's average goes into, which update_efb makes and replaces together.
VkFramebuffer target_framebuffer(backend::Texture& color, const backend::Texture* depth,
                                 const backend::Texture* resolve) noexcept {
  if (color.framebuffer != VK_NULL_HANDLE && color.framebufferDepth == depth) {
    return color.framebuffer;
  }
  if (color.framebuffer != VK_NULL_HANDLE) {
    retire([framebuffer = color.framebuffer] { vkDestroyFramebuffer(g_device, framebuffer, nullptr); });
    color.framebuffer = VK_NULL_HANDLE;
  }
  const std::array attachments{color.view, depth != nullptr ? depth->view : VK_NULL_HANDLE,
                               resolve != nullptr ? resolve->view : VK_NULL_HANDLE};
  const VkFramebufferCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = depth != nullptr ? scene_pass(Load::Load, Load::Load, DepthStore::Kept, false, resolve != nullptr)
                                     : color_pass(Load::Load),
      .attachmentCount = resolve != nullptr ? 3u : depth != nullptr ? 2u : 1u,
      .pAttachments = attachments.data(),
      .width = color.width,
      .height = color.height,
      .layers = 1,
  };
  if (succeeded(vkCreateFramebuffer(g_device, &info, nullptr, &color.framebuffer), "vkCreateFramebuffer")) {
    color.framebufferDepth = depth;
  }
  return color.framebuffer;
}

// The set a copy or the present samples a colour target through, filtered or not, or a copy or GXPeekZ's snapshot a
// depth target's depth, nearest.
VkDescriptorSet source_set(backend::Texture& texture, bool linear) noexcept {
  auto& set = texture.sourceSets[linear ? 1 : 0];
  if (set != VK_NULL_HANDLE) {
    return set;
  }
  set = allocate_set(source_set_layout());
  if (set == VK_NULL_HANDLE) {
    return set;
  }
  const auto filter = linear ? SamplerFilter::Linear : SamplerFilter::Nearest;
  const bool depth = texture.depthView != VK_NULL_HANDLE;
  const VkDescriptorImageInfo image{
      .sampler = sampler({.magFilter = filter, .minFilter = filter}),
      .imageView = depth ? texture.depthView : texture.view,
      .imageLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
  };
  const VkWriteDescriptorSet write{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &image,
  };
  vkUpdateDescriptorSets(g_device, 1, &write, 0, nullptr);
  return set;
}

namespace {
// A pass drawing the triangle over a target through layout's sets and push constants, then, given over, again through
// its pipeline and set: the present's UI over the EFB, in one pass.
void draw_triangle(VkCommandBuffer cmd, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extent,
                   const VkViewport& viewport, VkPipeline pipeline, VkPipelineLayout layout,
                   std::span<const VkDescriptorSet> sets, VkShaderStageFlags pushStages, std::span<const float> push,
                   VkPipeline overPipeline = VK_NULL_HANDLE, VkDescriptorSet overSet = VK_NULL_HANDLE) noexcept {
  const VkClearValue black{};
  const VkRenderPassBeginInfo begin{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = renderPass,
      .framebuffer = framebuffer,
      .renderArea = {{0, 0}, extent},
      .clearValueCount = 1,
      .pClearValues = &black,
  };
  vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, static_cast<uint32_t>(sets.size()),
                          sets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, layout, pushStages, 0, static_cast<uint32_t>(push.size_bytes()), push.data());
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  const VkRect2D scissor{{0, 0}, extent};
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  const VkBuffer vertices = triangle_buffer();
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &offset);
  vkCmdDraw(cmd, 3, 1, 0, 0);
  if (overPipeline != VK_NULL_HANDLE && overSet != VK_NULL_HANDLE) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &overSet, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);
  }
  vkCmdEndRenderPass(cmd);
}

} // namespace

void draw_texture(VkCommandBuffer cmd, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extent,
                  const VkViewport& viewport, VkPipeline pipeline, VkDescriptorSet set,
                  const std::array<float, 4>& source, VkPipeline overPipeline, VkDescriptorSet overSet) noexcept {
  draw_triangle(cmd, renderPass, framebuffer, extent, viewport, pipeline, copy_layout(), std::span{&set, 1},
                VK_SHADER_STAGE_VERTEX_BIT, source, overPipeline, overSet);
}

namespace {

// The bytes the game recorded into a CPU pool since the last copy, up to the op's high-water mark, into the GPU's.
void copy_recorded(const CpuPool& from, Buffer& to, uint32_t& copied, uint32_t highWater) noexcept {
  if (highWater > copied) {
    std::memcpy(to.mapped + copied, from.get() + copied, highWater - copied);
    copied = highWater;
  }
}

void upload_textures(FrameState& state, const detail::FrameOp& op, VkCommandBuffer cmd) noexcept {
  bool waited = false;
  for (size_t i = state.uploadsDone; i < op.textureUploads.size(); ++i) {
    const auto& upload = *op.textureUploads[i];
    backend::Texture* texture = upload.texture.get();
    if (texture == nullptr) {
      continue;
    }
    const VkBuffer source = upload.overflow ? upload.overflow->buffer : state.uploads.buffer;
    const VkDeviceSize offset = upload.overflow ? 0 : upload.offset;
    const uint32_t pitch = AURORA_ALIGN(upload.bytesPerRow, backend::limits().textureRowAlignment);
    const uint32_t level = 1u << upload.mipLevel;
    const bool rewrite = (texture->uploadedLevels & level) != 0;
    texture->uploadedLevels |= level;
    if (rewrite && !waited) {
      // The layer writes the texels with the CPU as it replays the copy, once the GPU has finished what came before
      // (vulkan4xenon's docs/SUPPORT.md, Commands), so the barrier records nothing there; kept for Vulkan's sake.
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 0, nullptr);
      waited = true;
    }
    // A BC1 row holds a block of 4 texels in 8 bytes, and a level under a block is copied at its own size, as Vulkan
    // asks. Its offset must be a multiple of 8, which queue_texture_upload_data gives.
    if (texture->bc1 && offset % 8 != 0) {
      Log.error("A BC1 upload at offset {} isn't on a block; skipped", offset);
      continue;
    }
    const VkBufferImageCopy region{
        .bufferOffset = offset,
        .bufferRowLength = texture->bc1 ? pitch / 8 * 4 : pitch / texture->texelBytes,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, upload.mipLevel, 0, 1},
        .imageExtent = {std::min(upload.width, std::max(texture->width >> upload.mipLevel, 1u)),
                        std::min(upload.height, std::max(texture->height >> upload.mipLevel, 1u)), 1},
    };
    vkCmdCopyBufferToImage(cmd, source, texture->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
  }
  state.uploadsDone = op.textureUploads.size();
}

// A pipeline that exists, so usable now; null while it is still being built, or when it couldn't be. create_pipeline
// makes no prepass.
std::shared_ptr<backend::Pipeline> usable_pipeline(PipelineRef ref) noexcept {
  CompiledPipeline pipeline;
  if (!get_pipeline(ref, pipeline)) {
    return nullptr;
  }
  return std::move(pipeline.main);
}

void use_pipeline(PipelineRef ref, std::shared_ptr<backend::Pipeline> pipeline, VkCommandBuffer cmd) noexcept {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->pipeline);
  // Another layout leaves the GX sets and push constants undefined.
  if (pipeline->layout != g_currentLayout) {
    g_currentLayout = pipeline->layout;
    g_boundTextures = VK_NULL_HANDLE;
    g_boundFogRange = VK_NULL_HANDLE;
    g_pushedMatrix = nullptr;
    g_pushedNormalMatrix = nullptr;
    g_pushedLineMask = NoLineMask;
  }
  g_currentPipeline = ref;
  g_currentPipelineObject = std::move(pipeline);
}

bool bind_pipeline(PipelineRef ref, VkCommandBuffer cmd) noexcept {
  if (ref == g_currentPipeline) {
    return true;
  }
  auto pipeline = usable_pipeline(ref);
  if (!pipeline) {
    return false;
  }
  use_pipeline(ref, std::move(pipeline), cmd);
  return true;
}

void reset_bindings() noexcept {
  g_currentPipeline = UINTPTR_MAX;
  g_currentPipelineObject.reset();
  g_currentLayout = VK_NULL_HANDLE;
  g_boundTextures = VK_NULL_HANDLE;
  g_boundFogRange = VK_NULL_HANDLE;
  g_pushedMatrix = nullptr;
  g_pushedNormalMatrix = nullptr;
  g_pushedLineMask = NoLineMask;
  g_lookedUpRef.reset();
}

// The set of a range fog table holding factors, one a column: one made before, or a new one, written by the CPU
// before any frame draws with it. Null when it can't be made.
VkDescriptorSet fog_range_set(std::span<const float> factors) noexcept {
  const uint64_t hash = XXH3_64bits(factors.data(), factors.size_bytes());
  for (auto it = g_fogRangeTables.begin(); it != g_fogRangeTables.end(); ++it) {
    if (it->hash == hash && it->factors.size() == factors.size() &&
        std::memcmp(it->factors.data(), factors.data(), factors.size_bytes()) == 0) {
      std::rotate(it, it + 1, g_fogRangeTables.end());
      return g_fogRangeTables.back().set;
    }
  }
  auto texture = make_texture({
      .width = glsl::FogRangeTableWidth,
      .height = glsl::FogRangeTableHeight,
      .mips = 1,
      .format = TextureFormat::R32Float,
      .gxFormat = InvalidTextureFormat,
      .usage = backend::TextureUsage::Sampled,
      .label = "Range fog table",
  });
  if (!texture) {
    return VK_NULL_HANDLE;
  }
  // Column x at texel (x % width, x / width), tiled as the layer tiles its images, each float a word in the CPU's
  // order, as the layer's copies take R32_SFLOAT's texels; columns past the target's repeat its last, unread. Mapped
  // memory is write-combined, so a word at a time (xsdk's CLAUDE.md), and vkQueueSubmit drains it for the GPU.
  auto* const texels = reinterpret_cast<volatile uint32_t*>(texture->memory.mapped);
  for (uint32_t x = 0; x < FogRangeColumns; ++x) {
    texels[tiled_index(x % glsl::FogRangeTableWidth, x / glsl::FogRangeTableWidth, glsl::FogRangeTableWidth)] =
        std::bit_cast<uint32_t>(factors[std::min<size_t>(x, factors.size() - 1)]);
  }
  const VkDescriptorSet set = allocate_set(source_set_layout());
  if (set == VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }
  const VkDescriptorImageInfo image{
      .sampler = sampler({.wrapS = GX_REPEAT}),
      .imageView = texture->view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
  };
  const VkWriteDescriptorSet write{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &image,
  };
  vkUpdateDescriptorSets(g_device, 1, &write, 0, nullptr);
  // A dropped table's texture and set go once the frames that may draw with it are done.
  if (g_fogRangeTables.size() == MaxFogRangeTables) {
    retire([set = g_fogRangeTables.front().set] { free_set(set); });
    g_fogRangeTables.erase(g_fogRangeTables.begin());
  }
  g_fogRangeTables.push_back({hash, {factors.begin(), factors.end()}, std::move(texture), set});
  return set;
}

// The set of the table the FIFO worker pushed at base, in words into the storage pool, for a target width columns
// wide; null, logged, when it can't be had.
VkDescriptorSet pass_fog_range(std::span<const uint8_t> storage, uint32_t base, uint32_t width) noexcept {
  for (const auto& [at, set] : g_passFogRanges) {
    if (at == base) {
      return set;
    }
  }
  VkDescriptorSet set = VK_NULL_HANDLE;
  const size_t offset = size_t{base} * sizeof(float);
  if (width == 0 || offset > storage.size() || storage.size() - offset < size_t{width} * sizeof(float)) {
    static std::once_flag once;
    std::call_once(once,
                   [&] { Log.error("No range fog table at word {} of the storage pool for {} columns", base, width); });
  } else {
    if (width > FogRangeColumns) {
      static std::once_flag once;
      std::call_once(once, [&] {
        Log.error("Range fog over {} columns: the table holds {}, and those past it read its last row", width,
                  FogRangeColumns);
      });
    }
    // The pool's arrays are 16-byte aligned (backend::Limits::storageAlignment).
    set = fog_range_set({reinterpret_cast<const float*>(storage.data() + offset), width});
  }
  g_passFogRanges.emplace_back(base, set);
  return set;
}

// Whether a draw's entry can hold an index list and its first vertices: a draw with no indices of its own, whose places
// fit 16 bits. The layer splits longer draws itself (vulkan4xenon's docs/SUPPORT.md, Commands).
bool listable(const gx::DrawData& data) noexcept { return data.indexCount == 0 && data.vtxCount <= MaxListedVertices; }

// A listed draw's index list, from its vertex bytes, which lay_out_pass keyed it by.
const decode::IndexList& listed_draw_list(size_t i, std::span<const uint8_t> verts) {
  const decode::KeyedDraw& keyed = g_keyedDraws[i];
  const uint32_t stride = keyed.config->vtxStride;
  const uint32_t count = keyed.draw->vtxCount;
  return g_lists.get(g_keys.vertex_hash(i), verts.subspan(keyed.draw->immediateData.vtxStart, size_t{count} * stride),
                     stride, count);
}

// An unlisted draw's bytes in the vertex cache: a line or point draw's corners.
VkDeviceSize stream_bytes(const decode::KeyedDraw& keyed) noexcept {
  return VkDeviceSize{glsl::stream_vertices(*keyed.config, keyed.draw->vtxCount)} * keyed.layout->stride;
}

// Where a draw's vertices, and a listed draw's list before them, are in the vertex cache.
void set_place(PassDraw& draw, VkDeviceSize offset, const decode::KeyedDraw& keyed) noexcept {
  draw.indices = keyed.listed ? offset : NoIndices;
  draw.offset = keyed.listed ? offset + decode::listed_indices_size(keyed.draw->vtxCount) : offset;
}

// A decode that read past one of its arrays read bytes its key doesn't cover, so no other draw may take its vertices:
// later frames' draws won't find them, and this pass's that already did get their own. Indices past an array are a
// game's bug, so this is rare.
void forget_reads_past_arrays(FrameState& state, std::span<const uint8_t> verts,
                              std::span<const uint8_t> storage) noexcept {
  std::vector<decode::DecodeKey> forgotten;
  for (size_t j = 0; j < g_decodeJobs.size(); ++j) {
    const auto& key = g_drawKeys[g_decodeDraws[j]];
    if (!g_decodeJobs[j].inside && key) {
      g_cache.forget(*key);
      forgotten.push_back(*key);
    }
  }
  for (size_t i = 0; i < g_keyedDraws.size() && !forgotten.empty(); ++i) {
    const auto& key = g_drawKeys[i];
    if (!g_drawHits[i] || !key || std::ranges::find(forgotten, *key) == forgotten.end()) {
      continue;
    }
    const decode::KeyedDraw& keyed = g_keyedDraws[i];
    PassDraw& draw = g_passDraws[g_keyedPlaces[i]];
    const decode::IndexList* list = keyed.listed ? &listed_draw_list(i, verts) : nullptr;
    const auto place = g_cache.place(
        std::nullopt, list != nullptr ? decode::listed_entry_size(*list, keyed.layout->stride) : stream_bytes(keyed));
    if (!place) {
      ++state.verticesSkipped;
      draw.decoded = false;
      continue;
    }
    set_place(draw, place->offset, keyed);
    decode::DecodeJob job{.config = keyed.config,
                          .layout = keyed.layout,
                          .draw = keyed.draw,
                          .out = g_vertexCache.mapped + place->offset,
                          .list = list};
    decode::decode_pass({&job, 1}, verts, storage);
  }
}

// Before a pass is recorded: each GX draw's pipeline and its vertices' place in the vertex cache. A draw whose inputs
// an earlier draw had, in this frame or an earlier one, finds that draw's vertices there; the rest are decoded into new
// space, on two threads (decode::decode_pass). The keys are hashed first, on the same two threads (decode::VertexKeys).
void lay_out_pass(const detail::FramePacket& frame, FrameState& state, const detail::FrameOp& op,
                  detail::RenderPass& pass) noexcept {
  g_passDraws.clear();
  g_keyedDraws.clear();
  g_keyedPlaces.clear();
  g_decodeJobs.clear();
  g_decodeDraws.clear();
  g_passFogRanges.clear();
  const std::span<const uint8_t> verts{frame.verts.data(), op.highWater.verts};
  const std::span<const uint8_t> storage{frame.storage.data(), op.highWater.storage};
  // Aurora's range fog table has a factor for each column of the pass's target (gx/command_processor.cpp).
  const uint32_t targetWidth = pass.colorAttachments[SceneColorAttachmentIndex].size.width;
  PipelineRef ref = UINTPTR_MAX;
  std::shared_ptr<backend::Pipeline> pipeline;
  for (auto& command : pass.commands) {
    if (command.type != detail::CommandType::Draw || command.data.draw.kind != detail::DrawKind::Gx) {
      continue;
    }
    const auto& data = detail::inline_payload<gx::DrawData>(command.data.draw.payload.data());
    PassDraw& draw = g_passDraws.emplace_back();
    if (data.vtxCount == 0) {
      continue;
    }
    if (data.pipeline != ref) {
      ref = data.pipeline;
      pipeline = usable_pipeline(ref);
    }
    if (!pipeline) {
      continue;
    }
    if (pipeline->fogRange) {
      draw.fogRange = pass_fog_range(storage, data.immediateData.fogRangeBase, targetWidth);
      if (draw.fogRange == VK_NULL_HANDLE) {
        continue;
      }
    }
    draw.pipeline = pipeline;
    g_keyedDraws.push_back(
        {pipeline->decodeFormat, &pipeline->shaderConfig, &pipeline->vertexLayout, &data, listable(data)});
    g_keyedPlaces.push_back(g_passDraws.size() - 1);
  }
  g_drawKeys.resize(g_keyedDraws.size());
  g_drawEnds.resize(g_keyedDraws.size());
  g_drawHits.assign(g_keyedDraws.size(), false);
  g_keys.compute(g_keyedDraws, verts, storage, g_drawKeys, g_drawEnds, backend::limits().contentHashes);
  for (size_t i = 0; i < g_keyedDraws.size(); ++i) {
    decode::KeyedDraw& keyed = g_keyedDraws[i];
    PassDraw& draw = g_passDraws[g_keyedPlaces[i]];
    // A draw with no key has no vertex bytes' hash to find its list by, and is decoded whole each frame.
    keyed.listed = keyed.listed && g_drawKeys[i].has_value();
    std::optional<decode::VertexCache::Place> place;
    const decode::IndexList* list = nullptr;
    if (keyed.listed) {
      // The entry's size depends on the list, needed only when the draw misses.
      place = g_cache.find(*g_drawKeys[i]);
      if (!place) {
        list = &listed_draw_list(i, verts);
        place = g_cache.place(g_drawKeys[i], decode::listed_entry_size(*list, keyed.layout->stride));
      }
    } else {
      place = g_cache.place(g_drawKeys[i], stream_bytes(keyed));
    }
    if (!place) {
      ++state.verticesSkipped;
      continue;
    }
    set_place(draw, place->offset, keyed);
    draw.decoded = true;
    g_drawHits[i] = place->decoded;
    if (!place->decoded) {
      g_decodeJobs.push_back({.config = keyed.config,
                              .layout = keyed.layout,
                              .draw = keyed.draw,
                              .out = g_vertexCache.mapped + place->offset,
                              .ends = g_drawEnds[i],
                              .list = list});
      g_decodeDraws.push_back(i);
    }
  }
  decode::decode_pass(g_decodeJobs, verts, storage);
  forget_reads_past_arrays(state, verts, storage);
}

// glsl.hpp's contract: one decoded stream, the draw's uniform pieces at their offsets, its textures, and its current
// position matrix pushed.
void draw_gx(const FrameState& state, const gx::DrawData& data, const PassDraw& draw, VkCommandBuffer cmd) noexcept {
  if (!draw.pipeline) {
    return;
  }
  if (data.pipeline != g_currentPipeline) {
    use_pipeline(data.pipeline, draw.pipeline, cmd);
  }
  if (!draw.decoded) {
    return;
  }
  const auto& pipeline = *g_currentPipelineObject;
  const VkDeviceSize offset = draw.offset;
  if (g_lookedUpRef != data.bindGroups.textureBindGroup) {
    g_lookedUpSet = texture_set(data.bindGroups.textureBindGroup);
    g_lookedUpRef = data.bindGroups.textureBindGroup;
  }
  const VkDescriptorSet textures = g_lookedUpSet;
  if (textures != g_boundTextures) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 1, 1, &textures, 0, nullptr);
    g_boundTextures = textures;
  }
  if (pipeline.fogRange && draw.fogRange != g_boundFogRange) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, glsl::FogRangeSet, 1, &draw.fogRange,
                            0, nullptr);
    g_boundFogRange = draw.fogRange;
  }
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &state.uniformSet,
                          data.uniformOffsets.size(), data.uniformOffsets.data());
  // The current matrix of the position matrices piece, 48 bytes each, from the CPU's copy of the uniform data.
  const uint8_t* matrix = state.uniformData.get() + data.uniformOffsets[gx::UniformPosMtx] +
                          size_t{data.immediateData.currentPnMtx} * sizeof(Mat3x4<float>);
  if (g_pushedMatrix != matrix) {
    vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Mat3x4<float>), matrix);
    g_pushedMatrix = matrix;
  }
  // A lit program's current normal matrix, after it: the normal matrices piece is there for every lit draw.
  if (pipeline.normalMatrix) {
    const uint8_t* normal = state.uniformData.get() + data.uniformOffsets[gx::UniformNrmMtx] +
                            size_t{data.immediateData.currentPnMtx} * sizeof(Mat3x4<float>);
    if (g_pushedNormalMatrix != normal) {
      vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(Mat3x4<float>), sizeof(Mat3x4<float>),
                         normal);
      g_pushedNormalMatrix = normal;
    }
  }
  // A line or point program's texture coordinates the offset moves, from the view piece's mask.
  if (!pipeline.vertexLayout.corners.empty()) {
    uint32_t mask = 0;
    std::memcpy(&mask,
                state.uniformData.get() + data.uniformOffsets[gx::UniformView] +
                    offsetof(gx::ViewUniform, lineTexCoordMask),
                sizeof(mask));
    if (g_pushedLineMask != mask) {
      std::array<float, glsl::LineMaskBytes / sizeof(float)> moved{};
      for (size_t i = 0; i < moved.size(); ++i) {
        moved[i] = (mask >> i & 1) != 0 ? 1.f : 0.f;
      }
      vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, glsl::LineMaskOffset, glsl::LineMaskBytes,
                         moved.data());
      g_pushedLineMask = mask;
    }
  }
  vkCmdBindVertexBuffers(cmd, 0, 1, &g_vertexCache.buffer, &offset);
  if (!pipeline.vertexLayout.corners.empty()) {
    // Two triangles a segment or point, whatever indices the draw came with.
    vkCmdDraw(cmd, glsl::stream_vertices(pipeline.shaderConfig, data.vtxCount), 1, 0, 0);
  } else if (draw.indices != NoIndices) {
    // Each vertex names its first equal vertex, decoded once: the same triangles, and the GPU transforms each distinct
    // vertex once.
    vkCmdBindIndexBuffer(cmd, g_vertexCache.buffer, draw.indices, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, data.vtxCount, 1, 0, 0, 0);
  } else if (data.indexCount != 0) {
    const VkBuffer indices = data.patternIndices ? g_indexPatterns.buffer : state.indices.buffer;
    vkCmdBindIndexBuffer(cmd, indices, data.idxRange.offset, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, data.indexCount, 1, 0, 0, 0);
  } else {
    vkCmdDraw(cmd, data.vtxCount, 1, 0, 0);
  }
}

// As gfx/clear.cpp draws it: the clear's depth as the depth range, over its rectangle of the target.
void draw_clear(const clear::DrawData& data, const TextureExtent& size, VkCommandBuffer cmd) noexcept {
  if (!bind_pipeline(data.pipeline, cmd)) {
    return;
  }
  vkCmdPushConstants(cmd, g_currentPipelineObject->layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(data.color),
                     data.color.data());
  const VkViewport viewport{0.f,        0.f,       static_cast<float>(size.width), static_cast<float>(size.height),
                            data.depth, data.depth};
  vkCmdSetViewport(cmd, 0, 1, &viewport);
  const auto scissor = to_scissor(data.rect, size);
  vkCmdSetScissor(cmd, 0, 1, &scissor);
  const VkBuffer vertices = triangle_buffer();
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &offset);
  vkCmdDraw(cmd, 3, 1, 0, 0);
}

// Render worker: the slot a snapshot of the frame's last pass goes into, its texture at the GX framebuffer's size; null
// when none is wanted yet, the other slot's frame isn't done, or the slot or its pipeline can't be made.
DepthPeekSlot* due_depth_peek() noexcept {
  const auto size = vi::configured_fb_size();
  std::lock_guard lock{g_depthPeekMutex};
  const auto now = std::chrono::steady_clock::now();
  if (!g_depthPeekRequested || now < g_nextDepthPeek || size.x == 0 || size.y == 0) {
    return nullptr;
  }
  auto& [first, second] = g_depthPeekSlots;
  const bool firstNewest = first.serial >= second.serial;
  const DepthPeekSlot& newest = firstNewest ? first : second;
  DepthPeekSlot& slot = firstNewest ? second : first;
  if (newest.serial != 0 && !frame_done(newest.serial)) {
    return nullptr;
  }
  if (!slot.texture || slot.texture->width != size.x || slot.texture->height != size.y) {
    slot.serial = 0;
    slot.texture = backend::create_texture({
        .width = size.x,
        .height = size.y,
        .mips = 1,
        .format = TextureFormat::RGBA8Unorm,
        .gxFormat = InvalidTextureFormat,
        .usage = backend::TextureUsage::Render,
        .label = "Depth peek",
    });
  }
  if (!slot.texture || depth_peek_pipeline() == VK_NULL_HANDLE) {
    return nullptr;
  }
  g_depthPeekRequested = false;
  g_nextDepthPeek = now + DepthPeekInterval;
  return &slot;
}

// Where the GX framebuffer of width by height sits in a source of sourceWidth by sourceHeight: depth_peek.cpp's
// make_params, offset and scale in source texels.
std::array<float, 4> peek_placement(uint32_t sourceWidth, uint32_t sourceHeight, uint32_t width,
                                    uint32_t height) noexcept {
  const auto policy = gx::g_gxState.viewportPolicy;
  if (policy == AURORA_VIEWPORT_NATIVE) {
    return {0.f, 0.f, 1.f, 1.f};
  }
  const float scaleX = static_cast<float>(sourceWidth) / static_cast<float>(width);
  const float scaleY = static_cast<float>(sourceHeight) / static_cast<float>(height);
  if (policy == AURORA_VIEWPORT_STRETCH) {
    return {0.f, 0.f, scaleX, scaleY};
  }
  const float scale = std::min(scaleX, scaleY);
  return {(static_cast<float>(sourceWidth) - static_cast<float>(width) * scale) * 0.5f,
          (static_cast<float>(sourceHeight) - static_cast<float>(height) * scale) * 0.5f, scale, scale};
}

// The snapshot's draw: each texel of the slot GX's z of the depth texel under its centre, as depth_peek.cpp's compute
// pass finds it, then the slot marked as this frame's.
void encode_depth_peek(DepthPeekSlot& slot, backend::Texture& depth, VkCommandBuffer cmd) noexcept {
  backend::Texture& target = *slot.texture;
  const VkFramebuffer framebuffer = target_framebuffer(target, nullptr);
  const VkDescriptorSet set = source_set(depth, false);
  if (framebuffer == VK_NULL_HANDLE || set == VK_NULL_HANDLE) {
    return;
  }
  const auto [offsetX, offsetY, scaleX, scaleY] = peek_placement(depth.width, depth.height, target.width, target.height);
  const auto width = static_cast<float>(depth.width);
  const auto height = static_cast<float>(depth.height);
  const std::array<float, 4> source{offsetX / width, offsetY / height, static_cast<float>(target.width) * scaleX / width,
                                    static_cast<float>(target.height) * scaleY / height};
  const VkExtent2D extent{target.width, target.height};
  const VkViewport viewport{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
  draw_texture(cmd, color_pass(Load::DontCare), framebuffer, extent, viewport, depth_peek_pipeline(), set, source);
  std::lock_guard lock{g_depthPeekMutex};
  slot.serial = next_frame_serial();
}

// A copy of the pass's colour, or of its depth where the pass left it to be read (depthRead): depth copies sample
// nearest, as tex_copy_conv.cpp's textureLoad does.
void encode_copy(const detail::RenderPass& pass, backend::Texture& color, backend::Texture* depth, bool depthRead,
                 VkCommandBuffer cmd) noexcept {
  const GXTexFmt format = pass.resolveFormat;
  const bool fromDepth = gx::is_depth_format(format);
  if (fromDepth && (!depthRead || !converts(format))) {
    static std::once_flag once;
    std::call_once(once, [format, depth] {
      Log.error("Depth copies (format {}) aren't supported{}", static_cast<int>(format),
                depth != nullptr && depth->depthView == VK_NULL_HANDLE ? " from depth the layer can't sample" : "");
    });
    return;
  }
  backend::Texture& source = fromDepth ? *depth : color;
  backend::Texture* target = pass.resolveTarget->gpu.get();
  if (target == nullptr) {
    return;
  }
  const VkPipeline pipeline = copy_pipeline(converts(format) ? format : GX_TF_RGBA8,
                                            !fromDepth && !gx::efb_has_alpha(pass.resolveSourceFormat));
  const VkFramebuffer framebuffer = target_framebuffer(*target, nullptr);
  const auto& rect = pass.resolveRect;
  const bool scaled =
      target->width != static_cast<uint32_t>(rect.width) || target->height != static_cast<uint32_t>(rect.height);
  const VkDescriptorSet set = source_set(source, scaled && !fromDepth);
  if (pipeline == VK_NULL_HANDLE || framebuffer == VK_NULL_HANDLE || set == VK_NULL_HANDLE) {
    return;
  }
  // The rectangle in the source, as recording.cpp's resolve_pass_into gives it to Dawn's conversions.
  const auto& sourceSize = pass.colorAttachments[SceneColorAttachmentIndex].size;
  const auto width = static_cast<float>(sourceSize.width);
  const auto height = static_cast<float>(sourceSize.height);
  const std::array<float, 4> sourceRect{rect.x / width, rect.y / height, rect.width / width, rect.height / height};
  const VkExtent2D extent{target->width, target->height};
  const VkViewport viewport{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
  draw_texture(cmd, color_pass(Load::DontCare), framebuffer, extent, viewport, pipeline, set, sourceRect);
}

// An EFB copy's indices through a palette into its RGBA8 texture, as the WebGPU backend's tex_palette_conv does.
void encode_palette_conv(const tex_palette_conv::ConvRequest& conv, VkCommandBuffer cmd) noexcept {
  // gx/texture.cpp asks for 8-bit and 4-bit indices only, from EFB copies.
  if (conv.variant == tex_palette_conv::Variant::Direct) {
    static std::once_flag once;
    std::call_once(once, [] { Log.error("Palette conversions of 16-bit indices aren't supported"); });
    return;
  }
  backend::Texture* const source = conv.src ? conv.src->gpu.get() : nullptr;
  backend::Texture* const target = conv.dst ? conv.dst->gpu.get() : nullptr;
  backend::Texture* const palette = conv.tlut ? conv.tlut->gpu.get() : nullptr;
  if (source == nullptr || target == nullptr || palette == nullptr || palette->width == 0) {
    return;
  }
  const VkPipeline pipeline = palette_pipeline(conv.variant == tex_palette_conv::Variant::FromFloat4);
  const VkFramebuffer framebuffer = target_framebuffer(*target, nullptr);
  const std::array sets{source_set(*source, false), source_set(*palette, false)};
  if (pipeline == VK_NULL_HANDLE || framebuffer == VK_NULL_HANDLE || sets[0] == VK_NULL_HANDLE ||
      sets[1] == VK_NULL_HANDLE) {
    return;
  }
  // The whole copy, texel for texel; then an index's scale and offset to the middle of its entry in the palette.
  const float entry = 1.f / static_cast<float>(palette->width);
  const std::array<float, 8> push{0.f, 0.f, 1.f, 1.f, entry, 0.f, 0.5f * entry, 0.5f};
  const VkExtent2D extent{target->width, target->height};
  const VkViewport viewport{0.f, 0.f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.f, 1.f};
  draw_triangle(cmd, color_pass(Load::DontCare), framebuffer, extent, viewport, pipeline, palette_layout(), sets,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, push);
}

// The multisampled EFB depth's sample 0 into the one-sample depth copies and GXPeekZ sample: the layer takes it out of
// eDRAM at the end of the pass just before, so it must follow vkCmdEndRenderPass (vulkan4xenon's docs/SUPPORT.md,
// Commands), and it ignores layouts.
void resolve_depth(const backend::Texture& depth, const backend::Texture& read, VkCommandBuffer cmd) noexcept {
  const VkImageResolve region{
      .srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
      .extent = {depth.width, depth.height, 1},
  };
  vkCmdResolveImage(cmd, depth.image, VK_IMAGE_LAYOUT_GENERAL, read.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

// The EFB copied into the swapchain image, then the UI image over it, premultiplied, where the UI drew: Dawn's present
// (webgpu/present.cpp:209-221), in one pass.
void present_efb(VkCommandBuffer cmd, VkFramebuffer framebuffer, VkExtent2D extent, const VkViewport& viewport,
                 bool covered, backend::Texture& efb, backend::Texture* ui) noexcept {
  const VkPipeline pipeline = present_pipeline();
  const VkDescriptorSet set = source_set(efb, true);
  if (pipeline == VK_NULL_HANDLE || set == VK_NULL_HANDLE) {
    return;
  }
  const VkPipeline composite = ui != nullptr ? ui_present_pipeline() : VK_NULL_HANDLE;
  const VkDescriptorSet uiSet = composite != VK_NULL_HANDLE ? source_set(*ui, false) : VK_NULL_HANDLE;
  draw_texture(cmd, present_pass(covered ? Load::DontCare : Load::Clear), framebuffer, extent, viewport, pipeline, set,
               WholeSource, composite, uiSet);
}

void encode_pass(const detail::FramePacket& frame, FrameState& state, const detail::FrameOp& op,
                 detail::RenderPass& pass, VkCommandBuffer cmd) noexcept {
  if (!pass.sealed) {
    return;
  }
  // Before the pass, where the WebGPU backend runs them (gfx/encoding.cpp, render).
  for (const auto& conv : pass.paletteConvs) {
    encode_palette_conv(conv, cmd);
  }
  if (pass.discardable) {
    return;
  }
  const auto& colorAttachment = pass.colorAttachments[SceneColorAttachmentIndex];
  backend::Texture* color = colorAttachment.target ? colorAttachment.target->gpu.get() : state.efbColor.get();
  backend::Texture* depth = pass.depthTarget ? pass.depthTarget->gpu.get() : state.efbDepth.get();
  if (color == nullptr || (pass.hasDepth && depth == nullptr)) {
    return;
  }
  // At 2 or 4 samples the EFB's passes, which always have depth (gfx/recording.cpp, set_efb_targets), draw into its
  // multisampled colour and depth, and readDepth is the one-sample depth that copies and GXPeekZ sample.
  backend::Texture* const multisampled = colorAttachment.target ? nullptr : state.efbMultisampled.get();
  backend::Texture* const readDepth = multisampled != nullptr ? state.efbDepthRead.get() : depth;
  // A depth copy or GXPeekZ's snapshot samples the pass's depth, which the layer copies out of eDRAM only from a pass
  // that leaves it to be read, or at 2 and 4 samples resolves after it; one with nothing to draw is encoded all the
  // same to leave it so.
  const bool sampledDepth = pass.hasDepth && readDepth->depthView != VK_NULL_HANDLE;
  const bool depthCopy = pass.resolveTarget && gx::is_depth_format(pass.resolveFormat);
  DepthPeekSlot* const peek = pass.captureDepthSnapshot && sampledDepth ? due_depth_peek() : nullptr;
  const bool depthRead = sampledDepth && (depthCopy || peek != nullptr);
  // A pass with nothing to draw leaves its targets as they are, so only its copy is encoded.
  if (pass.has_content() || depthRead) {
    const VkFramebuffer framebuffer = multisampled != nullptr
                                          ? target_framebuffer(*multisampled, depth, color)
                                          : target_framebuffer(*color, pass.hasDepth ? depth : nullptr);
    if (framebuffer == VK_NULL_HANDLE) {
      return;
    }
    const Load colorLoad = colorAttachment.clear ? Load::Clear : Load::Load;
    // The frame's last pass (recording.cpp's finish marks it) drops the EFB's depth unless a snapshot reads it: the
    // next frame's first pass clears it. A multisampled depth is kept for its resolve.
    const DepthStore depthStore = depthRead ? (multisampled != nullptr ? DepthStore::Kept : DepthStore::Read)
                                  : pass.captureDepthSnapshot && !pass.depthTarget ? DepthStore::Dropped
                                                                                   : DepthStore::Kept;
    const bool fromRead = pass.hasDepth && depth->depthLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    const std::array clearValues{
        VkClearValue{.color = {.float32 = {colorAttachment.clearValue.x(), colorAttachment.clearValue.y(),
                                           colorAttachment.clearValue.z(), colorAttachment.clearValue.w()}}},
        VkClearValue{.depthStencil = {.depth = pass.clearDepthValue}},
    };
    const VkRenderPassBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = pass.hasDepth ? scene_pass(colorLoad, pass.clearDepth ? Load::Clear : Load::Load, depthStore,
                                                 fromRead, multisampled != nullptr)
                                    : color_pass(colorLoad),
        .framebuffer = framebuffer,
        .renderArea = {{0, 0}, {color->width, color->height}},
        .clearValueCount = pass.hasDepth ? 2u : 1u,
        .pClearValues = clearValues.data(),
    };
    lay_out_pass(frame, state, op, pass);
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
    reset_bindings();
    size_t gxDraws = 0;
    for (auto& command : pass.commands) {
      switch (command.type) {
      case detail::CommandType::SetViewport: {
        const auto viewport = to_viewport(command.data.setViewport);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        break;
      }
      case detail::CommandType::SetScissor: {
        const auto scissor = to_scissor(command.data.setScissor, colorAttachment.size);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        break;
      }
      case detail::CommandType::Draw:
        switch (command.data.draw.kind) {
        case detail::DrawKind::Gx:
          draw_gx(state, detail::inline_payload<gx::DrawData>(command.data.draw.payload.data()), g_passDraws[gxDraws++],
                  cmd);
          break;
        case detail::DrawKind::Clear:
          draw_clear(detail::inline_payload<clear::DrawData>(command.data.draw.payload.data()), colorAttachment.size,
                     cmd);
          break;
        case detail::DrawKind::RmlUi: // recorded only on Dawn
          break;
        }
        break;
      case detail::CommandType::CustomDraw: // the WebGPU extension's, recorded only on Dawn
      case detail::CommandType::DebugMarker:
        break;
      }
    }
    vkCmdEndRenderPass(cmd);
    if (multisampled != nullptr && depthRead) {
      resolve_depth(*depth, *readDepth, cmd);
    }
    if (pass.hasDepth) {
      depth->depthLayout = depthStore == DepthStore::Read ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                          : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }
  }
  if (peek != nullptr) {
    encode_depth_peek(*peek, *readDepth, cmd);
  }
  if (pass.resolveTarget) {
    encode_copy(pass, *color, readDepth, depthRead, cmd);
  }
}
} // namespace

bool initialize_frames() noexcept {
  if (!create_buffer(VertexCacheSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true,
                     "vertex cache", g_vertexCache)) {
    return false;
  }
  if (const auto& patterns = gx::fifo::index_patterns(); !patterns.indices.empty()) {
    if (!create_buffer(patterns.indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "index patterns",
                       g_indexPatterns)) {
      return false;
    }
    std::memcpy(g_indexPatterns.mapped, patterns.indices.data(), patterns.indices.size());
  }
  for (auto& state : g_frames) {
    if (!create_buffer(UniformBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, "uniforms", state.uniforms) ||
        !create_buffer(IndexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, "indices", state.indices) ||
        !create_buffer(TextureUploadSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, "texture uploads", state.uploads)) {
      return false;
    }
    state.verts = new_cpu_pool(VertexBufferSize);
    state.storage = new_cpu_pool(StorageBufferSize);
    state.uniformData = new_cpu_pool(UniformBufferSize);
    state.indexData = new_cpu_pool(IndexBufferSize);
    state.uploadData = new_cpu_pool(TextureUploadSize);
    if (!state.verts || !state.storage || !state.uniformData || !state.indexData || !state.uploadData) {
      Log.error("No {} bytes in the heap for a frame's CPU pools",
                VertexBufferSize + StorageBufferSize + UniformBufferSize + IndexBufferSize + TextureUploadSize);
      return false;
    }
    state.uniformSet = allocate_set(uniform_set_layout());
    if (state.uniformSet == VK_NULL_HANDLE) {
      return false;
    }
    // finish() pads the pool by MaxUniformSize, so a draw's dynamic offset always leaves its piece (recording.cpp).
    std::array<VkDescriptorBufferInfo, gx::UniformPieceCount> buffers{};
    std::array<VkWriteDescriptorSet, gx::UniformPieceCount> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
      buffers[i] = {.buffer = state.uniforms.buffer, .offset = 0, .range = gx::UniformPieceSizes[i]};
      writes[i] = {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = state.uniformSet,
          .dstBinding = i,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .pBufferInfo = &buffers[i],
      };
    }
    vkUpdateDescriptorSets(g_device, writes.size(), writes.data(), 0, nullptr);
  }
  return update_efb();
}

void shutdown_frames() noexcept {
  for (auto& state : g_frames) {
    destroy_buffer(state.uniforms);
    destroy_buffer(state.indices);
    destroy_buffer(state.uploads);
    free_set(state.uniformSet);
    state = {};
  }
  destroy_buffer(g_vertexCache);
  destroy_buffer(g_indexPatterns);
  // What the cache held went with its buffer.
  g_cache = decode::VertexCache{VertexCacheSize, detail::FrameSlotCount, wait_for_frame};
  g_keys = {};
  g_lists = decode::IndexLists{IndexListBytes};
  g_efbColor.reset();
  g_efbDepth.reset();
  g_efbMultisampled.reset();
  g_efbDepthRead.reset();
  for (const auto& table : g_fogRangeTables) {
    free_set(table.set);
  }
  g_fogRangeTables.clear();
  g_passFogRanges.clear();
  reset_bindings();
  g_passDraws.clear();
  std::lock_guard lock{g_depthPeekMutex};
  g_depthPeekSlots = {};
  g_depthPeekRequested = false;
}

TextureExtent efb_size() noexcept { return g_efbSize; }
} // namespace aurora::gfx::vulkan4xenon

namespace aurora::gfx::backend {
using namespace vulkan4xenon;

void request_depth_snapshot() noexcept {
  std::lock_guard lock{g_depthPeekMutex};
  g_depthPeekRequested = true;
}

bool read_depth(uint16_t x, uint16_t y, uint32_t& z) noexcept {
  std::lock_guard lock{g_depthPeekMutex};
  const DepthPeekSlot* newest = nullptr;
  for (const auto& slot : g_depthPeekSlots) {
    if (slot.serial != 0 && frame_done(slot.serial) && (newest == nullptr || slot.serial > newest->serial)) {
      newest = &slot;
    }
  }
  if (newest == nullptr || x >= newest->texture->width || y >= newest->texture->height) {
    return false;
  }
  // Write-combined memory, a word at a time. A texel's bytes run red to alpha in memory (write_capture), so the word's
  // top three bytes are z, its high byte first.
  const auto* texels = reinterpret_cast<const volatile uint32_t*>(newest->texture->memory.mapped);
  z = texels[tiled_index(x, y, (newest->texture->width + 31) & ~31u)] >> 8;
  return true;
}

bool begin_frame(detail::FramePacket& frame) noexcept {
  auto& state = g_frames[frame.slot];
  if (!acquire_slot(frame.slot)) {
    return false;
  }
#ifdef AURORA_DIAGNOSTICS
  note_frame_path(frame.frameId, FramePath::SlotAcquired, std::chrono::steady_clock::now());
#endif
  if (!update_efb()) {
    release_slot(frame.slot);
    return false;
  }
  state.efbColor = g_efbColor;
  state.efbDepth = g_efbDepth;
  state.efbMultisampled = g_efbMultisampled;
  state.efbDepthRead = g_efbDepthRead;
  frame.verts = ByteBuffer{state.verts.get(), VertexBufferSize};
  frame.uniforms = ByteBuffer{state.uniformData.get(), UniformBufferSize};
  frame.indices = ByteBuffer{state.indexData.get(), IndexBufferSize};
  frame.storage = ByteBuffer{state.storage.get(), StorageBufferSize};
  frame.textureUpload = ByteBuffer{state.uploadData.get(), TextureUploadSize};
  return true;
}

void start_frame(detail::FramePacket& frame) noexcept {
  render_worker::enqueue_begin_frame(frame.frameId, [slot = frame.slot] {
    auto& state = g_frames[slot];
    state.uploadsDone = 0;
    state.uniformsCopied = 0;
    state.indicesCopied = 0;
    state.uploadsCopied = 0;
    state.verticesSkipped = 0;
    // Every frame before this one has been submitted: its end item came first.
    g_cache.begin_frame(next_frame_serial());
    g_keys.begin_frame();
    g_lists.begin_frame();
    const VkCommandBufferBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    state.recording = succeeded(vkBeginCommandBuffer(slot_commands(slot), &begin), "vkBeginCommandBuffer");
  });
}

void finish_frame(detail::FramePacket& frame) noexcept { (void)frame; }

// Texture copies and encoder tasks are the WebGPU extension's, recorded only on Dawn.
void encode(detail::FramePacket& frame, const detail::FrameOp& op) noexcept {
  auto& state = g_frames[frame.slot];
  if (!state.recording) {
    return;
  }
  const VkCommandBuffer cmd = slot_commands(frame.slot);
  copy_recorded(state.uniformData, state.uniforms, state.uniformsCopied, op.highWater.uniforms);
  copy_recorded(state.indexData, state.indices, state.indicesCopied, op.highWater.indices);
  copy_recorded(state.uploadData, state.uploads, state.uploadsCopied, op.highWater.textureUpload);
  upload_textures(state, op, cmd);
  if (op.renderPass != nullptr) {
    encode_pass(frame, state, op, *op.renderPass, cmd);
  }
}

void end_frame(detail::FramePacket& frame, const imgui::DrawData& overlay, const ui::Frame& ui) noexcept {
  auto& state = g_frames[frame.slot];
  const VkCommandBuffer cmd = slot_commands(frame.slot);
  // Drawn whether or not a swapchain image is acquired, so a capture shows it. A frame not recorded keeps RmlUi's
  // uploads for the next, as RmlUi holds their textures.
  backend::Texture* uiImage = nullptr;
  if (state.recording) {
    uiImage = encode_ui(frame.slot, ui, overlay, {state.efbColor->width, state.efbColor->height}, cmd);
  } else {
    keep_ui_uploads(ui);
  }
  uint32_t image = 0;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
#ifdef AURORA_DIAGNOSTICS
  const auto acquireStart = std::chrono::steady_clock::now();
#endif
  const bool present = state.recording && acquire_swapchain_image(image, framebuffer);
#ifdef AURORA_DIAGNOSTICS
  g_acquireNs.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - acquireStart).count(),
      std::memory_order_relaxed);
#endif
  if (present) {
    const VkExtent2D extent = swapchain_extent();
    const auto viewport = present_viewport(extent, state.efbColor->width, state.efbColor->height);
    const bool covered = viewport.width == extent.width && viewport.height == extent.height;
    present_efb(cmd, framebuffer, extent, viewport, covered, *state.efbColor, uiImage);
  }
#ifdef AURORA_DIAGNOSTICS
  compose_for_readback(cmd, *state.efbColor, uiImage);
#endif
  const uint64_t serial = next_frame_serial();
  submit_frame(frame.slot, image, present, frame.frameId);
  if (!frame.capturePath.empty()) {
    write_capture(*state.efbColor, uiImage, serial, frame.capturePath);
  }
#ifdef AURORA_DIAGNOSTICS
  read_back_ui(*state.efbColor, uiImage, serial);
#endif
  state.recording = false;
  reset_bindings();
  g_passDraws.clear();
#ifdef AURORA_DIAGNOSTICS
  if (g_cache.frame_bytes() > g_decodedPeak.load(std::memory_order_relaxed)) {
    g_decodedPeak.store(g_cache.frame_bytes(), std::memory_order_relaxed);
  }
  if (++g_cacheFrames == 1000) {
    g_cacheFrames = 0;
    const auto stats = g_cache.take_stats();
    constexpr double MiB = 1024.0 * 1024.0;
    const uint64_t bytes = stats.hitBytes + stats.missBytes;
    Log.info(
        "Vertex cache, last 1000 frames: {} of {} draws found their vertices, {:.2f}% of {:.1f} MiB; {:.1f} MiB "
        "decoded; {} entries evicted, {} forgotten, {} without room; at most {:.2f} MiB a frame and {:.2f} MiB "
        "held, of {:.2f}; waited {} times for frames in flight, {:.1f} ms in all, {:.2f} ms at most; {} index lists "
        "built, {} kept in {:.2f} MiB",
        stats.hits, stats.hits + stats.misses, bytes == 0 ? 0.0 : 100.0 * stats.hitBytes / bytes, bytes / MiB,
        stats.missBytes / MiB, stats.evictions, stats.forgotten, stats.failures, stats.peakFrameBytes / MiB,
        stats.peakUsedBytes / MiB, VertexCacheSize / MiB, stats.waits, stats.waitNs / 1e6, stats.longestWaitNs / 1e6,
        g_lists.take_built(), g_lists.size(), g_lists.bytes() / MiB);
  }
#endif
  // The cache waits for frames in flight until a draw fits, so a draw is skipped only when this frame's own vertices
  // leave no room once every frame before it is done: the cache is too small for the game.
  if (state.verticesSkipped != 0) {
    Log.error(
        "No room for the vertices of {} draws in the vertex cache's {} bytes beside this frame's own, with every "
        "frame before it done; they were skipped. AURORA_VULKAN4XENON_VERTEX_CACHE_BYTES is too small for this "
        "game",
        state.verticesSkipped, VertexCacheSize);
  }
  after_present();
  expire_texture_sets();
  collect_retired();
}
} // namespace aurora::gfx::backend

#ifdef AURORA_DIAGNOSTICS
// The render worker's time acquiring swapchain images since the last call, part of its busy time.
extern "C" unsigned long long aurora_vulkan4xenon_take_acquire_ns() {
  return aurora::gfx::vulkan4xenon::g_acquireNs.exchange(0, std::memory_order_relaxed);
}

// The most a frame's decoded vertices took of the vertex cache, and its size.
extern "C" void aurora_vulkan4xenon_vertices(unsigned long long* peak, unsigned long long* size) {
  *peak = aurora::gfx::vulkan4xenon::g_decodedPeak.load(std::memory_order_relaxed);
  *size = aurora::gfx::vulkan4xenon::VertexCacheSize;
}
#endif
