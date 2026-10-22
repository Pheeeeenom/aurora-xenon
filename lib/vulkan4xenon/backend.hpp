#pragma once
// The vulkan4xenon backend's own declarations, shared by its files, on the vulkan4xenon layer: the Vulkan 1.0 subset
// over the Xbox 360's GPU, which its docs/SUPPORT.md describes.

#include "../gfx/backend.hpp"
#include "../gfx/clear.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/ui.hpp"
#include "../gx/pipeline.hpp"
#include "../internal.hpp"
#include "decode.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>

namespace aurora::gfx::vulkan4xenon {
inline constexpr Module Log{"aurora::vulkan4xenon"};

// Logs a Vulkan call that failed, by name and result; true when it succeeded.
bool succeeded(VkResult result, const char* call) noexcept;

// A counting semaphore for wake-ups between threads.
class Semaphore {
public:
  explicit Semaphore(int count = 0) noexcept : m_count(count) {}
  void post() noexcept {
    {
      std::lock_guard lock{m_mutex};
      ++m_count;
    }
    m_cv.notify_one();
  }
  void wait() noexcept {
    std::unique_lock lock{m_mutex};
    m_cv.wait(lock, [&] { return m_count > 0; });
    --m_count;
  }

private:
  std::mutex m_mutex;
  std::condition_variable m_cv;
  int m_count;
};

// device.cpp: the instance, device, queue and swapchain, memory, and the frames' submission.
extern VkDevice g_device;
extern VkQueue g_queue;

bool initialize_device() noexcept;
void shutdown_device() noexcept;
// Whether the layer samples BC1 images, by the format's properties, as vulkan4xenon has since its milestone 20 (its
// docs/SUPPORT.md, Formats).
bool samples_bc1() noexcept;

// Sub-allocated from blocks of the layer's one memory type: whole blocks while it can, then exact ones. The UI's (ui)
// come from pools of their own, whose blocks go back to the layer as they empty.
struct Allocation {
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  VkDeviceSize size = 0;
  uint8_t* mapped = nullptr;
  uint32_t pool = 0;
};
bool allocate(const VkMemoryRequirements& requirements, Allocation& out, bool ui = false) noexcept;
void free_allocation(const Allocation& allocation) noexcept;

// A buffer with memory of its own, mapped for its whole life.
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  uint8_t* mapped = nullptr;
  VkDeviceSize size = 0;
  uint32_t pool = 0;
#ifdef AURORA_DIAGNOSTICS
  // The kernel block it took, rounded as the pool's are, for the memory peaks.
  VkDeviceSize allocated = 0;
#endif
};
bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool videoRam, const char* label, Buffer& out) noexcept;
void destroy_buffer(Buffer& buffer) noexcept;

// Runs `release` once the GPU has finished frame g_submitted + 1, the one being recorded, and so every frame that could
// have used what it releases; from any thread.
void retire(std::function<void()> release) noexcept;

enum class Load : uint8_t {
  Load,
  Clear,
  DontCare,
};
// How a scene pass leaves its depth: stored as an attachment for the passes after; stored and left to be read, which
// the layer copies out of eDRAM into the depth image for a depth copy or snapshot to sample (its docs/SUPPORT.md,
// Images); or dropped.
enum class DepthStore : uint8_t {
  Kept,
  Read,
  Dropped,
};
// RGBA8 colour, stored, and, for scenes, D24S8 depth, left as depthStore says. A loaded depth comes in as the pass
// before left it: to be read when fromRead is set, else as an attachment. Multisampled, both are at the EFB's samples,
// and the colour's average goes into a third attachment, at one sample.
VkRenderPass scene_pass(Load color, Load depth, DepthStore depthStore = DepthStore::Kept, bool fromRead = false,
                        bool multisampled = false) noexcept;
// The EFB's samples: 1, or the config's 2 or 4 where the layer draws the EFB's formats at that count.
VkSampleCountFlagBits efb_samples() noexcept;
VkRenderPass color_pass(Load color) noexcept;
// B8G8R8A8 colour, the swapchain's only format (vulkan4xenon's docs/SUPPORT.md, Display and swapchain), cleared or
// left as it is: what the present draws into.
VkRenderPass present_pass(Load color) noexcept;
// The UI's: RGBA8 colour and a D24S8 whose stencil holds RmlUi's clip mask, each loaded or cleared, and stored.
VkRenderPass ui_pass(Load color, Load depthStencil) noexcept;

VkExtent2D swapchain_extent() noexcept;
// Render worker: waits on the CPU for the next image to leave the screen (vulkan4xenon's docs/SUPPORT.md, Display
// and swapchain).
bool acquire_swapchain_image(uint32_t& index, VkFramebuffer& framebuffer) noexcept;

VkCommandBuffer slot_commands(size_t slot) noexcept;
// Game thread: takes the slot once the GPU has finished its previous frame; false if the wait failed. A slot taken
// for a frame that is then not submitted is given back.
bool acquire_slot(size_t slot) noexcept;
void release_slot(size_t slot) noexcept;
// Render worker: submits the slot's commands with its fence and, if they drew one, presents the image.
bool submit_frame(size_t slot, uint32_t imageIndex, bool present, uint64_t frameId) noexcept;
// Render worker: runs what was retired before every finished frame.
void collect_retired() noexcept;
// Render worker: the number the next frame it submits will get; submit_frame numbers frames from 1, in order.
uint64_t next_frame_serial() noexcept;
// Any thread: whether the GPU has finished the frame submitted as `serial`, without waiting.
bool frame_done(uint64_t serial) noexcept;
// The most anisotropy a sampler may ask for: 1 where the layer has no samplerAnisotropy.
float max_anisotropy() noexcept;
// Render worker: blocks until the frame submitted as `serial` is done on the GPU, and returns true. The GPU waiter
// marks frames done in order, and a submitted frame's fence depends on nothing the render worker does next, so a wait
// for a submitted frame ends. A frame not yet submitted is logged and not waited for: false.
bool wait_for_frame(uint64_t serial) noexcept;

// pipeline.cpp: pipeline layouts and pipelines.
bool initialize_pipelines() noexcept;
void shutdown_pipelines() noexcept;

// program.cpp: each program's SPIR-V, from the game's pack, the disk or glslang. Called under pipeline.cpp's lock.
using Program = std::span<const uint32_t>;
void initialize_programs() noexcept;
void shutdown_programs() noexcept;
// Empty when glslang refuses the program, which is logged once; its pipeline isn't made, and its draws are skipped.
Program find_program(const std::string& source, bool fragment) noexcept;

// Set 0 of the GX layout: the frame's uniform buffer, dynamic, once per uniform piece. Set 1: the 8 GX texture maps.
VkDescriptorSetLayout uniform_set_layout() noexcept;
VkDescriptorSetLayout texture_set_layout() noexcept;
// The one texture a copy or the present samples, and set 2 of the GX layout, the range fog table (glsl.hpp).
VkDescriptorSetLayout source_set_layout() noexcept;
VkPipelineLayout copy_layout() noexcept;
// A palette conversion's: the copy at set 0, the palette at set 1, and the copy's push constant with 1 / its entries.
VkPipelineLayout palette_layout() noexcept;
// Three vertices covering any framebuffer: positions and texture coordinates.
VkBuffer triangle_buffer() noexcept;

// Null when the pipeline can't be made.
std::shared_ptr<backend::Pipeline> make_pipeline(const clear::PipelineConfig& config,
                                                 const RenderTargetLayout& layout) noexcept;
// Null, with the reason logged, for a config the layer can't draw as GX does, glsl.hpp can't express or the game's
// shader pack doesn't hold; its draws are skipped. At more than one sample, for the EFB's passes.
std::shared_ptr<backend::Pipeline> make_pipeline(const gx::PipelineConfig& config,
                                                 VkSampleCountFlagBits samples) noexcept;
// Whether an EFB copy to this format draws a conversion rather than the texels as they are.
bool converts(GXTexFmt format) noexcept;
// Render worker: the pipeline drawing a copy to `format`, with alpha read as 1 when opaqueAlpha is set (copies from an
// EFB without alpha, as tex_copy_conv's are); null when it can't be made.
VkPipeline copy_pipeline(GXTexFmt format, bool opaqueAlpha) noexcept;
// Render worker: the pipeline converting a copy's 8-bit or 4-bit indices through a palette; null when it can't be made.
VkPipeline palette_pipeline(bool fourBit) noexcept;
// Render worker: the pipeline packing a depth image's GX z into an RGBA8 snapshot for GXPeekZ (glsl.hpp,
// depth_peek_fragment) through the copy layout; null when it can't be made.
VkPipeline depth_peek_pipeline() noexcept;
// The copy of the EFB into a swapchain image; null when its programs aren't in the pack.
VkPipeline present_pipeline() noexcept;
// Render worker: the UI's pipelines over glsl.hpp's ui programs, through ui_pipeline_layout, in ui_pass; null when they
// can't be made. RmlUi's by ui::Pipeline, its vertices Rml::Vertex; ImGui's, its vertices ImDrawVert; and the UI
// image's premultiplied composite over the swapchain image, through the copy layout in present_pass.
VkPipeline ui_pipeline(ui::Pipeline pipeline) noexcept;
VkPipeline imgui_pipeline() noexcept;
VkPipeline ui_present_pipeline() noexcept;
#ifdef AURORA_DIAGNOSTICS
// The present's composite into an RGBA8 target, for the console checks to read back.
VkPipeline ui_check_pipeline() noexcept;
#endif
// The UI's: set 0 the texture sampled, and the vertex stage's push constant, a matrix's columns then a translation.
VkPipelineLayout ui_pipeline_layout() noexcept;
inline constexpr uint32_t UiPushBytes = 80;

// backend.cpp: textures and descriptor sets.
bool initialize_textures() noexcept;
// Tiled, as every image the layer makes is. R32Float only sampled, as the range fog table; R8 and RG8 only sampled, as
// intensity, and as intensity and alpha. At 2 or 4 samples, the EFB's colour or depth, never sampled; transfer adds
// the usages a resolve takes.
// A depth target made with depthSampled false is never sampled, and takes no memory but eDRAM.
std::shared_ptr<backend::Texture> make_texture(const backend::TextureDesc& desc,
                                               VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT,
                                               VkImageUsageFlags transfer = 0, bool depthSampled = true) noexcept;
void shutdown_textures() noexcept;
VkDescriptorSet allocate_set(VkDescriptorSetLayout layout) noexcept;
void free_set(VkDescriptorSet set) noexcept;
VkSampler sampler(const SamplerConfig& config) noexcept;
// Render worker: the set prepare_textures made for a draw, or the empty one.
VkDescriptorSet texture_set(BindGroupRef ref) noexcept;
// Render worker, after a frame's submission.
void expire_texture_sets() noexcept;

// encoding.cpp: the frames.
bool initialize_frames() noexcept;
void shutdown_frames() noexcept;
TextureExtent efb_size() noexcept;
// A colour target's framebuffer, alone or with a depth target.
VkFramebuffer target_framebuffer(backend::Texture& color, const backend::Texture* depth,
                                 const backend::Texture* resolve = nullptr) noexcept;
// The set a copy or the present samples a texture through, filtered or not.
VkDescriptorSet source_set(backend::Texture& texture, bool linear) noexcept;
// A tiled R8G8B8A8 image's texel, its bytes red to alpha, read on the CPU; once the GPU is done with it.
std::array<uint8_t, 4> read_texel(const backend::Texture& image, uint32_t x, uint32_t y) noexcept;
// A pass drawing one texture over a target, for copies and the present. source is offset and scale in it.
void draw_texture(VkCommandBuffer cmd, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extent,
                  const VkViewport& viewport, VkPipeline pipeline, VkDescriptorSet set,
                  const std::array<float, 4>& source) noexcept;

// ui.cpp: RmlUi's list and ImGui's lists drawn into the UI image, made while something shows.
void shutdown_ui() noexcept;
// Render worker, before the frame's present: the textures staged for the UI copied in, then, if either draws, the
// UI's passes. The UI image they drew into, or null when nothing drew: then no pass was recorded, and the UI's
// targets are released.
backend::Texture* encode_ui(size_t slot, const ui::Frame& rml, const imgui::DrawData& imgui,
                            VkCommandBuffer cmd) noexcept;
#ifdef AURORA_DIAGNOSTICS
// Render worker, before a frame's submission: when a readback is asked for, the EFB and the UI image composited as the
// present composites them, into an image of the EFB's size.
void compose_for_readback(VkCommandBuffer cmd, backend::Texture& efb, backend::Texture* ui) noexcept;
// Render worker, after it: the EFB, the UI image and that composite read back once the frame is done.
void read_back_ui(const backend::Texture& efb, const backend::Texture* ui, uint64_t serial) noexcept;
#endif
} // namespace aurora::gfx::vulkan4xenon

namespace aurora::gfx::backend {
struct Texture {
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  vulkan4xenon::Allocation memory;
  uint32_t width = 0;
  uint32_t height = 0;
  // BC1 blocks of 4x4 texels, 8 bytes each, rather than texels.
  bool bc1 = false;
  // A texel's bytes: 1 for R8, 2 for RG8, else 4.
  uint32_t texelBytes = 4;
  // A depth target's depth alone, to sample; null where the layer can't sample it (make_texture).
  VkImageView depthView = VK_NULL_HANDLE;
  // Render worker only: the levels uploaded so far, so an upload over one waits for the scenes before it; the
  // texture as a colour target, with the depth it was made with or alone; and as a copy's source. A depth target's
  // layout as the last pass left it: as an attachment, or to be read.
  VkImageLayout depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t uploadedLevels = 0;
  VkFramebuffer framebuffer = VK_NULL_HANDLE;
  const Texture* framebufferDepth = nullptr;
  std::array<VkDescriptorSet, 2> sourceSets{};

  Texture() noexcept = default;
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;
  ~Texture();
};

struct Pipeline {
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  // GX pipelines: what decode_vertices needs, and what of it the vertex cache keys on (decode::decode_format).
  gx::ShaderConfig shaderConfig{};
  vulkan4xenon::decode::VertexLayout vertexLayout;
  uint64_t decodeFormat = 0;
  // The vertex program reads the current normal matrix, pushed after the position matrix (glsl.hpp).
  bool normalMatrix = false;
  // The fragment program samples the range fog table (glsl.hpp).
  bool fogRange = false;

  Pipeline() noexcept = default;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  ~Pipeline();
};

// Sub-allocated as images are: load frames overflow the pool with hundreds of uploads, and a kernel block apiece,
// 256 KiB at least in video RAM, ran video RAM out (hardware, diag3).
struct UploadBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  vulkan4xenon::Allocation memory;

  UploadBuffer() noexcept = default;
  UploadBuffer(const UploadBuffer&) = delete;
  UploadBuffer& operator=(const UploadBuffer&) = delete;
  ~UploadBuffer();
};
} // namespace aurora::gfx::backend
