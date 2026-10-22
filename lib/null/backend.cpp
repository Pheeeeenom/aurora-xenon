// A GX backend that discards the work, so Aurora runs with no window and no GPU.
#include "../gfx/backend.hpp"

#include "../gfx/frame.hpp"
#include "../gfx/frame_packet.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resources.hpp"
#include "../imgui.hpp"
#include "../internal.hpp"
#include "../window.hpp"
#ifdef AURORA_GLSL_RECORD_DIR
#include "../vulkan4xenon/glsl.hpp"
#endif

#include <aurora/aurora.h>
#include <aurora/gfx.h>
#include <aurora/imgui.h>
#include <imgui.h>

#include <initializer_list>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#ifdef AURORA_GLSL_RECORD_DIR
#include <set>
#include <string_view>
#include <tuple>
#endif

namespace aurora::gfx::backend {
struct Pipeline {};
struct Texture {};
struct UploadBuffer {};

namespace {
constexpr Module Log{"aurora::gfx::null"};

// What Dawn reports on the desktop (Vulkan) for the limits that shape a recorded frame, so textures convert and the
// pools fill as they do under Dawn: its "Using limits" and "Enabling features" log lines.
constexpr Limits NullLimits{
    .uniformAlignment = 64,
    .storageAlignment = 16,
    .textureRowAlignment = 256,
    .bcTextures = true,
    .cmprAsBc1 = false,
    .bcPowerOfTwo = false,
    .componentSwizzle = true,
    // As under Dawn, the pipeline cache's threads create the pipelines, which are empty here.
    .pipelineThreads = true,
};

constexpr uint64_t PoolBytes = VertexBufferSize + UniformBufferSize + IndexBufferSize + StorageBufferSize +
                               (UseTextureBuffer ? TextureUploadSize : 0);
// Nothing reads the pools back, so every frame slot records into the same memory.
std::unique_ptr<uint8_t[]> g_pools;
TextureExtent g_efbSize{};

void take_efb_size() noexcept {
  const auto size = window::get_window_size();
  g_efbSize = {size.fb_width, size.fb_height};
}

#ifdef AURORA_GLSL_RECORD_DIR
// Each size and format of target the game draws or copies into, once each: what the vulkan4xenon backend must draw
// into, beside the programs recorded (vulkan4xenon's ROADMAP.md, milestone 11).
void log_target(const TextureDesc& desc) noexcept {
  static constexpr std::array<std::string_view, 5> Usages{"", "Copy target", "Conversion target", "Offscreen target",
                                                          "Offscreen depth target"};
  static std::mutex lock;
  static std::set<std::tuple<TextureUsage, uint32_t, uint32_t, u32>> logged;
  if (desc.usage == TextureUsage::Sampled) {
    return;
  }
  std::lock_guard guard(lock);
  if (logged.emplace(desc.usage, desc.width, desc.height, desc.gxFormat).second) {
    Log.info("{} {}x{}, GX format {}", Usages[static_cast<size_t>(desc.usage)], desc.width, desc.height,
             desc.gxFormat == InvalidTextureFormat ? "none" : fmt::format("{}", desc.gxFormat));
  }
}
#endif
} // namespace

bool initialize(AuroraBackend& backend) noexcept {
  g_pools.reset(new (std::nothrow) uint8_t[PoolBytes]);
  if (!g_pools) {
    Log.error("Failed to allocate {} bytes of streaming pools", PoolBytes);
    return false;
  }
  take_efb_size();
  backend = BACKEND_NULL;
  Log.info("Null backend: GX work is recorded and discarded, with no window");
  return true;
}

void shutdown() noexcept { g_pools.reset(); }

bool uses_sdl_window() noexcept { return false; }

bool can_present() noexcept { return true; }

void resize(const AuroraWindowSize& size) noexcept { (void)size; }

void refresh_surface() noexcept {}

std::shared_ptr<Texture> create_texture(const TextureDesc& desc) noexcept {
#ifdef AURORA_GLSL_RECORD_DIR
  log_target(desc);
#else
  (void)desc;
#endif
  return std::make_shared<Texture>();
}

std::shared_ptr<UploadBuffer> stage_upload(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                                           bool ui) noexcept {
  (void)data;
  (void)bytesPerRow;
  (void)rowsPerImage;
  (void)ui;
  return std::make_shared<UploadBuffer>();
}

CompiledPipeline create_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout) noexcept {
  (void)layout;
#ifdef AURORA_GLSL_RECORD_DIR
  // Each config's GLSL for the vulkan4xenon backend, which a run on this backend records for the offline compile.
  vulkan4xenon::glsl::record(config.shaderConfig, AURORA_GLSL_RECORD_DIR);
#else
  (void)config;
#endif
  return {.main = std::make_shared<Pipeline>()};
}

std::shared_ptr<Pipeline> create_pipeline(const clear::PipelineConfig& config,
                                          const RenderTargetLayout& layout) noexcept {
  (void)config;
  (void)layout;
  return std::make_shared<Pipeline>();
}

void pipelines_loaded() noexcept {}

void prepare_textures(BindGroupRef ref, const TextureSet& set) noexcept {
  (void)ref;
  (void)set;
}

bool converts_copy(GXTexFmt format) noexcept {
  (void)format;
  return false;
}

void request_depth_snapshot() noexcept {}

bool read_depth(uint16_t x, uint16_t y, uint32_t& z) noexcept {
  (void)x;
  (void)y;
  (void)z;
  return false;
}

const Limits& limits() noexcept { return NullLimits; }

RenderTargetLayout efb_layout() noexcept {
  RenderTargetLayout layout{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{
          .semantic = ColorAttachmentSemantic::SceneColor,
          .format = TextureFormat::RGBA8Unorm,
          .width = g_efbSize.width,
          .height = g_efbSize.height,
      }}},
      .depthStencilFormat = TextureFormat::Depth32Float,
      .sampleCount = g_config.msaa,
  };
  detail::finalize_render_target_layout(layout);
  return layout;
}

bool begin_frame(detail::FramePacket& frame) noexcept {
  // The EFB follows the window between frames, as Dawn's does when a resize event reaches it.
  take_efb_size();
  size_t offset = 0;
  const auto point = [&](ByteBuffer& buf, uint64_t size) {
    buf = ByteBuffer{g_pools.get() + offset, static_cast<size_t>(size)};
    offset += size;
  };
  point(frame.verts, VertexBufferSize);
  point(frame.uniforms, UniformBufferSize);
  point(frame.indices, IndexBufferSize);
  point(frame.storage, StorageBufferSize);
  if constexpr (UseTextureBuffer) {
    point(frame.textureUpload, TextureUploadSize);
  }
  return true;
}

void start_frame(detail::FramePacket& frame) noexcept { (void)frame; }

void finish_frame(detail::FramePacket& frame) noexcept { (void)frame; }

void encode(detail::FramePacket& frame, const detail::FrameOp& op) noexcept {
  (void)frame;
  (void)op;
}

void end_frame(detail::FramePacket& frame, const imgui::DrawData& overlay, const ui::Frame& ui) noexcept {
  (void)overlay;
  (void)ui;
  if (!frame.capturePath.empty()) {
    static std::once_flag once;
    std::call_once(once, [] { Log.warn("Frame capture: the null backend draws nothing"); });
  }
  after_present();
}
} // namespace aurora::gfx::backend

namespace aurora::imgui {
void initialize() noexcept {}

void shutdown() noexcept { ImGui::DestroyContext(); }

void process_event(const SDL_Event& event) noexcept { (void)event; }

void new_frame(const AuroraWindowSize& size) noexcept {
  ImGuiIO& io = ImGui::GetIO();
  // A renderer would build the font atlas, and NewFrame needs it built.
  if (!io.Fonts->IsBuilt()) {
    io.Fonts->Build();
  }
  io.DisplaySize = {static_cast<float>(size.width), static_cast<float>(size.height)};
  ImGui::NewFrame();
}
} // namespace aurora::imgui

void aurora_enable_vsync(bool enabled) { (void)enabled; }

bool aurora_present_waits_for_vblank() { return false; }

void aurora_set_resampler(AuroraSampler sampler) { (void)sampler; }

const AuroraBackend* aurora_get_available_backends(size_t* count) {
  static constexpr AuroraBackend Backends[]{BACKEND_NULL};
  if (count != nullptr) {
    *count = std::size(Backends);
  }
  return Backends;
}

void aurora_gpu_frame_time(uint64_t* lastNs, uint64_t* meanNs, uint64_t* maxNs, uint64_t* count) {
  // No GPU, so no samples.
  for (uint64_t* out : {lastNs, meanNs, maxNs, count}) {
    if (out != nullptr) {
      *out = 0;
    }
  }
}

ImTextureID aurora_imgui_add_texture(uint32_t width, uint32_t height, const void* rgba8) {
  (void)width;
  (void)height;
  (void)rgba8;
  return ImTextureID{};
}
