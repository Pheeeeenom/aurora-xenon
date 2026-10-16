#include <aurora/aurora.h>
#include <aurora/time.hpp>
#include <cmath>

#include <string>
#include <vector>

#ifdef AURORA_ENABLE_GX
#include "gfx/resources.hpp"
#include "gfx/frame.hpp"
#include "gfx/recording.hpp"
#include "gfx/render_worker.hpp"
#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "gx/gx.hpp"
#include "gfx/backend.hpp"
#include "imgui.hpp"
#ifdef AURORA_GFX_BACKEND_DAWN
#include "webgpu/gpu.hpp"
#endif
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "gamepad.hpp"
#include "internal.hpp"
#include "thread.hpp"
#include "window.hpp"

#include <SDL3/SDL_filesystem.h>
#include <magic_enum.hpp>

#include "system_info.hpp"
#include "tracy/Tracy.hpp"

namespace aurora {
AuroraConfig g_config;
uint32_t g_sdlCustomEventsStart;
char g_gameName[4];

namespace {
constexpr Module Log{"aurora"};

#ifndef AURORA_ENABLE_GX
constexpr std::array<AuroraBackend, 0> PreferredBackendOrder{};
#endif

// Not magic_enum: AuroraBackend is a plain C enum with no fixed underlying
// type, and instantiating enum_name on one fails to compile under Apple clang.
constexpr const char* backend_name(AuroraBackend backend) noexcept {
  switch (backend) {
  case BACKEND_AUTO:
    return "auto";
  case BACKEND_D3D11:
    return "D3D11";
  case BACKEND_D3D12:
    return "D3D12";
  case BACKEND_METAL:
    return "Metal";
  case BACKEND_VULKAN:
    return "Vulkan";
  case BACKEND_OPENGL:
    return "OpenGL";
  case BACKEND_OPENGLES:
    return "OpenGL ES";
  case BACKEND_WEBGPU:
    return "WebGPU";
  case BACKEND_NULL:
    return "null";
  }
  return "?";
}

bool g_initialFrame = false;

AuroraInfo initialize(int argc, char* argv[], const AuroraConfig& config) noexcept {
  g_config = config;
  Log.info("Aurora initializing");
  log_system_information();
  if (g_config.appName == nullptr) {
    g_config.appName = "Aurora";
  } else {
    g_config.appName = strdup(g_config.appName);
  }
  if (g_config.userPath == nullptr) {
    g_config.userPath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.userPath = strdup(g_config.userPath);
  }
  if (g_config.cachePath == nullptr) {
    g_config.cachePath = SDL_GetPrefPath(nullptr, g_config.appName);
  } else {
    g_config.cachePath = strdup(g_config.cachePath);
  }
  if (g_config.resourcesPath == nullptr) {
    g_config.resourcesPath = SDL_GetBasePath();
  } else {
    g_config.resourcesPath = strdup(g_config.resourcesPath);
  }
  // WebGPU only guarantees sample counts 1 and 4, and Dawn rejects the rest
  // outright: an unsupported count aborts device creation rather than
  // degrading, so clamp here instead of trusting the caller.
  g_config.msaa = g_config.msaa > 1 ? 4 : 1;
  if (g_config.maxTextureAnisotropy == 0) {
    g_config.maxTextureAnisotropy = 16;
  }
  AURORA_ASSERT(window::initialize(), "Error initializing window");

  g_sdlCustomEventsStart = SDL_RegisterEvents(2);
  AURORA_ASSERT(g_sdlCustomEventsStart, "Failed to allocate user events: {}", SDL_GetError());
  AURORA_ASSERT(window::initialize_event_watch(), "Error initializing SDL event watch");

#ifdef AURORA_ENABLE_GX
  // The backend makes the window, device and surface.
  AuroraBackend selectedBackend = config.desiredBackend;
  AURORA_ASSERT(gfx::backend::initialize(selectedBackend), "Error creating window: {}", SDL_GetError());
  // A pinned backend that cannot come up silently falls through to the
  // preferred order, and the only trace of that was the per-attempt warnings
  // a few hundred lines earlier. Say it once, plainly, so a log tail still
  // shows the pin was not honoured.
  if (config.desiredBackend != BACKEND_AUTO && selectedBackend != config.desiredBackend) {
    Log.warn("Requested backend {} could not be initialized; running on {} instead",
             backend_name(config.desiredBackend), backend_name(selectedBackend));
  }
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  AURORA_ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();
  thread::set_current({
      .name = "Main thread",
      .priority = thread::Priority::High,
      .affinity = thread::Affinity::SharedCache,
  });

#ifdef AURORA_ENABLE_GX
  gfx::initialize();
  gx::fifo::init();
  imgui::create_context();
#endif
  const auto size = window::get_window_size();
  Log.info("Using framebuffer size {}x{} scale {}", size.fb_width, size.fb_height, size.scale);
#ifdef AURORA_ENABLE_GX
  if (g_config.imGuiInitCallback != nullptr) {
    g_config.imGuiInitCallback(&size);
  }
  imgui::initialize();
#endif

#ifdef AURORA_ENABLE_RMLUI
  rmlui::initialize(size);
#endif

  g_initialFrame = true;
  g_config.desiredBackend = selectedBackend;
  return {
      .backend = selectedBackend,
#ifdef AURORA_GFX_BACKEND_DAWN
      .adapterName = webgpu::g_adapterName.c_str(),
      .adapterDriver = webgpu::g_adapterDriver.c_str(),
      .adapterVendorId = webgpu::g_adapterInfo.vendorID,
      .adapterDeviceId = webgpu::g_adapterInfo.deviceID,
#else
      .adapterName = "",
      .adapterDriver = "",
      .adapterVendorId = 0,
      .adapterDeviceId = 0,
#endif
      .userPath = g_config.userPath,
      .cachePath = g_config.cachePath,
      .window = window::get_sdl_window(),
      .windowSize = size,
  };
}

void shutdown() noexcept {
#ifdef AURORA_ENABLE_GX
  gx::fifo::shutdown();
  gfx::render_worker::synchronize();
#ifdef AURORA_ENABLE_RMLUI
  rmlui::shutdown();
#endif
  imgui::shutdown();
  gfx::shutdown();
  gfx::backend::shutdown();
#endif
  gamepad::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    gamepad::initialize();
  }
#ifdef AURORA_ENABLE_GX
  gx::update();
#endif
  return window::poll_events();
}

#ifdef AURORA_ENABLE_GX
// The frame capture asked for on the game thread while it records a frame; end_frame hands it to that frame's present.
namespace capture {
std::string g_requestPath;
} // namespace capture
#endif

bool begin_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
  if (!gfx::backend::can_present()) {
    return false;
  }

#ifndef __EMSCRIPTEN__
  imgui::new_frame(window::get_window_size());
#endif
  const auto slot = gfx::begin_frame();
  if (!slot) {
    return false;
  }
  gx::begin_frame(*slot);
#endif
  return true;
}

#ifdef AURORA_ENABLE_GX
// Where the frame ended before the one ending is in the command stream.
uint64_t g_lastFrameEnd = 0;
#endif

// The FIFO processor ends the frame where its end is in the command stream: it seals the last pass, updates its caches
// and hands the frame to the render worker, which presents it with ImGui's overlay. The game thread goes on to the next
// frame meanwhile; it waits on the processor only where the game waits through GX, and for the frame before's stats.
void end_frame() noexcept {
  ZoneScoped;
#ifdef AURORA_ENABLE_GX
#ifdef __EMSCRIPTEN__
  imgui::DrawData imguiDrawData; // Browser controls live in the launcher.
#else
  auto imguiDrawData = imgui::freeze();
#endif
  std::string capturePath = std::move(capture::g_requestPath);
  capture::g_requestPath.clear();

  const auto ending = gfx::end_frame(std::move(imguiDrawData), std::move(capturePath));
  const uint64_t frameEnd = gx::fifo::end_frame({.slot = static_cast<uint8_t>(ending.slot), .close = ending.close});
  if (!ending.close) {
    // The backend records what it draws over the frame (RmlUi on Dawn).
    gx::fifo::drain();
    gfx::finish_frame(ending.slot);
  }
  // The stats published are the frame before's, as they were when this thread ended frames itself.
  gx::fifo::wait_processed(g_lastFrameEnd);
  g_lastFrameEnd = frameEnd;
  gfx::publish_frame_stats();
#endif
}
} // namespace
} // namespace aurora

// C API bindings
AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config) {
  return aurora::initialize(argc, argv, *config);
}
void aurora_shutdown() { aurora::shutdown(); }
AuroraWindowSize aurora_window_size() { return aurora::window::get_window_size(); }
void aurora_apply_frame_buffer_resize() { aurora::window::resize_frame_buffer_now(); }
void aurora_capture_frame(const char* path) {
#ifdef AURORA_ENABLE_GX
  if (path != nullptr) {
    aurora::capture::g_requestPath = path;
  }
#endif
}
const AuroraEvent* aurora_update() { return aurora::update(); }
bool aurora_begin_frame() { return aurora::begin_frame(); }
void aurora_end_frame() { aurora::end_frame(); }
SDL_Window* aurora_get_window() { return aurora::window::get_sdl_window(); }
AuroraWindowSize aurora_get_window_size() {
  return aurora::window::get_sdl_window() != nullptr ? aurora::window::get_window_size() : AuroraWindowSize{};
}
AuroraBackend aurora_get_backend() { return aurora::g_config.desiredBackend; }
#ifndef AURORA_ENABLE_GX
const AuroraBackend* aurora_get_available_backends(size_t* count) {
  if (count != nullptr) {
    *count = aurora::PreferredBackendOrder.size();
  }
  return aurora::PreferredBackendOrder.data();
}
#endif
void aurora_set_log_level(AuroraLogLevel level) { aurora::g_config.logLevel = level; }
void aurora_set_pause_on_focus_lost(bool value) { aurora::g_config.pauseOnFocusLost = value; }
void aurora_set_background_input(bool value) {
  aurora::g_config.allowJoystickBackgroundEvents = value;
  aurora::window::set_background_input(value);
}
#ifndef AURORA_ENABLE_GX
void aurora_set_resampler(AuroraSampler sampler) { (void)sampler; }
#endif
}
void aurora_preserve_frame_buffer(bool preserve) {
#ifdef AURORA_ENABLE_GX
  aurora::gfx::set_preserve_frame_buffer(preserve);
#else
  (void)preserve;
#endif
}
void aurora_set_timescale(float scale) { aurora::time::set_scale(scale); }
float aurora_get_timescale() { return aurora::time::scale(); }
