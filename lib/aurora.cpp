#include <aurora/aurora.h>
#include <aurora/time.hpp>

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
#endif

#ifdef AURORA_ENABLE_RMLUI
#include "rmlui.hpp"
#endif

#include "input.hpp"
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
  if (g_config.msaa == 0) {
    g_config.msaa = 1;
  }
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
#else
  AuroraBackend selectedBackend = BACKEND_NULL;
  AURORA_ASSERT(window::create_window(BACKEND_NULL), "Error creating window: {}", SDL_GetError());
  AURORA_ASSERT(window::create_renderer(), "Failed to initialize SDL renderer: {}", SDL_GetError());
#endif

  window::show_window();
  thread::set_current({
      .name = "Main thread",
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
  input::shutdown();
  window::shutdown();
}

const AuroraEvent* update() noexcept {
  ZoneScoped;
  if (g_initialFrame) {
    g_initialFrame = false;
    input::initialize();
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

  imgui::new_frame(window::get_window_size());
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
  auto imguiDrawData = imgui::freeze();
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
void aurora_set_timescale(float scale) { aurora::time::set_scale(scale); }
float aurora_get_timescale() { return aurora::time::scale(); }
