#include <dolphin/vi.h>

#include "../../window.hpp"
#include "aurora/math.hpp"
#include "vi_internal.hpp"

#include <optional>

namespace aurora::vi {
std::optional<GXRenderModeObj> g_renderMode;
ConfigureHook g_configureHook = nullptr;

Vec2<uint32_t> render_mode_size() noexcept {
  if (!g_renderMode) {
    return {640, 480};
  }
  return {g_renderMode->fbWidth, g_renderMode->efbHeight};
}

void configure(const GXRenderModeObj* rm) noexcept {
  const auto oldSize = render_mode_size();
  if (rm == nullptr) {
    g_renderMode.reset();
  } else {
    g_renderMode = *rm;
  }
  if (const auto size = render_mode_size(); size != oldSize) {
    window::request_frame_buffer_resize();
    if (g_configureHook != nullptr) {
      g_configureHook(size);
    }
  }
}

void set_configure_hook(ConfigureHook hook) noexcept { g_configureHook = hook; }

Vec2<uint32_t> configured_fb_size() noexcept {
  return render_mode_size();
}

// The display aspect VILockAspectRatio was given, 0 when unlocked. A GameCube framebuffer's pixels aren't square: a
// 640x448 framebuffer is a 1.43 rectangle of pixels a television showed at 4:3.
float g_lockedAspect = 0.f;

float locked_aspect() noexcept { return g_lockedAspect; }
} // namespace aurora::vi

extern "C" {
void VIInit() {}
void VIConfigure(const GXRenderModeObj* rm) { aurora::vi::configure(rm); }
void VIConfigurePan(u16 xOrg, u16 yOrg, u16 width, u16 height) {
  (void)xOrg;
  (void)yOrg;
  (void)width;
  (void)height;
}
u32 VIGetTvFormat() { return 0; }
void VIFlush() {}

void VISetWindowTitle(const char* title) { aurora::window::set_title(title); }
void VISetWindowFullscreen(bool fullscreen) { aurora::window::set_fullscreen(fullscreen); }
bool VIGetWindowFullscreen() { return aurora::window::get_fullscreen(); }
void VISetWindowSize(uint32_t width, uint32_t height) { aurora::window::set_window_size(width, height); }
void VISetWindowPosition(uint32_t x, uint32_t y) { aurora::window::set_window_position(x, y); }
void VICenterWindow() { aurora::window::center_window(); }
void VISetFrameBufferScale(float scale) { aurora::window::set_frame_buffer_scale(scale); }

void VILockAspectRatio(int width, int height) {
  const float aspect = (width > 0 && height > 0) ? static_cast<float>(width) / static_cast<float>(height) : 0.f;
  if (aspect == aurora::vi::g_lockedAspect) {
    return;
  }
  aurora::vi::g_lockedAspect = aspect;
  aurora::window::request_frame_buffer_resize();
}

void VIUnlockAspectRatio() { VILockAspectRatio(0, 0); }
}
