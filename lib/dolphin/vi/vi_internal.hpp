#pragma once

#include <dolphin/gx/GXStruct.h>

#include <cstdint>

#include <aurora/math.hpp>

namespace aurora::vi {
void configure(const GXRenderModeObj* rm) noexcept;
// The game thread's VI mode: its framebuffer size.
Vec2<uint32_t> configured_fb_size() noexcept;
// Called on the game thread when VIConfigure changes the framebuffer size, so the change reaches the FIFO processor in
// order with the frame's commands.
using ConfigureHook = void (*)(Vec2<uint32_t> fbSize) noexcept;
void set_configure_hook(ConfigureHook hook) noexcept;
// The display aspect VILockAspectRatio was given; 0 when unlocked.
float locked_aspect() noexcept;
} // namespace aurora::vi
