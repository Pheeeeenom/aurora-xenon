#pragma once

#include "aurora/aurora.h"

#include <SDL3/SDL_events.h>
#include <aurora/rmlui.hpp>

#ifdef AURORA_RMLUI_DRAWS
#include "webgpu/gpu.hpp"

#include <dawn/webgpu_cpp.h>
#else
#include "gfx/ui.hpp"
#endif

namespace aurora::rmlui {

#ifdef AURORA_RMLUI_DRAWS
struct RecordedFrame {
  wgpu::BindGroup bindGroup;
  bool overlay = false;
};
#endif

void initialize(const AuroraWindowSize& size) noexcept;
void handle_event(SDL_Event& event) noexcept;
#ifdef AURORA_RMLUI_DRAWS
RecordedFrame record_frame(const webgpu::Viewport& presentViewport) noexcept;
#else
// Runs the documents a frame and, where something shows, records what they draw, swapped into frame.
void update(gfx::ui::Frame& frame) noexcept;
#endif
void shutdown() noexcept;

} // namespace aurora::rmlui
