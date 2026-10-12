#pragma once

#include "aurora/aurora.h"

#include <SDL3/SDL_events.h>
#include <aurora/rmlui.hpp>

#ifdef AURORA_RMLUI_DRAWS
#include "webgpu/gpu.hpp"

#include <dawn/webgpu_cpp.h>
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
// Runs the documents a frame, for a backend that doesn't draw them.
void update() noexcept;
#endif
void shutdown() noexcept;

} // namespace aurora::rmlui
