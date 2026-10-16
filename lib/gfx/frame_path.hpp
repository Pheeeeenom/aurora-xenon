#pragma once
// Where each stage hands a frame on, for a game's benchmark (aurora_frame_path), with AURORA_ENABLE_DIAGNOSTICS. Apart
// from frame.hpp, so the render worker and its tests take only this.

#include <chrono>
#include <cstdint>

namespace aurora::gfx {
enum class FramePath : uint8_t {
  SlotAcquired,   // begin_frame has the frame's slots
  FifoDone,       // the game thread has seen the FIFO worker finish the frame's commands
  RenderBegin,    // the render worker begins the frame
  Submitted,      // the backend submits it
  FenceSignalled, // its fence has signalled
  Count,
};

#ifdef AURORA_DIAGNOSTICS
// Game thread, as begin_frame gives the frame its id: clears the frame's record before any stage can stamp it.
void start_frame_path(uint64_t frameId) noexcept;
void note_frame_path(uint64_t frameId, FramePath stage, std::chrono::steady_clock::time_point when) noexcept;
#endif
} // namespace aurora::gfx
