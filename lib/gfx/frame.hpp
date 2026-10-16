#pragma once

#include "resources.hpp"
#include "frame_packet.hpp"
#include "frame_path.hpp"

#include <optional>
#include <string>

namespace aurora::imgui {
class DrawData;
} // namespace aurora::imgui

// Frames recorded or drawing at once, each with its own copy of the streaming pools: the CMake setting
// AURORA_FRAME_SLOT_COUNT.
#ifndef AURORA_FRAME_SLOT_COUNT
#define AURORA_FRAME_SLOT_COUNT 2
#endif

namespace aurora::gfx::detail {

inline constexpr size_t FrameSlotCount = AURORA_FRAME_SLOT_COUNT;
static_assert(FrameSlotCount >= 1, "AURORA_FRAME_SLOT_COUNT must be at least 1");

AuroraStats& stats() noexcept;

// FIFO processor, at the frame's start in the command stream: the frame's recording begins.
void begin_frame_recording(size_t slot) noexcept;
// FIFO processor: it has processed the frame's commands (the frame path's FifoDone).
void note_fifo_done(size_t slot) noexcept;
// FIFO processor, at the frame's end in the command stream: seals the frame's last pass, and closes the frame if
// `close`.
void end_frame_recording(size_t slot, bool close) noexcept;
// Ends the frame's recording and hands it to the render worker to submit and present. On the FIFO processor, or on the
// game thread with the stream drained.
void close_frame(size_t slot) noexcept;

} // namespace aurora::gfx::detail

namespace aurora::gfx {
void initialize();
void shutdown();
// Game thread: takes a frame slot, which the backend readies. The frame's recording begins where its start is written
// into the command stream (gx::begin_frame). The slot, or none when the backend has none.
std::optional<size_t> begin_frame();
struct FrameEnding {
  size_t slot;
  // Whether the FIFO processor closes the frame at its end. If not, the game thread drains the command stream and
  // calls finish_frame.
  bool close;
};
// Game thread: the backend presents the frame begun last with the overlay, and writes it to capturePath when that isn't
// empty, once the frame is closed.
FrameEnding end_frame(imgui::DrawData overlay, std::string capturePath);
// Game thread, with the command stream drained: what the backend records after the frame's GX passes, then the close.
void finish_frame(size_t slot);
// Game thread, once the frame ended before the last one is closed: publishes its stats (aurora_get_stats).
void publish_frame_stats() noexcept;
uint32_t current_frame() noexcept;
// The id of the frame begun last.
uint64_t current_frame_id() noexcept;
void gpu_synchronize();
void after_present() noexcept;
float calculate_fps() noexcept;
} // namespace aurora::gfx
