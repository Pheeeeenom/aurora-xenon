#include "frame.hpp"

#include "backend.hpp"
#include "pipeline_cache.hpp"
#include "recording.hpp"
#include "render_worker.hpp"
#include "texture_replacement.hpp"
#include "ui.hpp"
#include "../gx/fifo.hpp"
#include "../gx/gx.hpp"
#include "../imgui.hpp"
#include "../window.hpp"
#ifdef AURORA_RMLUI_DRAWS
#include "../rmlui/pipeline.hpp"

#include <aurora/rmlui.hpp>
#elif defined(AURORA_ENABLE_RMLUI)
#include "../rmlui.hpp"
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tracy/Tracy.hpp>

namespace aurora::gfx {
using namespace detail;

namespace {
AuroraStats g_stats{};
// Each closed frame's stats, by frame id, until the game thread publishes them as the next frame ends.
std::array<AuroraStats, 4> g_closedStats{};
// Game thread: the framebuffer size the frame begun last had, the frame, and the id of the frame ended before it.
Vec2<uint32_t> g_beganFbSize{};
size_t g_beganSlot = 0;
uint64_t g_beganFrameId = 0;
uint64_t g_endedFrameId = 0;

void publish_stats(const AuroraStats& frameStats) noexcept {
  auto& stats = g_stats;
  stats.drawCallCount = frameStats.drawCallCount;
  stats.mergedDrawCallCount = frameStats.mergedDrawCallCount;
  stats.lastVertSize = frameStats.lastVertSize;
  stats.lastUniformSize = frameStats.lastUniformSize;
  stats.lastIndexSize = frameStats.lastIndexSize;
  stats.lastStorageSize = frameStats.lastStorageSize;
  stats.lastTextureUploadSize = frameStats.lastTextureUploadSize;

  TracyPlotConfig("aurora: lastVertSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastUniformSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastIndexSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastStorageSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlotConfig("aurora: lastTextureUploadSize", tracy::PlotFormatType::Memory, false, true, 0);
  TracyPlot("aurora: queuedPipelines", static_cast<int64_t>(stats.queuedPipelines));
  TracyPlot("aurora: createdPipelines", static_cast<int64_t>(stats.createdPipelines));
  TracyPlot("aurora: drawCallCount", static_cast<int64_t>(stats.drawCallCount));
  TracyPlot("aurora: mergedDrawCallCount", static_cast<int64_t>(stats.mergedDrawCallCount));
  TracyPlot("aurora: lastVertSize", static_cast<int64_t>(stats.lastVertSize));
  TracyPlot("aurora: lastUniformSize", static_cast<int64_t>(stats.lastUniformSize));
  TracyPlot("aurora: lastIndexSize", static_cast<int64_t>(stats.lastIndexSize));
  TracyPlot("aurora: lastStorageSize", static_cast<int64_t>(stats.lastStorageSize));
  TracyPlot("aurora: lastTextureUploadSize", static_cast<int64_t>(stats.lastTextureUploadSize));
}

uint32_t g_frameIndex = UINT32_MAX;

std::array<FramePacket, FrameSlotCount> g_framePackets;
uint64_t g_nextFrameId = 1;
render_worker::FrameSlotPool g_frameSlots{FrameSlotCount};
// What the game thread hands a frame's end, kept by slot until the frame is closed; RmlUi's list until the slot is
// taken again, as the render worker reads it.
struct FrameEndData {
  imgui::DrawData overlay;
  std::string capturePath;
  ui::Frame ui;
};
std::array<FrameEndData, FrameSlotCount> g_frameEnds;

using PresentClock = std::chrono::steady_clock;
constexpr auto PresentFpsWindow = std::chrono::seconds{1};
std::mutex g_presentStatsMutex;
std::deque<PresentClock::time_point> g_presentTimes;
std::atomic_int64_t g_lastPresentNs = 0;
std::atomic_int64_t g_presentPeriodNs = 0;
PresentClock::time_point g_cpuFrameStart;
constexpr auto MaxPacingSample = std::chrono::milliseconds{250};
constexpr uint32_t PacingEmaWeight = 8;

int64_t timestamp_ns(PresentClock::time_point time) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

int64_t duration_ns(PresentClock::duration duration) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

void update_ema(std::atomic_int64_t& value, int64_t sample) {
  if (sample <= 0 || sample > duration_ns(MaxPacingSample)) {
    return;
  }

  int64_t current = value.load(std::memory_order_acquire);
  while (true) {
    const int64_t next = current == 0 ? sample : current + (sample - current) / static_cast<int64_t>(PacingEmaWeight);
    if (value.compare_exchange_weak(current, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }
  }
}

void prune_present_times(PresentClock::time_point now) {
  while (!g_presentTimes.empty() && g_presentTimes.front() + PresentFpsWindow < now) {
    g_presentTimes.pop_front();
  }
}

} // namespace

namespace detail {

AuroraStats& stats() noexcept { return g_stats; }

} // namespace detail

RenderTargetLayout scene_render_target_layout() noexcept { return backend::efb_layout(); }

bool uses_reversed_z() noexcept { return gx::UseReversedZ; }

void initialize() {
  g_frameIndex = 0;
  g_lastPresentNs.store(0, std::memory_order_release);
  g_presentPeriodNs.store(0, std::memory_order_release);
  g_cpuFrameStart = {};
  {
    std::lock_guard lock{g_presentStatsMutex};
    g_presentTimes.clear();
  }
  render_worker::initialize();
#ifdef AURORA_RMLUI_DRAWS
  rmlui::initialize_pipeline();
#endif
  initialize_pipeline_cache();
}

void shutdown() {
  render_worker::synchronize();
  render_worker::shutdown();
  g_lastPresentNs.store(0, std::memory_order_release);
  g_presentPeriodNs.store(0, std::memory_order_release);
  g_cpuFrameStart = {};
  {
    std::lock_guard lock{g_presentStatsMutex};
    g_presentTimes.clear();
  }
  shutdown_pipeline_cache();
  texture_replacement::shutdown();
  gx::shutdown();
#ifdef AURORA_RMLUI_DRAWS
  rmlui::shutdown_pipeline();
#endif
  for (auto& packet : g_framePackets) {
    packet = {};
  }
  shutdown_recording();
  g_frameIndex = UINT32_MAX;
  g_frameSlots.reset();
}

size_t acquire_frame_slot() {
  ZoneScopedN("Acquire frame slot");
  const auto waitStart = PresentClock::now();
  // Blocks: the render worker releases a slot itself at the end of its frame task, never from a Dawn callback, so
  // nothing needs pumping meanwhile.
  const size_t slot = g_frameSlots.acquire();
  const auto waitDuration = PresentClock::now() - waitStart;
  const double waitMs = std::chrono::duration<double, std::milli>{waitDuration}.count();
  TracyPlot("aurora: frameSlotWaitMs", waitMs);
  return slot;
}

std::optional<size_t> begin_frame() {
  ZoneScoped;
  const size_t frameSlot = acquire_frame_slot();
  auto& frame = g_framePackets[frameSlot];
  frame = {};
  frame.frameId = g_nextFrameId++;
  frame.slot = frameSlot;
#ifdef AURORA_DIAGNOSTICS
  // Before the backend can hand the frame to the render worker. A backend that waits for a slot of its own stamps
  // SlotAcquired again once it has it.
  start_frame_path(frame.frameId);
  note_frame_path(frame.frameId, FramePath::SlotAcquired, PresentClock::now());
#endif
  // The EFB follows the window's framebuffer, so the FIFO processor finishes the frames before with the size they
  // began with.
  const auto windowSize = window::get_window_size();
  if (const Vec2<uint32_t> fbSize{windowSize.fb_width, windowSize.fb_height}; fbSize != g_beganFbSize) {
    gx::fifo::finish_published();
    g_beganFbSize = fbSize;
  }
  // The backend points the pools at its memory before recording starts.
  if (!backend::begin_frame(frame)) {
    g_frameSlots.release(frameSlot);
    return std::nullopt;
  }
  g_beganSlot = frameSlot;
  g_beganFrameId = frame.frameId;
  g_cpuFrameStart = PresentClock::now();
  return frameSlot;
}

FrameEnding end_frame(imgui::DrawData overlay, std::string capturePath) {
  ZoneScoped;
  const size_t slot = g_beganSlot;
  auto& ending = g_frameEnds[slot];
  ending.overlay = std::move(overlay);
  ending.capturePath = std::move(capturePath);
  if (g_cpuFrameStart.time_since_epoch().count() != 0) {
    const auto cpuFrameTime = PresentClock::now() - g_cpuFrameStart;
    const double cpuFrameTimeMs = std::chrono::duration<double, std::milli>{cpuFrameTime}.count();
    TracyPlot("aurora: cpuFrameTimeMs", cpuFrameTimeMs);
  }
  // RmlUi records its passes into the frame after the GX passes, here on the game thread.
#ifdef AURORA_RMLUI_DRAWS
  const bool close = !rmlui::is_initialized();
#else
#ifdef AURORA_ENABLE_RMLUI
  // RmlUi records what it draws, for a backend that draws it with the frame.
  rmlui::update(ending.ui);
#endif
  constexpr bool close = true;
#endif
  if (close) {
    backend::finish_frame(g_framePackets[slot]);
  }
  return {.slot = slot, .close = close};
}

void finish_frame(size_t slot) {
  backend::finish_frame(g_framePackets[slot]);
  close_frame(slot);
}

void publish_frame_stats() noexcept {
  // Published on the game thread, which reads them after end_frame, rather than by the render worker, which raced that
  // read. They are the frame before the one just ended: where a read after end_frame found them while the render
  // worker published them.
  publish_stats(g_closedStats[g_endedFrameId % g_closedStats.size()]);
  g_endedFrameId = g_beganFrameId;
}

namespace detail {
void begin_frame_recording(size_t slot) noexcept {
  auto& frame = g_framePackets[slot];
  frame.frameIndex = g_frameIndex;
  backend::start_frame(frame);
  begin_recording(frame, slot);
  begin_pipeline_frame();
}

void note_fifo_done([[maybe_unused]] size_t slot) noexcept {
#ifdef AURORA_DIAGNOSTICS
  note_frame_path(g_framePackets[slot].frameId, FramePath::FifoDone, PresentClock::now());
#endif
}

void end_frame_recording(size_t slot, bool close) noexcept {
  finish();
  if (close) {
    close_frame(slot);
  }
}

void close_frame(size_t slot) noexcept {
  const auto recorded = end_recording();
  auto& frame = *recorded.packet;
  auto& ending = g_frameEnds[slot];
  frame.capturePath = std::move(ending.capturePath);
  ending.capturePath.clear();
  g_closedStats[frame.frameId % g_closedStats.size()] = frame.stats;
  const uint64_t frameId = frame.frameId;
  end_pipeline_frame();
  ++g_frameIndex;

  // The backend submits and presents the packet; the slot is free once the packet is cleared.
  render_worker::enqueue_end_frame(frameId, [slot, overlay = std::move(ending.overlay)] {
    auto& packet = g_framePackets[slot];
    backend::end_frame(packet, overlay, g_frameEnds[slot].ui);
    packet = {};
    g_frameSlots.release(slot);
  });
  ending.overlay = {};
}
} // namespace detail

uint32_t current_frame() noexcept { return g_frameIndex; }

uint64_t current_frame_id() noexcept { return g_nextFrameId - 1; }

void gpu_synchronize() { render_worker::synchronize(); }

void synchronize() { render_worker::synchronize(); }

void after_present() noexcept {
  const auto now = PresentClock::now();
  const int64_t nowNs = timestamp_ns(now);
  const int64_t previousPresentNs = g_lastPresentNs.exchange(nowNs, std::memory_order_acq_rel);
  if (previousPresentNs != 0) {
    update_ema(g_presentPeriodNs, nowNs - previousPresentNs);
    const double presentPeriodMs = static_cast<double>(g_presentPeriodNs.load(std::memory_order_acquire)) / 1'000'000.0;
    TracyPlot("aurora: presentPeriodMs", presentPeriodMs);
  }
  std::lock_guard lock{g_presentStatsMutex};
  g_presentTimes.push_back(now);
  prune_present_times(now);
}

float calculate_fps() noexcept {
  const auto now = PresentClock::now();
  std::lock_guard lock{g_presentStatsMutex};
  prune_present_times(now);
  if (g_presentTimes.size() < 2) {
    return 0.f;
  }

  const auto elapsed = std::chrono::duration<float>(g_presentTimes.back() - g_presentTimes.front()).count();
  if (elapsed <= 0.f) {
    return 0.f;
  }
  return static_cast<float>(g_presentTimes.size() - 1) / elapsed;
}
} // namespace aurora::gfx

const AuroraStats* aurora_get_stats() { return &aurora::gfx::detail::stats(); }
void aurora_get_pipeline_counts(uint32_t* queued, uint32_t* created) {
  uint32_t q = 0;
  uint32_t c = 0;
  aurora::gfx::get_pipeline_counts(q, c);
  if (queued != nullptr) {
    *queued = q;
  }
  if (created != nullptr) {
    *created = c;
  }
}
float aurora_get_fps() { return aurora::gfx::calculate_fps(); }

#ifdef AURORA_DIAGNOSTICS
// The frame begun last, whose path aurora_frame_path (frame_path.cpp) gives.
extern "C" unsigned long long aurora_frame_path_last_id() { return aurora::gfx::current_frame_id(); }
#endif
