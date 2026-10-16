// Each frame's path, for a game's benchmark.
#include "frame_path.hpp"

#ifdef AURORA_DIAGNOSTICS
#include <array>
#include <atomic>
#include <cstddef>

namespace aurora::gfx {
namespace {
// The last 64 frames' paths, kept until the benchmark reads each, a few frames after its fence.
struct FramePathEntry {
  std::atomic<uint64_t> frameId{0};
  std::array<std::atomic<int64_t>, static_cast<size_t>(FramePath::Count)> ns{};
};
std::array<FramePathEntry, 64> g_framePaths;
} // namespace

void start_frame_path(uint64_t frameId) noexcept {
  auto& entry = g_framePaths[frameId % g_framePaths.size()];
  entry.frameId.store(0, std::memory_order_release);
  for (auto& ns : entry.ns) {
    ns.store(0, std::memory_order_relaxed);
  }
  entry.frameId.store(frameId, std::memory_order_release);
}

void note_frame_path(uint64_t frameId, FramePath stage, std::chrono::steady_clock::time_point when) noexcept {
  auto& entry = g_framePaths[frameId % g_framePaths.size()];
  if (entry.frameId.load(std::memory_order_acquire) == frameId) {
    entry.ns[static_cast<size_t>(stage)].store(when.time_since_epoch() / std::chrono::nanoseconds{1},
                                               std::memory_order_relaxed);
  }
}
} // namespace aurora::gfx

// A frame's path in steady_clock nanoseconds by FramePath, 0 where a stage wasn't seen; 0 once the frame has left the
// record.
extern "C" int aurora_frame_path(unsigned long long frameId, long long* ns) {
  using namespace aurora::gfx;
  const auto& entry = g_framePaths[frameId % g_framePaths.size()];
  for (size_t i = 0; i < entry.ns.size(); ++i) {
    ns[i] = entry.ns[i].load(std::memory_order_relaxed);
  }
  return entry.frameId.load(std::memory_order_acquire) == frameId ? 1 : 0;
}
#endif
