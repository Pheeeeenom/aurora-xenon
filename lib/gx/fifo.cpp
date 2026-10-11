#include "fifo.hpp"

#include "../dolphin/vi/vi_internal.hpp"
#include "../thread.hpp"
#include "command_processor.hpp"
#include "dolphin/gx/GXCommandList.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#include <tracy/Tracy.hpp>

// The command stream's memory, in bytes: fixed chunks the game thread writes while the FIFO processor reads those
// before them, each taken back once the processor has left it. The game thread waits only when every chunk holds
// commands not yet processed, as a GameCube CPU waited at its FIFO's high-water mark.
#ifndef AURORA_FIFO_RING_BYTES
#define AURORA_FIFO_RING_BYTES 2097152
#endif

namespace aurora::gx::fifo {
namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
} // namespace detail

namespace {
constexpr Module Log{"aurora::gx::fifo"};
constexpr auto kProcessingMode = ProcessingMode::Thread;
constexpr uint32_t kDrawBatchSize = 1;
// The single buffer's starting size. A run of commands longer than a chunk gets a chunk of its own.
constexpr uint32_t kChunkSize = 64 * 1024;
constexpr uint32_t kChunkCount = std::max<uint32_t>(AURORA_FIFO_RING_BYTES / kChunkSize, 2);

// A piece of the stream. The game thread writes one chunk at a time; the processor reads them in the order they're
// linked, each from its base to its end.
struct Chunk {
  uint8_t* data = nullptr;
  uint32_t capacity = 0;
  // The stream position of data[0], set before any of the chunk's bytes are published.
  uint64_t base = 0;
  // Where the chunk's commands end once the game thread has moved on to `next`; UINT64_MAX until then. `next` is
  // linked first.
  std::atomic<uint64_t> end{UINT64_MAX};
  std::atomic<Chunk*> next{nullptr};
  // From when the game thread takes the chunk until the processor leaves it.
  std::atomic<bool> inUse{false};
  // Allocated for one run of commands longer than a chunk; the processor frees it as it leaves it.
  bool own = false;
};

std::unique_ptr<Chunk[]> sRing;
uint8_t* sRingMemory = nullptr;
// Game thread: the ring chunk taken last, and the chunk being written.
uint32_t sRingIndex = 0;
Chunk* sWriteChunk = nullptr;
// Processor: the chunk being read.
Chunk* sReadChunk = nullptr;

bool sFrameActive = false;
uint32_t sPendingDraws = 0;
std::atomic<uint64_t> sPublished{0};
std::atomic<uint64_t> sProcessed{0};
// The stream position the game thread waits for the processor to reach, UINT64_MAX when it waits for none, and a count
// the processor bumps when it reaches it. The game thread is the only thread that waits for positions, and the
// processor wakes it only then, not at every step.
std::atomic<uint64_t> sAwaited{UINT64_MAX};
std::atomic<uint32_t> sReached{0};
uint64_t sWaiterWakes = 0;
std::atomic<uint32_t> sWorkerWake{0};
// Set by the processor before it waits for a wake, cleared after. A publish wakes it only while this is set: without
// a futex, libstdc++'s notify takes a mutex even with nobody waiting, which costs the game thread more than the
// commands it publishes.
std::atomic<bool> sWorkerSleeping{false};
thread::Thread sWorkerThread;
std::atomic<DrawDoneCallback> sDrawDoneCallback{nullptr};
// Game thread: where the last GXSetDrawDone's token ends in the stream.
uint64_t sDrawDonePosition = 0;
std::atomic<DrawSyncCallback> sDrawSyncCallback{nullptr};
std::atomic<uint16_t> sDrawSyncToken{0};
bool sLoggedOwnChunk = false;

void dispatch_draw_done() noexcept {
  if (const auto callback = sDrawDoneCallback.load(std::memory_order_acquire); callback != nullptr) {
    callback();
  }
}

// A display list called by reference, read from the game's memory now, as the GPU reads a called list when it reaches
// it. GX has a game keep a list unchanged until it has been drawn, so its bytes are still the call's.
void process_list(const uint8_t* data, uint32_t size) noexcept {
  uint32_t done = 0;
  while (done < size) {
    const ProcessResult result = process(data + done, size - done);
    AURORA_ASSERT(result.bytesProcessed > 0 && result.bytesProcessed <= size - done,
                  "Display list processing made invalid progress: processed {} of {} remaining bytes",
                  result.bytesProcessed, size - done);
    if (result.drawDone) {
      dispatch_draw_done();
    }
    if (result.list != nullptr) {
      Log.warn("Ignoring a display list called from a display list");
    }
    done += result.bytesProcessed;
  }
}

void wake_worker() noexcept {
  sWorkerWake.fetch_add(1, std::memory_order_release);
  sWorkerWake.notify_all();
}

// Processor, having processed up to `processed`: wakes the game thread if that's as far as it waits for.
void notify_reached(uint64_t processed) noexcept {
  uint64_t awaited = sAwaited.load(std::memory_order_seq_cst);
  if (processed >= awaited && sAwaited.compare_exchange_strong(awaited, UINT64_MAX, std::memory_order_seq_cst)) {
    sReached.fetch_add(1, std::memory_order_seq_cst);
    sReached.notify_all();
  }
}

// Game thread: waits until the processor has processed the stream up to `position`, which is published.
void await_position(uint64_t position) noexcept {
  while (true) {
    const uint32_t reached = sReached.load(std::memory_order_seq_cst);
    sAwaited.store(position, std::memory_order_seq_cst);
    if (sProcessed.load(std::memory_order_seq_cst) >= position) {
      sAwaited.store(UINT64_MAX, std::memory_order_relaxed);
      return;
    }
    sReached.wait(reached, std::memory_order_seq_cst);
    ++sWaiterWakes;
  }
}

// Processor: moves past the chunks the stream has left where processing stands, so the game thread can take them back.
void leave_finished_chunks(uint64_t processed) noexcept {
  while (sReadChunk->end.load(std::memory_order_acquire) == processed) {
    Chunk* chunk = sReadChunk;
    sReadChunk = chunk->next.load(std::memory_order_acquire);
    if (chunk->own) {
      std::free(chunk->data);
      delete chunk;
    } else {
      chunk->inUse.store(false, std::memory_order_release);
      chunk->inUse.notify_all();
    }
  }
}

void process_to(uint64_t target) noexcept {
  uint64_t processed = sProcessed.load(std::memory_order_relaxed);
  while (processed < target) {
    leave_finished_chunks(processed);
    const Chunk& chunk = *sReadChunk;
    // A published command never crosses a chunk's end: the game thread carries what isn't published yet over whole.
    const uint64_t stop = std::min(target, chunk.end.load(std::memory_order_acquire));
    AURORA_ASSERT(processed >= chunk.base && stop > processed && stop - chunk.base <= chunk.capacity,
                  "FIFO processing range [{}, {}) is outside its chunk [{}, {})", processed, stop, chunk.base,
                  chunk.base + chunk.capacity);
    const ProcessResult result =
        process(chunk.data + (processed - chunk.base), static_cast<uint32_t>(stop - processed));
    AURORA_ASSERT(result.bytesProcessed > 0 && result.bytesProcessed <= stop - processed,
                  "FIFO processor made invalid progress: processed {} of {} remaining bytes", result.bytesProcessed,
                  stop - processed);
    if (result.drawDone) {
      dispatch_draw_done();
    }
    // The list is the game's memory, not the stream's.
    if (result.list != nullptr) {
      process_list(result.list, result.listSize);
    }
    processed += result.bytesProcessed;
    // Sequentially consistent, as is await_position's store of the position it waits for: one of them sees the other.
    sProcessed.store(processed, std::memory_order_seq_cst);
    notify_reached(processed);
  }
  leave_finished_chunks(processed);
}

#ifdef AURORA_DIAGNOSTICS
// The worker's processing time, which a game's benchmark takes each frame (aurora_fifo_take_busy_ns).
std::atomic<uint64_t> sWorkerBusyNs{0};
#endif

void worker_main(std::stop_token token) noexcept {
  std::stop_callback wakeOnStop{token, wake_worker};
  while (true) {
    const uint32_t event = sWorkerWake.load(std::memory_order_acquire);
    const uint64_t processed = sProcessed.load(std::memory_order_relaxed);
    const uint64_t published = sPublished.load(std::memory_order_acquire);
    if (published != processed) {
#ifdef AURORA_DIAGNOSTICS
      const auto start = std::chrono::steady_clock::now();
#endif
      process_to(published);
#ifdef AURORA_DIAGNOSTICS
      sWorkerBusyNs.fetch_add(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count(),
          std::memory_order_relaxed);
#endif
      continue;
    }
    // The game thread may have moved on from the chunk being read since it was processed.
    leave_finished_chunks(processed);

    if (token.stop_requested()) {
      break;
    }
    // Dekker's handshake with publish_to: the processor stores sWorkerSleeping then loads sPublished, the game thread
    // stores sPublished then loads sWorkerSleeping, all sequentially consistent. In their single total order one of the
    // stores comes first, so either this load sees the new position and the processor doesn't wait, or the game thread
    // sees sWorkerSleeping set and bumps sWorkerWake, which `event` was read before, so the wait returns. A stale set
    // flag only costs a spurious wake.
    sWorkerSleeping.store(true, std::memory_order_seq_cst);
    if (sPublished.load(std::memory_order_seq_cst) != processed) {
      sWorkerSleeping.store(false, std::memory_order_relaxed);
      continue;
    }
    sWorkerWake.wait(event, std::memory_order_acquire);
    sWorkerSleeping.store(false, std::memory_order_relaxed);
  }
}

void start_worker() {
  if (kProcessingMode != ProcessingMode::Thread || sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread = thread::Thread{{
                                     .name = "Aurora FIFO processor",
                                     .affinity = thread::Affinity::SharedCache,
                                 },
                                 worker_main};
}

void stop_worker() {
  if (!sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread.request_stop();
  sWorkerThread.join();
}

void use_chunk(Chunk& chunk, uint64_t base) noexcept {
  chunk.base = base;
  chunk.end.store(UINT64_MAX, std::memory_order_relaxed);
  chunk.next.store(nullptr, std::memory_order_relaxed);
  chunk.inUse.store(true, std::memory_order_relaxed);
  sWriteChunk = &chunk;
  detail::sBufferData = chunk.data;
  detail::sBufferCapacity = chunk.capacity;
}

// With the processor stopped: frees the stream, including chunks of their own not yet left.
void free_stream() noexcept {
  for (Chunk* chunk = sReadChunk; chunk != nullptr;) {
    Chunk* next = chunk->next.load(std::memory_order_relaxed);
    if (chunk->own) {
      std::free(chunk->data);
      delete chunk;
    }
    chunk = next;
  }
  sReadChunk = nullptr;
  sWriteChunk = nullptr;
  sRing.reset();
  std::free(sRingMemory);
  sRingMemory = nullptr;
  detail::sBufferData = nullptr;
  detail::sBufferSize = 0;
  detail::sBufferCapacity = 0;
}

// With the processor stopped: the stream starts over at position 0, in the ring's first chunk.
void reset_stream() {
  free_stream();
  sRingMemory = static_cast<uint8_t*>(std::malloc(size_t{kChunkCount} * kChunkSize));
  AURORA_ASSERT(sRingMemory != nullptr, "fifo::init: failed to allocate {} bytes", size_t{kChunkCount} * kChunkSize);
  sRing = std::make_unique<Chunk[]>(kChunkCount);
  for (uint32_t i = 0; i < kChunkCount; ++i) {
    sRing[i].data = sRingMemory + size_t{i} * kChunkSize;
    sRing[i].capacity = kChunkSize;
  }
  sRingIndex = 0;
  use_chunk(sRing[0], 0);
  sReadChunk = sWriteChunk;
  sPendingDraws = 0;
  sPublished.store(0, std::memory_order_relaxed);
  sProcessed.store(0, std::memory_order_relaxed);
}

// Game thread: carries what isn't published yet over to the start of a chunk with room for `extra` bytes more, so a
// command is always whole in one chunk.
void next_chunk(uint32_t extra) {
  if (sWriteChunk == nullptr) {
    // Written before init: the stream starts here, and init starts it over.
    reset_stream();
    if (extra <= detail::sBufferCapacity) {
      return;
    }
  }
  Chunk& old = *sWriteChunk;
  const uint64_t published = sPublished.load(std::memory_order_relaxed);
  const auto tailOffset = static_cast<uint32_t>(published - old.base);
  const uint32_t tailSize = detail::sBufferSize - tailOffset;
  const uint64_t needed = uint64_t{tailSize} + extra;
  AURORA_ASSERT(needed <= std::numeric_limits<uint32_t>::max(), "fifo::write_data: {} bytes between publishes", needed);
  Chunk* next = nullptr;
  if (needed <= kChunkSize) {
    sRingIndex = (sRingIndex + 1) % kChunkCount;
    next = &sRing[sRingIndex];
    if (next->inUse.load(std::memory_order_acquire)) {
      ZoneScopedN("Wait for a FIFO chunk");
      wake_worker();
      do {
        next->inUse.wait(true, std::memory_order_acquire);
      } while (next->inUse.load(std::memory_order_acquire));
    }
  } else {
    // Twice what's needed, as the run likely goes on.
    const auto capacity = static_cast<uint32_t>(std::min<uint64_t>(needed * 2, std::numeric_limits<uint32_t>::max()));
    next = new Chunk;
    next->capacity = capacity;
    next->data = static_cast<uint8_t*>(std::malloc(capacity));
    AURORA_ASSERT(next->data != nullptr, "fifo::write_data: failed to allocate {} bytes", capacity);
    next->own = true;
    if (!sLoggedOwnChunk) {
      sLoggedOwnChunk = true;
      Log.warn("{} bytes written between publishes, more than a {} byte chunk: given a chunk of their own", needed,
               kChunkSize);
    }
  }
  std::memcpy(next->data, old.data + tailOffset, tailSize);
  use_chunk(*next, published);
  detail::sBufferSize = tailSize;
  old.next.store(next, std::memory_order_release);
  old.end.store(published, std::memory_order_release);
  // So the processor leaves the old chunk once it has processed it.
  if (kProcessingMode == ProcessingMode::Thread) {
    wake_worker();
  }
}

// Game thread, with everything written processed: the stream goes on from the first chunk's start, as the single buffer
// did after a drain, so that a frame's commands keep to the same memory.
void restart_stream(uint64_t position) {
  Chunk& first = sRing[0];
  if (sWriteChunk == &first) {
    // The processor reads the chunk again only after the next publish.
    first.base = position;
  } else {
    // The processor may not have left it yet.
    wake_worker();
    while (first.inUse.load(std::memory_order_acquire)) {
      first.inUse.wait(true, std::memory_order_acquire);
    }
    Chunk& old = *sWriteChunk;
    use_chunk(first, position);
    sRingIndex = 0;
    old.next.store(&first, std::memory_order_release);
    old.end.store(position, std::memory_order_release);
    wake_worker();
  }
  detail::sBufferSize = 0;
  sPendingDraws = 0;
}

// Game thread: Aurora's own commands, which go into the stream even while a display list is being recorded.
void append(const uint8_t* bytes, uint32_t length) {
  if (detail::sBufferCapacity - detail::sBufferSize < length) {
    next_chunk(length);
  }
  std::memcpy(detail::sBufferData + detail::sBufferSize, bytes, length);
  detail::sBufferSize += length;
}

void publish_to(uint64_t target) noexcept {
  if (target > sPublished.load(std::memory_order_relaxed)) {
    sPendingDraws = 0;
    // Sequentially consistent, for the handshake with the processor's wait (worker_main).
    sPublished.store(target, std::memory_order_seq_cst);
    if (kProcessingMode == ProcessingMode::Thread) {
      if (sWorkerSleeping.load(std::memory_order_seq_cst)) {
        wake_worker();
      }
    } else {
      process_to(target);
    }
  }
}

template <size_t N>
void put_u32(std::array<uint8_t, N>& out, size_t at, uint32_t value) noexcept {
  const uint32_t big = bswap(value);
  std::memcpy(out.data() + at, &big, sizeof(big));
}

std::array<uint8_t, 3> aurora_command(uint16_t subCommand) noexcept {
  return {GX_AURORA, static_cast<uint8_t>(subCommand >> 8), static_cast<uint8_t>(subCommand & 0xFF)};
}

// VIConfigure during a frame: the VI mode's size applies from here in the stream.
void on_vi_configure(Vec2<uint32_t> fbSize) noexcept {
  std::array<uint8_t, 11> command{};
  const auto head = aurora_command(GX_AURORA_SET_VI_SIZE);
  std::memcpy(command.data(), head.data(), head.size());
  put_u32(command, 3, fbSize.x);
  put_u32(command, 7, fbSize.y);
  append(command.data(), command.size());
}
} // namespace

ProcessingMode processing_mode() noexcept { return kProcessingMode; }

void init() {
  stop_worker();
  reset_stream();
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  sFrameActive = false;
  sWorkerWake.store(0, std::memory_order_relaxed);
  vi::set_configure_hook(on_vi_configure);
  start_worker();
}

void shutdown() { stop_worker(); }

void begin_frame(const FrameBegin& begin) noexcept {
  std::array<uint8_t, 13> command{};
  const auto head = aurora_command(GX_AURORA_BEGIN_FRAME);
  std::memcpy(command.data(), head.data(), head.size());
  command[3] = begin.slot;
  command[4] = static_cast<uint8_t>(begin.viewportPolicy);
  put_u32(command, 5, begin.viWidth);
  put_u32(command, 9, begin.viHeight);
  // Ahead of what was written since the last frame ended, which isn't published yet.
  if (detail::sBufferCapacity - detail::sBufferSize < command.size()) {
    next_chunk(command.size());
  }
  const auto at = static_cast<uint32_t>(sPublished.load(std::memory_order_relaxed) - sWriteChunk->base);
  std::memmove(detail::sBufferData + at + command.size(), detail::sBufferData + at, detail::sBufferSize - at);
  std::memcpy(detail::sBufferData + at, command.data(), command.size());
  detail::sBufferSize += command.size();
  sFrameActive = true;
  if (kProcessingMode != ProcessingMode::Drain) {
    publish_to(sPublished.load(std::memory_order_relaxed) + command.size());
  }
}

uint64_t end_frame(const FrameEnd& end) noexcept {
  std::array<uint8_t, 5> command{};
  const auto head = aurora_command(GX_AURORA_END_FRAME);
  std::memcpy(command.data(), head.data(), head.size());
  command[3] = end.slot;
  command[4] = end.close ? 1 : 0;
  append(command.data(), command.size());
  const uint64_t position = write_position();
  if (kProcessingMode != ProcessingMode::Drain) {
    publish_to(position);
  }
  sFrameActive = false;
  return position;
}

void wait_processed(uint64_t position) noexcept {
  if (kProcessingMode == ProcessingMode::Thread) {
    await_position(position);
  } else if (sProcessed.load(std::memory_order_relaxed) < position) {
    process_to(position);
  }
}

void finish_published() noexcept { wait_processed(sPublished.load(std::memory_order_relaxed)); }

void write_data_slow(const void* data, uint32_t length) {
  next_chunk(length);
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize += length;
}

namespace {
// Where a command that doesn't fit the display list being recorded is written, to be dropped at its commit.
uint8_t sDroppedCommand[MaxReservedCommand];
bool sDroppingCommand = false;
} // namespace

uint8_t* reserve_slow(uint32_t size) {
  if (!detail::sInDisplayList) {
    next_chunk(size);
    return detail::sBufferData + detail::sBufferSize;
  }
  if (size <= detail::sDlSize - detail::sDlWritePos) {
    return detail::sDlBuffer + detail::sDlWritePos;
  }
  AURORA_ASSERT(size <= MaxReservedCommand, "fifo::reserve: a {} byte command in a display list", size);
  sDroppingCommand = true;
  return sDroppedCommand;
}

void commit_list(const uint8_t* end) noexcept {
  if (sDroppingCommand) {
    sDroppingCommand = false;
    return;
  }
  detail::sDlWritePos = static_cast<uint32_t>(end - detail::sDlBuffer);
}

uint64_t write_position() noexcept {
  return sWriteChunk != nullptr ? sWriteChunk->base + detail::sBufferSize : detail::sBufferSize;
}

void publish() noexcept {
  if (!sFrameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }
  publish_to(write_position());
}

void publish_now() noexcept {
  if (!detail::sInDisplayList) {
    publish_to(write_position());
  }
}

DrawDoneCallback set_draw_done_callback(DrawDoneCallback callback) noexcept {
  return sDrawDoneCallback.exchange(callback, std::memory_order_acq_rel);
}

void set_draw_done() noexcept {
  if (!detail::sInDisplayList) {
    sDrawDonePosition = write_position();
  }
}

void wait_draw_done() noexcept {
  const uint64_t position = sDrawDonePosition;
  // Given to the processor even outside a frame, as a drain does.
  publish_to(position);
  wait_processed(position);
}

DrawSyncCallback set_draw_sync_callback(DrawSyncCallback callback) noexcept {
  return sDrawSyncCallback.exchange(callback, std::memory_order_acq_rel);
}

uint16_t draw_sync_token() noexcept { return sDrawSyncToken.load(std::memory_order_acquire); }

void pass_draw_sync(uint16_t token, bool interrupt) noexcept {
  sDrawSyncToken.store(token, std::memory_order_release);
  if (const auto callback = sDrawSyncCallback.load(std::memory_order_acquire); interrupt && callback != nullptr) {
    callback(token);
  }
}

void finish_draw() noexcept {
  if (!sFrameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }
  if (++sPendingDraws >= kDrawBatchSize) {
    publish();
  }
}

void patch_u32(uint64_t position, uint32_t val) {
  AURORA_ASSERT(!detail::sInDisplayList && sWriteChunk != nullptr && position >= sWriteChunk->base &&
                    position + sizeof(uint32_t) <= write_position(),
                "fifo::patch_u32: invalid patch position {} (written to {})", position, write_position());
  AURORA_ASSERT(position >= sPublished.load(std::memory_order_relaxed),
                "fifo::patch_u32: position {} is below the published watermark", position);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + (position - sWriteChunk->base), &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  const uint32_t bytesWritten = detail::sDlWritePos;
  const uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain() {
  if (sWriteChunk == nullptr) {
    return;
  }
  const uint64_t target = write_position();
  if (target != sProcessed.load(std::memory_order_acquire)) {
    ZoneScoped;
    switch (kProcessingMode) {
    case ProcessingMode::Drain:
    case ProcessingMode::Inline:
      sPublished.store(target, std::memory_order_relaxed);
      process_to(target);
      break;
    case ProcessingMode::Thread:
      sPublished.store(target, std::memory_order_release);
      wake_worker();
      await_position(target);
      break;
    }
  }
  restart_stream(target);
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }
uint64_t waiter_wakes() noexcept { return sWaiterWakes; }

void clear_buffer() {
  if (sWriteChunk == nullptr) {
    return;
  }
  const uint64_t processed = sProcessed.load(std::memory_order_acquire);
  AURORA_ASSERT(sPublished.load(std::memory_order_acquire) == processed,
                "fifo::clear_buffer: published commands are still pending");
  sWriteChunk->base = processed;
  detail::sBufferSize = 0;
  sPendingDraws = 0;
}

} // namespace aurora::gx::fifo

#ifdef AURORA_DIAGNOSTICS
// The FIFO worker's processing time since the last call.
extern "C" unsigned long long aurora_fifo_take_busy_ns() {
  return aurora::gx::fifo::sWorkerBusyNs.exchange(0, std::memory_order_relaxed);
}
#endif
