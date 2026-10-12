#pragma once

#include "../internal.hpp"

#include <dolphin/gx/GXAurora.h>

#include <cstring>

namespace aurora::gx::fifo {

namespace detail {
extern uint8_t* sBufferData;
extern uint32_t sBufferSize;
extern uint32_t sBufferCapacity;
extern bool sInDisplayList;
extern uint8_t* sDlBuffer;
extern uint32_t sDlSize;
extern uint32_t sDlWritePos;
} // namespace detail

enum class ProcessingMode : uint8_t {
  // Process FIFO synchronously at drain()
  Drain,
  // Process FIFO synchronously at publish() (useful for profiling)
  Inline,
  // Process FIFO asynchronously on a worker thread; drain() synchronizes
  Thread
};
ProcessingMode processing_mode() noexcept;

void init();
void shutdown();

// What the FIFO processor needs to begin recording a frame: its slot, and the settings the game thread changed between
// frames, which the processor applies before the frame's first command.
struct FrameBegin {
  uint8_t slot = 0;
  AuroraViewportPolicy viewportPolicy = AURORA_VIEWPORT_FIT;
  uint32_t viWidth = 640;
  uint32_t viHeight = 480;
};
// Game thread: the frame begins in the stream ahead of anything written since the last one ended, which belongs to it,
// and the FIFO processor is given commands from here on.
void begin_frame(const FrameBegin& begin = {}) noexcept;
// How the FIFO processor ends a frame. With close false it stops once the frame's last pass is sealed, and whoever
// drained the stream records what follows and closes the frame (gfx::detail::close_frame).
struct FrameEnd {
  uint8_t slot = 0;
  bool close = true;
};
// Game thread: the frame ends in the stream, which the FIFO processor is given up to here; commands written after it
// wait for the next frame. Returns the stream position after the frame's end.
uint64_t end_frame(const FrameEnd& end = {}) noexcept;

// Out-of-line slow path: carries what isn't published yet over to the next chunk of the stream, then appends data.
void write_data_slow(const void* data, uint32_t length);

inline void write_data(const void* data, const uint32_t length) {
  if (!detail::sInDisplayList)
    LIKELY {
      if (length <= detail::sBufferCapacity - detail::sBufferSize)
        LIKELY {
          std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
          detail::sBufferSize += length;
          return;
        }
      write_data_slow(data, length);
    }
  else if (length <= detail::sDlSize - detail::sDlWritePos) {
    std::memcpy(detail::sDlBuffer + detail::sDlWritePos, data, length);
    detail::sDlWritePos += length;
  }
}

inline void write_u8(const uint8_t val) {
  if (!detail::sInDisplayList)
    LIKELY {
      if (detail::sBufferSize < detail::sBufferCapacity)
        LIKELY {
          detail::sBufferData[detail::sBufferSize++] = val;
          return;
        }
      write_data_slow(&val, 1);
    }
  else if (detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = val;
  }
}

inline void write_u16(const uint16_t val) {
  const auto out = to_big_endian(val);
  write_data(&out, sizeof(out));
}

inline void write_u32(const uint32_t val) {
  const auto out = to_big_endian(val);
  write_data(&out, sizeof(out));
}

inline void write_u64(const uint64_t val) {
  const auto out = to_big_endian(val);
  write_data(&out, sizeof(out));
}

inline void write_f32(const float val) {
  const auto out = to_big_endian(val);
  write_data(&out, sizeof(out));
}

// A command's bytes, written through a local pointer with one room check for the whole command. Each write_* above
// checks the room and reloads the stream's globals for every value, which on an in-order CPU costs several times the
// store itself. Nothing else may write to the stream between reserve() and commit().
constexpr uint32_t MaxReservedCommand = 128;
// Out-of-line slow paths: a new chunk, or room in the display list being recorded. A command that doesn't fit the list
// is dropped whole.
uint8_t* reserve_slow(uint32_t size);
void commit_list(const uint8_t* end) noexcept;

inline uint8_t* reserve(const uint32_t size) {
  if (!detail::sInDisplayList && size <= detail::sBufferCapacity - detail::sBufferSize)
    LIKELY { return detail::sBufferData + detail::sBufferSize; }
  return reserve_slow(size);
}

inline void commit(const uint8_t* end) noexcept {
  if (!detail::sInDisplayList)
    LIKELY {
      detail::sBufferSize = static_cast<uint32_t>(end - detail::sBufferData);
      return;
    }
  commit_list(end);
}

inline uint8_t* put_u8(uint8_t* p, const uint8_t val) noexcept {
  *p = val;
  return p + 1;
}

inline uint8_t* put_u16(uint8_t* p, const uint16_t val) noexcept {
  const auto out = to_big_endian(val);
  std::memcpy(p, &out, sizeof(out));
  return p + sizeof(out);
}

inline uint8_t* put_u32(uint8_t* p, const uint32_t val) noexcept {
  const auto out = to_big_endian(val);
  std::memcpy(p, &out, sizeof(out));
  return p + sizeof(out);
}

inline uint8_t* put_u64(uint8_t* p, const uint64_t val) noexcept {
  const auto out = to_big_endian(val);
  std::memcpy(p, &out, sizeof(out));
  return p + sizeof(out);
}

inline uint8_t* put_f32(uint8_t* p, const float val) noexcept {
  const auto out = to_big_endian(val);
  std::memcpy(p, &out, sizeof(out));
  return p + sizeof(out);
}

inline uint8_t* put(uint8_t* p, const uint8_t val) noexcept { return put_u8(p, val); }
inline uint8_t* put(uint8_t* p, const uint16_t val) noexcept { return put_u16(p, val); }
inline uint8_t* put(uint8_t* p, const uint32_t val) noexcept { return put_u32(p, val); }
inline uint8_t* put(uint8_t* p, const uint64_t val) noexcept { return put_u64(p, val); }
inline uint8_t* put(uint8_t* p, const float val) noexcept { return put_f32(p, val); }

// One command of fixed-size values, each big-endian, sized by their types.
template <typename... T>
inline void write_command(const T... values) {
  uint8_t* p = reserve((sizeof(T) + ...));
  ((p = put(p, values)), ...);
  commit(p);
}

// The stream position the next byte is written at. Positions only grow, and a byte keeps its position when the slow
// path carries it over to another chunk.
uint64_t write_position() noexcept;

// Overwrites an unpublished u32 previously written at the given stream position.
void patch_u32(uint64_t position, uint32_t val);

// Marks a complete draw and publishes when the configured draw batch is full.
void finish_draw() noexcept;

// Makes commands written so far available to the FIFO processor.
void publish() noexcept;
// The same, outside a frame too, for a wait on what's written: commands written between frames are then processed
// before the next frame begins.
void publish_now() noexcept;

using DrawDoneCallback = void (*)();
DrawDoneCallback set_draw_done_callback(DrawDoneCallback callback) noexcept;
// Game thread: the draw-done token just written is the one wait_draw_done waits for the processor to pass.
void set_draw_done() noexcept;
void wait_draw_done() noexcept;

// Draw sync tokens (GXSetDrawSync): the one the processor passed last, and the callback it calls at each.
using DrawSyncCallback = void (*)(uint16_t token);
DrawSyncCallback set_draw_sync_callback(DrawSyncCallback callback) noexcept;
uint16_t draw_sync_token() noexcept;
// Processor: it has passed a draw sync token, and calls the callback for PE_TOKEN_INT.
void pass_draw_sync(uint16_t token, bool interrupt) noexcept;

// Display list recording
void begin_display_list(uint8_t* buf, uint32_t size);
uint32_t end_display_list();
bool in_display_list();

// Ensure all buffered commands have been processed.
void drain();
// Game thread: waits until the processor has processed the stream up to `position`, which is published. Only the game
// thread waits, and the processor wakes it only once it gets there.
void wait_processed(uint64_t position) noexcept;
// Game thread: waits until the processor has processed everything published, without publishing more: for changes to
// what the processor reads, such as the EFB's size, between frames.
void finish_published() noexcept;

// The chunk being written, from its start, for tests that read what GX calls wrote.
const uint8_t* get_buffer_data();
uint32_t get_buffer_size();
// How many times the processor has woken the game thread from a wait, for tests.
uint64_t waiter_wakes() noexcept;
// Drops what isn't published, once everything published has been processed.
void clear_buffer();

} // namespace aurora::gx::fifo
