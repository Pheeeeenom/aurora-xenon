#include "gx.hpp"
#include "__gx.h"

#include "../../gx/fifo.hpp"
#include "dolphin/gx/GXAurora.h"

#include <cstring>

static __GXData_struct sSavedGXData;

extern "C" {
void GXBeginDisplayList(void* list, u32 size) {
  CHECK(!aurora::gx::fifo::in_display_list(), "Display list began twice!");

  // Flush any pending dirty state before recording
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Save current shadow register state if requested
  if (__gx->dlSaveContext != 0) {
    std::memcpy(&sSavedGXData, __gx, sizeof(sSavedGXData));
  }

  __gx->inDispList = 1;

  // Redirect FIFO writes to the user-provided buffer
  aurora::gx::fifo::begin_display_list(static_cast<u8*>(list), size);
}

u32 GXEndDisplayList() {
  // Flush any pending dirty state into the display list
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // End FIFO redirection and get the byte count (ROUNDUP32)
  u32 bytesWritten = aurora::gx::fifo::end_display_list();

  // Restore saved shadow register state
  if (__gx->dlSaveContext != 0) {
    std::memcpy(__gx, &sSavedGXData, sizeof(*__gx));
  }

  __gx->inDispList = 0;

  return bytesWritten;
}

void GXCallDisplayList(const void* data, u32 nbytes) {
  // Flush any pending dirty state before calling
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Flush pending primitives
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }

  // A call inside a list being recorded can't name another list (GX has no nested calls), so its
  // bytes go into the recording.
  if (aurora::gx::fifo::in_display_list()) {
    aurora::gx::fifo::write_data(data, nbytes);
    return;
  }

  // The list's address, not its bytes: the FIFO processor reads them when it reaches the call, as
  // the SDK's GX_CMD_CALL_DL has the GPU do, which saves copying them here.
  aurora::gx::fifo::write_command(static_cast<uint8_t>(GX_AURORA), static_cast<uint16_t>(GX_AURORA_CALL_DISPLAY_LIST),
                                  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data)),
                                  static_cast<uint32_t>(nbytes));
  aurora::gx::fifo::publish();
}

}
