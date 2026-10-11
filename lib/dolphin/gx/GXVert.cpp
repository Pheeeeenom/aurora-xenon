#include "gx.hpp"
#include "__gx.h"
#include "../../gx/fifo.hpp"

namespace {
// Track vertex count between GXBegin/GXEnd for mismatch detection
u16 sBeginNVerts = 0;
u64 sBeginFifoSize = 0;
bool sInBegin = false;
// GX_AUTO: the stream position of the u32 byte-length placeholder to patch in GXEnd
u64 sBeginSizeOffset = 0;
bool sBeginAuto = false;

void pre_begin() {
  CHECK(!sInBegin, "GXBegin: called without matching GXEnd");

  // Flush dirty state before starting a draw
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Flush pending primitives if needed
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }
}

void post_begin(u16 nVerts) {
  sBeginNVerts = nVerts;
  sBeginFifoSize = aurora::gx::fifo::write_position();
  sInBegin = true;
}
} // namespace

extern "C" {

void GXBegin(GXPrimitive primitive, GXVtxFmt vtxFmt, u16 nVerts) {
  pre_begin();

  const u8 drawCmd = static_cast<u8>(vtxFmt) | static_cast<u8>(primitive);
  sBeginAuto = nVerts == GX_AUTO;
  if (sBeginAuto) {
    AURORA_ASSERT(!aurora::gx::fifo::in_display_list(), "GXBegin: GX_AUTO not supported in display lists");
    GX_WRITE_AURORA(GX_AURORA_DRAW_SIZED);
    GX_WRITE_U8(drawCmd);
    sBeginSizeOffset = aurora::gx::fifo::write_position();
    GX_WRITE_U32(0);
  } else {
    aurora::gx::fifo::write_command(static_cast<uint8_t>(drawCmd), static_cast<uint16_t>(nVerts));
  }

  post_begin(nVerts);
}

void GXBeginIndexed(GXVtxFmt vtxFmt, u16 nVerts, const u16* indices, u32 nIndices) {
  pre_begin();
  AURORA_ASSERT(nVerts != 0, "GXBeginIndexed: vertex count must be nonzero");
  AURORA_ASSERT(indices != nullptr, "GXBeginIndexed: indices must not be null");
  AURORA_ASSERT(nIndices != 0, "GXBeginIndexed: index count must be nonzero");
  AURORA_ASSERT(nIndices % 3 == 0, "GXBeginIndexed: index count must describe whole triangles");
  AURORA_ASSERT(nIndices <= UINT32_MAX / sizeof(u16), "GXBeginIndexed: index data size overflow");

  sBeginAuto = false;
  GX_WRITE_AURORA(GX_AURORA_DRAW_INDEXED);
  GX_WRITE_U8(static_cast<u8>(vtxFmt) | static_cast<u8>(GX_TRIANGLES));
  GX_WRITE_U16(nVerts);
  GX_WRITE_U32(nIndices);

  // Indices are host-endian even in a big-endian FIFO
  aurora::gx::fifo::write_data(indices, nIndices * sizeof(u16));

  post_begin(nVerts);
}

void GXEnd() {
  if (sInBegin) {
    const auto bytesWritten = static_cast<u32>(aurora::gx::fifo::write_position() - sBeginFifoSize);
    if (sBeginAuto) {
      aurora::gx::fifo::patch_u32(sBeginSizeOffset, bytesWritten);
      sBeginAuto = false;
    } else if (sBeginNVerts > 0 && bytesWritten > 0) {
      // We don't know the vertex size without processing the FIFO for vtxFmt changes
      // so this is just a best-effort check to find obvious issues
      if (bytesWritten % sBeginNVerts != 0) {
        Log.warn("GXEnd: vertex data not evenly divisible: {} bytes for {} vertices", bytesWritten, sBeginNVerts);
      }
    }
    sInBegin = false;
    aurora::gx::fifo::finish_draw();
  }
}

void GXPosition3f32(f32 x, f32 y, f32 z) {
  aurora::gx::fifo::write_command(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

void GXPosition3u16(u16 x, u16 y, u16 z) {
  aurora::gx::fifo::write_command(static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint16_t>(z));
}

void GXPosition3s16(s16 x, s16 y, s16 z) {
  aurora::gx::fifo::write_command(static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint16_t>(z));
}

void GXPosition3u8(u8 x, u8 y, u8 z) {
  aurora::gx::fifo::write_command(static_cast<uint8_t>(x), static_cast<uint8_t>(y), static_cast<uint8_t>(z));
}

void GXPosition3s8(s8 x, s8 y, s8 z) {
  aurora::gx::fifo::write_command(static_cast<uint8_t>(x), static_cast<uint8_t>(y), static_cast<uint8_t>(z));
}

void GXPosition2f32(f32 x, f32 y) { aurora::gx::fifo::write_command(static_cast<float>(x), static_cast<float>(y)); }

void GXPosition2u16(u16 x, u16 y) {
  aurora::gx::fifo::write_command(static_cast<uint16_t>(x), static_cast<uint16_t>(y));
}

void GXPosition2s16(s16 x, s16 y) {
  aurora::gx::fifo::write_command(static_cast<uint16_t>(x), static_cast<uint16_t>(y));
}

void GXPosition2u8(u8 x, u8 y) { aurora::gx::fifo::write_command(static_cast<uint8_t>(x), static_cast<uint8_t>(y)); }

void GXPosition2s8(s8 x, s8 y) { aurora::gx::fifo::write_command(static_cast<uint8_t>(x), static_cast<uint8_t>(y)); }

void GXPosition1x16(u16 idx) { aurora::gx::fifo::write_command(static_cast<uint16_t>(idx)); }

u8* GXReserveVertexData(u32 size) {
  // A list being recorded drops a value that doesn't fit, so the values go through their calls there.
  if (aurora::gx::fifo::in_display_list()) {
    return nullptr;
  }
  return aurora::gx::fifo::reserve(size);
}

void GXCommitVertexData(const u8* end) { aurora::gx::fifo::commit(end); }

void GXPosition1x8(u8 idx) { aurora::gx::fifo::write_command(static_cast<uint8_t>(idx)); }

void GXNormal3f32(f32 x, f32 y, f32 z) {
  aurora::gx::fifo::write_command(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

void GXNormal3s16(s16 x, s16 y, s16 z) {
  aurora::gx::fifo::write_command(static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint16_t>(z));
}

void GXNormal3s8(s8 x, s8 y, s8 z) {
  aurora::gx::fifo::write_command(static_cast<uint8_t>(x), static_cast<uint8_t>(y), static_cast<uint8_t>(z));
}

void GXNormal1x16(u16 index) { aurora::gx::fifo::write_command(static_cast<uint16_t>(index)); }

void GXNormal1x8(u8 index) { aurora::gx::fifo::write_command(static_cast<uint8_t>(index)); }

void GXColor4f32(f32 r, f32 g, f32 b, f32 a) {
  aurora::gx::fifo::write_command(static_cast<uint8_t>(r * 255.0), static_cast<uint8_t>(g * 255.0),
                                  static_cast<uint8_t>(b * 255.0), static_cast<uint8_t>(a * 255.0));
}

void GXColor4u8(u8 r, u8 g, u8 b, u8 a) {
  aurora::gx::fifo::write_command(static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b),
                                  static_cast<uint8_t>(a));
}

void GXColor3u8(u8 r, u8 g, u8 b) {
  aurora::gx::fifo::write_command(static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b));
}

void GXColor1u32(u32 clr) { aurora::gx::fifo::write_command(static_cast<uint32_t>(clr)); }

void GXColor1u16(u16 clr) { aurora::gx::fifo::write_command(static_cast<uint16_t>(clr)); }

void GXColor1x16(u16 index) { aurora::gx::fifo::write_command(static_cast<uint16_t>(index)); }

void GXColor1x8(u8 index) { aurora::gx::fifo::write_command(static_cast<uint8_t>(index)); }

void GXTexCoord2f32(f32 s, f32 t) { aurora::gx::fifo::write_command(static_cast<float>(s), static_cast<float>(t)); }

void GXTexCoord2u16(u16 s, u16 t) {
  aurora::gx::fifo::write_command(static_cast<uint16_t>(s), static_cast<uint16_t>(t));
}

void GXTexCoord2s16(s16 s, s16 t) {
  aurora::gx::fifo::write_command(static_cast<uint16_t>(s), static_cast<uint16_t>(t));
}

void GXTexCoord2u8(u8 s, u8 t) { aurora::gx::fifo::write_command(static_cast<uint8_t>(s), static_cast<uint8_t>(t)); }

void GXTexCoord2s8(s8 s, s8 t) { aurora::gx::fifo::write_command(static_cast<uint8_t>(s), static_cast<uint8_t>(t)); }

void GXTexCoord1f32(f32 s) { aurora::gx::fifo::write_command(static_cast<float>(s)); }

void GXTexCoord1u16(u16 s) { aurora::gx::fifo::write_command(static_cast<uint16_t>(s)); }

void GXTexCoord1s16(s16 s) { aurora::gx::fifo::write_command(static_cast<uint16_t>(s)); }

void GXTexCoord1u8(u8 s) { aurora::gx::fifo::write_command(static_cast<uint8_t>(s)); }

void GXTexCoord1s8(s8 s) { aurora::gx::fifo::write_command(static_cast<uint8_t>(s)); }

void GXTexCoord1x16(u16 index) { aurora::gx::fifo::write_command(static_cast<uint16_t>(index)); }

void GXTexCoord1x8(u8 index) { aurora::gx::fifo::write_command(static_cast<uint8_t>(index)); }

void GXCmd1u8(const u8 x) { aurora::gx::fifo::write_u8(x); }
void GXCmd1u16(const u16 x) { aurora::gx::fifo::write_u16(x); }
void GXCmd1u32(const u32 x) { aurora::gx::fifo::write_u32(x); }
void GXCmd1u64(const u64 x) { aurora::gx::fifo::write_u64(x); }

void GXParam1u8(const u8 x) { aurora::gx::fifo::write_u8(x); }
void GXParam1u16(const u16 x) { aurora::gx::fifo::write_u16(x); }
void GXParam1u32(const u32 x) { aurora::gx::fifo::write_u32(x); }
void GXParam1s8(const s8 x) { aurora::gx::fifo::write_u8(static_cast<uint8_t>(x)); }
void GXParam1s16(const s16 x) { aurora::gx::fifo::write_u16(static_cast<uint16_t>(x)); }
void GXParam1s32(const s32 x) { aurora::gx::fifo::write_u32(static_cast<uint32_t>(x)); }
void GXParam1f32(const f32 x) { aurora::gx::fifo::write_f32(x); }
void GXParam3f32(const f32 x, const f32 y, const f32 z) {
  aurora::gx::fifo::write_f32(x);
  aurora::gx::fifo::write_f32(y);
  aurora::gx::fifo::write_f32(z);
}
void GXParam4f32(const f32 x, const f32 y, const f32 z, const f32 w) {
  aurora::gx::fifo::write_f32(x);
  aurora::gx::fifo::write_f32(y);
  aurora::gx::fifo::write_f32(z);
  aurora::gx::fifo::write_f32(w);
}

} // extern "C"
