#include "gx.hpp"
#include "__gx.h"
#include "dolphin/mtx/GeoTypes.h"

namespace fifo = aurora::gx::fifo;

static inline void CacheProjectionVector(const f32* ptr, GXProjectionType type) {
  __gx->projType = type;
  for (int i = 0; i < 6; ++i) {
    __gx->projMtx[i] = ptr[i + 1];
  }
}

// XF bulk write: 6 params + projection type at 0x1020-0x1026
static inline void write_projection() {
  uint8_t* p = fifo::reserve(1 + 4 + 6 * 4 + 4);
  p = fifo::put_u8(p, 0x10);
  p = fifo::put_u32(p, 0x00061020);
  for (int i = 0; i < 6; ++i) {
    p = fifo::put_f32(p, __gx->projMtx[i]);
  }
  p = fifo::put_u32(p, __gx->projType);
  fifo::commit(p);
  __gx->bpSent = 0;
}

extern "C" {

void GXSetProjection(const void* mtx_, GXProjectionType type) {
  const auto& mtx = *reinterpret_cast<const aurora::Mat4x4<float>*>(mtx_);
  const f32 projVec[] = {
      static_cast<f32>(type == GX_ORTHOGRAPHIC),
      mtx[0][0],
      type == GX_ORTHOGRAPHIC ? mtx[0][3] : mtx[0][2],
      mtx[1][1],
      type == GX_ORTHOGRAPHIC ? mtx[1][3] : mtx[1][2],
      mtx[2][2],
      mtx[2][3],
  };
  CacheProjectionVector(projVec, type);
  write_projection();
}

void GXSetProjectionv(const f32* ptr) {
  CHECK(ptr != nullptr, "null projection vector");

  const GXProjectionType type = ptr[0] == 0.0f ? GX_PERSPECTIVE : GX_ORTHOGRAPHIC;
  CacheProjectionVector(ptr, type);
  write_projection();
}

void GXLoadPosMtxImm(const void* mtx_, u32 id) {
  // Any row of matrix memory up to the last slot, not only the ten named ones; see MaxPnMtx.
  CHECK(id <= (aurora::gx::MaxPnMtx - 1) * 3, "invalid pn mtx {}", static_cast<int>(id));
  const auto* mtx = reinterpret_cast<const f32*>(mtx_);

  uint8_t* p = fifo::reserve(1 + 4 + 12 * 4);
  p = fifo::put_u8(p, 0x10);
  p = fifo::put_u32(p, (id * 4) | 0xB0000);
  for (int i = 0; i < 12; i++) {
    p = fifo::put_f32(p, mtx[i]);
  }
  fifo::commit(p);
}

void GXLoadPosMtxIndx(u16 mtxIndx, u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", static_cast<int>(id));

  fifo::write_command(static_cast<uint8_t>(GX_LOAD_INDX_A), static_cast<uint16_t>(mtxIndx),
                      static_cast<uint16_t>((11u << 12) | id * 4));
}

void GXLoadNrmMtxImm(const void* mtx_, u32 id) {
  // As GXLoadPosMtxImm.
  CHECK(id <= (aurora::gx::MaxPnMtx - 1) * 3, "invalid pn mtx {}", static_cast<int>(id));
  const auto* mtx = reinterpret_cast<const f32*>(mtx_);

  uint8_t* p = fifo::reserve(1 + 4 + 9 * 4);
  p = fifo::put_u8(p, 0x10);
  p = fifo::put_u32(p, (id * 3 + 0x400) | 0x80000);
  // Write 3x3 from 3x4 matrix (skip translation column)
  for (int row = 0; row < 3; row++) {
    p = fifo::put_f32(p, mtx[row * 4]);
    p = fifo::put_f32(p, mtx[row * 4 + 1]);
    p = fifo::put_f32(p, mtx[row * 4 + 2]);
  }
  fifo::commit(p);
}

void GXSetCurrentMtx(u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", id);
  SET_REG_FIELD(0, __gx->matIdxA, 6, 0, id);
  __GXSetMatrixIndex(GX_VA_PNMTXIDX);
}

void GXLoadTexMtxImm(const void* mtx_, u32 id, GXTexMtxType type) {
  CHECK((id >= GX_TEXMTX0 && id <= GX_IDENTITY) || (id >= GX_PTTEXMTX0 && id <= GX_PTIDENTITY), "invalid tex mtx {}",
        id);

  u32 addr;
  if (id >= GX_PTTEXMTX0) {
    addr = (id - GX_PTTEXMTX0) * 4 + 0x500;
    CHECK(type == GX_MTX3x4, "invalid pt mtx type {}", underlying(type));
  } else {
    addr = id * 4;
  }

  u32 count = (type == GX_MTX2x4) ? 8 : 12;
  u32 reg = addr | ((count - 1) << 16);

  const auto* mtx = reinterpret_cast<const f32*>(mtx_);
  uint8_t* p = fifo::reserve(1 + 4 + count * 4);
  p = fifo::put_u8(p, 0x10);
  p = fifo::put_u32(p, reg);
  for (u32 i = 0; i < count; i++) {
    p = fifo::put_f32(p, mtx[i]);
  }
  fifo::commit(p);
}

void GXSetViewport(float left, float top, float width, float height, float nearZ, float farZ) {
  GXSetViewportJitter(left, top, width, height, nearZ, farZ, 1);
}

void GXSetViewportJitter(float left, float top, float width, float height, float nearZ, float farZ, u32 field) {
  float sx;
  float sy;
  float sz;
  float ox;
  float oy;
  float oz;
  float zmin;
  float zmax;

  if (field == 0) {
    top -= 0.5f;
  }

  sx = width / 2.0f;
  sy = -height / 2.0f;
  ox = 340.0f + (left + width / 2.0f);
  oy = 340.0f + (top + height / 2.0f);
  zmin = 1.6777215e7f * nearZ;
  zmax = 1.6777215e7f * farZ;
  sz = zmax - zmin;
  oz = zmax;

  __gx->vpLeft = left;
  __gx->vpTop = top;
  __gx->vpWd = width;
  __gx->vpHt = height;
  __gx->vpNearz = nearZ;
  __gx->vpFarz = farZ;

  // XF bulk write: the six viewport registers at 0x101A-0x101F
  fifo::write_command(static_cast<uint8_t>(0x10), static_cast<uint32_t>(0x0005101A), sx, sy, sz, ox, oy, oz);
  __gx->bpSent = 0;
}

void GXProject(f32 x, f32 y, f32 z, const f32 mtx[3][4], const f32* pm, const f32* vp, f32* sx,
               f32* sy, f32* sz) {
  Vec peye;
  f32 xc;
  f32 yc;
  f32 zc;
  f32 wc;

  peye.x = mtx[0][3] + ((mtx[0][2] * z) + ((mtx[0][0] * x) + (mtx[0][1] * y)));
  peye.y = mtx[1][3] + ((mtx[1][2] * z) + ((mtx[1][0] * x) + (mtx[1][1] * y)));
  peye.z = mtx[2][3] + ((mtx[2][2] * z) + ((mtx[2][0] * x) + (mtx[2][1] * y)));
  if (pm[0] == 0.0f) {
    xc = (peye.x * pm[1]) + (peye.z * pm[2]);
    yc = (peye.y * pm[3]) + (peye.z * pm[4]);
    zc = pm[6] + (peye.z * pm[5]);
    wc = 1.0f / -peye.z;
  } else {
    xc = pm[2] + (peye.x * pm[1]);
    yc = pm[4] + (peye.y * pm[3]);
    zc = pm[6] + (peye.z * pm[5]);
    wc = 1.0f;
  }
  *sx = (vp[2] / 2.0f) + (vp[0] + (wc * (xc * vp[2] / 2.0f)));
  *sy = (vp[3] / 2.0f) + (vp[1] + (wc * (-yc * vp[3] / 2.0f)));
  *sz = vp[5] + (wc * (zc * (vp[5] - vp[4])));
}

// TODO GXLoadNrmMtxImm3x3
void GXLoadNrmMtxIndx3x3(u16 mtxIndx, u32 id) {
  CHECK(id >= GX_PNMTX0 && id <= GX_PNMTX9, "invalid pn mtx {}", static_cast<int>(id));

  GX_WRITE_U8(GX_LOAD_INDX_B);
  GX_WRITE_U16(mtxIndx);
  GX_WRITE_U16((8u << 12) | (id * 3 + 0x400));
}

// TODO GXLoadTexMtxIndx
// TODO GXSetZScaleOffset
}
