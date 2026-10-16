// The command stream's bytes, whatever way the GX calls write them: a long run of GX calls with varied arguments
// gives the bytes the per-value writes gave before commands were written with one room check each (Aurora e3908bb).

#include "gx_test_common.hpp"
#include "gfx/texture.hpp"

#include <cstdint>
#include <vector>

namespace {
class Rng {
public:
  explicit Rng(uint32_t seed) : mState(seed) {}
  uint32_t next() {
    mState ^= mState << 13;
    mState ^= mState >> 17;
    mState ^= mState << 5;
    return mState;
  }
  uint32_t below(uint32_t n) { return next() % n; }
  float unit() { return static_cast<float>(next() >> 8) / 16777216.0f * 2.0f - 1.0f; }
  u8 byte() { return static_cast<u8>(next()); }

private:
  uint32_t mState;
};

uint64_t fnv1a(const uint8_t* data, size_t size) {
  uint64_t hash = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ data[i]) * 0x100000001b3ull;
  }
  return hash;
}

// Pointers the stream only carries, never reads: the same values in every build.
const void* fake_pointer(uint32_t value) { return reinterpret_cast<const void*>(static_cast<uintptr_t>(value)); }

GXColor color(Rng& rng) { return {rng.byte(), rng.byte(), rng.byte(), rng.byte()}; }

void matrices(Rng& rng) {
  f32 m[3][4];
  for (auto& row : m) {
    for (auto& v : row) {
      v = rng.unit() * 100.0f;
    }
  }
  const u32 slot = rng.below(13) * 3;
  GXLoadPosMtxImm(m, slot);
  GXLoadNrmMtxImm(m, slot);
  GXLoadTexMtxImm(m, GX_TEXMTX0 + rng.below(10) * 3, rng.below(2) != 0 ? GX_MTX2x4 : GX_MTX3x4);
  GXLoadTexMtxImm(m, GX_PTTEXMTX0 + rng.below(20) * 3, GX_MTX3x4);
  GXSetCurrentMtx(rng.below(10) * 3);
  f32 p[4][4];
  for (auto& row : p) {
    for (auto& v : row) {
      v = rng.unit();
    }
  }
  GXSetProjection(p, rng.below(2) != 0 ? GX_PERSPECTIVE : GX_ORTHOGRAPHIC);
  const f32 v[7] = {
      static_cast<f32>(rng.below(2)), rng.unit(), rng.unit(), rng.unit(), rng.unit(), rng.unit(), rng.unit()};
  GXSetProjectionv(v);
  GXSetViewport(rng.unit() * 10.0f, rng.unit() * 10.0f, 640.0f, 448.0f, 0.0f, 1.0f);
  GXSetViewportJitter(0.0f, 0.0f, 640.0f, 480.0f, rng.unit(), 1.0f, rng.below(2));
}

void lights(Rng& rng) {
  GXLightObj light;
  GXInitLightColor(&light, color(rng));
  GXInitLightPos(&light, rng.unit() * 50.0f, rng.unit() * 50.0f, rng.unit() * 50.0f);
  GXInitLightDir(&light, rng.unit(), rng.unit(), rng.unit());
  GXInitLightAttnA(&light, rng.unit(), rng.unit(), rng.unit());
  GXInitLightAttnK(&light, rng.unit(), rng.unit(), rng.unit());
  GXInitLightDistAttn(&light, 10.0f + rng.unit(), 0.5f, static_cast<GXDistAttnFn>(rng.below(4)));
  GXLoadLightObjImm(&light, static_cast<GXLightID>(1u << rng.below(8)));
  GXInitSpecularDir(&light, rng.unit(), rng.unit(), rng.unit());
  GXInitLightAttn(&light, rng.unit(), rng.unit(), rng.unit(), rng.unit(), rng.unit(), rng.unit());
  GXLoadLightObjImm(&light, static_cast<GXLightID>(1u << rng.below(8)));
  GXSetChanCtrl(GX_COLOR0A0, rng.below(2), static_cast<GXColorSrc>(rng.below(2)), static_cast<GXColorSrc>(rng.below(2)),
                rng.below(256), static_cast<GXDiffuseFn>(rng.below(3)), static_cast<GXAttnFn>(rng.below(3)));
  GXSetChanCtrl(GX_ALPHA1, rng.below(2), GX_SRC_REG, GX_SRC_VTX, rng.below(256), GX_DF_CLAMP, GX_AF_SPEC);
  GXSetChanAmbColor(GX_COLOR0A0, color(rng));
  GXSetChanMatColor(GX_COLOR1A1, color(rng));
  GXSetNumChans(rng.below(3));
}

void textures(Rng& rng) {
  GXTexObj obj;
  GXInitTexObj(&obj, fake_pointer(0x10000000 + rng.below(0x1000000) * 32), 8 << rng.below(6), 8 << rng.below(6),
               static_cast<GXTexFmt>(rng.below(7)), static_cast<GXTexWrapMode>(rng.below(3)),
               static_cast<GXTexWrapMode>(rng.below(3)), rng.below(2));
  GXInitTexObjFilter(&obj, static_cast<GXTexFilter>(rng.below(6)), static_cast<GXTexFilter>(rng.below(2)));
  GXInitTexObjWrapMode(&obj, static_cast<GXTexWrapMode>(rng.below(3)), static_cast<GXTexWrapMode>(rng.below(3)));
  GXInitTexObjMaxAniso(&obj, static_cast<GXAnisotropy>(rng.below(3)));
  // Object ids come from a counter that other tests in the process move on.
  reinterpret_cast<GXTexObj_*>(&obj)->texObjId = rng.next();
  GXLoadTexObj(&obj, static_cast<GXTexMapID>(rng.below(8)));
  GXTexObj ci;
  GXInitTexObjCI(&ci, fake_pointer(0x20000000 + rng.below(0x1000000) * 32), 64, 32, GX_TF_C8, GX_CLAMP, GX_REPEAT,
                 GX_FALSE, rng.below(16));
  GXInitTexObjTlut(&ci, rng.below(16));
  reinterpret_cast<GXTexObj_*>(&ci)->texObjId = rng.next();
  GXLoadTexObj(&ci, static_cast<GXTexMapID>(rng.below(8)));
  GXTlutObj tlut;
  GXInitTlutObj(&tlut, fake_pointer(0x30000000 + rng.below(0x100000) * 32), static_cast<GXTlutFmt>(rng.below(3)),
                16 << rng.below(5));
  reinterpret_cast<GXTlutObj_*>(&tlut)->tlutObjId = rng.next();
  GXLoadTlut(&tlut, rng.below(16));
}

void tev(Rng& rng) {
  const auto stage = static_cast<GXTevStageID>(rng.below(16));
  GXSetNumTevStages(1 + rng.below(16));
  GXSetTevOp(stage, static_cast<GXTevMode>(rng.below(5)));
  GXSetTevColorIn(stage, static_cast<GXTevColorArg>(rng.below(16)), static_cast<GXTevColorArg>(rng.below(16)),
                  static_cast<GXTevColorArg>(rng.below(16)), static_cast<GXTevColorArg>(rng.below(16)));
  GXSetTevAlphaIn(stage, static_cast<GXTevAlphaArg>(rng.below(8)), static_cast<GXTevAlphaArg>(rng.below(8)),
                  static_cast<GXTevAlphaArg>(rng.below(8)), static_cast<GXTevAlphaArg>(rng.below(8)));
  GXSetTevColorOp(stage, static_cast<GXTevOp>(rng.below(2)), static_cast<GXTevBias>(rng.below(3)),
                  static_cast<GXTevScale>(rng.below(4)), rng.below(2), static_cast<GXTevRegID>(rng.below(4)));
  GXSetTevAlphaOp(stage, static_cast<GXTevOp>(rng.below(2)), static_cast<GXTevBias>(rng.below(3)),
                  static_cast<GXTevScale>(rng.below(4)), rng.below(2), static_cast<GXTevRegID>(rng.below(4)));
  GXSetTevOrder(stage, static_cast<GXTexCoordID>(rng.below(8)), static_cast<GXTexMapID>(rng.below(8)),
                static_cast<GXChannelID>(4 + rng.below(3)));
  GXSetTevKColor(static_cast<GXTevKColorID>(rng.below(4)), color(rng));
  GXSetTevColor(static_cast<GXTevRegID>(1 + rng.below(3)), color(rng));
  GXSetTevColorS10(static_cast<GXTevRegID>(rng.below(4)),
                   {static_cast<s16>(rng.below(2048)), static_cast<s16>(rng.below(2048)),
                    static_cast<s16>(-static_cast<int>(rng.below(1024))), static_cast<s16>(rng.below(1024))});
  GXSetTevKColorSel(stage, static_cast<GXTevKColorSel>(rng.below(32)));
  GXSetTevKAlphaSel(stage, static_cast<GXTevKAlphaSel>(rng.below(32)));
  GXSetTevSwapMode(stage, static_cast<GXTevSwapSel>(rng.below(4)), static_cast<GXTevSwapSel>(rng.below(4)));
  GXSetTevSwapModeTable(static_cast<GXTevSwapSel>(rng.below(4)), static_cast<GXTevColorChan>(rng.below(4)),
                        static_cast<GXTevColorChan>(rng.below(4)), static_cast<GXTevColorChan>(rng.below(4)),
                        static_cast<GXTevColorChan>(rng.below(4)));
  GXSetTevDirect(stage);
  GXSetNumIndStages(rng.below(4));
  GXSetIndTexOrder(static_cast<GXIndTexStageID>(rng.below(4)), static_cast<GXTexCoordID>(rng.below(8)),
                   static_cast<GXTexMapID>(rng.below(8)));
  GXSetIndTexCoordScale(static_cast<GXIndTexStageID>(rng.below(4)), static_cast<GXIndTexScale>(rng.below(9)),
                        static_cast<GXIndTexScale>(rng.below(9)));
  GXSetAlphaCompare(static_cast<GXCompare>(rng.below(8)), rng.byte(), static_cast<GXAlphaOp>(rng.below(4)),
                    static_cast<GXCompare>(rng.below(8)), rng.byte());
}

void pixels(Rng& rng) {
  GXSetBlendMode(static_cast<GXBlendMode>(rng.below(4)), static_cast<GXBlendFactor>(rng.below(8)),
                 static_cast<GXBlendFactor>(rng.below(8)), static_cast<GXLogicOp>(rng.below(16)));
  GXSetZMode(rng.below(2), static_cast<GXCompare>(rng.below(8)), rng.below(2));
  GXSetColorUpdate(rng.below(2));
  GXSetAlphaUpdate(rng.below(2));
  GXSetZCompLoc(rng.below(2));
  GXSetDstAlpha(rng.below(2), rng.byte());
  GXSetDither(rng.below(2));
  GXSetFog(static_cast<GXFogType>(rng.below(2) != 0 ? GX_FOG_PERSP_LIN : GX_FOG_NONE), 10.0f + rng.unit(),
           1000.0f + rng.unit(), 0.25f, 2000.0f, color(rng));
  GXFogAdjTable table;
  for (auto& r : table.r) {
    r = static_cast<u16>(rng.below(4096));
  }
  GXSetFogRangeAdj(rng.below(2), static_cast<u16>(320 + rng.below(64)), &table);
  GXSetScissor(rng.below(64), rng.below(64), 64 + rng.below(512), 64 + rng.below(384));
  GXSetCullMode(static_cast<GXCullMode>(rng.below(4)));
  GXSetCoPlanar(rng.below(2));
  GXPixModeSync();
  GXTexModeSync();
  GXSetLineWidth(rng.byte(), static_cast<GXTexOffset>(rng.below(6)));
  GXSetPointSize(rng.byte(), static_cast<GXTexOffset>(rng.below(6)));
  GXEnableTexOffsets(static_cast<GXTexCoordID>(rng.below(8)), rng.below(2), rng.below(2));
  GXSetTexCopySrc(0, 0, 64 + rng.below(512), 64 + rng.below(384));
  GXSetTexCopyDst(64, 64, GX_TF_RGBA8, rng.below(2));
}

void geometry(Rng& rng, bool recording) {
  GXClearVtxDesc();
  for (int attr = GX_VA_PNMTXIDX; attr <= GX_VA_TEX7; ++attr) {
    if (rng.below(3) == 0) {
      GXSetVtxDesc(static_cast<GXAttr>(attr), static_cast<GXAttrType>(rng.below(4)));
    }
  }
  const auto fmt = static_cast<GXVtxFmt>(rng.below(8));
  GXSetVtxAttrFmt(fmt, GX_VA_POS, GX_POS_XYZ, static_cast<GXCompType>(rng.below(5)), rng.below(16));
  GXSetVtxAttrFmt(fmt, GX_VA_NRM, GX_NRM_XYZ, GX_S8, 6);
  GXSetVtxAttrFmt(fmt, GX_VA_CLR0, GX_CLR_RGBA, GX_RGBA8, 0);
  GXSetVtxAttrFmt(fmt, static_cast<GXAttr>(GX_VA_TEX0 + rng.below(8)), GX_TEX_ST, GX_F32, 0);
  for (int attr = GX_VA_POS; attr <= GX_VA_TEX7; ++attr) {
    if (rng.below(2) == 0) {
      GXSetArray(static_cast<GXAttr>(attr), fake_pointer(0x40000000 + rng.below(0x100000) * 4), rng.below(1 << 20),
                 static_cast<u8>(rng.below(64)), rng.below(2));
    }
  }
  GXSetArray(GX_VA_NBT, fake_pointer(0x50000000), 4096, 36, false);
  GXSetNumTexGens(rng.below(9));
  GXSetTexCoordGen2(static_cast<GXTexCoordID>(rng.below(8)), static_cast<GXTexGenType>(rng.below(11)),
                    static_cast<GXTexGenSrc>(rng.below(21)), GX_TEXMTX0 + rng.below(10) * 3, rng.below(2),
                    GX_PTTEXMTX0 + rng.below(20) * 3);
  GXSetTexCoordGen2(static_cast<GXTexCoordID>(rng.below(8)), GX_TG_MTX2x4, GX_TG_TEX0, GX_IDENTITY, GX_FALSE,
                    GX_PTIDENTITY);
  GXInvalidateVtxCache();
  GXBegin(static_cast<GXPrimitive>(GX_QUADS + 8 * rng.below(8)), fmt, 2);
  for (int i = 0; i < 2; ++i) {
    GXPosition3f32(rng.unit(), rng.unit(), rng.unit());
    GXPosition3u16(rng.next(), rng.next(), rng.next());
    GXPosition3s16(rng.next(), rng.next(), rng.next());
    GXPosition3u8(rng.byte(), rng.byte(), rng.byte());
    GXPosition3s8(rng.byte(), rng.byte(), rng.byte());
    GXPosition2f32(rng.unit(), rng.unit());
    GXPosition2u16(rng.next(), rng.next());
    GXPosition2s16(rng.next(), rng.next());
    GXPosition2u8(rng.byte(), rng.byte());
    GXPosition2s8(rng.byte(), rng.byte());
    GXPosition1x16(rng.next());
    GXPosition1x8(rng.byte());
    GXNormal3f32(rng.unit(), rng.unit(), rng.unit());
    GXNormal3s16(rng.next(), rng.next(), rng.next());
    GXNormal3s8(rng.byte(), rng.byte(), rng.byte());
    GXNormal1x16(rng.next());
    GXNormal1x8(rng.byte());
    GXColor4f32(rng.unit() * 0.5f + 0.5f, rng.unit() * 0.5f + 0.5f, 0.25f, 1.0f);
    GXColor4u8(rng.byte(), rng.byte(), rng.byte(), rng.byte());
    GXColor3u8(rng.byte(), rng.byte(), rng.byte());
    GXColor1u32(rng.next());
    GXColor1u16(rng.next());
    GXColor1x16(rng.next());
    GXColor1x8(rng.byte());
    GXTexCoord2f32(rng.unit(), rng.unit());
    GXTexCoord2u16(rng.next(), rng.next());
    GXTexCoord2s16(rng.next(), rng.next());
    GXTexCoord2u8(rng.byte(), rng.byte());
    GXTexCoord2s8(rng.byte(), rng.byte());
    GXTexCoord1f32(rng.unit());
    GXTexCoord1u16(rng.next());
    GXTexCoord1s16(rng.next());
    GXTexCoord1u8(rng.byte());
    GXTexCoord1s8(rng.byte());
    GXTexCoord1x16(rng.next());
    GXTexCoord1x8(rng.byte());
  }
  GXEnd();
  // While a list is recorded, a call copies the called list's bytes, so it needs real ones.
  static std::vector<u8> called(96, 0x5A);
  if (recording) {
    GXCallDisplayList(called.data(), static_cast<u32>(called.size()));
  } else {
    GXCallDisplayList(fake_pointer(0x60000000 + rng.below(0x100000) * 32), 32 * (1 + rng.below(64)));
  }
  GXLoadPosMtxIndx(static_cast<u16>(rng.next()), rng.below(10) * 3);
}

void write_everything(uint32_t seed, int rounds, bool recording) {
  Rng rng(seed);
  for (int i = 0; i < rounds; ++i) {
    matrices(rng);
    lights(rng);
    textures(rng);
    tev(rng);
    pixels(rng);
    geometry(rng, recording);
  }
  GXFlush();
}

struct Golden {
  uint32_t seed;
  size_t size;
  uint64_t hash;
};
} // namespace

// Before commands were written with one room check each (e3908bb): the stream and a recorded list. Recorded with
// encounter's afa388d applied there, which masks a value to its register field (SET_REG_FIELD).
constexpr Golden kStream[] = {
    {1, 33339, 0x5F1973F3944030FAull},
    {0x9E3779B9, 32817, 0xE3737D4B1464A3B6ull},
    {0x2545F491, 33681, 0x1C94F82CB30314E1ull},
};
constexpr Golden kList[] = {
    {1, 17248, 0x9A1BD2EE88F497D1ull},
    {0x9E3779B9, 17152, 0x076D6CE0100876A2ull},
};

TEST_F(GXFifoTest, GXCallsWriteTheStreamsBytesOfBefore) {
  for (const auto& golden : kStream) {
    aurora::gx::fifo::clear_buffer();
    write_everything(golden.seed, 20, false);
    const auto bytes = capture_fifo();
    const uint64_t hash = fnv1a(bytes.data(), bytes.size());
    std::printf("[golden] stream seed 0x%08X: %zu bytes, hash 0x%016llX\n", golden.seed, bytes.size(),
                static_cast<unsigned long long>(hash));
    EXPECT_EQ(bytes.size(), golden.size);
    EXPECT_EQ(hash, golden.hash);
  }
}

// A draw's indices through one reservation are the bytes of one GXPosition1x16 a value, past a chunk's end too (the
// unpublished stream is carried whole to the next chunk, so it stays in one buffer).
TEST_F(GXFifoTest, ReservedVertexDataWritesTheBytesOfOneCallAValue) {
  struct Case {
    u32 count;
    int draws;
  };
  for (const Case c : {Case{0, 1}, Case{1, 1}, Case{2, 1}, Case{127, 3}, Case{5400, 2}, Case{4000, 9}}) {
    Rng rng(0x51ED270Bu + c.count);
    std::vector<u16> indices(c.count);
    for (auto& index : indices) {
      index = static_cast<u16>(rng.next());
    }
    std::vector<u8> bytes[2];
    for (int way = 0; way < 2; ++way) {
      // No state left to flush, so each way's bytes are its draws'.
      GXClearVtxDesc();
      GXSetVtxDesc(GX_VA_POS, GX_INDEX16);
      __GXSetDirtyState();
      aurora::gx::fifo::clear_buffer();
      for (int d = 0; d < c.draws; ++d) {
        GXBegin(GX_TRIANGLES, GX_VTXFMT0, static_cast<u16>(c.count));
        if (way == 0) {
          for (const u16 index : indices) {
            GXPosition1x16(index);
          }
        } else {
          u8* out = GXReserveVertexData(c.count * 2);
          ASSERT_NE(out, nullptr);
          for (const u16 index : indices) {
            *out++ = static_cast<u8>(index >> 8);
            *out++ = static_cast<u8>(index);
          }
          GXCommitVertexData(out);
        }
        GXEnd();
      }
      bytes[way] = capture_fifo();
    }
    EXPECT_EQ(bytes[1], bytes[0]) << c.draws << " draws of " << c.count << " indices";
    EXPECT_EQ(bytes[0].size(), static_cast<size_t>(c.draws) * (3 + 2 * c.count));
  }
}

// While a list is recorded there is no reservation: the calls write the values, and drop any that don't fit.
TEST_F(GXFifoTest, NoVertexDataReservationInAList) {
  std::vector<u8> list(256);
  GXBeginDisplayList(list.data(), static_cast<u32>(list.size()));
  EXPECT_EQ(GXReserveVertexData(16), nullptr);
  GXEndDisplayList();
}

TEST_F(GXFifoTest, GXCallsWriteTheListsBytesOfBefore) {
  std::vector<u8> list(256 * 1024);
  for (const auto& golden : kList) {
    aurora::gx::fifo::clear_buffer();
    GXBeginDisplayList(list.data(), static_cast<u32>(list.size()));
    write_everything(golden.seed, 10, true);
    const u32 size = GXEndDisplayList();
    const uint64_t hash = fnv1a(list.data(), size);
    std::printf("[golden] list seed 0x%08X: %u bytes, hash 0x%016llX\n", golden.seed, size,
                static_cast<unsigned long long>(hash));
    EXPECT_EQ(size, golden.size);
    EXPECT_EQ(hash, golden.hash);
  }
}
