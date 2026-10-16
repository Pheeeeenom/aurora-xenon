#include <gtest/gtest.h>

#include <dolphin/gx.h>

namespace {

// NTSC interlaced with a double-strike field buffer: the vertical trim applies whole to every height.
TEST(GXAdjustForOverscanTest, TrimsAnInterlacedDoubleStrikeModeWhole) {
  GXRenderModeObj out{};
  GXAdjustForOverscan(&GXNtsc480Int, &out, 8, 16);
  EXPECT_EQ(out.fbWidth, 640 - 16);
  EXPECT_EQ(out.efbHeight, 480 - 32);
  EXPECT_EQ(out.xfbHeight, 480 - 32);
  EXPECT_EQ(out.viWidth, 640 - 16);
  EXPECT_EQ(out.viHeight, 480 - 32);
  EXPECT_EQ(out.viXOrigin, GXNtsc480Int.viXOrigin + 8);
  EXPECT_EQ(out.viYOrigin, GXNtsc480Int.viYOrigin + 16);
  EXPECT_EQ(out.xFBmode, GXNtsc480Int.xFBmode);
}

// An interlaced single-strike field buffer at 240 lines trims the XFB by half the amount.
TEST(GXAdjustForOverscanTest, HalvesTheXfbTrimForAnInterlacedSingleStrikeBuffer) {
  GXRenderModeObj in = GXNtsc480Int;
  in.xFBmode = VI_XFBMODE_SF;
  in.xfbHeight = 240;
  GXRenderModeObj out{};
  GXAdjustForOverscan(&in, &out, 0, 16);
  EXPECT_EQ(out.xfbHeight, 240 - 16);
  EXPECT_EQ(out.efbHeight, 480 - (32 * 480) / 240);
  EXPECT_EQ(out.viHeight, 480 - 32);
}

// Non-interlaced (NTSC 240 lines, single-strike): the XFB is trimmed whole and the VI height twice over.
TEST(GXAdjustForOverscanTest, TrimsANonInterlacedModeAsTheSdkDoes) {
  GXRenderModeObj in = GXNtsc480Int;
  in.viTVmode = VI_TVMODE_NTSC_DS;
  in.efbHeight = 240;
  in.xfbHeight = 240;
  in.xFBmode = VI_XFBMODE_SF;
  GXRenderModeObj out{};
  GXAdjustForOverscan(&in, &out, 0, 16);
  EXPECT_EQ(out.efbHeight, 240 - 32);
  EXPECT_EQ(out.xfbHeight, 240 - 32);
  EXPECT_EQ(out.viHeight, 480 - 64);
}

TEST(GXAdjustForOverscanTest, KeepsTheEfbHeightWithoutAnXfbHeight) {
  GXRenderModeObj in = GXNtsc480Int;
  in.xfbHeight = 0;
  GXRenderModeObj out{};
  GXAdjustForOverscan(&in, &out, 0, 16);
  EXPECT_EQ(out.efbHeight, 480);
}

} // namespace
