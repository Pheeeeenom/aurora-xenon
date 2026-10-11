#include <gtest/gtest.h>

#include "gfx/frame_packet.hpp"
#include "gfx/recording.hpp"
#include "gfx/texture.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <random>

namespace aurora::gfx {
namespace {

constexpr auto ColorFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr auto DepthFormat = wgpu::TextureFormat::Depth24Plus;

class GfxRecordingTest : public ::testing::Test {
protected:
  void SetUp() override {
    webgpu::g_graphicsConfig.surfaceConfiguration.format = ColorFormat;
    webgpu::g_graphicsConfig.depthFormat = DepthFormat;
    webgpu::g_graphicsConfig.msaaSamples = 1;
    webgpu::g_frameBuffer.size = {640, 480, 1};
    webgpu::g_frameBuffer.format = ColorFormat;
    webgpu::g_depthBuffer.size = {640, 480, 1};
    webgpu::g_depthBuffer.format = DepthFormat;
    detail::testing::suppress_render_worker(true);
    detail::begin_recording(frame, 0);
  }

  void TearDown() override {
    if (recordingActive) {
      if (is_offscreen()) {
        end_offscreen();
      }
      finish();
      detail::end_recording();
    }
    detail::shutdown_recording();
  }

  void seed(uint32_t width, uint32_t height) {
    detail::testing::seed_offscreen_cache(width, height, TextureFormat::RGBA8Unorm, TextureFormat::Depth32Float);
  }

  void copy_current_offscreen() {
    const auto& pass = frame.renderPasses.back();
    const auto& size = pass.colorAttachments[SceneColorAttachmentIndex].size;
    auto target = std::make_shared<TextureRef>(nullptr, TextureExtent{size.width, size.height},
                                               TextureFormat::RGBA8Unorm, 1, GX_TF_RGBA8);
    resolve_pass_into(std::move(target), {0, 0, static_cast<int32_t>(size.width), static_cast<int32_t>(size.height)},
                      false, false, false, {}, 1.f, GX_TF_RGBA8, GX_PF_RGBA6_Z24);
  }

  size_t count_efb_passes() const {
    return static_cast<size_t>(
        std::ranges::count_if(frame.renderPasses, [](const auto& pass) { return pass.label.starts_with("EFB"); }));
  }

  detail::FramePacket frame;
  bool recordingActive = true;
};

TEST_F(GfxRecordingTest, CreateRestoreReturnsToEfb) {
  seed(320, 180);
  begin_offscreen(320, 180);
  ASSERT_TRUE(is_offscreen());

  end_offscreen();

  EXPECT_FALSE(is_offscreen());
  ASSERT_EQ(frame.renderPasses.size(), 2u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[0].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, EfbPassUsesDiscoveredSceneLayout) {
  ASSERT_FALSE(frame.renderPasses.empty());
  const auto discovered = scene_render_target_layout();
  const auto targetLayout = frame.renderPasses.front().target_layout();

  EXPECT_EQ(targetLayout.key, discovered.key);
  EXPECT_EQ(targetLayout.colorAttachmentCount, discovered.colorAttachmentCount);
  EXPECT_EQ(targetLayout.colorAttachments[SceneColorAttachmentIndex].semantic, ColorAttachmentSemantic::SceneColor);
}

// The layout key a draw is keyed by is the current pass's, through offscreen passes and back. The offscreen targets'
// depth format differs from the EFB's, so a key left over from another pass would show.
TEST_F(GfxRecordingTest, LayoutKeyFollowsTheCurrentPass) {
  const auto key = [this] { return frame.renderPasses.back().target_layout().key; };
  const auto efbKey = get_render_target_layout_key();
  EXPECT_EQ(efbKey, key());
  EXPECT_EQ(has_normal_attachment(), false);
  seed(320, 180);
  begin_offscreen(320, 180);
  EXPECT_EQ(get_render_target_layout_key(), key());
  EXPECT_NE(get_render_target_layout_key(), efbKey);
  copy_current_offscreen();
  EXPECT_EQ(get_render_target_layout_key(), key());
  end_offscreen();
  EXPECT_EQ(get_render_target_layout_key(), key());
  EXPECT_EQ(get_render_target_layout_key(), efbKey);
  EXPECT_EQ(get_render_target_layout_key(), get_render_target_layout().key);
}

TEST_F(GfxRecordingTest, CopiedOffscreenPassIsRetainedOnRestore) {
  seed(320, 180);
  begin_offscreen(320, 180);
  copy_current_offscreen();

  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[0].has_consumer());
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
}

TEST_F(GfxRecordingTest, ReplacementRetainsCopiedPassesAndDiscardsContinuations) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  copy_current_offscreen();
  begin_offscreen(160, 90);
  copy_current_offscreen();

  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 5u);
  EXPECT_TRUE(frame.renderPasses[0].has_consumer());
  EXPECT_FALSE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
  EXPECT_TRUE(frame.renderPasses[2].has_consumer());
  EXPECT_FALSE(frame.renderPasses[2].discardable);
  EXPECT_TRUE(frame.renderPasses[3].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, RepeatedUncopiedCreatesDiscardEarlierPasses) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  begin_offscreen(160, 90);
  end_offscreen();

  ASSERT_EQ(frame.renderPasses.size(), 3u);
  EXPECT_TRUE(frame.renderPasses[0].sealed);
  EXPECT_TRUE(frame.renderPasses[0].discardable);
  EXPECT_TRUE(frame.renderPasses[1].sealed);
  EXPECT_TRUE(frame.renderPasses[1].discardable);
  EXPECT_EQ(count_efb_passes(), 1u);
}

TEST_F(GfxRecordingTest, PublicCreatePassRejectsExistingOffscreenPass) {
  seed(320, 180);
  seed(160, 90);
  ASSERT_TRUE(create_pass(320, 180));

  EXPECT_FALSE(create_pass(160, 90));

  ResolvedTargets ignored;
  EXPECT_TRUE(resolve_pass({.color = false, .depth = false}, ignored));
}

TEST_F(GfxRecordingTest, FinalizedPassesAreSealedOrDeliberatelyDiscarded) {
  seed(320, 180);
  seed(160, 90);
  begin_offscreen(320, 180);
  copy_current_offscreen();
  begin_offscreen(160, 90);
  end_offscreen();
  finish();

  ASSERT_FALSE(frame.renderPasses.empty());
  for (const auto& pass : frame.renderPasses) {
    EXPECT_TRUE(pass.sealed);
    if (!pass.has_consumer() && pass.label.starts_with("Offscreen")) {
      EXPECT_TRUE(pass.discardable);
    }
  }
  detail::end_recording();
  recordingActive = false;
}

// A clear without a copy (GX_AURORA_CLEAR_COPY_SRC) seals the pass with nothing to copy, and the next pass clears as
// after a copy with clear; a sealed pass that draws nothing and that nothing reads is discarded.
TEST_F(GfxRecordingTest, ClearWithoutACopy) {
  const ClipRect whole{0, 0, 640, 480};
  const Vec4<float> color{0.25f, 0.5f, 0.75f, 1.f};
  auto target =
      std::make_shared<TextureRef>(nullptr, TextureExtent{320, 240}, TextureFormat::RGBA8Unorm, 1, GX_TF_RGBA8);
  resolve_pass_into(std::move(target), whole, true, true, true, color, 0.5f, GX_TF_RGBA8, GX_PF_RGBA6_Z24);
  resolve_pass_into({}, whole, true, true, true, color, 0.5f, GX_TF_RGBA8, GX_PF_RGBA6_Z24);
  resolve_pass_into({}, whole, false, false, false, color, 0.5f, GX_TF_RGBA8, GX_PF_RGBA6_Z24);
  resolve_pass_into({}, whole, false, false, false, color, 0.5f, GX_TF_RGBA8, GX_PF_RGBA6_Z24);

  const auto& passes = frame.renderPasses;
  ASSERT_EQ(passes.size(), 5u);
  EXPECT_TRUE(passes[0].has_consumer());
  EXPECT_FALSE(passes[0].discardable);
  for (size_t i = 1; i < 4; ++i) {
    EXPECT_TRUE(passes[i].sealed) << i;
    EXPECT_FALSE(passes[i].resolveTarget) << i;
    EXPECT_EQ(passes[i].discardable, !passes[i].has_content()) << i;
  }
  // After the copy, and after the clear without one: cleared alike.
  for (size_t i = 1; i < 3; ++i) {
    const auto& color0 = passes[i].colorAttachments[SceneColorAttachmentIndex];
    EXPECT_TRUE(passes[i].clearDepth) << i;
    EXPECT_EQ(passes[i].clearDepthValue, 0.5f) << i;
    EXPECT_TRUE(color0.clear) << i;
    EXPECT_EQ(color0.clearValue, color) << i;
  }
  EXPECT_FALSE(passes[1].discardable);
  EXPECT_FALSE(passes[2].discardable);
  EXPECT_FALSE(passes[3].has_content());
  EXPECT_TRUE(passes[3].discardable);
  EXPECT_FALSE(passes[4].sealed);
}

// TextureSet::hash as it was: the fields streamed through Hasher::update.
BindGroupRef streamed_hash(const TextureSet& set) {
  Hasher hasher{};
  for (size_t i = 0; i < set.textures.size(); ++i) {
    const auto& sampler = set.samplers[i];
    hasher.update(reinterpret_cast<uintptr_t>(set.textures[i].get()));
    hasher.update(static_cast<uint32_t>(sampler.wrapS));
    hasher.update(static_cast<uint32_t>(sampler.wrapT));
    hasher.update(sampler.magFilter);
    hasher.update(sampler.minFilter);
    hasher.update(sampler.mipFilter);
    hasher.update(std::bit_cast<uint32_t>(sampler.minLod));
    hasher.update(std::bit_cast<uint32_t>(sampler.maxLod));
    hasher.update(sampler.maxAnisotropy);
  }
  return static_cast<BindGroupRef>(hasher.digest());
}

// A texture set's key is the one the streamed fields gave, for empty sets, full ones and everything between, with
// texture addresses and every sampler field at random.
TEST(TextureSetHash, AsTheStreamedFieldsGave) {
  std::mt19937_64 rng(1500);
  const auto texture = [](uint64_t address) {
    // Aliasing an empty owner: the address alone, which is all the key reads.
    return TextureHandle{TextureHandle{}, reinterpret_cast<TextureRef*>(static_cast<uintptr_t>(address))};
  };
  const std::array lods{0.f, -0.f, 1.f, 1000.f, 0.25f, -3.5f};
  for (int round = 0; round < 2000; ++round) {
    TextureSet set{};
    for (size_t i = 0; i < set.textures.size(); ++i) {
      if (round % 7 == 0 || rng() % 3 == 0) {
        continue;
      }
      set.textures[i] = texture(rng() & ~uint64_t{15});
      set.samplers[i] = {
          .wrapS = static_cast<GXTexWrapMode>(rng() % 3),
          .wrapT = static_cast<GXTexWrapMode>(rng() % 3),
          .magFilter = static_cast<SamplerFilter>(rng() % 2),
          .minFilter = static_cast<SamplerFilter>(rng() % 2),
          .mipFilter = static_cast<SamplerMipFilter>(rng() % 3),
          .minLod = lods[rng() % lods.size()],
          .maxLod = std::bit_cast<float>(static_cast<uint32_t>(rng())),
          .maxAnisotropy = static_cast<uint16_t>(rng()),
      };
    }
    ASSERT_EQ(set.hash(), streamed_hash(set)) << "round " << round;
  }
}

} // namespace
} // namespace aurora::gfx
