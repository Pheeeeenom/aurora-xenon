// RmlUi's recorder for the vulkan4xenon backend (lib/rmlui/RecordingRenderInterface.cpp): what it records against what
// Dawn's interface draws (lib/rmlui/WebGPURenderInterface.cpp), on RmlUi with no GPU.
#include "gfx/backend.hpp"
#include "rmlui/RecordingRenderInterface.hpp"

#include <RmlUi/Core.h>
#include <aurora/aurora.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

namespace aurora {
AuroraConfig g_config{};
} // namespace aurora

// Textures with nothing behind them, as the null backend makes.
namespace aurora::gfx::backend {
struct Texture {};
struct UploadBuffer {};
std::shared_ptr<Texture> create_texture(const TextureDesc&) noexcept { return std::make_shared<Texture>(); }
std::shared_ptr<UploadBuffer> stage_upload(const uint8_t*, uint32_t, uint32_t, bool) noexcept {
  return std::make_shared<UploadBuffer>();
}
} // namespace aurora::gfx::backend

namespace {
namespace ui = aurora::gfx::ui;
using aurora::rmlui::RecordingRenderInterface;

const Rml::Vector2i Size{1280, 720};

struct Recorder {
  RecordingRenderInterface rec;
  Rml::CompiledGeometryHandle triangle = 0;
  ui::Frame frame;

  Recorder() {
    const std::array<Rml::Vertex, 3> vertices{{{{10.f, 20.f}, {}, {}}, {{30.f, 20.f}, {}, {}}, {{10.f, 40.f}, {}, {}}}};
    const std::array<int, 3> indices{0, 1, 2};
    triangle = rec.CompileGeometry({vertices.data(), vertices.size()}, {indices.data(), indices.size()});
    rec.BeginFrame(Size);
  }
  ~Recorder() { rec.ReleaseGeometry(triangle); }

  const std::vector<ui::Op>& take() {
    rec.TakeFrame(frame);
    return frame.ops;
  }
};

void expect_draw(const ui::Op& op, ui::Pipeline pipeline, uint8_t stencilRef) {
  EXPECT_EQ(op.type, ui::OpType::Draw);
  EXPECT_EQ(op.pipeline, pipeline);
  EXPECT_EQ(op.stencilRef, stencilRef);
}

// The whole target, untransformed, as Dawn's EnsureClipResetGeometry makes it.
void expect_reset(const ui::Op& op, uint8_t stencilRef) {
  expect_draw(op, ui::Pipeline::ClipReplace, stencilRef);
  EXPECT_EQ(op.matrix, 0u);
  ASSERT_TRUE(op.geometry);
  const auto& vertices = op.geometry->vertices;
  const auto [minX, maxX] = std::ranges::minmax(vertices, {}, &ui::Vertex::x);
  const auto [minY, maxY] = std::ranges::minmax(vertices, {}, &ui::Vertex::y);
  EXPECT_EQ(minX.x, 0.f);
  EXPECT_EQ(maxX.x, static_cast<float>(Size.x));
  EXPECT_EQ(minY.y, 0.f);
  EXPECT_EQ(maxY.y, static_cast<float>(Size.y));
}

// Set: the target cleared to 0, the geometry set to 1; draws then pass where the stencil is 1. Dawn's references
// (WebGPURenderInterface.cpp, RenderToClipMask), its ClipReplace pipeline and its EqualKeep for masked draws.
TEST(RmlUiRecorder, ClipSetReplacesWithDawnsReferences) {
  Recorder r;
  r.rec.EnableClipMask(true);
  r.rec.RenderToClipMask(Rml::ClipMaskOperation::Set, r.triangle, {});
  r.rec.RenderGeometry(r.triangle, {}, 0);
  r.rec.EnableClipMask(false);
  r.rec.RenderGeometry(r.triangle, {}, 0);
  const auto& ops = r.take();
  ASSERT_EQ(ops.size(), 4u);
  expect_reset(ops[0], 0);
  expect_draw(ops[1], ui::Pipeline::ClipReplace, 1);
  expect_draw(ops[2], ui::Pipeline::BlendMasked, 1);
  expect_draw(ops[3], ui::Pipeline::Blend, 0);
}

// SetInverse: the target set to 1, the geometry cleared to 0.
TEST(RmlUiRecorder, ClipSetInverseClearsTheGeometry) {
  Recorder r;
  r.rec.EnableClipMask(true);
  r.rec.RenderToClipMask(Rml::ClipMaskOperation::SetInverse, r.triangle, {});
  r.rec.RenderGeometry(r.triangle, {}, 0);
  const auto& ops = r.take();
  ASSERT_EQ(ops.size(), 3u);
  expect_reset(ops[0], 1);
  expect_draw(ops[1], ui::Pipeline::ClipReplace, 0);
  expect_draw(ops[2], ui::Pipeline::BlendMasked, 1);
}

// Intersect increments where the stencil equals the reference, which then goes up one, until 255.
TEST(RmlUiRecorder, ClipIntersectNestsTo255) {
  Recorder r;
  r.rec.EnableClipMask(true);
  r.rec.RenderToClipMask(Rml::ClipMaskOperation::Set, r.triangle, {});
  for (int i = 0; i < 256; ++i) {
    r.rec.RenderToClipMask(Rml::ClipMaskOperation::Intersect, r.triangle, {});
  }
  r.rec.RenderGeometry(r.triangle, {}, 0);
  const auto& ops = r.take();
  // The reset and Set, 254 intersects from 1 to 254, then nothing past 255.
  ASSERT_EQ(ops.size(), 2u + 254u + 1u);
  for (size_t i = 0; i < 254; ++i) {
    expect_draw(ops[2 + i], ui::Pipeline::ClipIntersect, static_cast<uint8_t>(1 + i));
  }
  expect_draw(ops.back(), ui::Pipeline::BlendMasked, 255);
}

// The scissor clamped to the target, and the whole target when off, as Dawn's GetActiveScissorRegion.
TEST(RmlUiRecorder, ClampsTheScissorToTheTarget) {
  Recorder r;
  r.rec.SetScissorRegion(Rml::Rectanglei::FromPositionSize({-10, 700}, {100, 100}));
  r.rec.EnableScissorRegion(true);
  r.rec.RenderGeometry(r.triangle, {}, 0);
  r.rec.EnableScissorRegion(false);
  r.rec.RenderGeometry(r.triangle, {}, 0);
  const auto& ops = r.take();
  ASSERT_EQ(ops.size(), 2u);
  EXPECT_EQ(ops[0].scissor.x, 0);
  EXPECT_EQ(ops[0].scissor.y, 700);
  EXPECT_EQ(ops[0].scissor.width, 90);
  EXPECT_EQ(ops[0].scissor.height, 20);
  EXPECT_EQ(ops[1].scissor.x, 0);
  EXPECT_EQ(ops[1].scissor.y, 0);
  EXPECT_EQ(ops[1].scissor.width, Size.x);
  EXPECT_EQ(ops[1].scissor.height, Size.y);
}

// Where a pixel lands, from Vulkan's clip space, y down.
Rml::Vector2f vulkan_pixel(const std::array<float, 16>& m, Rml::Vector2f p) {
  const float x = m[0] * p.x + m[4] * p.y + m[12];
  const float y = m[1] * p.x + m[5] * p.y + m[13];
  const float w = m[3] * p.x + m[7] * p.y + m[15];
  return {(x / w + 1.f) * 0.5f * Size.x, (y / w + 1.f) * 0.5f * Size.y};
}

// Where Dawn puts it: its projection (WebGPURenderInterface.cpp, SetupRenderState) to WebGPU's clip space, y up.
Rml::Vector2f dawn_pixel(const Rml::Matrix4f& transform, Rml::Vector2f p) {
  const float l = 0.f, r = Size.x, t = 0.f, b = Size.y, n = -10000.f, f = 10000.f;
  const auto proj = Rml::Matrix4f::FromColumns({2.0f / (r - l), 0.0f, 0.0f, 0.0f}, {0.0f, 2.0f / (t - b), 0.0f, 0.0f},
                                               {0.0f, 0.0f, 1.0f / (f - n), 0.0f},
                                               {(r + l) / (l - r), (t + b) / (b - t), -n / (f - n), 1.0f});
  const Rml::Vector4f clip = (proj * transform) * Rml::Vector4f{p.x, p.y, 0.f, 1.f};
  return {(clip.x / clip.w + 1.f) * 0.5f * Size.x, (1.f - clip.y / clip.w) * 0.5f * Size.y};
}

// A transformed and translated point lands where Dawn puts it, and the untransformed one on its own pixel.
TEST(RmlUiRecorder, TransformsAndTranslatesAsDawn) {
  Recorder r;
  const Rml::Matrix4f transform = Rml::Matrix4f::Translate(300.f, 200.f, 0.f) * Rml::Matrix4f::RotateZ(30.f) *
                                  Rml::Matrix4f::Scale(1.5f, 0.75f, 1.f);
  r.rec.SetTransform(&transform);
  r.rec.RenderGeometry(r.triangle, {40.f, -15.f}, 0);
  r.rec.SetTransform(nullptr);
  r.rec.RenderGeometry(r.triangle, {40.f, -15.f}, 0);
  const auto& ops = r.take();
  ASSERT_EQ(ops.size(), 2u);
  for (const Rml::Vector2f vertex : {Rml::Vector2f{10.f, 20.f}, Rml::Vector2f{30.f, 20.f}, Rml::Vector2f{10.f, 40.f}}) {
    const Rml::Vector2f moved = vertex + Rml::Vector2f{ops[0].translation[0], ops[0].translation[1]};
    const auto got = vulkan_pixel(r.frame.matrices[ops[0].matrix], moved);
    const auto want = dawn_pixel(transform, moved);
    EXPECT_NEAR(got.x, want.x, 1e-3f);
    EXPECT_NEAR(got.y, want.y, 1e-3f);
    const auto plain = vulkan_pixel(r.frame.matrices[ops[1].matrix], moved);
    EXPECT_NEAR(plain.x, moved.x, 1e-3f);
    EXPECT_NEAR(plain.y, moved.y, 1e-3f);
  }
}

// A layer pushed, drawn into, composited onto the one below and popped; another saved as a texture; each op naming the
// layers Dawn's would touch.
TEST(RmlUiRecorder, PushCompositePopAndSaveTouchTheirLayers) {
  Recorder r;
  const Rml::LayerHandle layer = r.rec.PushLayer();
  r.rec.RenderGeometry(r.triangle, {}, 0);
  r.rec.CompositeLayers(layer, 0, Rml::BlendMode::Replace, {});
  r.rec.PopLayer();
  const Rml::LayerHandle saved = r.rec.PushLayer();
  r.rec.SetScissorRegion(Rml::Rectanglei::FromPositionSize({100, 50}, {40, 30}));
  r.rec.EnableScissorRegion(true);
  const Rml::TextureHandle texture = r.rec.SaveLayerAsTexture();
  r.rec.PopLayer();
  r.rec.RenderGeometry(r.triangle, {}, texture);
  const auto& ops = r.take();
  ASSERT_EQ(ops.size(), 8u);
  EXPECT_EQ(layer, 1u);
  EXPECT_EQ(saved, 1u);
  EXPECT_EQ(ops[0].type, ui::OpType::PushLayer);
  EXPECT_EQ(ops[0].destination, 1);
  EXPECT_EQ(ops[2].type, ui::OpType::Composite);
  EXPECT_EQ(ops[2].source, 1);
  EXPECT_EQ(ops[2].destination, 0);
  EXPECT_EQ(ops[2].pipeline, ui::Pipeline::Replace);
  EXPECT_EQ(ops[3].type, ui::OpType::PopLayer);
  EXPECT_EQ(ops[3].destination, 0);
  EXPECT_EQ(ops[5].type, ui::OpType::Save);
  EXPECT_EQ(ops[5].source, 1);
  EXPECT_EQ(ops[5].scissor.x, 100);
  EXPECT_EQ(ops[5].scissor.y, 50);
  EXPECT_EQ(ops[5].scissor.width, 40);
  EXPECT_EQ(ops[5].scissor.height, 30);
  ASSERT_TRUE(ops[5].texture);
  EXPECT_EQ(ops[6].type, ui::OpType::PopLayer);
  // The saved layer's texture is what the draw after samples.
  EXPECT_EQ(ops[7].texture, ops[5].texture);
  r.rec.ReleaseTexture(texture);
}

// Opacity folds into the composite's colour, filters multiplying; the others compile to nothing.
TEST(RmlUiRecorder, FoldsOpacityIntoTheComposite) {
  Recorder r;
  const auto half = r.rec.CompileFilter("opacity", {{"value", Rml::Variant(0.5f)}});
  const auto quarter = r.rec.CompileFilter("opacity", {{"value", Rml::Variant(0.25f)}});
  EXPECT_EQ(r.rec.CompileFilter("grayscale", {{"value", Rml::Variant(1.f)}}), 0u);
  const auto layer = r.rec.PushLayer();
  r.rec.EnableClipMask(true);
  const std::array<Rml::CompiledFilterHandle, 2> filters{half, quarter};
  r.rec.CompositeLayers(layer, 0, Rml::BlendMode::Blend, {filters.data(), filters.size()});
  r.rec.PopLayer();
  const auto& ops = r.take();
  ASSERT_EQ(ops.size(), 3u);
  EXPECT_EQ(ops[1].type, ui::OpType::Composite);
  EXPECT_FLOAT_EQ(ops[1].opacity, 0.125f);
  EXPECT_EQ(ops[1].pipeline, ui::Pipeline::BlendMasked);
  r.rec.ReleaseFilter(half);
  r.rec.ReleaseFilter(quarter);
}

// Blur and drop-shadow, as filters or a backdrop, push no layer; a fade does, and a box-shadow's blur adds none to the
// layer its texture is made in.
TEST(RmlUiRecorder, PushesNoLayerForBlurOrDropShadow) {
  RecordingRenderInterface rec;
  Rml::SetRenderInterface(&rec);
  ASSERT_TRUE(Rml::Initialise());
  aurora::rmlui::register_identity_filters();
  Rml::Context* context = Rml::CreateContext("test", Size);
  ASSERT_NE(context, nullptr);
  const auto layers = [&](const char* style) {
    const Rml::String source = Rml::String(
                                   "<rml><head><style>body { width: 100%; height: 100%; } div { width: 200px; "
                                   "height: 100px; background-color: #f00; ") +
                               style + " }</style></head><body><div/></body></rml>";
    Rml::ElementDocument* document = context->LoadDocumentFromMemory(source);
    document->Show();
    context->Update();
    rec.BeginFrame(Size);
    context->Render();
    ui::Frame frame;
    rec.TakeFrame(frame);
    document->Close();
    context->Update();
    return std::ranges::count(frame.ops, ui::OpType::PushLayer, &ui::Op::type);
  };
  EXPECT_EQ(layers("filter: blur(5px);"), 0);
  EXPECT_EQ(layers("backdrop-filter: blur(5px);"), 0);
  EXPECT_EQ(layers("filter: drop-shadow(#000 2px 2px 3px);"), 0);
  EXPECT_EQ(layers("filter: opacity(0.5);"), 1);
  EXPECT_EQ(layers("box-shadow: #000 0 0 10px 2px;"), 1);
  Rml::Shutdown();
}
} // namespace
