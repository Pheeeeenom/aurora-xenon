#pragma once
// What RmlUi draws a frame, for a backend that draws it on its own thread: recorded on the game thread
// (rmlui/RecordingRenderInterface.cpp), handed to backend::end_frame beside ImGui's frozen draw data, and read on the
// render worker. Only the vulkan4xenon backend draws it; Dawn's RmlUi records its own passes, and the null backend
// draws nothing.

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace aurora::gfx::backend {
struct Texture;
struct UploadBuffer;
} // namespace aurora::gfx::backend

namespace aurora::gfx::ui {
// RmlUi's vertex, Rml::Vertex: its position, its premultiplied colour as bytes in R, G, B, A order
// (rmlui:Include/RmlUi/Core/Vertex.h:12-18), and its texture coordinate.
struct Vertex {
  float x;
  float y;
  uint8_t colour[4];
  float u;
  float v;
};
static_assert(sizeof(Vertex) == 20);

struct Geometry {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};

// A rectangle of the target, already inside it.
struct Rect {
  int32_t x = 0;
  int32_t y = 0;
  int32_t width = 0;
  int32_t height = 0;
};

// The pipelines over the UI's program pair, as Dawn's (rmlui/WebGPURenderInterface.cpp, geometry_pipeline and
// blit_pipeline): premultiplied blending or replacing, through the clip mask or not; and the clip mask's two writes.
enum class Pipeline : uint8_t {
  Blend,
  BlendMasked,
  Replace,
  ReplaceMasked,
  ClipReplace,   // stencil always passes and takes the reference; no colour
  ClipIntersect, // stencil equal to the reference passes and increments; no colour
};

enum class OpType : uint8_t {
  Draw,      // geometry into the active layer
  PushLayer, // a new layer, transparent, takes the draws
  Composite, // a layer drawn over another, as a full-target triangle
  PopLayer,  // the layer below takes the draws again
  Save,      // the active layer's scissor region copied into a texture
};

struct Op {
  OpType type = OpType::Draw;
  Pipeline pipeline = Pipeline::Blend;
  uint8_t stencilRef = 0;
  // Composite: the layer read and the layer drawn into, 0 the UI image.
  uint8_t source = 0;
  uint8_t destination = 0;
  // Draw and Composite: the scissor. Save: the region saved.
  Rect scissor;
  // Draw: the transform's place in Frame::matrices, and the translation added before it.
  uint32_t matrix = 0;
  std::array<float, 2> translation{};
  // Composite: the opacity filters' product, which the layer is drawn at.
  float opacity = 1.f;
  std::shared_ptr<const Geometry> geometry;
  // Draw: the texture sampled, none for white. Save: the texture written.
  std::shared_ptr<backend::Texture> texture;
};

// A new texture's texels, staged on the game thread, copied in before the frame's UI is drawn.
struct Upload {
  std::shared_ptr<backend::Texture> texture;
  std::shared_ptr<backend::UploadBuffer> data;
  uint32_t width = 0;
  uint32_t height = 0;
};

struct Frame {
  // RmlUi's context: the UI image's size.
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<Op> ops;
  // Each a projection times a transform, column by column; the first is the projection alone.
  std::vector<std::array<float, 16>> matrices;
  std::vector<Upload> uploads;

  // Emptied, its vectors keeping their room for the next frame.
  void clear() noexcept {
    ops.clear();
    matrices.clear();
    uploads.clear();
  }
};
} // namespace aurora::gfx::ui
