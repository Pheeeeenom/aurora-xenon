#pragma once

#include "../gfx/ui.hpp"

#include <RmlUi/Core/RenderInterface.h>

#include <memory>
#include <string>
#include <vector>

namespace aurora::rmlui {
// RmlUi's render interface for a backend that draws it on its own thread (gfx/ui.hpp). It makes no graphics calls but
// for new textures: on the game thread it records what RmlUi asks into a list, which the backend replays with the
// frame. Its pipelines, stencil references and layers are Dawn's (WebGPURenderInterface.cpp); "blur" compiles to no
// filter, as the backend has none, so box-shadows are drawn without blur
// (rmlui:Source/Core/GeometryBoxShadow.cpp:194-200), and register_identity_filters keeps "blur" and "drop-shadow" from
// pushing layers.
class RecordingRenderInterface final : public Rml::RenderInterface {
public:
  // Before Context::Render: a frame of a target this size.
  void BeginFrame(Rml::Vector2i size) noexcept;
  // What was recorded since the last call, uploads included, swapped into frame; the recorder starts empty.
  void TakeFrame(gfx::ui::Frame& frame) noexcept;

  Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                                              Rml::Span<const int> indices) override;
  void RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
                      Rml::TextureHandle texture) override;
  void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;
  Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String& source) override;
  Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i dimensions) override;
  void ReleaseTexture(Rml::TextureHandle texture) override;
  void EnableScissorRegion(bool enable) override;
  void SetScissorRegion(Rml::Rectanglei region) override;
  void EnableClipMask(bool enable) override;
  void RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry,
                        Rml::Vector2f translation) override;
  void SetTransform(const Rml::Matrix4f* transform) override;
  Rml::LayerHandle PushLayer() override;
  void CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blendMode,
                       Rml::Span<const Rml::CompiledFilterHandle> filters) override;
  void PopLayer() override;
  Rml::TextureHandle SaveLayerAsTexture() override;
  Rml::CompiledFilterHandle CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters) override;
  void ReleaseFilter(Rml::CompiledFilterHandle filter) override;

private:
  Rml::TextureHandle make_texture(const Rml::byte* rgba, Rml::Vector2i size) noexcept;
  void draw(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, gfx::ui::Pipeline pipeline,
            uint32_t matrix) noexcept;
  gfx::ui::Rect scissor() const noexcept;
  gfx::ui::Pipeline masked(gfx::ui::Pipeline pipeline) const noexcept;

  gfx::ui::Frame m_frame;
  Rml::Vector2i m_size{1, 1};
  bool m_scissorEnabled = false;
  Rml::Rectanglei m_scissor = Rml::Rectanglei::MakeInvalid();
  uint32_t m_matrix = 0;
  bool m_clipEnabled = false;
  uint8_t m_stencilRef = 0;
  std::vector<uint8_t> m_layers{0};
  // The whole target, which a clip mask's Set and SetInverse clear first, as Dawn's EnsureClipResetGeometry.
  std::shared_ptr<gfx::ui::Geometry> m_reset;
  Rml::Vector2i m_resetSize{};
  std::vector<std::string> m_refusedFilters;
};

// RmlUi's "blur" and "drop-shadow" as filters that leave the element as it is and push no layer
// (rmlui:Source/Core/Factory.cpp:471-475). After Rml::Initialise.
void register_identity_filters() noexcept;
} // namespace aurora::rmlui
