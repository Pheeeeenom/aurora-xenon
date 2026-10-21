#include "RecordingRenderInterface.hpp"

#include "../gfx/backend.hpp"
#include "../gfx/texture.hpp"
#include "../logging.hpp"
#include "FileInterface_SDL.h"
#include "RuntimeTextureProvider.hpp"

#include <RmlUi/Core/CompiledFilterShader.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/Filter.h>
#include <RmlUi/Core/PropertyDefinition.h>
#include <RmlUi/Core/Vertex.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_surface.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace aurora::rmlui {
namespace {
Module Log("aurora::rmlui::RecordingRenderInterface");

static_assert(sizeof(Rml::Vertex) == sizeof(gfx::ui::Vertex));
static_assert(offsetof(Rml::Vertex, colour) == offsetof(gfx::ui::Vertex, colour));
static_assert(offsetof(Rml::Vertex, tex_coord) == offsetof(gfx::ui::Vertex, u));

// Dawn's depth range for transformed elements (WebGPURenderInterface.cpp, SetupRenderState).
constexpr float Near = -10000.f;
constexpr float Far = 10000.f;
// A stencil holds 8 bits, so a clip mask nests 255 deep, as on Dawn.
constexpr uint8_t MaxStencilRef = 255;

struct Texture {
  std::shared_ptr<gfx::backend::Texture> gpu;
};

struct Filter {
  float opacity;
};

// The target's pixels to Vulkan's clip space, y down: Dawn's projection with y the other way up, as WebGPU's points up.
Rml::Matrix4f projection(Rml::Vector2i size) noexcept {
  const auto width = static_cast<float>(std::max(size.x, 1));
  const auto height = static_cast<float>(std::max(size.y, 1));
  return Rml::Matrix4f::FromColumns({2.f / width, 0.f, 0.f, 0.f}, {0.f, 2.f / height, 0.f, 0.f},
                                    {0.f, 0.f, 1.f / (Far - Near), 0.f}, {-1.f, -1.f, -Near / (Far - Near), 1.f});
}

std::array<float, 16> columns(const Rml::Matrix4f& matrix) noexcept {
  Rml::ColumnMajorMatrix4f columnMajor;
  if constexpr (std::is_same_v<Rml::Matrix4f, Rml::RowMajorMatrix4f>) {
    columnMajor = matrix.Transpose();
  } else {
    columnMajor = matrix;
  }
  std::array<float, 16> out;
  std::memcpy(out.data(), columnMajor.data(), sizeof(out));
  return out;
}

// Premultiplied, as RmlUi's textures are, and Dawn's loader makes them (WebGPURenderInterface.cpp, get_image).
void premultiply(Rml::byte* texels, size_t size) noexcept {
  for (size_t i = 0; i + 3 < size; i += 4) {
    const uint32_t alpha = texels[i + 3];
    for (size_t channel = 0; channel < 3; ++channel) {
      texels[i + channel] = static_cast<Rml::byte>(texels[i + channel] * alpha / 255);
    }
  }
}

// An image file's texels as RGBA8, through SDL's PNG reader as Dawn's loader reads them; empty when it can't be read.
std::vector<Rml::byte> load_png(const Rml::String& source, Rml::Vector2i& size) {
  FileInterface_SDL fileInterface;
  const Rml::FileHandle file = fileInterface.Open(source);
  if (file == Rml::FileHandle{}) {
    return {};
  }
  SDL_Surface* loaded = SDL_LoadPNG_IO(reinterpret_cast<SDL_IOStream*>(file), true);
  if (loaded == nullptr) {
    Log.warn("Failed to load image '{}': {}", source, SDL_GetError());
    return {};
  }
  SDL_Surface* rgba = SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_RGBA32);
  SDL_DestroySurface(loaded);
  if (rgba == nullptr) {
    Log.warn("Failed to convert image '{}': {}", source, SDL_GetError());
    return {};
  }
  size = {rgba->w, rgba->h};
  const size_t row = static_cast<size_t>(rgba->w) * 4;
  std::vector<Rml::byte> texels(row * static_cast<size_t>(rgba->h));
  for (int y = 0; y < rgba->h; ++y) {
    std::memcpy(texels.data() + row * y,
                static_cast<const Rml::byte*>(rgba->pixels) + static_cast<size_t>(rgba->pitch) * y, row);
  }
  SDL_DestroySurface(rgba);
  return texels;
}

// "blur" and "drop-shadow" taking the stock instancers' properties, so the style sheets parse as before
// (rmlui:Source/Core/FilterBlur.cpp, FilterDropShadow.cpp), and drawing the element as it is.
class IdentityFilter final : public Rml::Filter {
public:
  Rml::CompiledFilter CompileFilter(Rml::Element*) const override { return {}; }
  Rml::FilterLayerOperation GetLayerOperation(Rml::Element*, Rml::FilterLayerState) const override {
    return Rml::FilterLayerOperation::Identity;
  }
};

class IdentityFilterInstancer final : public Rml::FilterInstancer {
public:
  explicit IdentityFilterInstancer(bool dropShadow) {
    if (dropShadow) {
      RegisterProperty("color", "transparent").AddParser("color");
      RegisterProperty("offset-x", "0px").AddParser("length");
      RegisterProperty("offset-y", "0px").AddParser("length");
    }
    RegisterProperty("sigma", "0px").AddParser("length");
    RegisterShorthand("filter", dropShadow ? "color, offset-x, offset-y, sigma" : "sigma",
                      Rml::ShorthandType::FallThrough);
  }
  Rml::SharedPtr<Rml::Filter> InstanceFilter(const Rml::String&, const Rml::PropertyDictionary&) override {
    return m_filter;
  }

private:
  Rml::SharedPtr<Rml::Filter> m_filter = Rml::MakeShared<IdentityFilter>();
};
} // namespace

void RecordingRenderInterface::BeginFrame(Rml::Vector2i size) noexcept {
  m_size = {std::max(size.x, 1), std::max(size.y, 1)};
  m_frame.width = static_cast<uint32_t>(m_size.x);
  m_frame.height = static_cast<uint32_t>(m_size.y);
  m_frame.matrices.push_back(columns(projection(m_size)));
  m_matrix = static_cast<uint32_t>(m_frame.matrices.size() - 1);
  m_scissorEnabled = false;
  m_clipEnabled = false;
  m_stencilRef = 0;
  m_layers.assign(1, 0);
}

void RecordingRenderInterface::TakeFrame(gfx::ui::Frame& frame) noexcept {
  std::swap(m_frame, frame);
  m_frame.clear();
}

Rml::CompiledGeometryHandle RecordingRenderInterface::CompileGeometry(Rml::Span<const Rml::Vertex> vertices,
                                                                      Rml::Span<const int> indices) {
  // Copied, as the frame that draws it may still be drawing when RmlUi frees its own.
  auto* geometry = new std::shared_ptr<gfx::ui::Geometry>(std::make_shared<gfx::ui::Geometry>());
  auto& data = **geometry;
  data.vertices.resize(vertices.size());
  std::memcpy(data.vertices.data(), vertices.data(), vertices.size() * sizeof(Rml::Vertex));
  data.indices.assign(indices.begin(), indices.end());
  return reinterpret_cast<Rml::CompiledGeometryHandle>(geometry);
}

void RecordingRenderInterface::draw(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
                                    gfx::ui::Pipeline pipeline, uint32_t matrix) noexcept {
  m_frame.ops.push_back({
      .type = gfx::ui::OpType::Draw,
      .pipeline = pipeline,
      .stencilRef = m_stencilRef,
      .scissor = scissor(),
      .matrix = matrix,
      .translation = {translation.x, translation.y},
      .geometry = *reinterpret_cast<const std::shared_ptr<gfx::ui::Geometry>*>(geometry),
  });
}

void RecordingRenderInterface::RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation,
                                              Rml::TextureHandle texture) {
  draw(geometry, translation, masked(gfx::ui::Pipeline::Blend), m_matrix);
  if (texture != 0) {
    m_frame.ops.back().texture = reinterpret_cast<const Texture*>(texture)->gpu;
  }
}

void RecordingRenderInterface::ReleaseGeometry(Rml::CompiledGeometryHandle geometry) {
  delete reinterpret_cast<std::shared_ptr<gfx::ui::Geometry>*>(geometry);
}

Rml::TextureHandle RecordingRenderInterface::make_texture(const Rml::byte* rgba, Rml::Vector2i size) noexcept {
  if (size.x <= 0 || size.y <= 0) {
    return 0;
  }
  const auto width = static_cast<uint32_t>(size.x);
  const auto height = static_cast<uint32_t>(size.y);
  auto gpu = gfx::backend::create_texture({
      .width = width,
      .height = height,
      .mips = 1,
      .format = gfx::TextureFormat::RGBA8Unorm,
      .gxFormat = gfx::InvalidTextureFormat,
      .usage = gfx::backend::TextureUsage::Sampled,
      .label = "RmlUi texture",
  });
  auto data = gpu ? gfx::backend::stage_upload(rgba, width * 4, height) : nullptr;
  if (!data) {
    Log.error("No memory for a {}x{} texture", width, height);
    return 0;
  }
  m_frame.uploads.push_back({.texture = gpu, .data = std::move(data), .width = width, .height = height});
  return reinterpret_cast<Rml::TextureHandle>(new Texture{std::move(gpu)});
}

// Runtime textures are made as Dawn makes them (WebGPURenderInterface.cpp, LoadTexture), without the mips some ask for.
Rml::TextureHandle RecordingRenderInterface::LoadTexture(Rml::Vector2i& dimensions, const Rml::String& source) {
  if (const auto runtime = load_runtime_texture(source)) {
    const size_t size = size_t{runtime->width} * runtime->height * 4;
    if (runtime->width == 0 || runtime->height == 0 || runtime->rgba8.size() < size) {
      Log.error("Runtime texture provider returned invalid texture! Path: {}", source);
      return 0;
    }
    std::vector<Rml::byte> texels(size);
    std::memcpy(texels.data(), runtime->rgba8.data(), size);
    if (!runtime->premultipliedAlpha) {
      premultiply(texels.data(), texels.size());
    }
    dimensions = {static_cast<int>(runtime->width), static_cast<int>(runtime->height)};
    return make_texture(texels.data(), dimensions);
  }
  Rml::Vector2i size;
  auto texels = load_png(source, size);
  if (texels.empty()) {
    Log.error("Failed to load texture! Path: {}", source);
    return 0;
  }
  premultiply(texels.data(), texels.size());
  dimensions = size;
  return make_texture(texels.data(), size);
}

Rml::TextureHandle RecordingRenderInterface::GenerateTexture(Rml::Span<const Rml::byte> source,
                                                             Rml::Vector2i dimensions) {
  return make_texture(source.data(), dimensions);
}

void RecordingRenderInterface::ReleaseTexture(Rml::TextureHandle texture) {
  delete reinterpret_cast<Texture*>(texture);
}

void RecordingRenderInterface::EnableScissorRegion(bool enable) { m_scissorEnabled = enable; }

void RecordingRenderInterface::SetScissorRegion(Rml::Rectanglei region) { m_scissor = region; }

// The scissor inside the target, all of it when off, as Dawn clamps it (WebGPURenderInterface.cpp,
// GetActiveScissorRegion).
gfx::ui::Rect RecordingRenderInterface::scissor() const noexcept {
  if (!m_scissorEnabled || !m_scissor.Valid()) {
    return {0, 0, m_size.x, m_size.y};
  }
  const int left = std::clamp(m_scissor.Left(), 0, m_size.x);
  const int top = std::clamp(m_scissor.Top(), 0, m_size.y);
  const int right = std::clamp(m_scissor.Right(), left, m_size.x);
  const int bottom = std::clamp(m_scissor.Bottom(), top, m_size.y);
  return {left, top, right - left, bottom - top};
}

gfx::ui::Pipeline RecordingRenderInterface::masked(gfx::ui::Pipeline pipeline) const noexcept {
  if (!m_clipEnabled) {
    return pipeline;
  }
  return pipeline == gfx::ui::Pipeline::Replace ? gfx::ui::Pipeline::ReplaceMasked : gfx::ui::Pipeline::BlendMasked;
}

void RecordingRenderInterface::EnableClipMask(bool enable) {
  m_clipEnabled = enable;
  if (!enable) {
    m_stencilRef = 0;
  }
}

// As Dawn's (WebGPURenderInterface.cpp, RenderToClipMask): Set and SetInverse draw the whole target, untransformed,
// then the geometry, each replacing the stencil with a reference; Intersect increments where the stencil equals it.
void RecordingRenderInterface::RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry,
                                                Rml::Vector2f translation) {
  if (operation == Rml::ClipMaskOperation::Intersect) {
    if (m_stencilRef == MaxStencilRef) {
      Log.warn("RmlUi clip mask nesting exceeded stencil capacity; further nested clipping may be incorrect");
      return;
    }
    draw(geometry, translation, gfx::ui::Pipeline::ClipIntersect, m_matrix);
    ++m_stencilRef;
    return;
  }
  if (!m_reset || m_resetSize != m_size) {
    const auto width = static_cast<float>(m_size.x);
    const auto height = static_cast<float>(m_size.y);
    m_reset = std::make_shared<gfx::ui::Geometry>(gfx::ui::Geometry{
        .vertices = {{0.f, 0.f, {}, 0.f, 0.f},
                     {width, 0.f, {}, 0.f, 0.f},
                     {width, height, {}, 0.f, 0.f},
                     {0.f, height, {}, 0.f, 0.f}},
        .indices = {0, 1, 2, 0, 2, 3},
    });
    m_resetSize = m_size;
  }
  const bool inverse = operation == Rml::ClipMaskOperation::SetInverse;
  m_stencilRef = inverse ? 1 : 0;
  m_frame.ops.push_back({
      .type = gfx::ui::OpType::Draw,
      .pipeline = gfx::ui::Pipeline::ClipReplace,
      .stencilRef = m_stencilRef,
      .scissor = scissor(),
      .matrix = 0,
      .geometry = m_reset,
  });
  m_stencilRef = inverse ? 0 : 1;
  draw(geometry, translation, gfx::ui::Pipeline::ClipReplace, m_matrix);
  m_stencilRef = 1;
}

void RecordingRenderInterface::SetTransform(const Rml::Matrix4f* transform) {
  if (transform == nullptr) {
    m_matrix = 0;
    return;
  }
  m_frame.matrices.push_back(columns(projection(m_size) * *transform));
  m_matrix = static_cast<uint32_t>(m_frame.matrices.size() - 1);
}

Rml::LayerHandle RecordingRenderInterface::PushLayer() {
  const auto layer = static_cast<uint8_t>(m_layers.size());
  m_layers.push_back(layer);
  m_frame.layers = std::max(m_frame.layers, layer);
  m_frame.ops.push_back({.type = gfx::ui::OpType::PushLayer, .destination = layer});
  return layer;
}

// Opacity, the only filter compiled, folds into the colour the layer is drawn at.
void RecordingRenderInterface::CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination,
                                               Rml::BlendMode blendMode,
                                               Rml::Span<const Rml::CompiledFilterHandle> filters) {
  if (source == destination || source >= m_layers.size() || destination >= m_layers.size()) {
    Log.warn("RmlUi composite of layer {} onto {} with {} layers isn't drawn", source, destination, m_layers.size());
    return;
  }
  float opacity = 1.f;
  for (const Rml::CompiledFilterHandle filter : filters) {
    opacity *= reinterpret_cast<const Filter*>(filter)->opacity;
  }
  m_frame.ops.push_back({
      .type = gfx::ui::OpType::Composite,
      .pipeline = masked(blendMode == Rml::BlendMode::Replace ? gfx::ui::Pipeline::Replace : gfx::ui::Pipeline::Blend),
      .stencilRef = m_stencilRef,
      .source = static_cast<uint8_t>(source),
      .destination = static_cast<uint8_t>(destination),
      .scissor = scissor(),
      .opacity = std::clamp(opacity, 0.f, 1.f),
  });
}

void RecordingRenderInterface::PopLayer() {
  if (m_layers.size() <= 1) {
    Log.warn("RmlUi requested PopLayer with no pushed layer");
    return;
  }
  m_layers.pop_back();
  m_frame.ops.push_back({.type = gfx::ui::OpType::PopLayer, .destination = m_layers.back()});
}

// The scissor region of the layer on top, which RmlUi sets to what it saves
// (rmlui:Source/Core/CallbackTexture.cpp:38-55).
Rml::TextureHandle RecordingRenderInterface::SaveLayerAsTexture() {
  const gfx::ui::Rect region = scissor();
  if (region.width <= 0 || region.height <= 0) {
    return 0;
  }
  auto gpu = gfx::backend::create_texture({
      .width = static_cast<uint32_t>(region.width),
      .height = static_cast<uint32_t>(region.height),
      .mips = 1,
      .format = gfx::TextureFormat::RGBA8Unorm,
      .gxFormat = gfx::InvalidTextureFormat,
      .usage = gfx::backend::TextureUsage::Render,
      .label = "RmlUi saved layer",
  });
  if (!gpu) {
    Log.error("No memory for a {}x{} saved layer", region.width, region.height);
    return 0;
  }
  m_frame.ops.push_back({
      .type = gfx::ui::OpType::Save,
      .source = m_layers.back(),
      .scissor = region,
      .texture = gpu,
  });
  return reinterpret_cast<Rml::TextureHandle>(new Texture{std::move(gpu)});
}

Rml::CompiledFilterHandle RecordingRenderInterface::CompileFilter(const Rml::String& name,
                                                                  const Rml::Dictionary& parameters) {
  if (name == "opacity") {
    const auto it = parameters.find("value");
    return reinterpret_cast<Rml::CompiledFilterHandle>(
        new Filter{it != parameters.end() ? it->second.Get<float>() : 1.f});
  }
  // Box-shadows ask for "blur" each time they're made, and the rest are drawn as nothing (milestone 16's scope).
  if (name != "blur" && std::ranges::find(m_refusedFilters, name) == m_refusedFilters.end()) {
    m_refusedFilters.push_back(name);
    Log.warn("RmlUi filter '{}' isn't drawn on this backend", name);
  }
  return 0;
}

void RecordingRenderInterface::ReleaseFilter(Rml::CompiledFilterHandle filter) {
  delete reinterpret_cast<Filter*>(filter);
}

void register_identity_filters() noexcept {
  static IdentityFilterInstancer blur{false};
  static IdentityFilterInstancer dropShadow{true};
  Rml::Factory::RegisterFilterInstancer("blur", &blur);
  Rml::Factory::RegisterFilterInstancer("drop-shadow", &dropShadow);
}
} // namespace aurora::rmlui
