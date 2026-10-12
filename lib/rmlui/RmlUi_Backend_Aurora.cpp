#include <RmlUi_Backend.h>
#include <RmlUi_Platform_SDL.h>

#include "SystemInterface_Aurora.h"
#ifdef AURORA_RMLUI_DRAWS
#include "WebGPURenderInterface.hpp"
#endif

namespace Backend {

#ifdef AURORA_RMLUI_DRAWS
using RenderInterface = aurora::rmlui::WebGPURenderInterface;
#else
// A backend that doesn't draw RmlUi: geometry and textures get handles, and nothing reaches the screen.
class RenderInterface final : public Rml::RenderInterface {
public:
  Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override {
    return ++m_lastHandle;
  }
  void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
  void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
  Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
    dimensions = {1, 1};
    return ++m_lastHandle;
  }
  Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return ++m_lastHandle; }
  void ReleaseTexture(Rml::TextureHandle) override {}
  void EnableScissorRegion(bool) override {}
  void SetScissorRegion(Rml::Rectanglei) override {}

private:
  uintptr_t m_lastHandle = 0;
};
#endif

RenderInterface* g_RenderInterface = nullptr;
aurora::rmlui::SystemInterface_Aurora* g_SystemInterface = nullptr;

bool Initialize(const char* window_name, int width, int height, bool allow_resize) {
  g_RenderInterface = new RenderInterface();
  g_SystemInterface = new aurora::rmlui::SystemInterface_Aurora();
  return true;
}

void Shutdown() {
  delete g_SystemInterface;
  delete g_RenderInterface;
}

Rml::SystemInterface* GetSystemInterface() { return g_SystemInterface; }

Rml::RenderInterface* GetRenderInterface() { return g_RenderInterface; }

bool ProcessEvents(Rml::Context* context, KeyDownCallback key_down_callback, bool power_save) {
  return true; // events handled directly by aurora
}

void RequestExit() {}

void BeginFrame() {}

void PresentFrame() {}

} // namespace Backend
