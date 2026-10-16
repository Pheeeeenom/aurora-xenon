#include "imgui.hpp"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <SDL3/SDL_events.h>
#include <imgui.h>

#include "internal.hpp"

#include "tracy/Tracy.hpp"

namespace aurora::imgui {
namespace {
std::string g_imguiSettings{};
std::string g_imguiLog{};
bool g_cursorManaged = false;
} // namespace

struct DrawData::Impl {
  ImDrawData drawData;
  std::vector<std::unique_ptr<ImDrawList>> drawLists;
};

void create_context() noexcept {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  g_imguiSettings = std::string{g_config.userPath} + "/imgui.ini";
  g_imguiLog = std::string{g_config.cachePath} + "/imgui.log";
  io.IniFilename = g_imguiSettings.c_str();
  io.LogFilename = g_imguiLog.c_str();
}

void set_cursor_managed(bool managed) noexcept {
  if (ImGui::GetCurrentContext() == nullptr) {
    return;
  }
  auto& flags = ImGui::GetIO().ConfigFlags;
  if (managed && (flags & ImGuiConfigFlags_NoMouseCursorChange) == 0) {
    flags |= ImGuiConfigFlags_NoMouseCursorChange;
    g_cursorManaged = true;
  } else if (!managed && g_cursorManaged) {
    flags &= ~ImGuiConfigFlags_NoMouseCursorChange;
    g_cursorManaged = false;
  }
}

bool DrawData::empty() const noexcept { return !m_impl || m_impl->drawData.CmdListsCount == 0; }

ImDrawData* DrawData::data() const noexcept { return m_impl ? &m_impl->drawData : nullptr; }

DrawData freeze() noexcept {
  ZoneScoped;
  ImGui::Render();

  auto* data = ImGui::GetDrawData();
  data->FramebufferScale = ImGui::GetIO().DisplayFramebufferScale;
  auto frozen = std::make_shared<DrawData::Impl>();
  frozen->drawData = *data;
  frozen->drawLists.reserve(data->CmdListsCount);
  frozen->drawData.CmdLists.resize(data->CmdListsCount);
  for (int i = 0; i < data->CmdListsCount; ++i) {
    frozen->drawLists.emplace_back(data->CmdLists[i]->CloneOutput());
    frozen->drawData.CmdLists[i] = frozen->drawLists.back().get();
  }
  return DrawData{std::move(frozen)};
}
} // namespace aurora::imgui
