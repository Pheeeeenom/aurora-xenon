#include "ui_layout.hpp"

#include <algorithm>

namespace aurora::gfx::vulkan4xenon::ui_layout {
static_assert(sizeof(ImDrawIdx) == 2 && sizeof(ImDrawVert) == 20);

void atlas_bytes(std::span<const uint32_t> words, uint8_t* out) noexcept {
  for (const uint32_t word : words) {
    *out++ = static_cast<uint8_t>(word);
    *out++ = static_cast<uint8_t>(word >> 8);
    *out++ = static_cast<uint8_t>(word >> 16);
    *out++ = static_cast<uint8_t>(word >> 24);
  }
}

// The clip rectangle as imgui_impl_vulkan.cpp's ImGui_ImplVulkan_RenderDrawData makes it a scissor.
void lay_out_imgui(const ImDrawData& data, uint32_t width, uint32_t height, ImGuiLayout& layout) {
  layout.lists.clear();
  layout.listVertices.clear();
  layout.draws.clear();
  uint32_t at = 0;
  for (int i = 0; i < data.CmdListsCount; ++i) {
    const ImDrawList* list = data.CmdLists[i];
    const uint32_t vertices = at;
    layout.lists.push_back(list);
    layout.listVertices.push_back(vertices);
    // 20 bytes a vertex, so the indices after them start on 4 bytes.
    at += static_cast<uint32_t>(list->VtxBuffer.Size) * sizeof(ImDrawVert);
    for (const ImDrawCmd& cmd : list->CmdBuffer) {
      if (cmd.UserCallback != nullptr || cmd.GetTexID() == ImTextureID{} || cmd.ElemCount == 0) {
        continue;
      }
      const float left = std::max((cmd.ClipRect.x - data.DisplayPos.x) * data.FramebufferScale.x, 0.f);
      const float top = std::max((cmd.ClipRect.y - data.DisplayPos.y) * data.FramebufferScale.y, 0.f);
      const float right =
          std::min((cmd.ClipRect.z - data.DisplayPos.x) * data.FramebufferScale.x, static_cast<float>(width));
      const float bottom =
          std::min((cmd.ClipRect.w - data.DisplayPos.y) * data.FramebufferScale.y, static_cast<float>(height));
      if (right <= left || bottom <= top) {
        continue;
      }
      layout.draws.push_back({
          .vertices = vertices,
          .indices = at,
          .indexCount = cmd.ElemCount,
          .vertexOffset = cmd.VtxOffset,
          .x = static_cast<int32_t>(left),
          .y = static_cast<int32_t>(top),
          .width = static_cast<int32_t>(right - left),
          .height = static_cast<int32_t>(bottom - top),
          .texture = cmd.GetTexID(),
          .source = list->IdxBuffer.Data + cmd.IdxOffset,
      });
      at += (cmd.ElemCount * sizeof(ImDrawIdx) + 3) & ~3u;
    }
  }
  layout.bytes = at;
}

uint8_t composite(uint8_t efb, uint8_t ui, uint8_t alpha) noexcept {
  const uint32_t below = (uint32_t{efb} * (255u - alpha) + 127u) / 255u;
  return static_cast<uint8_t>(std::min(below + ui, 255u));
}
} // namespace aurora::gfx::vulkan4xenon::ui_layout
