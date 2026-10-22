#pragma once
// The UI's data as the vulkan4xenon backend lays it out, without Vulkan, so the host tests take it: ImGui's atlas and
// draw lists, and the present's composite.

#include <imgui.h>

#include <cstdint>
#include <span>
#include <vector>

namespace aurora::gfx::vulkan4xenon::ui_layout {
// ImGui's atlas words, IM_COL32's 0xAABBGGRR (imgui:imgui.h:2882-2889), as bytes in R, G, B, A order on any host,
// which R8G8B8A8_UNORM reads; out may be the words' own memory, each word read before its bytes are written.
void atlas_bytes(std::span<const uint32_t> words, uint8_t* out) noexcept;

// One ImGui command: its indices, 16-bit, copied to start on 4 bytes, the only alignment the console has drawn index
// lists from (xsdk:docs/GUIDE.md:93-95).
struct ImGuiDraw {
  uint32_t vertices; // its list's vertices, in bytes from the start
  uint32_t indices;  // in bytes from the start, a multiple of 4
  uint32_t indexCount;
  uint32_t vertexOffset;
  int32_t x, y, width, height; // the scissor, inside the target
  ImTextureID texture;
  const ImDrawIdx* source; // the indices to copy
};

struct ImGuiLayout {
  std::vector<const ImDrawList*> lists;
  std::vector<uint32_t> listVertices; // where each list's vertices go, in bytes from the start
  std::vector<ImGuiDraw> draws;
  uint32_t bytes = 0;
};

// Where ImGui's lists and commands go, from a start on 4 bytes, in a target of width by height. Commands with a
// callback, no texture, or a clip rectangle with nothing inside the target are left out.
void lay_out_imgui(const ImDrawData& data, uint32_t width, uint32_t height, ImGuiLayout& layout);

// One channel of a premultiplied UI texel over an EFB texel, as the present blends them: ui + efb * (1 - alpha).
uint8_t composite(uint8_t efb, uint8_t ui, uint8_t alpha) noexcept;
} // namespace aurora::gfx::vulkan4xenon::ui_layout
