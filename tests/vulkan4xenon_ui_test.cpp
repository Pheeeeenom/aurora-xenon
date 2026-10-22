// The vulkan4xenon backend's UI data (lib/vulkan4xenon/ui_layout.cpp): ImGui's atlas and draw lists, and the present's
// composite, which need no GPU.
#include "vulkan4xenon/ui_layout.hpp"

#include <gtest/gtest.h>
#include <imgui.h>

#include <array>
#include <cstdint>
#include <cstring>

namespace {
namespace ui_layout = aurora::gfx::vulkan4xenon::ui_layout;

constexpr ImTextureID Atlas = 1;

ImDrawCmd command(uint32_t indexOffset, uint32_t count, ImVec4 clip = {0.f, 0.f, 1280.f, 720.f}) {
  ImDrawCmd cmd;
  cmd.ClipRect = clip;
  cmd.TextureId = Atlas;
  cmd.IdxOffset = indexOffset;
  cmd.ElemCount = count;
  return cmd;
}

// One list of four vertices and nine indices, its commands as given, on a 1280x720 display.
struct Lists {
  ImDrawList list{nullptr};
  ImDrawData data;

  explicit Lists(std::initializer_list<ImDrawCmd> commands) {
    list.VtxBuffer.resize(4);
    for (ImDrawIdx i = 0; i < 9; ++i) {
      list.IdxBuffer.push_back(i);
    }
    for (const ImDrawCmd& cmd : commands) {
      list.CmdBuffer.push_back(cmd);
    }
    data.CmdLists.push_back(&list);
    data.CmdListsCount = 1;
    data.TotalVtxCount = list.VtxBuffer.Size;
    data.TotalIdxCount = list.IdxBuffer.Size;
    data.DisplaySize = {1280.f, 720.f};
    data.FramebufferScale = {1.f, 1.f};
  }
};

// IM_COL32's words, 0xAABBGGRR (imgui:imgui.h:2882-2889), come out red first, as R8G8B8A8_UNORM reads them, whatever
// the host's byte order: the atlas's white texels FF FF FF and their alpha. In place, as new_frame converts them.
TEST(ImGuiAtlas, WordsBecomeBytesRedFirst) {
  std::array<uint32_t, 2> words{IM_COL32(255, 255, 255, 0x40), IM_COL32(1, 2, 3, 4)};
  ui_layout::atlas_bytes(words, reinterpret_cast<uint8_t*>(words.data()));
  std::array<uint8_t, 8> bytes{};
  std::memcpy(bytes.data(), words.data(), bytes.size());
  EXPECT_EQ(bytes, (std::array<uint8_t, 8>{0xFF, 0xFF, 0xFF, 0x40, 1, 2, 3, 4}));
}

// A command whose indices start at an odd one is copied to start on 4 bytes, after its list's vertices.
TEST(ImGuiLayout, CopiesACommandAtAnOddIndexOntoFourBytes) {
  Lists lists{command(0, 3), command(3, 3), command(6, 3)};
  ui_layout::ImGuiLayout layout;
  ui_layout::lay_out_imgui(lists.data, 1280, 720, layout);
  ASSERT_EQ(layout.draws.size(), 3u);
  EXPECT_EQ(layout.listVertices[0], 0u);
  uint32_t end = 4 * sizeof(ImDrawVert);
  for (size_t i = 0; i < layout.draws.size(); ++i) {
    const auto& draw = layout.draws[i];
    EXPECT_EQ(draw.indices % 4, 0u) << i;
    EXPECT_GE(draw.indices, end) << i;
    EXPECT_EQ(draw.source, lists.list.IdxBuffer.Data + 3 * i) << i;
    EXPECT_EQ(draw.indexCount, 3u) << i;
    end = draw.indices + draw.indexCount * sizeof(ImDrawIdx);
  }
  EXPECT_GE(layout.bytes, end);
}

// A clip rectangle with nothing inside the target draws nothing, and the rest are clamped to it.
TEST(ImGuiLayout, SkipsAnEmptyClipRectangle) {
  Lists lists{command(0, 3, {10.f, 10.f, 10.f, 50.f}), command(3, 3, {1300.f, 0.f, 1400.f, 50.f}),
              command(6, 3, {-20.f, 700.f, 40.f, 800.f})};
  ui_layout::ImGuiLayout layout;
  ui_layout::lay_out_imgui(lists.data, 1280, 720, layout);
  ASSERT_EQ(layout.draws.size(), 1u);
  const auto& draw = layout.draws[0];
  EXPECT_EQ(draw.source, lists.list.IdxBuffer.Data + 6);
  EXPECT_EQ(draw.x, 0);
  EXPECT_EQ(draw.y, 700);
  EXPECT_EQ(draw.width, 40);
  EXPECT_EQ(draw.height, 20);
}

// A premultiplied UI texel over an EFB texel: the UI's channel plus the EFB's times one minus the UI's alpha.
TEST(PresentComposite, AddsTheUiToTheEfbUnderItsAlpha) {
  EXPECT_EQ(ui_layout::composite(200, 0, 0), 200);
  EXPECT_EQ(ui_layout::composite(200, 255, 255), 255);
  EXPECT_EQ(ui_layout::composite(200, 64, 128), 64 + 100);
  EXPECT_EQ(ui_layout::composite(30, 64, 64), 64 + 22);
}
} // namespace
