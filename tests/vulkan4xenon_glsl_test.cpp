// The vulkan4xenon backend's GLSL generator (lib/vulkan4xenon/glsl.cpp), on configs made to hit what it writes.
// With AURORA_GLSL_TEST_DIR set, each config's programs are also recorded there, for glslang and xost to compile.
#include "gx/gx.hpp"
#include "vulkan4xenon/glsl.hpp"

#include <aurora/aurora.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <string_view>

// What shader_info.cpp's uniform building reads, which the generator doesn't call.
namespace aurora {
AuroraConfig g_config{};
} // namespace aurora
namespace aurora::gx {
GXState g_gxState{};
const gfx::TextureBind& get_texture(GXTexMapID id) noexcept { return g_gxState.textures[id]; }
} // namespace aurora::gx
namespace aurora::gfx {
uint8_t* map_uniform(size_t length, Range& range) {
  (void)length;
  (void)range;
  return nullptr;
}
} // namespace aurora::gfx

namespace {
namespace gx = aurora::gx;
namespace glsl = aurora::gfx::vulkan4xenon::glsl;

// One TEV stage whose colour op takes its a and b from C0 and C1, and whose alpha op compares A0 and A1 by op,
// picking the constant 1.
gx::ShaderConfig compare_config(GXTevOp op) {
  gx::ShaderConfig config{};
  config.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.tevStageCount = 1;
  auto& stage = config.tevStages[0];
  stage.colorPass = {GX_CC_C0, GX_CC_C1, GX_CC_ZERO, GX_CC_ZERO};
  stage.alphaPass = {GX_CA_A0, GX_CA_A1, GX_CA_KONST, GX_CA_ZERO};
  stage.alphaOp.op = op;
  if (const char* dir = std::getenv("AURORA_GLSL_TEST_DIR")) {
    glsl::record(config, dir);
  }
  return config;
}

// The fragment program's line for stage 0's alpha.
std::string alpha_line(const gx::ShaderConfig& config) {
  const auto programs = glsl::build_programs(config);
  const auto start = programs.fragment.find("float s0_alpha = ");
  if (start == std::string::npos) {
    return {};
  }
  return programs.fragment.substr(start, programs.fragment.find('\n', start) - start);
}

// TEVREG0 and TEVREG1 are the material's tevreg[1] and tevreg[2], after TEVPREV.
constexpr std::string_view C0 = "umat.tevreg[1].rgb";
constexpr std::string_view C1 = "umat.tevreg[2].rgb";
constexpr std::string_view A0 = "umat.tevreg[1].a";
constexpr std::string_view A1 = "umat.tevreg[2].a";

struct PackedCompare {
  GXTevOp op;
  std::string_view key; // what the key reads of an operand, as gx/shader.cpp's tev_op writes it
};

class PackedAlphaCompare : public testing::TestWithParam<PackedCompare> {};

// A packed compare in the alpha op compares the colour operands a and b, as gx/shader.cpp does since encounter's
// 3094e04: the key is from the colours' channels, and the alpha operands a and b aren't read.
TEST_P(PackedAlphaCompare, ComparesTheColourOperands) {
  const auto line = alpha_line(compare_config(GetParam().op));
  ASSERT_FALSE(line.empty());
  EXPECT_NE(line.find(fmt::format("({}){}", C0, GetParam().key)), std::string::npos) << line;
  EXPECT_NE(line.find(fmt::format("({}){}", C1, GetParam().key)), std::string::npos) << line;
  EXPECT_EQ(line.find(A0), std::string::npos) << line;
  EXPECT_EQ(line.find(A1), std::string::npos) << line;
}

INSTANTIATE_TEST_SUITE_P(
    Ops, PackedAlphaCompare,
    testing::Values(PackedCompare{GX_TEV_COMP_R8_GT, ".r * 255.0"}, PackedCompare{GX_TEV_COMP_R8_EQ, ".r * 255.0"},
                    PackedCompare{GX_TEV_COMP_GR16_GT, ".rg * 255.0, vec2(1.0, 256.0)"},
                    PackedCompare{GX_TEV_COMP_GR16_EQ, ".rg * 255.0, vec2(1.0, 256.0)"},
                    PackedCompare{GX_TEV_COMP_BGR24_GT, ".rgb * 255.0, vec3(1.0, 256.0, 65536.0)"},
                    PackedCompare{GX_TEV_COMP_BGR24_EQ, ".rgb * 255.0, vec3(1.0, 256.0, 65536.0)"}));

// The alpha op's RGB8 compare is A8: it compares the alpha operands.
TEST(AlphaCompareA8, ComparesTheAlphaOperands) {
  for (const GXTevOp op : {GX_TEV_COMP_A8_GT, GX_TEV_COMP_A8_EQ}) {
    const auto line = alpha_line(compare_config(op));
    ASSERT_FALSE(line.empty());
    EXPECT_NE(line.find(fmt::format("floor(({}) * 255.0 + 0.5)", A0)), std::string::npos) << line;
    EXPECT_NE(line.find(fmt::format("floor(({}) * 255.0 + 0.5)", A1)), std::string::npos) << line;
    EXPECT_EQ(line.find(".rgb"), std::string::npos) << line;
  }
}
} // namespace
