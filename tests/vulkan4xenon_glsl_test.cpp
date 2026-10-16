// The vulkan4xenon backend's GLSL generator (lib/vulkan4xenon/glsl.cpp), on configs made to hit what it writes.
// With AURORA_GLSL_TEST_DIR set, each config's programs are also recorded there, for glslang and xost to compile.
#include "gx/gx.hpp"
#include "vulkan4xenon/glsl.hpp"

#include <aurora/aurora.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
using namespace std::string_literals;
using namespace std::string_view_literals;

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
  const auto config = compare_config(GetParam().op);
  const auto line = alpha_line(config);
  ASSERT_FALSE(line.empty());
  // The operands as the stage reads them: TEVREG0 and TEVREG1 wrapped, as no stage has written them (milestone 18).
  EXPECT_NE(line.find(fmt::format("s0_color_a{}", GetParam().key)), std::string::npos) << line;
  EXPECT_NE(line.find(fmt::format("s0_color_b{}", GetParam().key)), std::string::npos) << line;
  const auto fragment = glsl::build_programs(config).fragment;
  EXPECT_NE(fragment.find(fmt::format("vec3 s0_color_a = (({}) * 255.0 - floor", C0)), std::string::npos) << fragment;
  EXPECT_NE(fragment.find(fmt::format("vec3 s0_color_b = (({}) * 255.0 - floor", C1)), std::string::npos) << fragment;
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
    const auto config = compare_config(op);
    const auto line = alpha_line(config);
    ASSERT_FALSE(line.empty());
    EXPECT_NE(line.find("floor(s0_alpha_a * 255.0 + 0.5)"), std::string::npos) << line;
    EXPECT_NE(line.find("floor(s0_alpha_b * 255.0 + 0.5)"), std::string::npos) << line;
    EXPECT_EQ(line.find(".rgb"), std::string::npos) << line;
    const auto fragment = glsl::build_programs(config).fragment;
    EXPECT_NE(fragment.find(fmt::format("float s0_alpha_a = (({}) * 255.0 - floor", A0)), std::string::npos)
        << fragment;
    EXPECT_NE(fragment.find(fmt::format("float s0_alpha_b = (({}) * 255.0 - floor", A1)), std::string::npos)
        << fragment;
  }
}

// Milestone 11, step 5: fog, lighting, matrices picked per vertex and indirect texturing, as gx/shader.cpp's WGSL
// writes them.

bool has(std::string_view program, std::string_view text) { return program.find(text) != std::string_view::npos; }

// The programs of config, recorded for glslang and xost when AURORA_GLSL_TEST_DIR is set.
glsl::Programs built(const gx::ShaderConfig& config) {
  if (const char* dir = std::getenv("AURORA_GLSL_TEST_DIR")) {
    glsl::record(config, dir);
  }
  return glsl::build_programs(config);
}

// One TEV stage drawing colour channel 0 as it's rasterized.
gx::ShaderConfig rasterized() {
  gx::ShaderConfig config{};
  config.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.vtxStride = 12;
  config.tevStageCount = 1;
  auto& stage = config.tevStages[0];
  stage.colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_RASC};
  stage.alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA};
  stage.channelId = GX_COLOR0A0;
  return config;
}

// The same with its colour, or its alpha, lit by the lights of mask, with normals.
gx::ShaderConfig lit(GXDiffuseFn diffFn, GXAttnFn attnFn, u8 mask, bool alpha = false, GXColorSrc ambSrc = GX_SRC_REG) {
  auto config = rasterized();
  config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32, .offset = 12};
  config.vtxStride = 24;
  auto& cc = config.colorChannels[alpha ? GX_ALPHA0 : GX_COLOR0];
  cc = {.ambSrc = ambSrc, .diffFn = diffFn, .attnFn = attnFn, .lightingEnabled = true, .lightMask = mask};
  if (ambSrc == GX_SRC_VTX) {
    config.attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8, .offset = 24};
    config.vtxStride = 28;
  }
  return config;
}

// A light's position is its first vec4 in the lights piece.
std::string light_position(u32 light) { return fmt::format("ulights.lights[{}]", light * 5); }

// Each light of the mask is read, and no other: the masks are in the key, so the lights are unrolled.
TEST(Lighting, SumsTheMasksLightsAlone) {
  for (const u8 mask : {0xffu, 0x01u, 0x7eu, 0xfcu, 0x00u}) {
    const auto programs = built(lit(GX_DF_CLAMP, GX_AF_SPOT, mask));
    ASSERT_FALSE(programs.vertex.empty()) << int{mask};
    for (u32 light = 0; light < 8; ++light) {
      EXPECT_EQ(has(programs.vertex, light_position(light) + ".xyz - mv_pos4.xyz"), (mask >> light & 1) != 0)
          << int{mask} << " light " << light;
    }
    // The current normal matrix is pushed after the position matrix, for a channel that sums any light.
    EXPECT_EQ(programs.normalMatrix, mask != 0);
    EXPECT_EQ(has(programs.vertex, "vec4 nrm_mtx[3];"), mask != 0);
    EXPECT_EQ(has(programs.vertex, "dot(upush.nrm_mtx[0], nrm4)"), mask != 0);
  }
}

struct LightingFns {
  GXDiffuseFn diffFn;
  GXAttnFn attnFn;
};

class LightingFunctions : public testing::TestWithParam<LightingFns> {};

// Each diffuse and attenuation function, on the colour and on the alpha: the normal is read where the WGSL reads it.
TEST_P(LightingFunctions, AreWritten) {
  const auto [diffFn, attnFn] = GetParam();
  const bool normal = diffFn != GX_DF_NONE || attnFn == GX_AF_SPEC;
  for (const bool alpha : {false, true}) {
    const auto programs = built(lit(diffFn, attnFn, 0x05, alpha));
    ASSERT_FALSE(programs.vertex.empty());
    ASSERT_FALSE(programs.fragment.empty());
    EXPECT_EQ(programs.normalMatrix, normal);
    EXPECT_EQ(has(programs.vertex, "in_nrm"), normal);
    EXPECT_EQ(has(programs.vertex, "_attn0 = "), attnFn != GX_AF_NONE);
    EXPECT_EQ(has(programs.vertex, "_diff2 = "), diffFn != GX_DF_NONE);
    EXPECT_TRUE(has(programs.vertex, alpha ? "float cc0a_lit = uview.channel_colors[3].a * clamp(cc0a_sum2, 0.0, 1.0);"
                                           : "vec4 cc0_lit = uview.channel_colors[1] * clamp(cc0_sum2, 0.0, 1.0);"));
  }
}

INSTANTIATE_TEST_SUITE_P(Fns, LightingFunctions,
                         testing::Values(LightingFns{GX_DF_NONE, GX_AF_NONE}, LightingFns{GX_DF_SIGN, GX_AF_NONE},
                                         LightingFns{GX_DF_CLAMP, GX_AF_NONE}, LightingFns{GX_DF_NONE, GX_AF_SPOT},
                                         LightingFns{GX_DF_SIGN, GX_AF_SPOT}, LightingFns{GX_DF_CLAMP, GX_AF_SPOT},
                                         LightingFns{GX_DF_NONE, GX_AF_SPEC}, LightingFns{GX_DF_SIGN, GX_AF_SPEC},
                                         LightingFns{GX_DF_CLAMP, GX_AF_SPEC}));

// The ambient colour from the vertex, as the scene's one such channel has it.
TEST(Lighting, TakesTheAmbientFromTheVertex) {
  const auto programs = built(lit(GX_DF_CLAMP, GX_AF_SPOT, 0xfe, false, GX_SRC_VTX));
  ASSERT_FALSE(programs.vertex.empty());
  EXPECT_TRUE(has(programs.vertex, "vec4 cc0_sum1 = in_clr0 + "));
}

// The position matrix picked per vertex, read at the row the vertex names, and with it the normal matrix: nothing is
// pushed but the unused current position matrix.
TEST(MatricesPerVertex, ReadThePositionAndNormalMatrixAtTheVertexsRow) {
  auto config = lit(GX_DF_CLAMP, GX_AF_SPOT, 0xff);
  config.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 24};
  config.vtxStride = 25;
  const auto programs = built(config);
  ASSERT_FALSE(programs.vertex.empty());
  EXPECT_TRUE(has(programs.vertex, "int pnmtx = int(in_pnmtxidx);"));
  EXPECT_TRUE(has(programs.vertex, "dot(upos.mtx[pnmtx + 2], pos4)"));
  EXPECT_TRUE(has(programs.vertex, "dot(unrm.mtx[pnmtx + 1], nrm4)"));
  EXPECT_FALSE(programs.normalMatrix);
  EXPECT_FALSE(has(programs.vertex, "upush.pos_mtx"));
  const auto& attributes = programs.layout.attributes;
  ASSERT_FALSE(attributes.empty());
  EXPECT_EQ(attributes.back().attr, GX_VA_PNMTXIDX);
  EXPECT_EQ(attributes.back().format, glsl::AttrFormat::F32x1);
}

// A texture matrix picked per vertex: a position matrix's slot or a texture matrix's, the scene's TEX1MTXIDX.
TEST(MatricesPerVertex, PickTheTexgensMatrix) {
  for (const GXTexGenType type : {GX_TG_MTX2x4, GX_TG_MTX3x4}) {
    auto config = rasterized();
    config.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 12};
    config.attrs[GX_VA_TEX1MTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 13};
    config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 14};
    config.vtxStride = 22;
    config.tcgs[1] = {.type = type, .src = GX_TG_TEX0, .mtx = GX_TEXMTX3};
    auto& stage = config.tevStages[0];
    stage.texCoordId = GX_TEXCOORD1;
    stage.texMapId = GX_TEXMAP0;
    stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
    const auto programs = built(config);
    ASSERT_FALSE(programs.vertex.empty());
    EXPECT_TRUE(has(programs.vertex, "int tc1_pos_row = int(min(in_tex1mtxidx, 27.0));"));
    EXPECT_TRUE(has(programs.vertex, "int tc1_tex_row = int(clamp(in_tex1mtxidx - 30.0, 0.0, 27.0));"));
    EXPECT_TRUE(has(programs.vertex, "dot(uview.tex_mtx[tc1_tex_row + 2], tc1)"));
  }
}

struct FogCase {
  u8 type;
  bool range;
  std::string_view z; // what fog_z is
};

class Fog : public testing::TestWithParam<FogCase> {};

// Each fog type, perspective and orthographic, with and without range: the depth from clip z and w, the range factor
// from the fragment's column.
TEST_P(Fog, IsWritten) {
  for (const bool ortho : {false, true}) {
    auto config = rasterized();
    config.fogType = GetParam().type | (ortho ? 0x08 : 0);
    config.fogRangeEnabled = GetParam().range;
    const auto programs = built(config);
    ASSERT_FALSE(programs.fragment.empty());
    EXPECT_TRUE(has(programs.vertex, "out_fog_zw = clip_pos.zw;"));
    EXPECT_TRUE(has(programs.fragment, "vec4 depth_range;"));
    EXPECT_TRUE(has(programs.fragment, ortho ? "fog_base = uview.fog[1].x * fog_depth;"
                                             : "fog_base = uview.fog[1].x / (uview.fog[1].y - fog_depth);"));
    EXPECT_EQ(has(programs.fragment, "gl_FragCoord.x"), GetParam().range);
    EXPECT_EQ(has(programs.fragment, "layout(set = 2, binding = 0) uniform sampler2D fog_range_table;"),
              GetParam().range);
    EXPECT_EQ(programs.fogRange, GetParam().range);
    EXPECT_TRUE(has(programs.fragment, GetParam().z)) << programs.fragment;
    EXPECT_TRUE(has(programs.fragment, "o_color = vec4(fog_color, "));
  }
}

INSTANTIATE_TEST_SUITE_P(Types, Fog,
                         testing::Values(FogCase{GX_FOG_PERSP_LIN, false, "fog_z = fog_f;"},
                                         FogCase{GX_FOG_PERSP_LIN, true, "fog_z = fog_f;"},
                                         FogCase{GX_FOG_PERSP_EXP, false, "fog_z = 1.0 - exp2(-8.0 * fog_f);"},
                                         FogCase{GX_FOG_PERSP_EXP2, false, "fog_z = 1.0 - exp2(-8.0 * fog_f * fog_f);"},
                                         FogCase{GX_FOG_PERSP_REVEXP, true, "fog_z = exp2(-8.0 * (1.0 - fog_f));"},
                                         FogCase{GX_FOG_PERSP_REVEXP2, false, "fog_z = exp2(-8.0 * fog_g * fog_g);"},
                                         FogCase{GX_FOG_PERSP_REVEXP2, true, "fog_z = exp2(-8.0 * fog_g * fog_g);"}));

// Fog on a colour its stage didn't clamp is mixed first, then wrapped, as the WGSL does.
TEST(FogOrder, IsMixedBeforeTheWrap) {
  auto config = rasterized();
  config.fogType = GX_FOG_PERSP_LIN;
  config.tevStages[0].colorOp.clamp = false;
  const auto programs = built(config);
  EXPECT_TRUE(has(programs.fragment, "vec3 out_color = ((fog_color) * 255.0"));
}

// Indirect texturing on stage 0, from indirect stage 0's texture on map 1 at coordinate 1, onto map 0 at coordinate 0.
gx::ShaderConfig indirect(GXIndTexMtxID mtx, GXIndTexFormat format = GX_ITF_8, GXIndTexBiasSel bias = GX_ITB_NONE) {
  auto config = rasterized();
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 12};
  config.attrs[GX_VA_TEX1] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 20};
  config.vtxStride = 28;
  config.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0};
  config.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1};
  config.numIndStages = 1;
  config.indStages[0] = {GX_TEXCOORD1, GX_TEXMAP1, GX_ITS_2, GX_ITS_4};
  auto& stage = config.tevStages[0];
  stage.texCoordId = GX_TEXCOORD0;
  stage.texMapId = GX_TEXMAP0;
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  stage.indTexStage = GX_INDTEXSTAGE0;
  stage.indTexMtxId = mtx;
  stage.indTexFormat = format;
  stage.indTexBiasSel = bias;
  return config;
}

// The indirect texture sampled once, then each matrix's offset, as the WGSL works it out.
TEST(Indirect, OffsetsTheStagesCoordinate) {
  {
    const auto programs = built(indirect(GX_ITM_0, GX_ITF_8, GX_ITB_STU));
    ASSERT_FALSE(programs.fragment.empty());
    EXPECT_TRUE(has(programs.fragment,
                    "vec3 ind_tex0 = 255.0 * texture(tex1, tex1_uv * umat.texcoord_scale[1].xy *"
                    " vec2((1.0 / 2.0), (1.0 / 4.0)) / umat.tex_size_bias[1].xy).abg;"));
    EXPECT_TRUE(has(programs.fragment, "vec3 ind0_biased = ind0_coord + vec3(1.0, 1.0, 1.0) * -128.0;"));
    EXPECT_TRUE(has(programs.fragment, "vec2 ind0_uv = tev_coord0 / umat.tex_size_bias[0].xy;"));
    EXPECT_TRUE(has(programs.fragment, "vec4 sampled0 = texture(tex0, ind0_uv);"));
  }
  {
    const auto programs = built(indirect(GX_ITM_S1, GX_ITF_5, GX_ITB_T));
    ASSERT_FALSE(programs.fragment.empty());
    EXPECT_TRUE(has(programs.fragment, "vec3 ind0_coord = floor(ind_tex0 / 8.0);"));
    EXPECT_TRUE(has(programs.fragment, "ind0_biased.x * uview.ind_mtx[3].z / 256.0"));
  }
  {
    const auto programs = built(indirect(GX_ITM_T2, GX_ITF_3));
    ASSERT_FALSE(programs.fragment.empty());
    EXPECT_TRUE(has(programs.fragment, "ind0_coord.y * uview.ind_mtx[5].z / 256.0"));
  }
}

// Wraps, the WGSL's truncating %, and a second stage adding to the first's coordinate.
TEST(Indirect, WrapsAndAddsToThePreviousStage) {
  auto config = indirect(GX_ITM_1);
  config.tevStages[0].indTexWrapS = GX_ITW_64;
  config.tevStages[0].indTexWrapT = GX_ITW_0;
  config.tevStageCount = 2;
  auto& second = config.tevStages[1];
  second = config.tevStages[0];
  second.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_CPREV, GX_CC_ZERO};
  second.indTexWrapS = GX_ITW_OFF;
  second.indTexWrapT = GX_ITW_OFF;
  second.indTexAddPrev = true;
  const auto programs = built(config);
  ASSERT_FALSE(programs.fragment.empty());
  EXPECT_TRUE(has(programs.fragment, "vec2((ind0_texel.x - 64.0 * trunc(ind0_texel.x / 64.0)), 0.0)"));
  EXPECT_TRUE(has(programs.fragment, "vec2 tev_coord1 = tev_coord0 + ("));
  EXPECT_TRUE(has(programs.fragment, "vec4 sampled1 = texture(tex0, ind1_uv);"));
}

// The alpha bump: an indirect component's bits the format keeps, as the WGSL masks them; BUMPN scaled to 1.
TEST(Indirect, BumpsTheAlpha) {
  for (const GXChannelID channel : {GX_ALPHA_BUMP, GX_ALPHA_BUMPN}) {
    auto config = indirect(GX_ITM_0, GX_ITF_5);
    auto& stage = config.tevStages[0];
    stage.channelId = channel;
    stage.indTexAlphaSel = GX_ITBA_T;
    const auto programs = built(config);
    ASSERT_FALSE(programs.fragment.empty());
    const auto masked = "((floor(ind_tex0.y) - mod(floor(ind_tex0.y), 32.0)) / 255.0)"s;
    EXPECT_TRUE(
        has(programs.fragment, channel == GX_ALPHA_BUMPN ? fmt::format("({} * (255.0 / 248.0))", masked) : masked));
  }
}

// The scene's commonest needs together, as step 1 counted them: matrices picked per vertex for the position and
// texture coordinate 1, fog adjusted by range, and COLOR0 lit by eight spot lights, clamped.
gx::ShaderConfig scene() {
  auto config = lit(GX_DF_CLAMP, GX_AF_SPOT, 0xff);
  config.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 24};
  config.attrs[GX_VA_TEX1MTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 25};
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 26};
  config.vtxStride = 34;
  config.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_TEXMTX0};
  config.fogType = GX_FOG_PERSP_LIN;
  config.fogRangeEnabled = true;
  auto& stage = config.tevStages[0];
  stage.texCoordId = GX_TEXCOORD1;
  stage.texMapId = GX_TEXMAP0;
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  return config;
}

TEST(SceneConfigs, AreWritten) {
  const auto programs = built(scene());
  EXPECT_FALSE(programs.vertex.empty());
  EXPECT_FALSE(programs.fragment.empty());
}

// The configs Aurora's tools/check_glsl_math.py runs on xost's simulator against its own reference of the WGSL's
// math, each written as <name>.vert and <name>.frag when AURORA_GLSL_NAMED_DIR is set. A case's config is here and
// what the tool feeds it and expects is there, under the same name.
void named_programs(std::string_view name, const std::string& vertex, const std::string& fragment) {
  ASSERT_FALSE(vertex.empty()) << name;
  ASSERT_FALSE(fragment.empty()) << name;
  if (const char* dir = std::getenv("AURORA_GLSL_NAMED_DIR")) {
    for (const auto& [suffix, text] : {std::pair{".vert"sv, &vertex}, std::pair{".frag"sv, &fragment}}) {
      const auto path = fmt::format("{}/{}{}", dir, name, suffix);
      std::FILE* file = std::fopen(path.c_str(), "wb");
      ASSERT_NE(file, nullptr) << path;
      std::fwrite(text->data(), 1, text->size(), file);
      std::fclose(file);
    }
  }
}

void named(std::string_view name, const gx::ShaderConfig& config) {
  const auto programs = built(config);
  named_programs(name, programs.vertex, programs.fragment);
}

// TEV stage 0 sampling map 0 at texture coordinate 1, generated from TEX0 by texture matrix 3 or the matrix each
// vertex picks with TEX1MTXIDX, times the rasterized colour.
gx::ShaderConfig textured(bool perVertex) {
  auto config = rasterized();
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 12};
  config.vtxStride = 20;
  if (perVertex) {
    config.attrs[GX_VA_TEX1MTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 20};
    config.vtxStride = 21;
  }
  config.tcgs[1] = {.type = GX_TG_MTX3x4, .src = GX_TG_TEX0, .mtx = GX_TEXMTX3};
  auto& stage = config.tevStages[0];
  stage.texCoordId = GX_TEXCOORD1;
  stage.texMapId = GX_TEXMAP0;
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  return config;
}

gx::ShaderConfig fogged(u8 type, bool range) {
  auto config = rasterized();
  config.fogType = type;
  config.fogRangeEnabled = range;
  return config;
}

// GX lines, a line strip or points (lineMode 1 to 3), TEV stage 0 sampling map 0 at texture coordinate 0, TEX0 as it
// is, times the rasterized colour; by the current position matrix or the one each vertex picks.
gx::ShaderConfig lines(u8 lineMode, bool perVertex) {
  auto config = rasterized();
  config.lineMode = lineMode;
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 12};
  config.vtxStride = 20;
  if (perVertex) {
    config.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 20};
    config.vtxStride = 21;
  }
  config.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_IDENTITY};
  auto& stage = config.tevStages[0];
  stage.texCoordId = GX_TEXCOORD0;
  stage.texMapId = GX_TEXMAP0;
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  return config;
}

TEST(Lines, AreQuadsAcrossEachSegment) {
  for (const u8 mode : {1, 2}) {
    for (const bool perVertex : {false, true}) {
      const auto programs = built(lines(mode, perVertex));
      ASSERT_FALSE(programs.vertex.empty()) << int{mode} << perVertex;
      const auto& corners = programs.layout.corners;
      ASSERT_EQ(corners.size(), perVertex ? 3u : 2u);
      EXPECT_EQ(corners[0].value, glsl::CornerValue::Corner);
      EXPECT_EQ(corners[1].value, glsl::CornerValue::OtherPos);
      EXPECT_TRUE(has(programs.vertex, "vec4 line_mask[2];"));
      EXPECT_TRUE(has(programs.vertex, "vec2(line_dir.y * -1.0, line_dir.x) * (line_width * in_corner.x)"));
      EXPECT_EQ(has(programs.vertex, "upos.mtx[other_pnmtx]"), perVertex);
      EXPECT_TRUE(has(programs.vertex, "vec3(0.0, in_corner.y, 0.0) * (uview.line_params.z * upush.line_mask[0].x)"));
    }
  }
}

TEST(Points, AreSquaresAboutEachPoint) {
  const auto programs = built(lines(3, false));
  ASSERT_EQ(programs.layout.corners.size(), 1u);
  EXPECT_TRUE(has(programs.vertex, "vec2 corner_offset = in_corner * line_width / uview.viewport_sizes.xy;"));
  EXPECT_TRUE(has(programs.vertex, "vec3(in_corner * 0.5 + 0.5, 0.0) * (uview.line_params.z * upush.line_mask[0].x)"));
}

TEST(Corners, CountSixASegmentOrPoint) {
  EXPECT_EQ(glsl::stream_vertices(lines(1, false), 5), 12u);
  EXPECT_EQ(glsl::stream_vertices(lines(2, false), 4), 18u);
  EXPECT_EQ(glsl::stream_vertices(lines(2, false), 1), 0u);
  EXPECT_EQ(glsl::stream_vertices(lines(3, false), 3), 18u);
  EXPECT_EQ(glsl::stream_vertices(rasterized(), 7), 7u);
}

// Each corner write_corners writes, from decoded vertices whose position's x is their number and TEX0 ten times it:
// the end it holds, the other end, and its corner values.
struct Corner {
  float end;
  float other;
  float x;
  float y;
};

std::vector<Corner> corners_of(const gx::ShaderConfig& config, uint32_t count) {
  const auto layout = glsl::build_programs(config).layout;
  std::vector<uint8_t> decoded(size_t{count} * layout.stride);
  for (uint32_t v = 0; v < count; ++v) {
    const float values[5]{static_cast<float>(v), 0.f, 0.f, 10.f * v, 0.f};
    std::memcpy(decoded.data() + size_t{v} * layout.stride, values, sizeof(values));
  }
  std::vector<uint8_t> out(size_t{glsl::stream_vertices(config, count)} * layout.stride);
  glsl::write_corners(config, layout, decoded.data(), count, out.data());
  std::vector<Corner> corners;
  for (size_t i = 0; i < out.size(); i += layout.stride) {
    Corner corner{};
    float tex = 0.f;
    std::memcpy(&corner.end, out.data() + i, sizeof(float));
    std::memcpy(&tex, out.data() + i + 12, sizeof(float));
    EXPECT_EQ(tex, 10.f * corner.end);
    corner.other = -1.f;
    for (const auto& value : layout.corners) {
      if (value.value == glsl::CornerValue::Corner) {
        std::memcpy(&corner.x, out.data() + i + value.offset, sizeof(float));
        std::memcpy(&corner.y, out.data() + i + value.offset + 4, sizeof(float));
      } else if (value.value == glsl::CornerValue::OtherPos) {
        std::memcpy(&corner.other, out.data() + i + value.offset, sizeof(float));
      }
    }
    corners.push_back(corner);
  }
  return corners;
}

void expect_corners(const std::vector<Corner>& got, const std::vector<Corner>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].end, want[i].end) << i;
    EXPECT_EQ(got[i].other, want[i].other) << i;
    EXPECT_EQ(got[i].x, want[i].x) << i;
    EXPECT_EQ(got[i].y, want[i].y) << i;
  }
}

// WebGPU's corners 0, 1, 3, 3, 2, 0: 2 and 3 at the second end, odd ones on the side the texture offset moves. Seen
// from the second end, that side is the other one across.
TEST(Corners, AreEachSegmentsEndsWithTheOtherEnd) {
  const std::vector<Corner> segment01{{0, 1, -1, 0}, {0, 1, 1, 1}, {1, 0, -1, 1},
                                      {1, 0, -1, 1}, {1, 0, 1, 0}, {0, 1, -1, 0}};
  auto twoSegments = segment01;
  for (const auto& corner : segment01) {
    twoSegments.push_back({corner.end + 2, corner.other + 2, corner.x, corner.y});
  }
  expect_corners(corners_of(lines(1, false), 4), twoSegments);
  auto strip = segment01;
  for (const auto& corner : segment01) {
    strip.push_back({corner.end + 1, corner.other + 1, corner.x, corner.y});
  }
  expect_corners(corners_of(lines(2, false), 3), strip);
}

TEST(Corners, AreEachPointsFourCorners) {
  expect_corners(corners_of(lines(3, false), 1),
                 {{0, -1, -1, -1}, {0, -1, 1, -1}, {0, -1, 1, 1}, {0, -1, 1, 1}, {0, -1, -1, 1}, {0, -1, -1, -1}});
}

// Milestone 18, step 1: TEV stage 0 sampling map 0 at texture coordinate `coord`, times the rasterized colour, with
// TEX0 and, when `nbt`, an NBT normal (normal, binormal, tangent), and the texgens given.
gx::ShaderConfig texgens(std::initializer_list<std::pair<u32, gx::TcgConfig>> tcgs, u32 coord, bool nbt = false) {
  auto config = rasterized();
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 12};
  config.vtxStride = 20;
  if (nbt) {
    config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 9, .compType = GX_F32, .offset = 20};
    config.vtxStride = 56;
  }
  for (const auto& [i, tcg] : tcgs) {
    config.tcgs[i] = tcg;
  }
  auto& stage = config.tevStages[0];
  stage.texCoordId = static_cast<GXTexCoordID>(coord);
  stage.texMapId = GX_TEXMAP0;
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  return config;
}

// Texture coordinate 1 embossed from texture coordinate 0, TEX0 as it is, by light 2.
gx::ShaderConfig embossed() {
  return texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_IDENTITY}},
                  {1, {.type = GX_TG_BUMP2, .src = GX_TG_TEXCOORD0, .embossSrc = 0}}},
                 1, true);
}

// A texgen from TEXCOORD0 takes texgen 0's source, TEX0 (gx/shader.cpp): written once texgen 0 is, or as it would be.
gx::ShaderConfig prior(bool sampled) {
  auto config = texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_TEXMTX3}},
                         {1, {.type = GX_TG_MTX2x4, .src = GX_TG_TEXCOORD0, .mtx = GX_TEXMTX2}}},
                        sampled ? 0 : 1);
  if (sampled) {
    // Stage 0 samples coordinate 0 and stage 1 coordinate 1 times it.
    config.tevStageCount = 2;
    config.tevStages[0].colorPass = {GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO, GX_CC_TEXC};
    auto& stage = config.tevStages[1];
    stage = config.tevStages[0];
    stage.texCoordId = GX_TEXCOORD1;
    stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_CPREV, GX_CC_ZERO};
  }
  return config;
}

TEST(Texgens, EmbossReadsTheBinormalAndTangentAfterTheNormal) {
  const auto programs = built(embossed());
  ASSERT_FALSE(programs.vertex.empty());
  std::vector<glsl::NbtSlice> slices;
  for (const auto& attribute : programs.layout.attributes) {
    if (attribute.attr == GX_VA_NRM) {
      slices.push_back(attribute.slice);
    }
  }
  EXPECT_EQ(slices, (std::vector{glsl::NbtSlice::Binormal, glsl::NbtSlice::Tangent}));
  EXPECT_TRUE(has(programs.vertex, "in vec3 in_binrm;"));
  EXPECT_TRUE(has(programs.vertex, "in vec3 in_tangent;"));
  EXPECT_TRUE(has(programs.vertex, "normalize(ulights.lights[10].xyz - mv_pos4.xyz)"));
  EXPECT_TRUE(has(programs.vertex, "upush.nrm_mtx"));
}

TEST(Texgens, ReadZeroForWhatTheVertexLacks) {
  // No binormals without an NBT normal, no position, and no TEX1.
  auto config = texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_BINRM, .mtx = GX_TEXMTX3}},
                         {1, {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1, .mtx = GX_TEXMTX3}}},
                        0);
  config.attrs[GX_VA_POS] = {};
  config.tevStageCount = 2;
  config.tevStages[1] = config.tevStages[0];
  config.tevStages[1].texCoordId = GX_TEXCOORD1;
  const auto programs = built(config);
  ASSERT_FALSE(programs.vertex.empty());
  EXPECT_TRUE(has(programs.vertex, "vec4 pos4 = vec4(vec3(0.0, 0.0, 0.0), 1.0);"));
  EXPECT_TRUE(has(programs.vertex, "vec4 tc0 = vec4(vec3(0.0, 0.0, 0.0), 1.0);"));
  EXPECT_TRUE(has(programs.vertex, "vec4 tc1 = vec4(vec2(0.0, 0.0), 1.0, 1.0);"));
  EXPECT_FALSE(has(programs.vertex, "in_binrm"));
  EXPECT_FALSE(has(programs.vertex, "in_tex1_uv"));
}

TEST(Texgens, TakeAPriorTexgensSource) {
  EXPECT_TRUE(has(built(prior(true)).vertex, "vec4 tc1 = tc0;"));
  EXPECT_TRUE(has(built(prior(false)).vertex, "vec4 tc1 = vec4(in_tex0_uv, 1.0, 1.0);"));
}

TEST(Texgens, FitElevenInterpolants) {
  // Both colours, eight 3x4 texgens and the fog: GX's most.
  auto config = rasterized();
  config.attrs[GX_VA_TEX0] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32, .offset = 12};
  config.vtxStride = 20;
  config.fogType = GX_FOG_PERSP_LIN;
  config.tevStageCount = 8;
  for (u32 i = 0; i < 8; ++i) {
    config.tcgs[i] = {.type = GX_TG_MTX3x4, .src = GX_TG_TEX0, .mtx = GX_TEXMTX3};
    auto& stage = config.tevStages[i];
    stage = config.tevStages[0];
    stage.texCoordId = static_cast<GXTexCoordID>(i);
    stage.texMapId = GX_TEXMAP0;
    stage.channelId = i == 1 ? GX_COLOR1A1 : GX_COLOR0A0;
    stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, i == 0 ? GX_CC_ZERO : GX_CC_CPREV};
  }
  const auto programs = built(config);
  EXPECT_FALSE(programs.vertex.empty());
  EXPECT_TRUE(has(programs.vertex, "out vec2 out_fog_zw;"));
  EXPECT_TRUE(has(programs.vertex, "out vec3 out_tex7_uvw;"));
  EXPECT_TRUE(has(programs.vertex, "out vec4 out_cc1;"));
}

TEST(MathCases, AreWritten) {
  named("lit_spot_clamp", lit(GX_DF_CLAMP, GX_AF_SPOT, 0x05));
  named("lit_spec_sign_alpha", lit(GX_DF_SIGN, GX_AF_SPEC, 0x06, true));
  named("lit_spot_ambient_vtx", lit(GX_DF_CLAMP, GX_AF_SPOT, 0xfe, false, GX_SRC_VTX));
  named("lit_none", lit(GX_DF_NONE, GX_AF_NONE, 0x81));
  {
    auto config = lit(GX_DF_CLAMP, GX_AF_SPOT, 0xff);
    config.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8, .offset = 24};
    config.vtxStride = 25;
    named("pnmtx_lit", config);
  }
  named("texmtx_fixed", textured(false));
  named("texmtx_picked", textured(true));
  named("fog_lin_range", fogged(GX_FOG_PERSP_LIN, true));
  named("fog_revexp2", fogged(GX_FOG_PERSP_REVEXP2, false));
  named("fog_ortho_exp", fogged(GX_FOG_ORTHO_EXP, false));
  {
    auto config = indirect(GX_ITM_0, GX_ITF_8, GX_ITB_STU);
    config.tevStages[0].indTexWrapS = GX_ITW_64;
    named("indirect_mtx", config);
  }
  named("indirect_dynamic_s", indirect(GX_ITM_S1, GX_ITF_5, GX_ITB_T));
  {
    auto config = indirect(GX_ITM_0, GX_ITF_5);
    config.tevStages[0].channelId = GX_ALPHA_BUMPN;
    config.tevStages[0].indTexAlphaSel = GX_ITBA_T;
    config.tevStages[0].alphaPass = {GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO, GX_CA_RASA};
    named("indirect_bump", config);
  }
  named("scene", scene());
  named("line_current", lines(1, false));
  named("line_picked", lines(2, true));
  named("points", lines(3, false));
  named("emboss", embossed());
  named("texgen_binormal", texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_BINRM, .mtx = GX_TEXMTX3}}}, 0, true));
  named("texgen_tangent", texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_TANGENT, .mtx = GX_TEXMTX4}}}, 0, true));
  named("texgen_prior", prior(true));
  named("texgen_prior_inline", prior(false));
  // A 3x4 texgen of the position, normalized: at a position of 0, nothing to normalize and a q of 0.
  named("texgen_zero",
        texgens({{0, {.type = GX_TG_MTX3x4, .src = GX_TG_POS, .mtx = GX_IDENTITY, .normalize = true}}}, 0));
  // A 3x4 texgen through post-transform matrix 0, whose q check_glsl_math.py's matrix makes 0.
  named("texgen_q_zero",
        texgens({{0, {.type = GX_TG_MTX3x4, .src = GX_TG_TEX0, .mtx = GX_IDENTITY, .postMtx = GX_PTTEXMTX0}}}, 0));
  {
    // No position, and a texgen from TEX1, which the vertex lacks.
    auto config = texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1, .mtx = GX_TEXMTX3}}}, 0);
    config.attrs[GX_VA_POS] = {};
    named("attr_defaults", config);
  }
  {
    // Milestone 18, step 2: TEVREG0 as stage 0's a, set past [0, 1] and written by no stage before, so wrapped.
    auto config = rasterized();
    config.tevStages[0].colorPass = {GX_CC_C0, GX_CC_ZERO, GX_CC_ZERO, GX_CC_ZERO};
    config.tevStages[0].alphaPass = {GX_CA_A0, GX_CA_ZERO, GX_CA_ZERO, GX_CA_ZERO};
    named("tev_reg_wrap", config);
  }
  named_programs("depth_z8", glsl::copy_vertex(), glsl::copy_fragment(GX_TF_Z8, false));
  named_programs("depth_z16", glsl::copy_vertex(), glsl::copy_fragment(GX_TF_Z16, false));
  named_programs("depth_peek", glsl::copy_vertex(), glsl::depth_peek_fragment());
}

// Milestone 18's console check (vulkan4xenon's tests/test_aurora_points.c, from tools/gen_aurora_points.py): step 1's
// math cases as GX points, so each interpolant holds the one vertex's value over the point, written as
// point_<name>.vert and .frag when AURORA_GLSL_CONSOLE_DIR is set.
TEST(ConsoleCases, AreWritten) {
  const char* dir = std::getenv("AURORA_GLSL_CONSOLE_DIR");
  const auto point = [&](std::string_view name, gx::ShaderConfig config) {
    config.lineMode = 3;
    const auto programs = built(config);
    ASSERT_FALSE(programs.vertex.empty()) << name;
    for (const auto& [suffix, text] :
         {std::pair{".vert"sv, &programs.vertex}, std::pair{".frag"sv, &programs.fragment}}) {
      if (dir == nullptr) {
        continue;
      }
      const auto path = fmt::format("{}/point_{}{}", dir, name, suffix);
      std::FILE* file = std::fopen(path.c_str(), "wb");
      ASSERT_NE(file, nullptr) << path;
      std::fwrite(text->data(), 1, text->size(), file);
      std::fclose(file);
    }
  };
  point("emboss", embossed());
  point("texgen_binormal", texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_BINRM, .mtx = GX_TEXMTX3}}}, 0, true));
  point("texgen_tangent", texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_TANGENT, .mtx = GX_TEXMTX4}}}, 0, true));
  point("texgen_prior", prior(true));
  point("texgen_prior_inline", prior(false));
  point("texgen_zero",
        texgens({{0, {.type = GX_TG_MTX3x4, .src = GX_TG_POS, .mtx = GX_IDENTITY, .normalize = true}}}, 0));
  point("texgen_q_zero",
        texgens({{0, {.type = GX_TG_MTX3x4, .src = GX_TG_TEX0, .mtx = GX_IDENTITY, .postMtx = GX_PTTEXMTX0}}}, 0));
  auto defaults = texgens({{0, {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1, .mtx = GX_TEXMTX3}}}, 0);
  defaults.attrs[GX_VA_POS] = {};
  point("attr_defaults", defaults);
}

// Depth copies to Z8 and Z16, and GXPeekZ's snapshot, take GX's z from the sampled depth's red, reversed, as
// tex_copy_conv.cpp's gx_z24 does; the other depth formats aren't converted. Their numbers are check_glsl_math.py's.
TEST(DepthPrograms, TakeGXsZFromTheSampledDepth) {
  for (const auto format : {GX_TF_Z8, GX_TF_Z16}) {
    EXPECT_TRUE(has(glsl::copy_fragment(format, false), "clamp(1.0 - texture(source, in_uv).rrrr, 0.0, 1.0)"));
    EXPECT_TRUE(glsl::converts(format));
  }
  EXPECT_TRUE(has(glsl::copy_fragment(GX_TF_Z8, false), "o_color = high * (1.0 / 255.0);"));
  EXPECT_TRUE(has(glsl::copy_fragment(GX_TF_Z16, false), "o_color = vec4(high.xyz, middle.x) * (1.0 / 255.0);"));
  EXPECT_TRUE(has(glsl::depth_peek_fragment(), "o_color = vec4(high.x, middle.x, low.x, 0.0) * (1.0 / 255.0);"));
  for (const auto format : {GX_TF_Z24X8, GX_CTF_Z4, GX_CTF_Z8M, GX_CTF_Z8L, GX_CTF_Z16L}) {
    EXPECT_FALSE(glsl::converts(format)) << format;
    EXPECT_TRUE(glsl::copy_fragment(format, false).empty()) << format;
  }
  const auto programs = glsl::fixed_programs();
  for (const auto& fragment :
       {glsl::copy_fragment(GX_TF_Z8, false), glsl::copy_fragment(GX_TF_Z16, false), glsl::depth_peek_fragment()}) {
    EXPECT_NE(std::ranges::find(programs, std::pair{fragment, true}), programs.end());
  }
}

// An EFB copy's index, 8 or 4 bits, scaled back from the copy's texel and looked up halfway into its palette entry.
TEST(PaletteConversion, ScalesTheIndexByItsBitsAndReadsTheEntrysMiddle) {
  EXPECT_TRUE(has(glsl::palette_fragment(false), "floor(texture(source, in_uv) * 255.0 + 0.5)"));
  EXPECT_TRUE(has(glsl::palette_fragment(true), "floor(texture(source, in_uv) * 15.0 + 0.5)"));
  EXPECT_TRUE(has(glsl::palette_fragment(false), "index.xx * upush.palette_scale.xy + upush.palette_scale.zw"));
  const auto programs = glsl::fixed_programs();
  for (const bool fourBit : {false, true}) {
    EXPECT_NE(std::ranges::find(programs, std::pair{glsl::palette_fragment(fourBit), true}), programs.end());
  }
}
} // namespace
