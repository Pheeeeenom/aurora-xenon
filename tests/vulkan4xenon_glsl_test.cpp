// The vulkan4xenon backend's GLSL generator (lib/vulkan4xenon/glsl.cpp), on configs made to hit what it writes.
// With AURORA_GLSL_TEST_DIR set, each config's programs are also recorded there, for glslang and xost to compile.
#include "gx/gx.hpp"
#include "vulkan4xenon/glsl.hpp"

#include <aurora/aurora.h>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>

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
void named(std::string_view name, const gx::ShaderConfig& config) {
  const auto programs = built(config);
  ASSERT_FALSE(programs.vertex.empty()) << name;
  ASSERT_FALSE(programs.fragment.empty()) << name;
  if (const char* dir = std::getenv("AURORA_GLSL_NAMED_DIR")) {
    for (const auto& [suffix, text] :
         {std::pair{".vert"sv, &programs.vertex}, std::pair{".frag"sv, &programs.fragment}}) {
      const auto path = fmt::format("{}/{}{}", dir, name, suffix);
      std::FILE* file = std::fopen(path.c_str(), "wb");
      ASSERT_NE(file, nullptr) << path;
      std::fwrite(text->data(), 1, text->size(), file);
      std::fclose(file);
    }
  }
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
}
} // namespace
