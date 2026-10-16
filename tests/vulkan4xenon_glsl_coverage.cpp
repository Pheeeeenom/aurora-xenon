// The vulkan4xenon backend's GLSL generator over GX's configs (vulkan4xenon's ROADMAP.md, milestone 18): every value of
// each ShaderConfig field the WGSL reads, one at a time on a plain config, then random configs, each of which must get
// programs. Only a config whose alpha compare fails every fragment may get none, as nothing of it is drawn. With
// AURORA_GLSL_COVERAGE_DIR set, each distinct program is written there under its hash, with manifest.txt naming each
// case's two, for tools/check_glsl_coverage.py to compile in glslang and lower in xost.
#include "gx/gx.hpp"
#include "vulkan4xenon/glsl.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {
namespace gx = aurora::gx;
namespace glsl = aurora::gfx::vulkan4xenon::glsl;

// GX's values for each field, from GXEnum.h.
constexpr std::array ColorArgs{GX_CC_CPREV, GX_CC_APREV, GX_CC_C0,    GX_CC_A0,   GX_CC_C1,   GX_CC_A1,
                               GX_CC_C2,    GX_CC_A2,    GX_CC_TEXC,  GX_CC_TEXA, GX_CC_RASC, GX_CC_RASA,
                               GX_CC_ONE,   GX_CC_HALF,  GX_CC_KONST, GX_CC_ZERO};
constexpr std::array AlphaArgs{GX_CA_APREV, GX_CA_A0,   GX_CA_A1,    GX_CA_A2,
                               GX_CA_TEXA,  GX_CA_RASA, GX_CA_KONST, GX_CA_ZERO};
constexpr std::array ColorOps{GX_TEV_ADD,          GX_TEV_SUB,          GX_TEV_COMP_R8_GT,    GX_TEV_COMP_R8_EQ,
                              GX_TEV_COMP_GR16_GT, GX_TEV_COMP_GR16_EQ, GX_TEV_COMP_BGR24_GT, GX_TEV_COMP_BGR24_EQ,
                              GX_TEV_COMP_RGB8_GT, GX_TEV_COMP_RGB8_EQ};
constexpr std::array Biases{GX_TB_ZERO, GX_TB_ADDHALF, GX_TB_SUBHALF};
constexpr std::array Scales{GX_CS_SCALE_1, GX_CS_SCALE_2, GX_CS_SCALE_4, GX_CS_DIVIDE_2};
constexpr std::array Regs{GX_TEVPREV, GX_TEVREG0, GX_TEVREG1, GX_TEVREG2};
constexpr std::array Channels{GX_COLOR0A0, GX_COLOR1A1, GX_COLOR_ZERO, GX_ALPHA_BUMP, GX_ALPHA_BUMPN, GX_COLOR_NULL};
constexpr std::array Compares{GX_NEVER, GX_LESS, GX_EQUAL, GX_LEQUAL, GX_GREATER, GX_NEQUAL, GX_GEQUAL, GX_ALWAYS};
constexpr std::array AlphaOps{GX_AOP_AND, GX_AOP_OR, GX_AOP_XOR, GX_AOP_XNOR};
constexpr std::array FogTypes{GX_FOG_NONE,         GX_FOG_PERSP_LIN,     GX_FOG_PERSP_EXP,    GX_FOG_PERSP_EXP2,
                              GX_FOG_PERSP_REVEXP, GX_FOG_PERSP_REVEXP2, GX_FOG_ORTHO_LIN,    GX_FOG_ORTHO_EXP,
                              GX_FOG_ORTHO_EXP2,   GX_FOG_ORTHO_REVEXP,  GX_FOG_ORTHO_REVEXP2};
constexpr std::array IndMtxs{GX_ITM_OFF, GX_ITM_0,  GX_ITM_1,  GX_ITM_2,  GX_ITM_S0,
                             GX_ITM_S1,  GX_ITM_S2, GX_ITM_T0, GX_ITM_T1, GX_ITM_T2};

std::vector<u32> kcsels() {
  std::vector<u32> values;
  for (u32 v = 0; v <= 0x07; ++v) {
    values.push_back(v);
  }
  for (u32 v = 0x0C; v <= 0x1F; ++v) {
    values.push_back(v);
  }
  return values;
}

std::vector<u32> kasels() {
  std::vector<u32> values;
  for (u32 v = 0; v <= 0x07; ++v) {
    values.push_back(v);
  }
  for (u32 v = 0x10; v <= 0x1F; ++v) {
    values.push_back(v);
  }
  return values;
}

// A texgen's matrix: a position matrix, a texture matrix or the identity.
std::vector<u32> texgen_matrices() {
  std::vector<u32> values;
  for (u32 v = 0; v <= GX_TEXMTX9; v += 3) {
    values.push_back(v);
  }
  values.push_back(GX_IDENTITY);
  return values;
}

std::vector<u32> post_matrices() {
  std::vector<u32> values;
  for (u32 v = GX_PTTEXMTX0; v <= GX_PTTEXMTX19; v += 3) {
    values.push_back(v);
  }
  values.push_back(GX_PTIDENTITY);
  return values;
}

// Every vertex attribute present, the normal an NBT one, and texgen 0, TEX0 as it is, sampled by stage 0 with map 0
// times the rasterized colour 0.
gx::ShaderConfig plain() {
  gx::ShaderConfig config{};
  config.attrs[GX_VA_POS] = {.attrType = GX_DIRECT, .cnt = 3, .compType = GX_F32};
  config.attrs[GX_VA_NRM] = {.attrType = GX_DIRECT, .cnt = 9, .compType = GX_F32};
  config.attrs[GX_VA_CLR0] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  config.attrs[GX_VA_CLR1] = {.attrType = GX_DIRECT, .cnt = 4, .compType = GX_RGBA8};
  for (u32 attr = GX_VA_TEX0; attr <= GX_VA_TEX7; ++attr) {
    config.attrs[attr] = {.attrType = GX_DIRECT, .cnt = 2, .compType = GX_F32};
  }
  config.tevStageCount = 1;
  config.tcgs[0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX0, .mtx = GX_IDENTITY};
  auto& stage = config.tevStages[0];
  stage.colorPass = {GX_CC_ZERO, GX_CC_TEXC, GX_CC_RASC, GX_CC_ZERO};
  stage.alphaPass = {GX_CA_ZERO, GX_CA_TEXA, GX_CA_RASA, GX_CA_ZERO};
  stage.texCoordId = GX_TEXCOORD0;
  stage.texMapId = GX_TEXMAP0;
  stage.channelId = GX_COLOR0A0;
  return config;
}

bool compare(GXCompare comp, u32 a, u32 ref) {
  switch (comp) {
  case GX_NEVER:
    return false;
  case GX_LESS:
    return a < ref;
  case GX_EQUAL:
    return a == ref;
  case GX_LEQUAL:
    return a <= ref;
  case GX_GREATER:
    return a > ref;
  case GX_NEQUAL:
    return a != ref;
  case GX_GEQUAL:
    return a >= ref;
  default:
    return true;
  }
}

// Whether the alpha compare fails at every alpha GX can have.
bool fails_every_fragment(const gx::AlphaCompare& ac) {
  for (u32 a = 0; a < 256; ++a) {
    const bool c0 = compare(ac.comp0, a, ac.ref0);
    const bool c1 = compare(ac.comp1, a, ac.ref1);
    const bool pass = ac.op == GX_AOP_AND   ? c0 && c1
                      : ac.op == GX_AOP_OR  ? c0 || c1
                      : ac.op == GX_AOP_XOR ? c0 != c1
                                            : c0 == c1;
    if (pass) {
      return false;
    }
  }
  return true;
}

struct Case {
  std::string name;
  gx::ShaderConfig config;
};

// Each field's values on the plain config, one at a time. A stage that samples keeps its coordinates and map.
std::vector<Case> sweep() {
  std::vector<Case> cases;
  const auto add = [&](std::string name, const std::function<void(gx::ShaderConfig&)>& change) {
    auto config = plain();
    change(config);
    cases.push_back({std::move(name), config});
  };
  for (u32 attr = GX_VA_PNMTXIDX; attr <= GX_VA_TEX7; ++attr) {
    for (u32 type = GX_NONE; type <= GX_INDEX16; ++type) {
      add(fmt::format("attrs[{}].attrType={}", attr, type), [&](gx::ShaderConfig& c) {
        c.attrs[attr].attrType = type;
        if (type != GX_NONE && c.attrs[attr].cnt == 0xFF) {
          c.attrs[attr].cnt = 1;
          c.attrs[attr].compType = GX_U8;
        }
        if (attr >= GX_VA_TEX0 && attr <= GX_VA_TEX7) {
          c.tcgs[0].src = static_cast<GXTexGenSrc>(GX_TG_TEX0 + (attr - GX_VA_TEX0));
        }
        if (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX) {
          c.tcgs[0].mtx = GX_TEXMTX2;
        }
      });
    }
  }
  for (u32 cnt : {3u, 9u}) {
    add(fmt::format("attrs[NRM].cnt={}", cnt), [&](gx::ShaderConfig& c) {
      c.attrs[GX_VA_NRM].cnt = cnt;
      c.tcgs[0].src = GX_TG_BINRM;
    });
  }
  for (u32 i = 0; i < gx::MaxTexCoord; ++i) {
    const auto at = [i](gx::ShaderConfig& c) {
      c.tcgs[0] = {};
      c.tevStages[0].texCoordId = static_cast<GXTexCoordID>(i);
    };
    for (u32 type = GX_TG_MTX3x4; type <= GX_TG_SRTG; ++type) {
      for (u32 src = GX_TG_POS; src <= GX_TG_COLOR1; ++src) {
        add(fmt::format("tcgs[{}].type={}.src={}", i, type, src), [&](gx::ShaderConfig& c) {
          at(c);
          c.tcgs[i] = {.type = static_cast<GXTexGenType>(type),
                       .src = static_cast<GXTexGenSrc>(src),
                       .mtx = GX_TEXMTX1,
                       .embossSrc = static_cast<u8>(i == 0 ? 1 : 0)};
          // The emboss source's own texgen.
          c.tcgs[i == 0 ? 1 : 0] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1, .mtx = GX_IDENTITY};
        });
      }
    }
    for (u32 mtx : texgen_matrices()) {
      for (u32 type : {GX_TG_MTX2x4, GX_TG_MTX3x4}) {
        add(fmt::format("tcgs[{}].mtx={}.type={}", i, mtx, type), [&](gx::ShaderConfig& c) {
          at(c);
          c.tcgs[i] = {.type = static_cast<GXTexGenType>(type), .src = GX_TG_TEX0, .mtx = static_cast<GXTexMtx>(mtx)};
        });
      }
    }
    for (u32 post : post_matrices()) {
      for (bool normalize : {false, true}) {
        add(fmt::format("tcgs[{}].postMtx={}.normalize={}", i, post, normalize), [&](gx::ShaderConfig& c) {
          at(c);
          c.tcgs[i] = {.type = GX_TG_MTX3x4,
                       .src = GX_TG_POS,
                       .mtx = GX_TEXMTX0,
                       .postMtx = static_cast<GXPTTexMtx>(post),
                       .normalize = normalize};
        });
      }
    }
    for (u32 src = 0; src < gx::MaxTexCoord; ++src) {
      add(fmt::format("tcgs[{}].embossSrc={}", i, src), [&](gx::ShaderConfig& c) {
        at(c);
        c.tcgs[src] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX2, .mtx = GX_TEXMTX4};
        c.tcgs[i] = {.type = GX_TG_BUMP5,
                     .src = static_cast<GXTexGenSrc>(GX_TG_TEXCOORD0 + std::min(src, 6u)),
                     .embossSrc = static_cast<u8>(src)};
      });
    }
  }
  for (u32 slot = 0; slot < 4; ++slot) {
    for (auto arg : ColorArgs) {
      add(fmt::format("tevStages[0].colorPass[{}]={}", slot, static_cast<int>(arg)),
          [&](gx::ShaderConfig& c) { (&c.tevStages[0].colorPass.a)[slot] = arg; });
    }
    for (auto arg : AlphaArgs) {
      add(fmt::format("tevStages[0].alphaPass[{}]={}", slot, static_cast<int>(arg)),
          [&](gx::ShaderConfig& c) { (&c.tevStages[0].alphaPass.a)[slot] = arg; });
    }
  }
  for (bool alpha : {false, true}) {
    const auto op = [alpha](gx::ShaderConfig& c) -> gx::TevOp& {
      return alpha ? c.tevStages[0].alphaOp : c.tevStages[0].colorOp;
    };
    const auto name = alpha ? "alphaOp" : "colorOp";
    for (auto v : ColorOps) {
      for (auto bias : Biases) {
        add(fmt::format("tevStages[0].{}.op={}.bias={}", name, static_cast<int>(v), static_cast<int>(bias)),
            [&](gx::ShaderConfig& c) {
              op(c).op = v;
              op(c).bias = bias;
            });
      }
    }
    for (auto v : Scales) {
      for (bool clamp : {false, true}) {
        add(fmt::format("tevStages[0].{}.scale={}.clamp={}", name, static_cast<int>(v), clamp),
            [&](gx::ShaderConfig& c) {
              op(c).scale = v;
              op(c).clamp = clamp;
            });
      }
    }
    for (auto v : Regs) {
      add(fmt::format("tevStages[0].{}.outReg={}", name, static_cast<int>(v)),
          [&](gx::ShaderConfig& c) { op(c).outReg = v; });
    }
  }
  for (u32 v : kcsels()) {
    add(fmt::format("tevStages[0].kcSel={}", v), [&](gx::ShaderConfig& c) {
      c.tevStages[0].kcSel = static_cast<GXTevKColorSel>(v);
      c.tevStages[0].colorPass.a = GX_CC_KONST;
    });
  }
  for (u32 v : kasels()) {
    add(fmt::format("tevStages[0].kaSel={}", v), [&](gx::ShaderConfig& c) {
      c.tevStages[0].kaSel = static_cast<GXTevKAlphaSel>(v);
      c.tevStages[0].alphaPass.a = GX_CA_KONST;
    });
  }
  for (u32 map = 0; map <= gx::MaxTextures; ++map) {
    add(fmt::format("tevStages[0].texMapId={}", map), [&](gx::ShaderConfig& c) {
      c.tevStages[0].texMapId = map == gx::MaxTextures ? GX_TEXMAP_NULL : static_cast<GXTexMapID>(map);
      if (map == gx::MaxTextures) {
        c.tevStages[0].texCoordId = GX_TEXCOORD_NULL;
      }
    });
  }
  for (auto channel : Channels) {
    add(fmt::format("tevStages[0].channelId={}", static_cast<int>(channel)),
        [&](gx::ShaderConfig& c) { c.tevStages[0].channelId = channel; });
  }
  for (u32 sel = 0; sel < 4; ++sel) {
    add(fmt::format("tevStages[0].tevSwapRas={}", sel),
        [&](gx::ShaderConfig& c) { c.tevStages[0].tevSwapRas = static_cast<GXTevSwapSel>(sel); });
    add(fmt::format("tevStages[0].tevSwapTex={}", sel),
        [&](gx::ShaderConfig& c) { c.tevStages[0].tevSwapTex = static_cast<GXTevSwapSel>(sel); });
    for (u32 comp = 0; comp < 4; ++comp) {
      for (u32 chan = 0; chan < 4; ++chan) {
        add(fmt::format("tevSwapTable[{}][{}]={}", sel, comp, chan), [&](gx::ShaderConfig& c) {
          (&c.tevSwapTable[sel].red)[comp] = static_cast<GXTevColorChan>(chan);
          c.tevStages[0].tevSwapRas = static_cast<GXTevSwapSel>(sel);
          c.tevStages[0].tevSwapTex = static_cast<GXTevSwapSel>(sel);
        });
      }
    }
  }
  // Indirect texturing: stage 0 offset by indirect stage `ind`'s sample of map 1 at texcoord 1.
  const auto indirect = [](gx::ShaderConfig& c, u32 ind) {
    c.numIndStages = ind + 1;
    for (u32 k = 0; k <= ind; ++k) {
      c.indStages[k] = {GX_TEXCOORD1, GX_TEXMAP1, GX_ITS_2, GX_ITS_4};
    }
    c.tcgs[1] = {.type = GX_TG_MTX2x4, .src = GX_TG_TEX1, .mtx = GX_IDENTITY};
    c.tevStages[0].indTexStage = static_cast<GXIndTexStageID>(ind);
    c.tevStages[0].indTexMtxId = GX_ITM_0;
  };
  for (u32 ind = 0; ind < gx::MaxIndStages; ++ind) {
    for (u32 s = 0; s <= GX_ITS_256; ++s) {
      add(fmt::format("indStages[{}].scale={}", ind, s), [&](gx::ShaderConfig& c) {
        indirect(c, ind);
        c.indStages[ind].scaleS = static_cast<GXIndTexScale>(s);
        c.indStages[ind].scaleT = static_cast<GXIndTexScale>(GX_ITS_256 - s);
      });
    }
  }
  for (u32 v = 0; v <= GX_ITF_3; ++v) {
    add(fmt::format("tevStages[0].indTexFormat={}", v), [&](gx::ShaderConfig& c) {
      indirect(c, 0);
      c.tevStages[0].indTexFormat = static_cast<GXIndTexFormat>(v);
    });
  }
  for (u32 v = 0; v <= GX_ITB_STU; ++v) {
    add(fmt::format("tevStages[0].indTexBiasSel={}", v), [&](gx::ShaderConfig& c) {
      indirect(c, 0);
      c.tevStages[0].indTexBiasSel = static_cast<GXIndTexBiasSel>(v);
    });
  }
  for (u32 v = 0; v <= GX_ITBA_U; ++v) {
    for (auto channel : {GX_ALPHA_BUMP, GX_ALPHA_BUMPN}) {
      add(fmt::format("tevStages[0].indTexAlphaSel={}.channelId={}", v, static_cast<int>(channel)),
          [&](gx::ShaderConfig& c) {
            indirect(c, 0);
            c.tevStages[0].indTexAlphaSel = static_cast<GXIndTexAlphaSel>(v);
            c.tevStages[0].channelId = channel;
          });
    }
  }
  for (auto v : IndMtxs) {
    add(fmt::format("tevStages[0].indTexMtxId={}", static_cast<int>(v)), [&](gx::ShaderConfig& c) {
      indirect(c, 0);
      c.tevStages[0].indTexMtxId = v;
    });
  }
  for (u32 v = 0; v <= GX_ITW_0; ++v) {
    add(fmt::format("tevStages[0].indTexWrap={}", v), [&](gx::ShaderConfig& c) {
      indirect(c, 0);
      c.tevStages[0].indTexWrapS = static_cast<GXIndTexWrap>(v);
      c.tevStages[0].indTexWrapT = static_cast<GXIndTexWrap>(GX_ITW_0 - v);
    });
  }
  for (bool addPrev : {false, true}) {
    for (bool origLod : {false, true}) {
      add(fmt::format("tevStages[0].indTexAddPrev={}.indTexUseOrigLOD={}", addPrev, origLod), [&](gx::ShaderConfig& c) {
        indirect(c, 0);
        c.tevStages[0].indTexAddPrev = addPrev;
        c.tevStages[0].indTexUseOrigLOD = origLod;
      });
    }
  }
  for (u32 count = 1; count <= gx::MaxTevStages; ++count) {
    add(fmt::format("tevStageCount={}", count), [&](gx::ShaderConfig& c) {
      c.tevStageCount = count;
      for (u32 s = 1; s < count; ++s) {
        c.tevStages[s] = c.tevStages[0];
        c.tevStages[s].colorPass.d = GX_CC_CPREV;
        c.tevStages[s].alphaPass.d = GX_CA_APREV;
      }
    });
  }
  for (u32 ch = 0; ch < gx::MaxColorChannels; ++ch) {
    const auto channel = [ch](gx::ShaderConfig& c) {
      c.tevStages[0].channelId = ch % 2 == 0 ? GX_COLOR0A0 : GX_COLOR1A1;
    };
    for (u32 diff = GX_DF_NONE; diff <= GX_DF_CLAMP; ++diff) {
      for (u32 attn = GX_AF_SPEC; attn <= GX_AF_NONE; ++attn) {
        for (u32 mask : {0x00u, 0x01u, 0x80u, 0xFFu}) {
          add(fmt::format("colorChannels[{}].diffFn={}.attnFn={}.lightMask={}", ch, diff, attn, mask),
              [&](gx::ShaderConfig& c) {
                channel(c);
                c.colorChannels[ch] = {.diffFn = static_cast<GXDiffuseFn>(diff),
                                       .attnFn = static_cast<GXAttnFn>(attn),
                                       .lightingEnabled = true,
                                       .lightMask = static_cast<u8>(mask)};
              });
        }
      }
    }
    for (u32 mat : {GX_SRC_REG, GX_SRC_VTX}) {
      for (u32 amb : {GX_SRC_REG, GX_SRC_VTX}) {
        for (bool lit : {false, true}) {
          add(fmt::format("colorChannels[{}].matSrc={}.ambSrc={}.lit={}", ch, mat, amb, lit), [&](gx::ShaderConfig& c) {
            channel(c);
            c.colorChannels[ch] = {.matSrc = static_cast<GXColorSrc>(mat),
                                   .ambSrc = static_cast<GXColorSrc>(amb),
                                   .diffFn = GX_DF_CLAMP,
                                   .attnFn = GX_AF_SPOT,
                                   .lightingEnabled = lit,
                                   .lightMask = 0x21};
          });
        }
      }
    }
  }
  for (auto comp0 : Compares) {
    for (auto comp1 : Compares) {
      for (auto op : AlphaOps) {
        for (u32 ref : {0u, 100u, 255u}) {
          add(fmt::format("alphaCompare={},{},{},{}", static_cast<int>(comp0), ref, static_cast<int>(op),
                          static_cast<int>(comp1)),
              [&](gx::ShaderConfig& c) { c.alphaCompare = {comp0, ref, op, comp1, 255 - ref}; });
        }
      }
    }
  }
  for (auto fog : FogTypes) {
    for (bool range : {false, true}) {
      add(fmt::format("fogType={}.fogRangeEnabled={}", static_cast<int>(fog), range), [&](gx::ShaderConfig& c) {
        c.fogType = fog;
        c.fogRangeEnabled = range;
      });
    }
  }
  for (u32 mode = 0; mode <= 3; ++mode) {
    for (bool perVertex : {false, true}) {
      add(fmt::format("lineMode={}.pnmtxidx={}", mode, perVertex), [&](gx::ShaderConfig& c) {
        c.lineMode = mode;
        if (perVertex) {
          c.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
        }
      });
    }
  }
  for (u32 op = GX_ZT_DISABLE; op <= GX_ZT_REPLACE; ++op) {
    for (u32 fmt = 0; fmt <= 2; ++fmt) {
      for (bool before : {false, true}) {
        add(fmt::format("zTexOp={}.zTexFmt={}.zCompLocBeforeTex={}", op, fmt, before), [&](gx::ShaderConfig& c) {
          c.zTexOp = op;
          c.zTexFmt = fmt;
          c.zCompLocBeforeTex = before;
          c.zTexBias = 0x123456;
        });
      }
    }
  }
  return cases;
}

// Random configs over the same values: stages, texgens, channels, indirect stages, the alpha compare, fog and lines.
std::vector<Case> random_configs(u32 count, u32 seed) {
  std::mt19937 rng(seed);
  const auto pick = [&](u32 n) { return static_cast<u32>(std::uniform_int_distribution<u32>(0, n - 1)(rng)); };
  const auto chance = [&](u32 percent) { return pick(100) < percent; };
  const auto kc = kcsels();
  const auto ka = kasels();
  const auto mtxs = texgen_matrices();
  const auto posts = post_matrices();
  std::vector<Case> cases;
  for (u32 n = 0; n < count; ++n) {
    gx::ShaderConfig c = plain();
    for (u32 attr = GX_VA_POS; attr <= GX_VA_TEX7; ++attr) {
      if (chance(15)) {
        c.attrs[attr] = {};
      }
    }
    if (chance(50)) {
      c.attrs[GX_VA_NRM].cnt = 3;
    }
    if (chance(20)) {
      c.attrs[GX_VA_PNMTXIDX] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
    }
    const u32 texgens = 1 + pick(gx::MaxTexCoord);
    for (u32 i = 0; i < gx::MaxTexCoord; ++i) {
      c.tcgs[i] = {};
      if (i >= texgens) {
        continue;
      }
      auto& t = c.tcgs[i];
      t.type = static_cast<GXTexGenType>(pick(GX_TG_SRTG + 1));
      t.src = static_cast<GXTexGenSrc>(pick(GX_TG_COLOR1 + 1));
      t.mtx = static_cast<GXTexMtx>(mtxs[pick(mtxs.size())]);
      t.postMtx = static_cast<GXPTTexMtx>(posts[pick(posts.size())]);
      t.normalize = chance(30);
      t.embossSrc = static_cast<u8>(pick(texgens));
      if (chance(15)) {
        c.attrs[GX_VA_TEX0MTXIDX + i] = {.attrType = GX_DIRECT, .cnt = 1, .compType = GX_U8};
      }
    }
    c.numIndStages = pick(gx::MaxIndStages + 1);
    for (u32 k = 0; k < c.numIndStages; ++k) {
      c.indStages[k] = {static_cast<GXTexCoordID>(pick(texgens)), static_cast<GXTexMapID>(pick(gx::MaxTextures)),
                        static_cast<GXIndTexScale>(pick(GX_ITS_256 + 1)),
                        static_cast<GXIndTexScale>(pick(GX_ITS_256 + 1))};
    }
    c.tevStageCount = 1 + pick(gx::MaxTevStages);
    for (u32 s = 0; s < c.tevStageCount; ++s) {
      auto& st = c.tevStages[s];
      st = {};
      st.colorPass = {ColorArgs[pick(ColorArgs.size())], ColorArgs[pick(ColorArgs.size())],
                      ColorArgs[pick(ColorArgs.size())], ColorArgs[pick(ColorArgs.size())]};
      st.alphaPass = {AlphaArgs[pick(AlphaArgs.size())], AlphaArgs[pick(AlphaArgs.size())],
                      AlphaArgs[pick(AlphaArgs.size())], AlphaArgs[pick(AlphaArgs.size())]};
      for (auto* op : {&st.colorOp, &st.alphaOp}) {
        *op = {ColorOps[pick(ColorOps.size())], Biases[pick(Biases.size())], Scales[pick(Scales.size())],
               Regs[pick(Regs.size())], chance(70)};
      }
      st.kcSel = static_cast<GXTevKColorSel>(kc[pick(kc.size())]);
      st.kaSel = static_cast<GXTevKAlphaSel>(ka[pick(ka.size())]);
      if (chance(80)) {
        st.texCoordId = static_cast<GXTexCoordID>(pick(texgens));
        st.texMapId = static_cast<GXTexMapID>(pick(gx::MaxTextures));
      }
      st.channelId = Channels[pick(Channels.size())];
      st.tevSwapRas = static_cast<GXTevSwapSel>(pick(4));
      st.tevSwapTex = static_cast<GXTevSwapSel>(pick(4));
      if (c.numIndStages != 0 && chance(40)) {
        st.indTexStage = static_cast<GXIndTexStageID>(pick(c.numIndStages));
        st.indTexFormat = static_cast<GXIndTexFormat>(pick(GX_ITF_3 + 1));
        st.indTexBiasSel = static_cast<GXIndTexBiasSel>(pick(GX_ITB_STU + 1));
        st.indTexAlphaSel = static_cast<GXIndTexAlphaSel>(pick(GX_ITBA_U + 1));
        st.indTexMtxId = IndMtxs[pick(IndMtxs.size())];
        st.indTexWrapS = static_cast<GXIndTexWrap>(pick(GX_ITW_0 + 1));
        st.indTexWrapT = static_cast<GXIndTexWrap>(pick(GX_ITW_0 + 1));
        st.indTexAddPrev = chance(30);
      }
    }
    for (u32 sel = 0; sel < 4; ++sel) {
      for (u32 comp = 0; comp < 4; ++comp) {
        (&c.tevSwapTable[sel].red)[comp] = static_cast<GXTevColorChan>(chance(70) ? comp : pick(4));
      }
    }
    for (auto& cc : c.colorChannels) {
      cc = {.matSrc = static_cast<GXColorSrc>(pick(2)),
            .ambSrc = static_cast<GXColorSrc>(pick(2)),
            .diffFn = static_cast<GXDiffuseFn>(pick(3)),
            .attnFn = static_cast<GXAttnFn>(pick(3)),
            .lightingEnabled = chance(50),
            .lightMask = static_cast<u8>(pick(256))};
    }
    c.alphaCompare = {Compares[pick(Compares.size())], pick(256), AlphaOps[pick(AlphaOps.size())],
                      Compares[pick(Compares.size())], pick(256)};
    c.fogType = FogTypes[pick(FogTypes.size())];
    c.fogRangeEnabled = chance(30);
    c.lineMode = chance(80) ? 0 : 1 + pick(3);
    c.zTexOp = pick(3);
    c.zTexFmt = pick(3);
    c.zCompLocBeforeTex = chance(50);
    cases.push_back({fmt::format("random {} of seed {}", n, seed), c});
  }
  return cases;
}

void write_text(const std::string& path, const std::string& text) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  ASSERT_NE(file, nullptr) << path;
  std::fwrite(text.data(), 1, text.size(), file);
  std::fclose(file);
}

TEST(Coverage, EveryConfigGetsPrograms) {
  const char* dir = std::getenv("AURORA_GLSL_COVERAGE_DIR");
  const char* countText = std::getenv("AURORA_GLSL_COVERAGE_RANDOM");
  auto cases = sweep();
  const u32 sweepCount = static_cast<u32>(cases.size());
  for (auto& c : random_configs(countText != nullptr ? static_cast<u32>(std::atoi(countText)) : 500, 18)) {
    cases.push_back(std::move(c));
  }
  std::map<uint64_t, bool> written;
  std::string manifest;
  u32 refused = 0;
  for (const auto& [name, config] : cases) {
    const auto programs = glsl::build_programs(config);
    // A compare that fails every pixel may still get programs, which discard every pixel as they run.
    if (programs.vertex.empty() && !fails_every_fragment(config.alphaCompare)) {
      ++refused;
      ADD_FAILURE() << name << ": no programs";
      continue;
    }
    if (dir == nullptr || programs.vertex.empty()) {
      continue;
    }
    std::array<uint64_t, 2> hashes{};
    for (u32 stage = 0; stage < 2; ++stage) {
      const auto& text = stage == 0 ? programs.vertex : programs.fragment;
      hashes[stage] = XXH3_64bits(text.data(), text.size());
      if (!written[hashes[stage]]) {
        written[hashes[stage]] = true;
        write_text(fmt::format("{}/{:016x}.{}", dir, hashes[stage], stage == 0 ? "vert" : "frag"), text);
      }
    }
    manifest += fmt::format("{:016x} {:016x} {}\n", hashes[0], hashes[1], name);
  }
  if (dir != nullptr) {
    write_text(fmt::format("{}/manifest.txt", dir), manifest);
  }
  std::printf("coverage: %zu configs (%u swept), %zu distinct programs, %u wrong\n", cases.size(), sweepCount,
              written.size(), refused);
}
} // namespace
