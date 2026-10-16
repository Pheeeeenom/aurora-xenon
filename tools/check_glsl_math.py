#!/usr/bin/env python3
"""Checks the vulkan4xenon backend's GLSL by its numbers: runs the programs Aurora's tests/vulkan4xenon_glsl_test.cpp
writes for its named configs (MathCases, AURORA_GLSL_NAMED_DIR) on xost's simulator, compiled as the game's pack is,
glslang to SPIR-V and xost to Xenos microcode, and compares what they compute with this tool's own reference of
lib/gx/shader.cpp's WGSL: lighting, fog with its range adjustment, matrices picked per vertex and indirect texturing
(vulkan4xenon's milestone 11, step 5), the corners of lines and points (milestone 14), GX's z in depth copies
and GXPeekZ's snapshot (milestone 14, step 5), and emboss, binormal, tangent and prior-texgen sources with the
zero guards and missing attributes (milestone 18, step 1).

    check_glsl_math.py --xost <xost tree> [--glslang <glslang>] <directory of named programs>

Each case runs its vertex program at one vertex, then its fragment program on what the vertex program wrote, at a
pixel. The uniforms are made up for the check; each piece is placed where xost put its block, member for member as
the program's GLSL declares it.
"""

import argparse
import math
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def single(x):
    return struct.unpack("<f", struct.pack("<f", x))[0] if math.isfinite(x) else x


# ---- vectors, as plain lists ----

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def scale(a, s):
    return [x * s for x in a]


def mul(a, b):
    return [x * y for x, y in zip(a, b)]


def clamp(x, lo=0.0, hi=1.0):
    return min(max(x, lo), hi)


def clamp4(v):
    return [clamp(x) for x in v]


def normalize(v):
    return scale(v, 1.0 / math.sqrt(dot(v, v)))


def mix(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b)]


def rows3x4(rows, v):
    """A Mat3x4's rows times v, a vec4."""
    return [dot(rows[k], v) for k in range(3)]


def wgsl_mod(x, y):
    """WGSL's % on f32, which truncates."""
    return x - y * math.trunc(x / y)


# ---- the made-up state ----

MAX_PNMTX, MAX_TEXMTX = 10, 10
TEXMTX0 = 30

PROJ = [[1.5, 0.0, 0.2, 0.0], [0.0, 2.0, -0.1, 0.0], [0.0, 0.0, 0.05, 1.2], [0.0, 0.0, -1.0, 0.0]]


def pos_matrix(k):
    """Position matrix k: a turn about z by k/10, then a move back."""
    c, s = math.cos(0.3 + 0.1 * k), math.sin(0.3 + 0.1 * k)
    return [[c, -s, 0.0, 0.5 - 0.05 * k], [s, c, 0.0, -0.25 + 0.02 * k], [0.0, 0.0, 1.0, -6.0 - 0.1 * k]]


def nrm_matrix(k):
    """Normal matrix k: the same turn, scaled by 1.3 so the normal needs normalizing."""
    return [[1.3 * x for x in row[:3]] + [0.0] for row in pos_matrix(k)]


def tex_matrix(k):
    return [[0.5 + 0.1 * k, 0.1, 0.0, 0.05 * k], [-0.2, 0.75, 0.05 * k, 0.1], [0.0, 0.02 * k, 1.0, 0.25 + 0.05 * k]]


# Eight lights: position, direction, colour, angle and distance coefficients. Even ones are spot lights and odd ones
# points with distance attenuation, as GX's GXInitLightSpot and GXInitLightDistAttn make them.
LIGHTS = []
for j in range(8):
    angle = 0.7 * j
    position = [3.0 * math.cos(angle), 2.0 * math.sin(angle), -4.0 + 0.5 * j]
    direction = normalize([-math.cos(angle), -math.sin(angle) * 0.5, -0.4])
    color = [0.08 + 0.03 * j, 0.24 - 0.02 * j, 0.12 + 0.03 * (j % 3), 0.15 + 0.02 * j]
    cos_att = [-2.0, 3.0, 0.0] if j % 2 == 0 else [1.0, 0.0, 0.0]
    dist_att = [1.0, 0.05 * j, 0.01] if j % 2 == 0 else [0.8, 0.0, 0.02 * j]
    LIGHTS.append((position, direction, color, cos_att, dist_att))

CHANNEL_COLORS = [[0.1, 0.15, 0.2, 0.3], [0.9, 0.8, 0.7, 0.6], [0.25, 0.3, 0.35, 0.4], [0.5, 0.55, 0.6, 0.75],
                  [0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4]
FOG_COLOR = [0.3, 0.4, 0.5, 1.0]
FOG_A, FOG_B, FOG_C = 0.5, 1.6, 0.2
# GX's range_k registers, as GXSetFogRangeAdj writes them: small, so range fog's factor at PIXEL is well past 1.
RANGE_K_REGS = [10, 14, 18, 24, 30, 38, 46, 56, 66, 78]
RANGE_CENTER = 0.1
DEPTH_RANGE = [0.1, 0.9]
VIEWPORT = [640.0, 480.0, 640.0, 480.0]
TEXCOORD_SCALE = [[64.0, 32.0, 0.0, 0.0], [128.0, 96.0, 0.0, 0.0]] + [[1.0, 1.0, 0.0, 0.0]] * 6
TEX_SIZE = [[128.0, 64.0, 0.0, 0.0], [32.0, 16.0, 0.0, 0.0]] + [[1.0, 1.0, 0.0, 0.0]] * 6
IND_MTX = [[0.5, -0.25, 0.125, 0.75], [0.25, -0.5, 1.0 / 64.0, 0.0],
           [0.75, 0.25, -0.5, 0.125], [-0.125, 0.5, 1.0 / 32.0, 0.0],
           [0.25, 0.25, 0.5, -0.5], [0.5, 0.75, 1.0 / 16.0, 0.0]]


def range_k():
    """range_k as gx/shader_info.cpp puts it in the fog uniform, each pair of registers swapped, over 64."""
    return [RANGE_K_REGS[(i & ~1) | (1 - (i & 1))] / 64.0 for i in range(10)]


def uniforms():
    """Each piece's members, as glsl.cpp declares them: name -> vec4s."""
    k = range_k()
    fog = [FOG_COLOR, [FOG_A, FOG_B, FOG_C, RANGE_CENTER], k[0:4], k[4:8], [k[8], k[9], k[9], k[9]]]
    lights = []
    for position, direction, color, cos_att, dist_att in LIGHTS:
        lights += [position + [0.0], direction + [0.0], color, cos_att + [0.0], dist_att + [0.0]]
    return {
        "uview.viewport_sizes": [VIEWPORT],
        "uview.proj": PROJ,
        "uview.tex_mtx": [row for k in range(MAX_TEXMTX) for row in tex_matrix(k)],
        "uview.channel_colors": CHANNEL_COLORS,
        "uview.fog": fog,
        "uview.ind_mtx": IND_MTX,
        "uview.depth_range": [DEPTH_RANGE + [0.0, 0.0]],
        "upos.mtx": [row for k in range(MAX_PNMTX) for row in pos_matrix(k)],
        "unrm.mtx": [row for k in range(MAX_PNMTX) for row in nrm_matrix(k)],
        "ulights.lights": lights,
        "umat.texcoord_scale": TEXCOORD_SCALE,
        "umat.tex_size_bias": TEX_SIZE,
        "upush.pos_mtx": pos_matrix(CURRENT_PNMTX),
        "upush.nrm_mtx": nrm_matrix(CURRENT_PNMTX),
    }


CURRENT_PNMTX = 1

# The vertex.
POSITION = [0.3, -0.2, 0.5]
NORMAL = [0.2, 0.9, 0.4]
COLOR0 = [0.4, 0.5, 0.6, 0.7]
TEX0 = [0.35, 0.65]
PIXEL = [163.0, 101.0]  # the fragment's column and row: range fog's index between its fourth and fifth k


# ---- the reference: lib/gx/shader.cpp's WGSL ----

def ref_position(pnmtx):
    mv = rows3x4(pos_matrix(pnmtx), POSITION + [1.0])
    clip = [dot(row, mv + [1.0]) for row in PROJ]
    return mv, clip


def ref_normal(pnmtx):
    n = rows3x4(nrm_matrix(pnmtx), NORMAL + [0.0])
    return normalize(n) if dot(n, n) > 1e-10 else n


SATURATED = []  # lit sums and fog factors a clamp hides, which would hide a term from the check


def ref_lighting(mv, n, mask, diff_fn, attn_fn, ambient, material, alpha=False):
    """lighting_func: the channel's ambient colour plus each light of mask's, clamped, times its material colour. The
    components the channel uses, its colour's or its alpha, mustn't clamp."""
    lighting = list(ambient)
    for j in range(8):
        if not mask & (1 << j):
            continue
        position, direction, color, cos_att, dist_att = LIGHTS[j]
        ldir = sub(position, mv)
        dist2 = dot(ldir, ldir)
        dist = math.sqrt(dist2)
        ldir = scale(ldir, 1.0 / dist)
        if attn_fn == "none":
            attn = 1.0
        elif attn_fn == "spot":
            cosraw = dot(ldir, direction)
            cosine = min(cosraw, 1.0) if cosraw > 0.0 else 0.0
            attn = max(0.0, dot(cos_att, [1.0, cosine, cosine * cosine]) / dot(dist_att, [1.0, dist, dist2]))
        else:  # spec
            cosraw = dot(n, direction)
            a = min(cosraw, 1.0) if cosraw > 0.0 and dot(n, ldir) >= 0.0 else 0.0
            terms = [1.0, a, a * a]
            dist_attn = max(0.0, dot(normalize(dist_att) if diff_fn != "none" else dist_att, terms))
            attn = max(0.0, dot(cos_att, terms) / dist_attn)
        diff = {"none": 1.0, "sign": dot(ldir, n), "clamp": max(0.0, dot(ldir, n))}[diff_fn]
        lighting = add(lighting, scale(color, attn * diff))
    if any(not 0.0 < x < 1.0 for x in (lighting[3:] if alpha else lighting[:3])):
        SATURATED.append(lighting)
    return mul(material, clamp4(lighting))


def range_factor(x):
    """gx/command_processor.cpp's build_fog_range_lut at column x, the render viewport VIEWPORT[0] wide."""
    k = range_k()
    offset = ((x + 0.5) / VIEWPORT[0]) * 2.0 - 1.0 - RANGE_CENTER
    index = clamp(9.0 - abs(offset) * 9.0, 0.0, 9.0)
    lower = int(index)
    upper = min(lower + 1, 9)
    fraction = index - lower
    kk = max(k[lower] * (1.0 - fraction) + k[upper] * fraction, 0.000001)
    return math.sqrt(offset * offset + kk * kk) / kk


# The backend's range fog table at set 2, binding 0 (lib/vulkan4xenon/glsl.hpp): range_factor's for each column x at
# texel (x % 64, x / 64) of 64 by 32, sampled nearest with u repeating and v clamped.
FOG_TABLE = (2, 0)
FOG_TABLE_SIZE = (64, 32)


def fog_table(u, v):
    width, height = FOG_TABLE_SIZE
    tx = int(math.floor((u - math.floor(u)) * width))
    ty = min(max(int(math.floor(v * height)), 0), height - 1)
    return [range_factor(tx + width * ty), 0.0, 0.0, 1.0]


def fog_table_uv():
    """Where a program should sample the table for PIXEL's column: the middle of its texel."""
    width, height = FOG_TABLE_SIZE
    column = PIXEL[0] + 0.5
    return [column / width, (math.floor(column / width) + 0.5) / height]


def ref_fog(color, clip, kind, ortho, ranged):
    """The fog's colour mixed in: its depth as the GPU's viewport maps it, reversed (UseReversedZ), and range's factor
    from gx/command_processor.cpp's build_fog_range_lut at the pixel's column."""
    ndc = clip[2] / clip[3]
    min_depth, max_depth = 1.0 - DEPTH_RANGE[1], 1.0 - DEPTH_RANGE[0]
    depth = 1.0 - (min_depth + ndc * (max_depth - min_depth))
    base = FOG_A * depth if ortho else FOG_A / (FOG_B - depth)
    if ranged:
        base *= range_factor(PIXEL[0])
    if not 0.0 < base - FOG_C < 1.0:
        SATURATED.append(base - FOG_C)
    f = clamp(base - FOG_C)
    z = {"lin": f, "exp": 1.0 - 2.0 ** (-8.0 * f), "exp2": 1.0 - 2.0 ** (-8.0 * f * f),
         "revexp": 2.0 ** (-8.0 * (1.0 - f)), "revexp2": 2.0 ** (-8.0 * (1.0 - f) * (1.0 - f))}[kind]
    return mix(color, FOG_COLOR[:3], clamp(z))


def ref_texgen(index_byte):
    """Texture coordinate 1: TEX0 by texture matrix 3, or by the matrix TEX1MTXIDX picks (shader.cpp's select), as a
    3x4 texgen's uvw."""
    tc = TEX0 + [1.0, 1.0]
    if index_byte is None:
        return rows3x4(tex_matrix(3), tc)
    if index_byte >= TEXMTX0:
        return rows3x4(tex_matrix(min((index_byte - TEXMTX0) // 3, MAX_TEXMTX - 1)), tc)
    return rows3x4(pos_matrix(min(index_byte // 3, MAX_PNMTX - 1)), tc)


def decoded_texmtx(index_byte):
    """The attribute decode.cpp writes for a TEXnMTXIDX byte: the slot's first row."""
    slot = MAX_PNMTX + (index_byte - TEXMTX0) // 3 if index_byte >= TEXMTX0 else index_byte // 3
    return slot * 3.0


def image(u, v):
    """Map 0: a smooth image."""
    return [0.25 + 0.5 * u, 0.75 - 0.25 * v, 0.5 * (u + v) % 1.0, 1.0]


# Map 1, the indirect texture: each texel half a step of 255 from a whole one, so the floors and masks of 255 times it
# don't depend on rounding.
INDIRECT = [(37 + 0.5) / 255.0, (200 + 0.5) / 255.0, (91 + 0.5) / 255.0, (166 + 0.5) / 255.0]


def ref_indirect(case):
    """The coordinate stage 0 samples map 0 at, and the indirect sample's coordinate, as shader.cpp works them out;
    the alpha bump's value for indirect_bump."""
    tex0_uv, tex1_uv = TEX0, TEX0                 # both coordinates generated from TEX0 by identity
    ind_uv = [tex1_uv[0] * TEXCOORD_SCALE[1][0] * 0.5 / TEX_SIZE[1][0],
              tex1_uv[1] * TEXCOORD_SCALE[1][1] * 0.25 / TEX_SIZE[1][1]]
    t = [255.0 * INDIRECT[3], 255.0 * INDIRECT[2], 255.0 * INDIRECT[1]]   # .abg
    if case == "indirect_mtx":                    # ITM_0, ITF_8, bias STU, wrap S 64
        coord = [x - 128.0 for x in t]
        c0, c1 = IND_MTX[0], IND_MTX[1]
    elif case == "indirect_dynamic_s":            # ITM_S1, ITF_5, bias T
        coord = [math.floor(x / 8.0) for x in t]
        coord[1] += 1.0
    else:                                         # indirect_bump: ITM_0, ITF_5
        coord = [math.floor(x / 8.0) for x in t]
        c0, c1 = IND_MTX[0], IND_MTX[1]
    texel = [tex0_uv[0] * TEXCOORD_SCALE[0][0], tex0_uv[1] * TEXCOORD_SCALE[0][1]]
    if case == "indirect_dynamic_s":
        offset = scale(texel, coord[0] * IND_MTX[3][2] / 256.0)
    else:
        offset = scale([dot([c0[0], c0[2], c1[0]], coord), dot([c0[1], c0[3], c1[1]], coord)], c1[2])
    if case == "indirect_mtx":
        texel = [wgsl_mod(texel[0], 64.0), texel[1]]
    t_coord = add(texel, offset)
    uv = [t_coord[0] / TEX_SIZE[0][0], t_coord[1] / TEX_SIZE[0][1]]
    bump = None
    if case == "indirect_bump":                   # alpha bump from T, ITF_5's mask 0xE0, BUMPN's scale
        value = int(math.floor(t[1])) & 0xE0
        bump = value / 255.0 * (255.0 / 248.0)
    return uv, ind_uv, bump


# ---- running a case's programs ----

class Programs:
    def __init__(self, xost, glslang, directory, name, work):
        sys.path.insert(0, str(xost))
        from xasm.program import XSDK_REGISTERS
        from xsc import spirv
        from xsc.lower import lower_ex
        from xsc.resolve import resolve_interface
        self.name = name
        self.stages = {}
        for stage in ("vert", "frag"):
            source = directory / f"{name}.{stage}"
            output = work / f"{name}.{stage}.spv"
            result = subprocess.run([str(glslang), "-V", str(source), "-o", str(output)], capture_output=True,
                                    text=True)
            if result.returncode != 0:
                raise SystemExit(f"glslang refused {source}:\n{result.stdout}{result.stderr}")
            module = spirv.Module.parse(output.read_bytes())
            blob, manifest = lower_ex(module, XSDK_REGISTERS)
            self.stages[stage] = (blob, manifest, resolve_interface(spirv.Module.parse(output.read_bytes())),
                                  source.read_text())

    def constants(self, stage, values):
        """Each block's members at xost's registers for it, from values by name; literals from the manifest."""
        blob, manifest, interface, text = self.stages[stage]
        consts = {}
        for block in interface.blocks:
            if block.kind == "push":
                var, members = "upush", block_members(text, r"layout\(push_constant, std140\) uniform Push")
            else:
                var, members = block_var(text, block.binding)
            offset = 0
            for name, count in members:
                data = values.get(f"{var}.{name}", [])
                for i in range(count):
                    consts[block.reg + offset + i] = list(data[i]) if i < len(data) else [0.0] * 4
                offset += count
        consts.update({reg: list(v) for reg, v in manifest})
        return consts

    def vertex(self, values, attributes):
        """The vertex program's exports at a vertex: interpolators by name, and the position."""
        import xdis.simulate as simulate
        blob, _, interface, _ = self.stages["vert"]
        fetched = list(reversed(interface.vfetch))   # xsc fetches the inputs last first (lower.py)

        def fetch(n, instruction, index):
            var = fetched[n][0]
            value = attributes[var.name]
            return list(value) + [0.0, 0.0, 1.0][len(value) - 1:] if len(value) < 4 else list(value)

        result = simulate.run(blob, registers={0: [0.0] * 4}, constants=self.constants("vert", values),
                              vertex_fetch=fetch)
        outputs = {var.name: result.exports.get(o) for var, o, _ in interface.interp_out}
        return outputs, result.exports.get(62)

    def fragment(self, values, outputs, textures, sampled):
        """The fragment program's colour at PIXEL, its inputs the vertex program's outputs."""
        import xdis.simulate as simulate
        blob, _, interface, _ = self.stages["frag"]
        registers = {}
        for var, r, _ in interface.interp_in:
            registers[r] = list(outputs["out_" + var.name[len("in_"):]])
        if interface.frag_coord is not None:
            registers[max(registers, default=-1) + 1] = PIXEL + [0.0, 0.0]
        slots = {slot: (var.descriptor_set, var.binding) for var, slot in interface.samplers}

        def fetch(slot, coordinates):
            key = slots[slot]
            sampled.setdefault(key, []).append(coordinates[:2])
            return textures[key](coordinates[0], coordinates[1])

        result = simulate.run(blob, registers=registers, constants=self.constants("frag", values), texture_fetch=fetch)
        return result.exports.get(0), result.killed


def block_members(text, header):
    """A block's members in declaration order, each a vec4 or an array of them: (name, count)."""
    match = re.search(header + r" \{\n(.*?)\n\}", text, re.S)
    members = []
    for line in match.group(1).splitlines():
        m = re.match(r"\s*vec4 (\w+)(?:\[(\d+)\])?;", line)
        members.append((m.group(1), int(m.group(2) or 1)))
    return members


def block_var(text, binding):
    match = re.search(rf"layout\(set = 0, binding = {binding}, std140\) uniform (\w+) \{{\n(.*?)\n\}} (\w+);", text,
                      re.S)
    header = rf"layout\(set = 0, binding = {binding}, std140\) uniform {match.group(1)}"
    return match.group(3), block_members(text, header)


# ---- the cases ----

def close(got, want, eps=2e-3):
    return got is not None and len(got) >= len(want) and \
        all(abs(g - w) <= eps * max(1.0, abs(w)) for g, w in zip(got, want))


def check_position(position, clip, failures, name):
    want = [clip[0], -clip[1], clip[2], clip[3]]
    if not close(position, want):
        failures.append(f"{name}: position {position}, want {want}")


LIT = {
    "lit_spot_clamp": (0x05, "clamp", "spot", False, "reg"),
    "lit_spec_sign_alpha": (0x06, "sign", "spec", True, "reg"),
    "lit_spot_ambient_vtx": (0xfe, "clamp", "spot", False, "vtx"),
    "lit_none": (0x81, "none", "none", False, "reg"),
    "pnmtx_lit": (0xff, "clamp", "spot", False, "reg"),
}


def check_lit(programs, name, failures):
    mask, diff_fn, attn_fn, alpha, ambient_src = LIT[name]
    pnmtx = 2 if name == "pnmtx_lit" else CURRENT_PNMTX
    attributes = {"in_pos": POSITION, "in_nrm": NORMAL, "in_clr0": COLOR0, "in_pnmtxidx": [pnmtx * 3.0]}
    outputs, position = programs.vertex(uniforms(), attributes)
    mv, clip = ref_position(pnmtx)
    check_position(position, clip, failures, name)
    n = ref_normal(pnmtx)
    if alpha:
        ambient = CHANNEL_COLORS[2]
        lit = ref_lighting(mv, n, mask, diff_fn, attn_fn, ambient, CHANNEL_COLORS[3], alpha=True)
        want = CHANNEL_COLORS[1][:3] + [lit[3]]
    else:
        ambient = COLOR0 if ambient_src == "vtx" else CHANNEL_COLORS[0]
        lit = ref_lighting(mv, n, mask, diff_fn, attn_fn, ambient, CHANNEL_COLORS[1])
        want = lit[:3] + [CHANNEL_COLORS[3][3]]
    if not close(outputs.get("out_cc0"), want):
        failures.append(f"{name}: cc0 {outputs.get('out_cc0')}, want {want}")
    color, _ = programs.fragment(uniforms(), outputs, {}, {})
    if not close(color, clamp4(want)):
        failures.append(f"{name}: colour {color}, want {clamp4(want)}")


FOG = {"fog_lin_range": ("lin", False, True), "fog_revexp2": ("revexp2", False, False),
       "fog_ortho_exp": ("exp", True, False)}


def check_fog(programs, name, failures):
    kind, ortho, ranged = FOG[name]
    outputs, position = programs.vertex(uniforms(), {"in_pos": POSITION})
    mv, clip = ref_position(CURRENT_PNMTX)
    check_position(position, clip, failures, name)
    rasterized = CHANNEL_COLORS[1][:3] + [CHANNEL_COLORS[3][3]]
    want = ref_fog(clamp4(rasterized)[:3], clip, kind, ortho, ranged) + [clamp(rasterized[3])]
    sampled = {}
    color, _ = programs.fragment(uniforms(), outputs, {FOG_TABLE: fog_table}, sampled)
    if ranged and not close(sampled.get(FOG_TABLE, [None])[0], fog_table_uv(), 1e-6):
        failures.append(f"{name}: range fog table sampled at {sampled.get(FOG_TABLE)}, want {fog_table_uv()}")
    if not close(color, want):
        failures.append(f"{name}: colour {color}, want {want}")


def check_texmtx(programs, name, failures):
    rasterized = CHANNEL_COLORS[1][:3] + [CHANNEL_COLORS[3][3]]
    for byte in ([None] if name == "texmtx_fixed" else [12, 39]):
        attributes = {"in_pos": POSITION, "in_tex0_uv": TEX0}
        if byte is not None:
            attributes["in_tex1mtxidx"] = [decoded_texmtx(byte)]
        outputs, _ = programs.vertex(uniforms(), attributes)
        uvw = ref_texgen(byte)
        if not close(outputs.get("out_tex1_uvw"), uvw):
            failures.append(f"{name} {byte}: tex1_uvw {outputs.get('out_tex1_uvw')}, want {uvw}")
        uv = [uvw[0] / uvw[2], uvw[1] / uvw[2]]
        want = mul(image(*uv)[:3], rasterized[:3]) + [rasterized[3]]
        color, _ = programs.fragment(uniforms(), outputs, {(1, 0): image}, {})
        if not close(color, clamp4(want)):
            failures.append(f"{name} {byte}: colour {color}, want {clamp4(want)}")


def check_indirect(programs, name, failures):
    outputs, _ = programs.vertex(uniforms(), {"in_pos": POSITION, "in_tex0_uv": TEX0, "in_tex1_uv": TEX0})
    uv, ind_uv, bump = ref_indirect(name)
    sampled = {}
    color, _ = programs.fragment(uniforms(), outputs, {(1, 0): image, (1, 1): lambda u, v: INDIRECT}, sampled)
    # Coordinates are checked closer than colours: an offset's small terms move them by less than a colour shows.
    if not close(sampled.get((1, 1), [None])[0], ind_uv, 1e-5):
        failures.append(f"{name}: indirect texture sampled at {sampled.get((1, 1))}, want {ind_uv}")
    if not close(sampled.get((1, 0), [None])[0], uv, 1e-5):
        failures.append(f"{name}: texture sampled at {sampled.get((1, 0))}, want {uv}")
    rasterized = CHANNEL_COLORS[1][:3] + [CHANNEL_COLORS[3][3]]
    if bump is None:
        want = mul(image(*uv)[:3], rasterized[:3]) + [rasterized[3]]
    else:
        want = scale(image(*uv)[:3], bump) + [bump]
    if not close(color, clamp4(want)):
        failures.append(f"{name}: colour {color}, want {clamp4(want)}")


def check_scene(programs, name, failures):
    pnmtx, byte = 4, 39
    attributes = {"in_pos": POSITION, "in_nrm": NORMAL, "in_tex0_uv": TEX0, "in_pnmtxidx": [pnmtx * 3.0],
                  "in_tex1mtxidx": [decoded_texmtx(byte)]}
    outputs, position = programs.vertex(uniforms(), attributes)
    mv, clip = ref_position(pnmtx)
    check_position(position, clip, failures, name)
    lit = ref_lighting(mv, ref_normal(pnmtx), 0xff, "clamp", "spot", CHANNEL_COLORS[0], CHANNEL_COLORS[1])
    cc0 = lit[:3] + [CHANNEL_COLORS[3][3]]
    if not close(outputs.get("out_cc0"), cc0):
        failures.append(f"{name}: cc0 {outputs.get('out_cc0')}, want {cc0}")
    tc = rows3x4(tex_matrix((byte - TEXMTX0) // 3), TEX0 + [1.0, 1.0])
    uv = tc[:2]                                  # a 2x4 texgen
    color, _ = programs.fragment(uniforms(), outputs, {(1, 0): image, FOG_TABLE: fog_table}, {})
    textured = clamp4(mul(image(*uv)[:3], clamp4(cc0)[:3]))
    want = ref_fog(textured, clip, "lin", False, True) + [clamp(cc0[3])]
    if not close(color, want):
        failures.append(f"{name}: colour {color}, want {want}")


# Texgens (milestone 18, step 1): emboss, binormal and tangent sources, a prior texgen's source, the guards for a zero
# vector and a q of 0, and attributes the vertex lacks. The vertex's NBT normal has this binormal and tangent after it.
BINORMAL = [0.7, -0.1, 0.3]
TANGENT = [-0.2, 0.4, 0.85]
# Post-transform matrix 0 for texgen_q_zero: a third row of 0, so q is 0, and x and y away from 0, inside -2 to 2.
POST_Q_ZERO = [[0.8, 0.1, 0.0, 0.3], [0.2, 1.1, 0.0, -0.4], [0.0, 0.0, 0.0, 0.0]]


def ref_emboss():
    """shader.cpp's emboss: texgen 0's coordinates, TEX0 as it is, moved by the direction to light 2 from the vertex in
    view space, along the tangent and the binormal the current normal matrix turns."""
    mv, _ = ref_position(CURRENT_PNMTX)
    ldir = normalize(sub(LIGHTS[2][0], mv))
    tangent = rows3x4(nrm_matrix(CURRENT_PNMTX), TANGENT + [0.0])
    binormal = rows3x4(nrm_matrix(CURRENT_PNMTX), BINORMAL + [0.0])
    return [TEX0[0] + dot(ldir, tangent), TEX0[1] + dot(ldir, binormal)]


def ref_texgen_uvs(name):
    """Where the stages sample map 0, in their order."""
    tex0 = TEX0 + [1.0, 1.0]
    if name == "emboss":
        return [ref_emboss()]
    if name == "texgen_binormal":
        return [rows3x4(tex_matrix(3), BINORMAL + [1.0])[:2]]
    if name == "texgen_tangent":
        return [rows3x4(tex_matrix(4), TANGENT + [1.0])[:2]]
    if name == "texgen_prior":                    # texgen 1 takes texgen 0's source, TEX0, by its own matrix
        return [rows3x4(tex_matrix(3), tex0)[:2], rows3x4(tex_matrix(2), tex0)[:2]]
    if name == "texgen_prior_inline":
        return [rows3x4(tex_matrix(2), tex0)[:2]]
    if name == "texgen_zero":                     # nothing to normalize, then a q of 0 at the vertex and the pixel
        return [[0.0, 0.0]]
    if name == "texgen_q_zero":                   # q is 0: half of x and y, held in -1 to 1, and no divide
        proj = rows3x4(POST_Q_ZERO, tex0)         # the post matrix times vec4(tc.xyz, 1), tc.xyz = (u, v, 1)
        return [[clamp(proj[0] * 0.5, -1.0, 1.0), clamp(proj[1] * 0.5, -1.0, 1.0)]]
    return [rows3x4(tex_matrix(3), [0.0, 0.0, 1.0, 1.0])[:2]]   # attr_defaults: TEX1 is 0


def check_texgen(programs, name, failures):
    # texgen_zero's position is 0; attr_defaults has none, so what its slot holds mustn't be read.
    at_zero = name in ("texgen_zero", "attr_defaults")
    position_in = {"texgen_zero": [0.0, 0.0, 0.0], "attr_defaults": [9.0, 9.0, 9.0]}.get(name, POSITION)
    attributes = {"in_pos": position_in, "in_nrm": NORMAL, "in_binrm": BINORMAL, "in_tangent": TANGENT,
                  "in_tex0_uv": TEX0}
    values = uniforms()
    if name == "texgen_q_zero":
        values["uview.postmtx"] = POST_Q_ZERO
    outputs, position = programs.vertex(values, attributes)
    mv = rows3x4(pos_matrix(CURRENT_PNMTX), ([0.0, 0.0, 0.0] if at_zero else POSITION) + [1.0])
    check_position(position, [dot(row, mv + [1.0]) for row in PROJ], failures, name)
    uvs = ref_texgen_uvs(name)
    sampled = {}
    color, _ = programs.fragment(values, outputs, {(1, 0): image}, sampled)
    got = sampled.get((1, 0), [])
    # In any order: xost may fetch a later stage's texel first.
    if len(got) != len(uvs) or not all(close(g, w, 1e-5) for g, w in zip(sorted(got), sorted(uvs))):
        failures.append(f"{name}: map 0 sampled at {got}, want {uvs}")
    rasterized = CHANNEL_COLORS[1][:3] + [CHANNEL_COLORS[3][3]]
    if name == "texgen_prior":                    # stage 1's texel times stage 0's
        want = mul(image(*uvs[1])[:3], image(*uvs[0])[:3]) + [rasterized[3]]
    else:
        want = mul(image(*uvs[0])[:3], rasterized[:3]) + [rasterized[3]]
    if not close(color, clamp4(want)):
        failures.append(f"{name}: colour {color}, want {clamp4(want)}")


# A TEV register set past [0, 1], as GXSetTevColorS10 sets one, read by a stage before any stage writes it (milestone 18,
# step 2): as a, b or c the stage takes its low 8 bits, as tev_overflow wraps it. TEVREG0's s10 values.
TEVREG0_S10 = [300, -64, 128, 1000]


def check_tev_wrap(programs, name, failures):
    values = uniforms()
    values["umat.tevreg"] = [[0.0] * 4, [c / 255.0 for c in TEVREG0_S10], [0.0] * 4, [0.0] * 4]
    outputs, _ = programs.vertex(values, {"in_pos": POSITION})
    color, _ = programs.fragment(values, outputs, {}, {})
    want = [(c % 256) / 255.0 for c in TEVREG0_S10]
    if not close(color, want):
        failures.append(f"{name}: colour {color}, want {want}")


# Lines and points (milestone 14): GX's line width 54, in sixths of a pixel, and a texture offset; the render viewport
# twice the logical one across and half again down, so the width scales by the smaller, 1.5. A segment's two ends, or
# the point at the first: position, TEX0 and the matrix it picks.
LINE_WIDTH = 9.0
LINE_TEX_OFFSET = 0.25
LINE_VIEWPORT = [1280.0, 720.0, 640.0, 480.0]
LINE_ENDS = [([0.3, -0.2, 0.5], [0.35, 0.65], 2), ([-0.4, 0.25, 0.1], [0.8, 0.15], 5)]


def line_uniforms(mask):
    values = uniforms()
    values["uview.viewport_sizes"] = [LINE_VIEWPORT]
    values["uview.line_params"] = [[LINE_WIDTH, 1.0, LINE_TEX_OFFSET, 0.0]]
    values["upush.line_mask"] = [[1.0 if mask >> i & 1 else 0.0 for i in range(k, k + 4)] for k in (0, 4)]
    return values


def clip_of(position, pnmtx):
    mv = rows3x4(pos_matrix(pnmtx), position + [1.0])
    return [dot(row, mv + [1.0]) for row in PROJ]


def ref_corner(points, vidx, picked, mask):
    """shader.cpp's corner vidx of the segment between LINE_ENDS, or of the point at the first: its clip position, and
    TEX0 moved by the texture offset where the mask says."""
    render, logical = LINE_VIEWPORT[:2], LINE_VIEWPORT[2:]
    width = LINE_WIDTH * min(render[0] / max(logical[0], 1.0), render[1] / max(logical[1], 1.0))

    def clip_at(end):
        return clip_of(LINE_ENDS[end][0], LINE_ENDS[end][2] if picked else CURRENT_PNMTX)

    if points:
        clip = clip_at(0)
        sign = [1.0 if vidx & 1 else -1.0, 1.0 if vidx >= 2 else -1.0]
        offset = [s * (width / 2.0) * 2.0 / r for s, r in zip(sign, render)]
        uv = list(LINE_ENDS[0][1])
        if mask & 1:
            uv = [uv[0] + (LINE_TEX_OFFSET if vidx & 1 else 0.0), uv[1] + (LINE_TEX_OFFSET if vidx >= 2 else 0.0)]
    else:
        clip_a, clip_b = clip_at(0), clip_at(1)
        delta = [(b[k] / b[3] - a[k] / a[3]) / 2.0 * render[k] for a, b, k in ((clip_a, clip_b, 0), (clip_a, clip_b, 1))]
        direction = normalize(delta) if dot(delta, delta) > 1e-10 else [1.0, 0.0]
        perp = [-direction[1], direction[0]]
        side = 1.0 if vidx & 1 else -1.0
        offset = [p * (width / 2.0) * side * 2.0 / r for p, r in zip(perp, render)]
        clip = clip_b if vidx >= 2 else clip_a
        uv = list(LINE_ENDS[1 if vidx >= 2 else 0][1])
        if mask & 1 and vidx & 1:
            uv[1] += LINE_TEX_OFFSET
    return [clip[0] + offset[0] * clip[3], clip[1] + offset[1] * clip[3], clip[2], clip[3]], uv


def corner_attributes(points, vidx, picked):
    """What lib/vulkan4xenon/glsl.cpp's write_corners puts in corner vidx's vertex."""
    second = vidx >= 2 and not points
    end, other = (1, 0) if second else (0, 1)
    side = 1.0 if vidx & 1 else -1.0
    corner = [side, 1.0 if vidx >= 2 else -1.0] if points else [-side if second else side, 1.0 if vidx & 1 else 0.0]
    attributes = {"in_pos": LINE_ENDS[end][0], "in_tex0_uv": LINE_ENDS[end][1], "in_corner": corner}
    if not points:
        attributes["in_other_pos"] = LINE_ENDS[other][0]
    if picked:
        attributes["in_pnmtxidx"] = [LINE_ENDS[end][2] * 3.0]
        attributes["in_other_pnmtx"] = [LINE_ENDS[other][2] * 3.0]
    return attributes


def check_lines(programs, name, failures):
    points, picked = name == "points", name == "line_picked"
    for mask in (0, 1):
        for vidx in range(4):
            where = f"{name} mask {mask} corner {vidx}"
            outputs, position = programs.vertex(line_uniforms(mask), corner_attributes(points, vidx, picked))
            clip, uv = ref_corner(points, vidx, picked, mask)
            want = [clip[0], -clip[1], clip[2], clip[3]]
            # Closer than check_position: the corner's move is a few hundredths of the position.
            if not close(position, want, 1e-5):
                failures.append(f"{where}: position {position}, want {want}")
            if not close(outputs.get("out_tex0_uv"), uv, 1e-5):
                failures.append(f"{where}: tex0_uv {outputs.get('out_tex0_uv')}, want {uv}")


# ---- depth copies and GXPeekZ's snapshot (milestone 14, step 5) ----

# Depths as the layer returns them in red: the ends, halves and thirds, a step of D24 from each end, ones a step under
# a byte's edge in z, and two besides.
DEPTHS = [0.0, 1.0, 0.5, 0.25, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 16777215.0, 1.0 - 1.0 / 16777215.0,
          1.0 - 65535.0 / 16777215.0, 1.0 - 255.0 / 16777215.0, 0.123456, 0.987654]
DEPTH_UV = [0.25, 0.75]


def ref_z24(depth):
    """tex_copy_conv.cpp's gx_z24 in f32 from reversed Z, then u32 and its min."""
    d = single(depth)
    return min(int(single(single(clamp(single(1.0 - d)) * 16777215.0) + 0.5)), 0xFFFFFF)


def ref_depth_output(name, z24):
    """FragZ8's high byte as intensity, FragZ16's high byte as intensity and the next as alpha, and the snapshot's
    three bytes, high first, with alpha 0."""
    high, middle, low = z24 >> 16, z24 >> 8 & 0xFF, z24 & 0xFF
    if name == "depth_z8":
        return [high / 255.0] * 4
    if name == "depth_z16":
        return [high / 255.0] * 3 + [middle / 255.0]
    return [high / 255.0, middle / 255.0, low / 255.0, 0.0]


def check_depth(programs, name, failures):
    outputs, _ = programs.vertex({"upush.uv_transform": [[0.0, 0.0, 1.0, 1.0]]},
                                 {"in_position": [-1.0, 1.0], "in_uv": DEPTH_UV})
    for depth in DEPTHS:
        sampled = {}
        got, _ = programs.fragment({}, outputs, {(0, 0): lambda u, v, d=depth: [single(d), 0.0, 0.0, 1.0]}, sampled)
        z24 = ref_z24(depth)
        want = ref_depth_output(name, z24)
        # A byte off is 1/255 away: closer than that, so each byte must be right.
        if not close(got, want, 1e-4):
            failures.append(f"{name} at depth {depth!r} (z {z24:06X}): {got}, want {want}")
        if not all(close(at, DEPTH_UV, 1e-6) for at in sampled.get((0, 0), [None])):
            failures.append(f"{name}: sampled at {sampled.get((0, 0))}, not {DEPTH_UV}")


CHECKS = {**{name: check_lit for name in LIT}, **{name: check_fog for name in FOG},
          "texmtx_fixed": check_texmtx, "texmtx_picked": check_texmtx, "indirect_mtx": check_indirect,
          "indirect_dynamic_s": check_indirect, "indirect_bump": check_indirect, "scene": check_scene,
          "line_current": check_lines, "line_picked": check_lines, "points": check_lines,
          "depth_z8": check_depth, "depth_z16": check_depth, "depth_peek": check_depth,
          **{name: check_texgen for name in ("emboss", "texgen_binormal", "texgen_tangent", "texgen_prior",
                                             "texgen_prior_inline", "texgen_zero", "texgen_q_zero", "attr_defaults")},
          "tev_reg_wrap": check_tev_wrap}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xost", type=Path, required=True)
    parser.add_argument("--glslang", type=Path, help="by default xost's tools/glslang/bin/glslang.exe")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    glslang = args.glslang or args.xost / "tools" / "glslang" / "bin" / "glslang.exe"
    names = sorted(p.stem for p in args.directory.glob("*.vert"))
    missing = sorted(set(CHECKS) - set(names))
    failures = [f"{name}: not written" for name in missing]
    with tempfile.TemporaryDirectory() as work:
        for name in names:
            if name not in CHECKS:
                failures.append(f"{name}: no check for it")
                continue
            before = len(failures)
            SATURATED.clear()
            CHECKS[name](Programs(args.xost, glslang, args.directory, name, Path(work)), name, failures)
            if SATURATED:
                failures.append(f"{name}: {SATURATED[0]} clamps, so the check can't see what it sums")
            print(f"  {'PASS' if len(failures) == before else 'FAIL'} {name}")
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        raise SystemExit("FAILED")
    print(f"OK: {len(names)} cases")


if __name__ == "__main__":
    main()
