#!/usr/bin/env python3
"""Compiles the programs Aurora's tests/vulkan4xenon_glsl_coverage.cpp writes for GX's configs (vulkan4xenon's
milestone 18) as the game's pack is built: glslang to SPIR-V, then xost to Xenos microcode inside its register budget.
Each distinct program is compiled once; a failure names the configs that use it.

    check_glsl_coverage.py --xost <xost tree> --glslang <glslang> [--jobs N] <AURORA_GLSL_COVERAGE_DIR>
"""

import argparse
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

XOST = None


def lower(args):
    """One program: glslang's verdict, then xost's; None when both take it, else what refused it."""
    glslang, source, work = args
    output = Path(work) / (source.name + ".spv")
    result = subprocess.run([str(glslang), "-V", str(source), "-o", str(output)], capture_output=True, text=True)
    if result.returncode != 0:
        return f"glslang: {(result.stdout + result.stderr).strip().splitlines()[-1]}"
    sys.path.insert(0, str(XOST))
    from xasm.program import XSDK_REGISTERS
    from xsc import spirv
    from xsc.lower import lower_ex
    try:
        lower_ex(spirv.Module.parse(output.read_bytes()), XSDK_REGISTERS)
    except Exception as error:  # xost's refusals and its register budget
        return f"xost: {error}"
    return None


def init(xost):
    global XOST
    XOST = xost


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xost", type=Path, required=True)
    parser.add_argument("--glslang", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    users = defaultdict(list)
    for line in (args.directory / "manifest.txt").read_text().splitlines():
        vert, frag, name = line.split(" ", 2)
        users[f"{vert}.vert"].append(name)
        users[f"{frag}.frag"].append(name)
    sources = sorted(args.directory / name for name in users)
    with tempfile.TemporaryDirectory() as work, ProcessPoolExecutor(args.jobs, initializer=init,
                                                                     initargs=(args.xost,)) as pool:
        verdicts = list(pool.map(lower, [(args.glslang, source, work) for source in sources], chunksize=16))
    failures = [(source.name, verdict) for source, verdict in zip(sources, verdicts) if verdict is not None]
    for name, verdict in failures:
        cases = users[name]
        print(f"  FAIL {name}: {verdict}")
        print(f"       used by {len(cases)} configs, as {cases[0]}" + (f" and {cases[1]}" if len(cases) > 1 else ""))
    print(f"{len(sources) - len(failures)} of {len(sources)} programs compile and lower, "
          f"for {sum(1 for _ in (args.directory / 'manifest.txt').read_text().splitlines())} configs")
    if failures:
        raise SystemExit("FAILED")


if __name__ == "__main__":
    main()
