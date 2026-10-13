#!/usr/bin/env python3
"""Build and run every verification/tests/test_*.cu under several nvcc
configurations (compiler differential testing) and compare the outputs.

Each configuration must (1) match the CPU reference and (2) agree with the
baseline configuration (-G, i.e. device optimisations off).

  python3 verification/run_matrix.py --compile-only sm_80,sm_90 --update-readme
  python3 verification/run_matrix.py --no-run --cc 75     # build only, no GPU needed
"""
import argparse, datetime, json, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
DTYPES = {"float32": np.float32, "float64": np.float64, "int32": np.int32,
          "uint32": np.uint32, "uint8": np.uint8}
ICON = {"PASS": "✅", "BUILT": "🔨", "CPU_MISMATCH": "❌ cpu", "DIFF": "❌ diff",
        "BUILD_FAIL": "🔧❌", "RUN_FAIL": "💥", "WARN": "⚠️"}
FAILING = {"CPU_MISMATCH", "DIFF", "BUILD_FAIL", "RUN_FAIL"}
BASELINE = "debug-G"
EXE = "test.exe" if sys.platform == "win32" else "test"


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def detect_gpu():
    r = sh(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"])
    if r.returncode != 0:
        sys.exit("nvidia-smi failed: a CUDA GPU is required (or use --no-run --cc XX)")
    name, cc = [s.strip() for s in r.stdout.splitlines()[0].split(",")]
    return name, cc.replace(".", "")


def nvcc_version(ccbin):
    m = re.search(r"release (\S+?),", sh(["nvcc", "--version"]).stdout)
    return m.group(1) if m else "unknown"


def lowest_supported_compute():
    archs = [a for a in sh(["nvcc", "--list-gpu-arch"]).stdout.split()
             if re.fullmatch(r"compute_\d+", a)]
    return min(int(a.split("_")[1]) for a in archs)


def make_configs(cc, ptx_arch):
    sm = f"-arch=sm_{cc}"

    def opt(n):  # -O is host-only; -Xptxas -O controls the device assembler
        return [f"-O{n}", "-Xptxas", f"-O{n}"]

    return {
        BASELINE:             {"flags": ["-G", "-O0", sm]},
        "O0":                 {"flags": opt(0) + [sm]},
        "O1":                 {"flags": opt(1) + [sm]},
        "O2":                 {"flags": opt(2) + [sm]},
        "O3":                 {"flags": opt(3) + [sm]},
        "O3-nofma":           {"flags": opt(3) + ["--fmad=false", sm]},
        f"O3-ptx{ptx_arch}":  {"flags": opt(3) + ["-gencode",
                               f"arch=compute_{ptx_arch},code=compute_{ptx_arch}"]},
        # fast-math is expected to change float results: report, don't fail
        "O3-fastmath":        {"flags": opt(3) + ["-use_fast_math", sm], "lenient": True},
    }


def build_and_run(test, cname, flags, ccbin, run):
    bdir = BUILD / test.stem[5:] / cname
    bdir.mkdir(parents=True, exist_ok=True)
    exe, out = bdir / EXE, bdir / "out.bin"
    cmd = ["nvcc", *flags, "-I", str(ROOT), "-o", str(exe), str(test)]
    if ccbin:
        cmd[1:1] = ["-ccbin", ccbin]
    b = sh(cmd)
    if b.returncode:
        return {"status": "BUILD_FAIL", "detail": b.stderr[-400:]}
    if not run:
        return {"status": "BUILT"}
    try:
        r = sh([str(exe), str(out)], timeout=120)
    except subprocess.TimeoutExpired:
        return {"status": "RUN_FAIL", "detail": "timeout"}
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"status": "RUN_FAIL", "detail": (r.stderr or "no output")[-400:]}
    if r.returncode not in (0, 1):
        return {"status": "RUN_FAIL", "detail": r.stderr[-400:]}
    info["status"] = "PASS" if info["pass"] else "CPU_MISMATCH"
    info["out"] = str(out)
    info["stderr"] = r.stderr[-400:]
    return info


def outputs_agree(a, b):
    dt = DTYPES[a["dtype"]]
    x, y = np.fromfile(a["out"], dtype=dt), np.fromfile(b["out"], dtype=dt)
    if x.shape != y.shape:
        return False
    if a["abs_tol"] == 0 and a["rel_tol"] == 0:
        return x.tobytes() == y.tobytes()                       # bit-exact
    xf, yf = x.astype(np.float64), y.astype(np.float64)
    tol = 2 * (a["abs_tol"] + a["rel_tol"] * np.abs(yf))        # each is within tol of CPU
    return bool(np.all(np.abs(xf - yf) <= tol))


def render_table(results, configs, header):
    extra = sorted({c for r in results.values() for c in r} - set(configs))
    cols = list(configs) + extra
    lines = [header, "",
             "| Kernel | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for k, row in sorted(results.items()):
        lines.append(f"| {k} | " + " | ".join(
            ICON.get(row.get(c, {}).get("status"), "—") for c in cols) + " |")
    lines += ["", "✅ matches CPU reference and baseline · ⚠️ lenient config differs (expected) · "
              "❌ cpu = GPU differs from CPU · ❌ diff = differs from baseline · "
              "🔧❌ build failed · 💥 crashed · 🔨 built only (not run)"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="*", help="kernel names to run (default: all)")
    ap.add_argument("--compile-only", default="", help="extra build-only targets, e.g. sm_80,sm_90")
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="build only; no GPU needed")
    ap.add_argument("--cc", help="compute capability without dot, e.g. 75 (default: detect)")
    ap.add_argument("--ccbin", help="host compiler for nvcc, e.g. g++-12")
    args = ap.parse_args()

    if args.no_run:
        gpu, cc = "none (build only)", args.cc or "75"
    else:
        gpu, cc = detect_gpu()
        cc = args.cc or cc
    configs = make_configs(cc, lowest_supported_compute())
    tests = sorted((ROOT / "tests").glob("test_*.cu"))
    if args.tests:
        tests = [t for t in tests if t.stem[5:] in args.tests]
    compile_only = [a for a in args.compile_only.split(",") if a]
    BUILD.mkdir(parents=True, exist_ok=True)

    results = {}
    for t in tests:
        name = t.stem[5:]
        results[name], base = {}, None
        for cname, cfg in configs.items():
            res = build_and_run(t, cname, cfg["flags"], args.ccbin, not args.no_run)
            if cname == BASELINE:
                base = res if res["status"] == "PASS" else None
            elif res["status"] == "PASS" and base is not None and not outputs_agree(res, base):
                res["status"] = "DIFF"
            if cfg.get("lenient") and res["status"] in ("CPU_MISMATCH", "DIFF"):
                res["status"] = "WARN"
            results[name][cname] = res
            print(f"{name:20s} {cname:16s} {res['status']}", flush=True)
            if res["status"] in FAILING and res.get("detail"):
                print("   ", res["detail"].strip().splitlines()[-1] if res["detail"].strip() else "")
        for arch in compile_only:
            cmd = ["nvcc", "-O3", f"-arch={arch}", "-I", str(ROOT), "-c", str(t),
                   "-o", str(BUILD / f"{name}_{arch}.o")]
            if args.ccbin:
                cmd[1:1] = ["-ccbin", args.ccbin]
            r = sh(cmd)
            results[name][f"build-{arch}"] = {"status": "BUILT" if r.returncode == 0 else "BUILD_FAIL"}
            print(f"{name:20s} build-{arch:10s} {results[name][f'build-{arch}']['status']}", flush=True)

    header = f"_nvcc {nvcc_version(args.ccbin)} · {gpu} (sm_{cc}) · {datetime.date.today()}_"
    table = render_table(results, configs, header)
    print("\n" + table)

    if not args.no_run:   # never write result files from a build-only run
        (ROOT / "RESULTS.md").write_text(table + "\n", encoding="utf-8")
        if args.update_readme:
            readme = ROOT.parent / "README.md"
            text = readme.read_text(encoding="utf-8")
            new = f"<!-- VERIFY:START -->\n{table}\n<!-- VERIFY:END -->"
            text = re.sub(r"<!-- VERIFY:START -->.*?<!-- VERIFY:END -->", lambda m: new, text, flags=re.S)
            readme.write_text(text, encoding="utf-8")

    bad = any(r["status"] in FAILING for row in results.values() for r in row.values())
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
