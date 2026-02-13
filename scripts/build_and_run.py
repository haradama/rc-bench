#!/usr/bin/env python3
import argparse, csv, os, subprocess, sys
from pathlib import Path

def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    if "stdout" in kw:
        subprocess.run(cmd, check=True, **kw)
        return ""
    else:
        return subprocess.check_output(cmd, text=True, **kw).strip()

def get_macos_sdk_path():
    try:
        return subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    except:
        return None

def detect_linalg_lowering_pass(mlir_opt: str) -> str:
    """
    Detect an available linalg lowering pass in mlir-opt.
    Different builds expose different pass names.
    Returns the *single* pass arg (e.g., '-convert-linalg-to-loops').
    """
    help_txt = subprocess.check_output([mlir_opt, "--help"], text=True)

    # Prefer the most common / direct ones first.
    candidates = [
        "-convert-linalg-to-loops",
        "-convert-linalg-to-parallel-loops",
        "-convert-linalg-to-affine-loops",
        "-convert-linalg-to-scf",
    ]
    for p in candidates:
        if p in help_txt:
            return p

    raise RuntimeError(
        "No known linalg lowering pass found in mlir-opt --help.\n"
        "Please run:\n"
        "  mlir-opt --help | grep -E 'convert-linalg|linalg-to|lower-linalg'\n"
        "and update candidates in detect_linalg_lowering_pass()."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out-dir", default="results_phase1")
    ap.add_argument("--mlir-translate", default="mlir-translate")
    ap.add_argument("--clang", default="clang++")
    ap.add_argument("--rc-opt", default=None)
    ap.add_argument("--mlir-opt", default="mlir-opt")   # ★復活：本家 mlir-opt を使う
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--Din", type=int, default=64)
    ap.add_argument("--leak", type=float, default=0.3)
    args = ap.parse_args()

    build = Path(args.build_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mlir").mkdir(exist_ok=True)
    (out / "llvm").mkdir(exist_ok=True)
    (out / "bin").mkdir(exist_ok=True)

    candidate1 = build / "bin" / "rc-opt"
    candidate2 = build / "lib" / "rc-opt"
    rc_opt = args.rc_opt or (str(candidate1) if candidate1.exists() else str(candidate2))

    mlir_opt = args.mlir_opt
    linalg_lower = detect_linalg_lowering_pass(mlir_opt)
    print(f"[info] Using mlir-opt linalg lowering pass: {linalg_lower}")

    driver = Path("runtime/driver.cpp")
    rt_c = Path("runtime/rc_runtime.c")

    clang_flags = ["-O3"]
    if sys.platform == "darwin":
        sdk_path = os.environ.get("SDKROOT") or get_macos_sdk_path()
        if sdk_path:
            clang_flags.extend(["-isysroot", sdk_path])

    # Lower pipeline (run by mlir-opt, not rc-opt)
    LOWER_TO_LLVM = [
        "-canonicalize",
        "-cse",
        linalg_lower,
        "-convert-scf-to-cf",
        "-convert-math-to-llvm",
        "-convert-arith-to-llvm",
        "-finalize-memref-to-llvm",
        "-convert-func-to-llvm",
        "-convert-cf-to-llvm",
        "-reconcile-unrealized-casts",
    ]

    cases = []
    for N in [512, 1024, 2048]:
        for B in [1, 16]:
            cases.append(("dense", N, B))

    rows = []
    for kind, N, B in cases:
        a_mlir = out / "mlir" / f"{kind}_A_rc_N{N}_B{B}.mlir"
        b_mlir = out / "mlir" / f"{kind}_B_linalg_N{N}_B{B}.mlir"

        run(["python3", "scripts/gen_case.py", "--out", str(a_mlir),
             "--mode", "dense_rc", "--N", str(N), "--B", str(B),
             "--Din", str(args.Din), "--T", str(args.T), "--leak", str(args.leak)])

        run(["python3", "scripts/gen_case.py", "--out", str(b_mlir),
             "--mode", "dense_linalg", "--N", str(N), "--B", str(B),
             "--Din", str(args.Din), "--T", str(args.T), "--leak", str(args.leak)])

        # ---------- A side ----------
        # 1) rc-opt: rc -> linalg
        a_low = out / "mlir" / f"{kind}_A_low_N{N}_B{B}.mlir"
        run([rc_opt, str(a_mlir), "-convert-rc-to-linalg", "-o", str(a_low)])

        # 2) mlir-opt: linalg -> llvm dialect
        a_llvm_dialect = out / "mlir" / f"{kind}_A_llvm_N{N}_B{B}.mlir"
        with open(a_llvm_dialect, "w") as f:
            run([mlir_opt, str(a_low), *LOWER_TO_LLVM, "-o", "-"], stdout=f)

        # 3) mlir-translate: llvm dialect -> LLVM IR
        a_ll = out / "llvm" / f"{kind}_A_N{N}_B{B}.ll"
        with open(a_ll, "w") as f:
            run([args.mlir_translate, "--mlir-to-llvmir", str(a_llvm_dialect)], stdout=f)

        # ---------- B side ----------
        b_llvm_dialect = out / "mlir" / f"{kind}_B_llvm_N{N}_B{B}.mlir"
        with open(b_llvm_dialect, "w") as f:
            run([mlir_opt, str(b_mlir), *LOWER_TO_LLVM, "-o", "-"], stdout=f)

        b_ll = out / "llvm" / f"{kind}_B_N{N}_B{B}.ll"
        with open(b_ll, "w") as f:
            run([args.mlir_translate, "--mlir-to-llvmir", str(b_llvm_dialect)], stdout=f)

        # Compile
        a_exe = out / "bin" / f"{kind}_A_N{N}_B{B}"
        b_exe = out / "bin" / f"{kind}_B_N{N}_B{B}"
        run([args.clang, *clang_flags, str(driver), str(rt_c), str(a_ll), "-lm", "-o", str(a_exe)])
        run([args.clang, *clang_flags, str(driver), str(rt_c), str(b_ll), "-lm", "-o", str(b_exe)])

        # Run
        a_out = run([str(a_exe), str(args.warmup), str(args.runs)])
        b_out = run([str(b_exe), str(args.warmup), str(args.runs)])

        def parse(line):
            parts = dict(p.split("=", 1) for p in line.split(","))
            return int(parts["best_ns"]), float(parts["avg_ns"])

        a_best, a_avg = parse(a_out)
        b_best, b_avg = parse(b_out)
        speedup_best = (b_best / a_best) if a_best > 0 else 0.0
        speedup_avg = (b_avg / a_avg) if a_avg > 0 else 0.0

        print(f"Result: N={N} B={B} -> A={a_avg:.2f}ns B={b_avg:.2f}ns (Speedup {speedup_avg:.2f}x)")

        rows.append({
            "kind": kind, "N": N, "B": B, "Din": args.Din, "T": args.T, "leak": args.leak,
            "A_best_ns": a_best, "A_avg_ns": a_avg,
            "B_best_ns": b_best, "B_avg_ns": b_avg,
            "speedup_best(B/A)": speedup_best,
            "speedup_avg(B/A)": speedup_avg,
        })

    csv_path = out / "phase1_dense.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote:", csv_path)

if __name__ == "__main__":
    main()
