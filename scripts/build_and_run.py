#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
import numpy as np


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


def _pick_first(help_txt: str, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in help_txt:
            return c
    return None


def _pick_all(help_txt: str, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in help_txt]


def build_b2_pipeline(mlir_opt: str, linalg_lower: str) -> list[str]:
    help_txt = subprocess.check_output([mlir_opt, "--help"], text=True)

    def has(p: str) -> bool:
        return p in help_txt

    def pick_first(cands: list[str]) -> str | None:
        for c in cands:
            if has(c):
                return c
        return None

    def pick_all(cands: list[str]) -> list[str]:
        return [c for c in cands if has(c)]

    linalg_opts: list[str] = []

    fuse_elem = pick_first([
        "-linalg-fuse-elementwise-ops",
        "-linalg-fuse-elementwise",
    ])
    if fuse_elem:
        linalg_opts.append(fuse_elem)

    fold_unit = "-linalg-fold-unit-extent-dims" if has(
        "-linalg-fold-unit-extent-dims") else None

    expand_strided = "-expand-strided-metadata" if has(
        "-expand-strided-metadata") else None

    if fold_unit:
        if expand_strided:
            linalg_opts.append(fold_unit)
        else:
            print(
                "[info] B2: skip -linalg-fold-unit-extent-dims (no -expand-strided-metadata available)")

    linalg_opts += pick_all([
        "-linalg-canonicalize",
    ])

    loop_opts = pick_all([
        "-loop-invariant-code-motion",
        "-licm",
        "-sccp",
        "-symbol-dce",
        "-canonicalize",
        "-cse",
    ])

    affine_lower = pick_first([
        "-lower-affine",
        "-convert-affine-to-scf",
        "-convert-affine-to-standard",
    ])

    lower_rest: list[str] = []

    if expand_strided:
        lower_rest += [expand_strided, "-canonicalize", "-cse"]

    if affine_lower:
        lower_rest += [affine_lower, "-canonicalize", "-cse"]

    lower_rest += [
        "-convert-scf-to-cf",
        "-convert-math-to-llvm",
        "-convert-arith-to-llvm",
        "-finalize-memref-to-llvm",
        "-convert-func-to-llvm",
        "-convert-cf-to-llvm",
        "-reconcile-unrealized-casts",
    ]

    pipeline = []
    pipeline += ["-canonicalize", "-cse"]
    pipeline += linalg_opts
    pipeline += ["-canonicalize", "-cse"]
    pipeline += [linalg_lower]
    pipeline += loop_opts
    pipeline += lower_rest

    used = {
        "linalg_lower": linalg_lower,
        "linalg_opts": linalg_opts,
        "expand_strided": expand_strided,
        "affine_lower": affine_lower,
        "loop_opts(extra)": [p for p in loop_opts if p not in ("-canonicalize", "-cse")],
    }
    print("[info] B2 pipeline picks:", used)

    return pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["dense", "esn_mg"], default="dense")
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--out-dir", default="results_phase1")
    ap.add_argument("--mlir-translate", default="mlir-translate")
    ap.add_argument("--clang", default="clang++")
    ap.add_argument("--rc-opt", default=None)
    ap.add_argument("--mlir-opt", default="mlir-opt")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--Din", type=int, default=64)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--assets", default=None,
                    help="(task=esn_mg) path to ESN assets .npz (e.g., results_phase2/esn_mg_assets.npz)")

    ap.add_argument("--enable-b2", action="store_true", default=True)
    ap.add_argument("--disable-b2", action="store_true", default=False)

    args = ap.parse_args()
    enable_b2 = args.enable_b2 and (not args.disable_b2)

    build = Path(args.build_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mlir").mkdir(exist_ok=True)
    (out / "llvm").mkdir(exist_ok=True)
    (out / "bin").mkdir(exist_ok=True)

    candidate1 = build / "bin" / "rc-opt"
    candidate2 = build / "lib" / "rc-opt"
    rc_opt = args.rc_opt or (
        str(candidate1) if candidate1.exists() else str(candidate2))

    mlir_opt = args.mlir_opt
    linalg_lower = detect_linalg_lowering_pass(mlir_opt)
    print(f"[info] Using mlir-opt linalg lowering pass: {linalg_lower}")

    B2_PIPELINE = build_b2_pipeline(
        mlir_opt, linalg_lower) if enable_b2 else []

    driver_dense = Path("runtime/driver.cpp")
    driver_esn = Path("runtime/driver_esn.cpp")
    rt_c = Path("runtime/rc_runtime.c")

    clang_flags = ["-O3"]
    if sys.platform == "darwin":
        sdk_path = os.environ.get("SDKROOT") or get_macos_sdk_path()
        if sdk_path:
            clang_flags.extend(["-isysroot", sdk_path])

    LOWER_TO_LLVM_CORE = [
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

    def parse_kv(line: str) -> dict:
        parts = {}
        for p in line.split(","):
            if "=" in p:
                k, v = p.split("=", 1)
                parts[k.strip()] = v.strip()
        return parts

    if args.task == "dense":
        driver = driver_dense
        cases = []
        for N in [32, 64, 128, 512, 1024, 2048]:
            for B in [1, 16]:
                cases.append(("dense", N, B))
    else:
        driver = driver_esn
        if args.assets is None:
            raise RuntimeError(
                "--task esn_mg requires --assets path/to/esn_mg_assets.npz")
        z = np.load(args.assets)
        esn_N = int(z["N"])
        esn_Din = int(z["Din"])
        esn_Ttest = int(z["Ttest"])
        esn_lr = float(z["lr"])
        cases = [("esn_mg", esn_N, 1)]

    rows = []
    for kind, N, B in cases:
        if args.task == "esn_mg":
            if args.assets is None:
                raise RuntimeError(
                    "--task esn_mg requires --assets path/to/esn_mg_assets.npz")

            tag = f"N{esn_N}_T{esn_Ttest}_lr{esn_lr:.6g}"
            a_mlir = out / "mlir" / f"{kind}_A_rc_{tag}.mlir"
            b_mlir = out / "mlir" / f"{kind}_B_linalg_{tag}.mlir"

            run(["python3", "scripts/gen_case.py", "--out", str(a_mlir),
                "--mode", "esn_mg_rc", "--assets", str(args.assets)])
            run(["python3", "scripts/gen_case.py", "--out", str(b_mlir),
                "--mode", "esn_mg_linalg", "--assets", str(args.assets)])
        else:
            tag = f"N{N}_B{B}"
            a_mlir = out / "mlir" / f"{kind}_A_rc_{tag}.mlir"
            b_mlir = out / "mlir" / f"{kind}_B_linalg_{tag}.mlir"

            run(["python3", "scripts/gen_case.py", "--out", str(a_mlir),
                "--mode", "dense_rc", "--N", str(N), "--B", str(B),
                 "--Din", str(args.Din), "--T", str(args.T), "--leak", str(args.leak)])

            run(["python3", "scripts/gen_case.py", "--out", str(b_mlir),
                "--mode", "dense_linalg", "--N", str(N), "--B", str(B),
                 "--Din", str(args.Din), "--T", str(args.T), "--leak", str(args.leak)])

        a_low = out / "mlir" / f"{kind}_A_low_{tag}.mlir"
        run([rc_opt, str(a_mlir), "-convert-rc-to-linalg", "-o", str(a_low)])

        a_llvm_dialect = out / "mlir" / f"{kind}_A_llvm_{tag}.mlir"
        with open(a_llvm_dialect, "w") as f:
            run([mlir_opt, str(a_low), *LOWER_TO_LLVM_CORE, "-o", "-"], stdout=f)

        a_ll = out / "llvm" / f"{kind}_A_{tag}.ll"
        with open(a_ll, "w") as f:
            run([args.mlir_translate, "--mlir-to-llvmir",
                str(a_llvm_dialect)], stdout=f)

        b0_llvm_dialect = out / "mlir" / f"{kind}_B0_llvm_{tag}.mlir"
        with open(b0_llvm_dialect, "w") as f:
            run([mlir_opt, str(b_mlir), *LOWER_TO_LLVM_CORE, "-o", "-"], stdout=f)

        b0_ll = out / "llvm" / f"{kind}_B0_{tag}.ll"
        with open(b0_ll, "w") as f:
            run([args.mlir_translate, "--mlir-to-llvmir",
                str(b0_llvm_dialect)], stdout=f)

        b2_best = b2_avg = None
        b2_kv = None
        if enable_b2:
            b2_llvm_dialect = out / "mlir" / f"{kind}_B2_llvm_{tag}.mlir"
            with open(b2_llvm_dialect, "w") as f:
                run([mlir_opt, str(b_mlir), *B2_PIPELINE, "-o", "-"], stdout=f)

            b2_ll = out / "llvm" / f"{kind}_B2_{tag}.ll"
            with open(b2_ll, "w") as f:
                run([args.mlir_translate, "--mlir-to-llvmir",
                    str(b2_llvm_dialect)], stdout=f)

        # Compile
        a_exe = out / "bin" / f"{kind}_A_{tag}"
        b0_exe = out / "bin" / f"{kind}_B0_{tag}"
        run([args.clang, *clang_flags, str(driver),
            str(rt_c), str(a_ll), "-lm", "-o", str(a_exe)])
        run([args.clang, *clang_flags, str(driver),
            str(rt_c), str(b0_ll), "-lm", "-o", str(b0_exe)])

        b2_exe = None
        if enable_b2:
            b2_exe = out / "bin" / f"{kind}_B2_{tag}"
            run([args.clang, *clang_flags, str(driver),
                str(rt_c), str(b2_ll), "-lm", "-o", str(b2_exe)])

        # Run
        a_out = run([str(a_exe), str(args.warmup), str(args.runs)])
        b0_out = run([str(b0_exe), str(args.warmup), str(args.runs)])
        b2_out = run([str(b2_exe), str(args.warmup),
                     str(args.runs)]) if enable_b2 else None

        a_kv = parse_kv(a_out)
        b0_kv = parse_kv(b0_out)
        a_best, a_avg = int(a_kv["best_ns"]), float(a_kv["avg_ns"])
        b0_best, b0_avg = int(b0_kv["best_ns"]), float(b0_kv["avg_ns"])

        speedup_b0_best = (a_best / b0_best) if b0_best > 0 else 0.0
        speedup_b0_avg = (a_avg / b0_avg) if b0_avg > 0 else 0.0

        speedup_b2_best = speedup_b2_avg = None
        if enable_b2 and b2_out is not None:
            b2_kv = parse_kv(b2_out)
            b2_best, b2_avg = int(b2_kv["best_ns"]), float(b2_kv["avg_ns"])
            speedup_b2_best = (a_best / b2_best) if b2_best > 0 else 0.0
            speedup_b2_avg = (a_avg / b2_avg) if b2_avg > 0 else 0.0

        Din_val = args.Din
        T_val = args.T
        leak_val = args.leak
        if args.task == "esn_mg":
            Din_val = esn_Din
            T_val = esn_Ttest
            leak_val = esn_lr

        row = {
            "kind": kind, "N": N, "B": B, "Din": Din_val, "T": T_val, "leak": leak_val,
            "A_best_ns": a_best, "A_avg_ns": a_avg,
            "B0_best_ns": b0_best, "B0_avg_ns": b0_avg,
            "speedup_B0_over_A_best": speedup_b0_best,
            "speedup_B0_over_A_avg": speedup_b0_avg,
        }

        print(
            f"Result: kind={kind} N={N} B={B} Din={Din_val} T={T_val} leak={float(leak_val):.6g}\n"
            f"  A vs B1: A_avg={a_avg:.2f}ns  B1_avg={b0_avg:.2f}ns  (Speedup {speedup_b0_avg:.2f}x)"
            f"  [best: A={a_best:.0f}ns B1={b0_best:.0f}ns ({speedup_b0_best:.2f}x)]"
        )

        if args.task == "esn_mg":
            row["A_rmse"] = float(a_kv.get("rmse", "nan"))
            row["A_r2"] = float(a_kv.get("r2", "nan"))
            row["B0_rmse"] = float(b0_kv.get("rmse", "nan"))
            row["B0_r2"] = float(b0_kv.get("r2", "nan"))
            if enable_b2 and b2_out is not None:
                row["B2_rmse"] = float(b2_kv.get("rmse", "nan"))
                row["B2_r2"] = float(b2_kv.get("r2", "nan"))

        if enable_b2:
            row.update({
                "B2_best_ns": b2_best,
                "B2_avg_ns": b2_avg,
                "speedup_B2_over_A_best": speedup_b2_best,
                "speedup_B2_over_A_avg": speedup_b2_avg,
            })

            print(
                f"  A vs B2: A_avg={a_avg:.2f}ns  B2_avg={b2_avg:.2f}ns  (Speedup {speedup_b2_avg:.2f}x)"
                f"  [best: A={a_best:.0f}ns B2={b2_best:.0f}ns ({speedup_b2_best:.2f}x)]"
            )

        rows.append(row)

    csv_path = out / ("phase1_dense.csv" if args.task ==
                      "dense" else "phase2_esn_mg.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Wrote:", csv_path)


if __name__ == "__main__":
    main()
