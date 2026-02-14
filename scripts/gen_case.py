#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def gen_case_dense_rc(N, B, Din, T, leak):
    return f"""
module {{
  // init helper
  func.func private @rc_fill2d(memref<*xf32>, i32, i32, f32) -> ()

  func.func @bench() -> i32 {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %T_idx = arith.constant {T} : index
    %B_idx = arith.constant {B} : index
    %N_idx = arith.constant {N} : index

    %N_i32   = arith.constant {N} : i32
    %B_i32   = arith.constant {B} : i32
    %Din_i32 = arith.constant {Din} : i32
    %leak = arith.constant {leak:.6f} : f32

    // main buffers
    %W    = memref.alloc() : memref<{N}x{N}xf32>
    %Win  = memref.alloc() : memref<{Din}x{N}xf32>
    %u    = memref.alloc() : memref<{B}x{Din}xf32>
    %x    = memref.alloc() : memref<{B}x{N}xf32>
    %x2   = memref.alloc() : memref<{B}x{N}xf32>

    // scratch (allocated once, reused)
    %tmp1 = memref.alloc() : memref<{B}x{N}xf32>
    %tmp2 = memref.alloc() : memref<{B}x{N}xf32>
    %pre  = memref.alloc() : memref<{B}x{N}xf32>

    // unranked for init helper
    %W_u   = memref.cast %W   : memref<{N}x{N}xf32>   to memref<*xf32>
    %Win_u = memref.cast %Win : memref<{Din}x{N}xf32> to memref<*xf32>
    %u_u   = memref.cast %u   : memref<{B}x{Din}xf32> to memref<*xf32>
    %x_u   = memref.cast %x   : memref<{B}x{N}xf32>   to memref<*xf32>

    %sW   = arith.constant 0.05 : f32
    %sWin = arith.constant 0.10 : f32
    %sU   = arith.constant 1.00 : f32
    %sX   = arith.constant 1.00 : f32

    func.call @rc_fill2d(%W_u,   %N_i32,   %N_i32,   %sW)   : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%Win_u, %Din_i32, %N_i32,   %sWin) : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%u_u,   %B_i32,   %Din_i32, %sU)   : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%x_u,   %B_i32,   %N_i32,   %sX)   : (memref<*xf32>, i32, i32, f32) -> ()

    scf.for %t = %c0 to %T_idx step %c1 {{
      // A side: single rc op (lowered to linalg by -convert-rc-to-linalg)
      rc.reservoir_step_dense %W, %Win, %u, %x, %x2, %tmp1, %tmp2, %pre {{ leak = {leak:.6f} : f32 }}
        : (memref<{N}x{N}xf32>, memref<{Din}x{N}xf32>, memref<{B}x{Din}xf32>, memref<{B}x{N}xf32>,
           memref<{B}x{N}xf32>, memref<{B}x{N}xf32>, memref<{B}x{N}xf32>, memref<{B}x{N}xf32>) -> ()

      // copy x2 -> x
      scf.for %i = %c0 to %B_idx step %c1 {{
        scf.for %j = %c0 to %N_idx step %c1 {{
          %v = memref.load %x2[%i, %j] : memref<{B}x{N}xf32>
          memref.store %v, %x[%i, %j] : memref<{B}x{N}xf32>
        }}
      }}
    }}

    %e = memref.load %x[%c0, %c0] : memref<{B}x{N}xf32>
    %k = arith.fptosi %e : f32 to i32

    memref.dealloc %W    : memref<{N}x{N}xf32>
    memref.dealloc %Win  : memref<{Din}x{N}xf32>
    memref.dealloc %u    : memref<{B}x{Din}xf32>
    memref.dealloc %x    : memref<{B}x{N}xf32>
    memref.dealloc %x2   : memref<{B}x{N}xf32>
    memref.dealloc %tmp1 : memref<{B}x{N}xf32>
    memref.dealloc %tmp2 : memref<{B}x{N}xf32>
    memref.dealloc %pre  : memref<{B}x{N}xf32>

    return %k : i32
  }}
}}
"""


def gen_case_dense_linalg(N, B, Din, T, leak):
    return f"""
module {{
  func.func private @rc_fill2d(memref<*xf32>, i32, i32, f32) -> ()

  func.func @bench() -> i32 {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %T_idx = arith.constant {T} : index
    %B_idx = arith.constant {B} : index
    %N_idx = arith.constant {N} : index

    %N_i32   = arith.constant {N} : i32
    %B_i32   = arith.constant {B} : i32
    %Din_i32 = arith.constant {Din} : i32

    %leak = arith.constant {leak:.6f} : f32
    %one  = arith.constant 1.000000 : f32
    %one_minus = arith.subf %one, %leak : f32
    %c0f = arith.constant 0.000000 : f32

    %W    = memref.alloc() : memref<{N}x{N}xf32>
    %Win  = memref.alloc() : memref<{Din}x{N}xf32>
    %u    = memref.alloc() : memref<{B}x{Din}xf32>
    %x    = memref.alloc() : memref<{B}x{N}xf32>
    %x2   = memref.alloc() : memref<{B}x{N}xf32>

    %tmp1 = memref.alloc() : memref<{B}x{N}xf32>
    %tmp2 = memref.alloc() : memref<{B}x{N}xf32>
    %pre  = memref.alloc() : memref<{B}x{N}xf32>

    %W_u   = memref.cast %W   : memref<{N}x{N}xf32>   to memref<*xf32>
    %Win_u = memref.cast %Win : memref<{Din}x{N}xf32> to memref<*xf32>
    %u_u   = memref.cast %u   : memref<{B}x{Din}xf32> to memref<*xf32>
    %x_u   = memref.cast %x   : memref<{B}x{N}xf32>   to memref<*xf32>

    %sW   = arith.constant 0.05 : f32
    %sWin = arith.constant 0.10 : f32
    %sU   = arith.constant 1.00 : f32
    %sX   = arith.constant 1.00 : f32

    func.call @rc_fill2d(%W_u,   %N_i32,   %N_i32,   %sW)   : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%Win_u, %Din_i32, %N_i32,   %sWin) : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%u_u,   %B_i32,   %Din_i32, %sU)   : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%x_u,   %B_i32,   %N_i32,   %sX)   : (memref<*xf32>, i32, i32, f32) -> ()

    scf.for %t = %c0 to %T_idx step %c1 {{
      linalg.fill ins(%c0f : f32) outs(%tmp1 : memref<{B}x{N}xf32>)
      linalg.fill ins(%c0f : f32) outs(%tmp2 : memref<{B}x{N}xf32>)

      linalg.matmul ins(%x, %W : memref<{B}x{N}xf32>, memref<{N}x{N}xf32>)
                   outs(%tmp1 : memref<{B}x{N}xf32>)

      linalg.matmul ins(%u, %Win : memref<{B}x{Din}xf32>, memref<{Din}x{N}xf32>)
                   outs(%tmp2 : memref<{B}x{N}xf32>)

      linalg.generic
        {{ indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
           iterator_types = ["parallel", "parallel"] }}
        ins(%tmp1, %tmp2 : memref<{B}x{N}xf32>, memref<{B}x{N}xf32>)
        outs(%pre : memref<{B}x{N}xf32>) {{
          ^bb0(%a: f32, %b: f32, %out: f32):
            %s = arith.addf %a, %b : f32
            linalg.yield %s : f32
        }}

      linalg.generic
        {{ indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
           iterator_types = ["parallel", "parallel"] }}
        ins(%pre : memref<{B}x{N}xf32>)
        outs(%pre : memref<{B}x{N}xf32>) {{
          ^bb0(%a: f32, %out: f32):
            %th = math.tanh %a : f32
            linalg.yield %th : f32
        }}

      linalg.generic
        {{ indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
           iterator_types = ["parallel", "parallel"] }}
        ins(%x, %pre : memref<{B}x{N}xf32>, memref<{B}x{N}xf32>)
        outs(%x2 : memref<{B}x{N}xf32>) {{
          ^bb0(%xv: f32, %pv: f32, %out: f32):
            %t1 = arith.mulf %xv, %one_minus : f32
            %t2 = arith.mulf %pv, %leak : f32
            %y  = arith.addf %t1, %t2 : f32
            linalg.yield %y : f32
        }}

      scf.for %i = %c0 to %B_idx step %c1 {{
        scf.for %j = %c0 to %N_idx step %c1 {{
          %v = memref.load %x2[%i, %j] : memref<{B}x{N}xf32>
          memref.store %v, %x[%i, %j] : memref<{B}x{N}xf32>
        }}
      }}
    }}

    %e = memref.load %x[%c0, %c0] : memref<{B}x{N}xf32>
    %k = arith.fptosi %e : f32 to i32

    memref.dealloc %W    : memref<{N}x{N}xf32>
    memref.dealloc %Win  : memref<{Din}x{N}xf32>
    memref.dealloc %u    : memref<{B}x{Din}xf32>
    memref.dealloc %x    : memref<{B}x{N}xf32>
    memref.dealloc %x2   : memref<{B}x{N}xf32>
    memref.dealloc %tmp1 : memref<{B}x{N}xf32>
    memref.dealloc %tmp2 : memref<{B}x{N}xf32>
    memref.dealloc %pre  : memref<{B}x{N}xf32>

    return %k : i32
  }}
}}
"""


def _fmt_f32(x: float) -> str:
    return f"{np.float32(x):.8g}"


def _dense_attr(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        xs = ", ".join(_fmt_f32(v) for v in arr.tolist())
        return f"dense<[{xs}]>"
    if arr.ndim == 2:
        rows = []
        for r in arr.tolist():
            rows.append("[" + ", ".join(_fmt_f32(v) for v in r) + "]")
        return "dense<[" + ", ".join(rows) + "]>"
    raise ValueError(f"ndim not supported: {arr.ndim}")


def emit_memref_global(f, name: str, arr: np.ndarray, constant=True, visibility="private", init_splat_zero=False):
    arr = np.asarray(arr, dtype=np.float32)
    shape = "x".join(str(d) for d in arr.shape)
    ty = f"memref<{shape}xf32>"
    const_kw = " constant" if constant else ""
    init = "dense<0.0>" if init_splat_zero else _dense_attr(arr)
    f.write(
        f'  memref.global "{visibility}"{const_kw} @{name} : {ty} = {init}\n')


def gen_esn_mg_mlir(out_path: str, assets_path: str, use_rc: bool):
    z = np.load(assets_path)

    N = int(z["N"])
    Din = int(z["Din"])
    Dout = int(z["Dout"])
    Ttest = int(z["Ttest"])
    lr = float(z["lr"])

    W = z["W"].astype(np.float32)        # (N,N)
    WinT = z["WinT"].astype(np.float32)     # (Din,N)
    Wout = z["Wout"].astype(np.float32)     # (N,Dout)
    bout = z["bias_out"].astype(np.float32)  # (1,Dout)
    Xtest = z["x_test"].astype(np.float32)   # (Ttest,Din)
    Ytest = z["y_test"].astype(np.float32)   # (Ttest,Dout)
    x0 = z["x0"].astype(np.float32)       # (1,N)

    assert W.shape == (N, N)
    assert WinT.shape == (Din, N)
    assert Wout.shape == (N, Dout)
    assert bout.shape == (1, Dout)
    assert Xtest.shape == (Ttest, Din)
    assert Ytest.shape == (Ttest, Dout)
    assert x0.shape == (1, N)

    def emit_infer(f, store_pred: bool, fn_name: str):
        f.write(f"  func.func @{fn_name}() -> i32 {{\n")
        f.write("    %c0 = arith.constant 0 : index\n")
        f.write("    %c1 = arith.constant 1 : index\n")
        f.write(f"    %cT = arith.constant {Ttest} : index\n")
        f.write(f"    %cDin = arith.constant {Din} : index\n")
        f.write(f"    %cDout = arith.constant {Dout} : index\n")
        f.write(f"    %cN = arith.constant {N} : index\n")
        f.write("    %c0f = arith.constant 0.0 : f32\n")
        f.write(f"    %leak = arith.constant {lr:.8g} : f32\n")
        f.write("    %one = arith.constant 1.0 : f32\n")
        f.write("    %one_minus = arith.subf %one, %leak : f32\n")

        # globals
        f.write(f"    %WT = memref.get_global @WT : memref<{N}x{N}xf32>\n")
        f.write(
            f"    %Win = memref.get_global @WinT : memref<{Din}x{N}xf32>\n")
        f.write(
            f"    %Wout = memref.get_global @Wout : memref<{N}x{Dout}xf32>\n")
        f.write(
            f"    %bout = memref.get_global @bout : memref<1x{Dout}xf32>\n")
        f.write(
            f"    %X = memref.get_global @Xtest : memref<{Ttest}x{Din}xf32>\n")
        f.write(f"    %x0 = memref.get_global @x0 : memref<1x{N}xf32>\n")
        if store_pred:
            f.write(
                f"    %pred = memref.get_global @pred : memref<{Ttest}x{Dout}xf32>\n")

        # locals (B=1)
        f.write(f"    %u  = memref.alloc() : memref<1x{Din}xf32>\n")
        f.write(f"    %x  = memref.alloc() : memref<1x{N}xf32>\n")
        f.write(f"    %x2 = memref.alloc() : memref<1x{N}xf32>\n")
        f.write(f"    %y  = memref.alloc() : memref<1x{Dout}xf32>\n")
        f.write(f"    %tmp1 = memref.alloc() : memref<1x{N}xf32>\n")
        f.write(f"    %tmp2 = memref.alloc() : memref<1x{N}xf32>\n")
        f.write(f"    %pre  = memref.alloc() : memref<1x{N}xf32>\n")

        # init x <- x0
        f.write(
            f"    memref.copy %x0, %x : memref<1x{N}xf32> to memref<1x{N}xf32>\n")
        f.write("    %sink0 = arith.constant 0 : i32\n")
        f.write(
            "    %sink = scf.for %t = %c0 to %cT step %c1 iter_args(%acc = %sink0) -> (i32) {\n")

        # load u from X[t,:]
        f.write("      scf.for %d = %c0 to %cDin step %c1 {\n")
        f.write(
            f"        %v = memref.load %X[%t, %d] : memref<{Ttest}x{Din}xf32>\n")
        f.write(
            f"        memref.store %v, %u[%c0, %d] : memref<1x{Din}xf32>\n")
        f.write("      }\n")

        if use_rc:
            f.write(
                f"      rc.reservoir_step_dense %WT, %Win, %u, %x, %x2, %tmp1, %tmp2, %pre {{ leak = {lr:.8g} : f32 }}\n")
            f.write(
                f"        : (memref<{N}x{N}xf32>, memref<{Din}x{N}xf32>, memref<1x{Din}xf32>, memref<1x{N}xf32>,\n")
            f.write(
                f"           memref<1x{N}xf32>, memref<1x{N}xf32>, memref<1x{N}xf32>, memref<1x{N}xf32>) -> ()\n")
        else:
            f.write("      linalg.fill ins(%c0f : f32) outs(%tmp1 : memref<1x{N}xf32>)\n".replace(
                "{N}", str(N)))
            f.write("      linalg.fill ins(%c0f : f32) outs(%tmp2 : memref<1x{N}xf32>)\n".replace(
                "{N}", str(N)))

            f.write("      linalg.matmul ins(%x, %WT : memref<1x{N}xf32>, memref<{N}x{N}xf32>) outs(%tmp1 : memref<1x{N}xf32>)\n"
                    .replace("{N}", str(N)))
            f.write("      linalg.matmul ins(%u, %Win : memref<1x{Din}xf32>, memref<{Din}x{N}xf32>) outs(%tmp2 : memref<1x{N}xf32>)\n"
                    .replace("{Din}", str(Din)).replace("{N}", str(N)))

            f.write(
                "      linalg.generic { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = [\"parallel\", \"parallel\"] }\n")
            f.write("        ins(%tmp1, %tmp2 : memref<1x{N}xf32>, memref<1x{N}xf32>) outs(%pre : memref<1x{N}xf32>) {\n"
                    .replace("{N}", str(N)))
            f.write("          ^bb0(%a0: f32, %b0: f32, %out0: f32):\n")
            f.write("            %sum0 = arith.addf %a0, %b0 : f32\n")
            f.write("            linalg.yield %sum0 : f32\n")

            f.write("        }\n")

            f.write(
                "      linalg.generic { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = [\"parallel\", \"parallel\"] }\n")
            f.write("        ins(%pre : memref<1x{N}xf32>) outs(%pre : memref<1x{N}xf32>) {\n".replace(
                "{N}", str(N)))
            f.write("          ^bb0(%in0: f32, %out0: f32):\n")
            f.write("            %th0 = math.tanh %in0 : f32\n")
            f.write("            linalg.yield %th0 : f32\n")

            f.write("        }\n")

            f.write(
                "      linalg.generic { indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>], iterator_types = [\"parallel\", \"parallel\"] }\n")
            f.write("        ins(%x, %pre : memref<1x{N}xf32>, memref<1x{N}xf32>) outs(%x2 : memref<1x{N}xf32>) {\n"
                    .replace("{N}", str(N)))
            f.write("          ^bb0(%xv0: f32, %pv0: f32, %out0: f32):\n")
            f.write("            %t1_0 = arith.mulf %xv0, %one_minus : f32\n")
            f.write("            %t2_0 = arith.mulf %pv0, %leak : f32\n")
            f.write("            %upd0 = arith.addf %t1_0, %t2_0 : f32\n")
            f.write("            linalg.yield %upd0 : f32\n")

            f.write("        }\n")

        # copy x2 -> x
        f.write("      scf.for %j = %c0 to %cN step %c1 {\n")
        f.write(
            f"        %vv = memref.load %x2[%c0, %j] : memref<1x{N}xf32>\n")
        f.write(f"        memref.store %vv, %x[%c0, %j] : memref<1x{N}xf32>\n")
        f.write("      }\n")

        # readout: y = x2@Wout + bout
        f.write(
            f"      linalg.fill ins(%c0f : f32) outs(%y : memref<1x{Dout}xf32>)\n")
        f.write(
            f"      linalg.matmul ins(%x2, %Wout : memref<1x{N}xf32>, memref<{N}x{Dout}xf32>) outs(%y : memref<1x{Dout}xf32>)\n")
        f.write("      scf.for %j = %c0 to %cDout step %c1 {\n")
        f.write(
            f"        %bias = memref.load %bout[%c0, %j] : memref<1x{Dout}xf32>\n")
        f.write(
            f"        %yv = memref.load %y[%c0, %j] : memref<1x{Dout}xf32>\n")
        f.write("        %yvb = arith.addf %yv, %bias : f32\n")
        f.write(
            f"        memref.store %yvb, %y[%c0, %j] : memref<1x{Dout}xf32>\n")
        if store_pred:
            f.write(
                f"        memref.store %yvb, %pred[%t, %j] : memref<{Ttest}x{Dout}xf32>\n")
        f.write("      }\n")

        # sink
        f.write(
            f"      %y00 = memref.load %y[%c0, %c0] : memref<1x{Dout}xf32>\n")
        f.write("      %scale = arith.constant 1000000.0 : f32\n")
        f.write("      %ys = arith.mulf %y00, %scale : f32\n")
        f.write("      %yi = arith.fptosi %ys : f32 to i32\n")
        f.write("      %acc2 = arith.xori %acc, %yi : i32\n")
        f.write("      scf.yield %acc2 : i32\n")
        f.write("    }\n")  # end for

        # deallocs
        f.write(f"    memref.dealloc %pre : memref<1x{N}xf32>\n")
        f.write(f"    memref.dealloc %tmp2 : memref<1x{N}xf32>\n")
        f.write(f"    memref.dealloc %tmp1 : memref<1x{N}xf32>\n")
        f.write(f"    memref.dealloc %y : memref<1x{Dout}xf32>\n")
        f.write(f"    memref.dealloc %x2 : memref<1x{N}xf32>\n")
        f.write(f"    memref.dealloc %x : memref<1x{N}xf32>\n")
        f.write(f"    memref.dealloc %u : memref<1x{Din}xf32>\n")

        f.write("    return %sink : i32\n")
        f.write("  }\n")

    def emit_rmse_r2(f):
        # RMSE
        f.write("  func.func @rmse() -> f64 {\n")
        f.write("    %c0 = arith.constant 0 : index\n")
        f.write("    %c1 = arith.constant 1 : index\n")
        f.write(f"    %cT = arith.constant {Ttest} : index\n")
        f.write(f"    %cD = arith.constant {Dout} : index\n")
        f.write("    %z = arith.constant 0.0 : f64\n")
        f.write(
            f"    %pred = memref.get_global @pred : memref<{Ttest}x{Dout}xf32>\n")
        f.write(
            f"    %Y = memref.get_global @Ytest : memref<{Ttest}x{Dout}xf32>\n")
        f.write(
            "    %sse = scf.for %t = %c0 to %cT step %c1 iter_args(%acc = %z) -> (f64) {\n")
        f.write(
            "      %acc2 = scf.for %j = %c0 to %cD step %c1 iter_args(%a2 = %acc) -> (f64) {\n")
        f.write(
            f"        %p = memref.load %pred[%t, %j] : memref<{Ttest}x{Dout}xf32>\n")
        f.write(
            f"        %y = memref.load %Y[%t, %j] : memref<{Ttest}x{Dout}xf32>\n")
        f.write("        %pf = arith.extf %p : f32 to f64\n")
        f.write("        %yf = arith.extf %y : f32 to f64\n")
        f.write("        %d = arith.subf %pf, %yf : f64\n")
        f.write("        %d2 = arith.mulf %d, %d : f64\n")
        f.write("        %a3 = arith.addf %a2, %d2 : f64\n")
        f.write("        scf.yield %a3 : f64\n")
        f.write("      }\n")
        f.write("      scf.yield %acc2 : f64\n")
        f.write("    }\n")
        n = float(Ttest * Dout)
        f.write(f"    %n = arith.constant {n:.1f} : f64\n")
        f.write("    %mse = arith.divf %sse, %n : f64\n")
        f.write("    %rmse = math.sqrt %mse : f64\n")
        f.write("    return %rmse : f64\n")
        f.write("  }\n")

        # R^2
        f.write("  func.func @r2() -> f64 {\n")
        f.write("    %c0 = arith.constant 0 : index\n")
        f.write("    %c1 = arith.constant 1 : index\n")
        f.write(f"    %cT = arith.constant {Ttest} : index\n")
        f.write(f"    %cD = arith.constant {Dout} : index\n")
        f.write("    %z = arith.constant 0.0 : f64\n")
        f.write(
            f"    %pred = memref.get_global @pred : memref<{Ttest}x{Dout}xf32>\n")
        f.write(
            f"    %Y = memref.get_global @Ytest : memref<{Ttest}x{Dout}xf32>\n")

        n = float(Ttest * Dout)
        f.write(f"    %n = arith.constant {n:.1f} : f64\n")

        # mean(Y)
        f.write(
            "    %sum = scf.for %t = %c0 to %cT step %c1 iter_args(%acc = %z) -> (f64) {\n")
        f.write(
            "      %acc2 = scf.for %j = %c0 to %cD step %c1 iter_args(%a2 = %acc) -> (f64) {\n")
        f.write(
            f"        %y = memref.load %Y[%t, %j] : memref<{Ttest}x{Dout}xf32>\n")
        f.write("        %yf = arith.extf %y : f32 to f64\n")
        f.write("        %a3 = arith.addf %a2, %yf : f64\n")
        f.write("        scf.yield %a3 : f64\n")
        f.write("      }\n")
        f.write("      scf.yield %acc2 : f64\n")
        f.write("    }\n")
        f.write("    %mean = arith.divf %sum, %n : f64\n")

        # ss_tot and ss_res
        f.write(
            "    %pair:2 = scf.for %t = %c0 to %cT step %c1 iter_args(%tot = %z, %res = %z) -> (f64, f64) {\n")
        f.write(
            "      %pair2:2 = scf.for %j = %c0 to %cD step %c1 iter_args(%t2 = %tot, %r2 = %res) -> (f64, f64) {\n")
        f.write(
            f"        %y = memref.load %Y[%t, %j] : memref<{Ttest}x{Dout}xf32>\n")
        f.write(
            f"        %p = memref.load %pred[%t, %j] : memref<{Ttest}x{Dout}xf32>\n")
        f.write("        %yf = arith.extf %y : f32 to f64\n")
        f.write("        %pf = arith.extf %p : f32 to f64\n")
        f.write("        %dy = arith.subf %yf, %mean : f64\n")
        f.write("        %dy2 = arith.mulf %dy, %dy : f64\n")
        f.write("        %d = arith.subf %yf, %pf : f64\n")
        f.write("        %d2 = arith.mulf %d, %d : f64\n")
        f.write("        %t3 = arith.addf %t2, %dy2 : f64\n")
        f.write("        %r3 = arith.addf %r2, %d2 : f64\n")
        f.write("        scf.yield %t3, %r3 : f64, f64\n")
        f.write("      }\n")
        f.write("      scf.yield %pair2#0, %pair2#1 : f64, f64\n")
        f.write("    }\n")

        f.write("    %z2 = arith.constant 0.0 : f64\n")
        f.write("    %is0 = arith.cmpf oeq, %pair#0, %z2 : f64\n")

        f.write("    %out = scf.if %is0 -> (f64) {\n")
        f.write("      scf.yield %z2 : f64\n")
        f.write("    } else {\n")
        f.write("      %frac = arith.divf %pair#1, %pair#0 : f64\n")
        f.write("      %one = arith.constant 1.0 : f64\n")
        f.write("      %r = arith.subf %one, %frac : f64\n")
        f.write("      scf.yield %r : f64\n")
        f.write("    }\n")
        f.write("    return %out : f64\n")
        f.write("  }\n")

    with open(out_path, "w") as f:
        f.write("module {\n")
        emit_memref_global(f, "W", W, constant=True)
        emit_memref_global(f, "WinT", WinT, constant=True)
        emit_memref_global(f, "Wout", Wout, constant=True)
        emit_memref_global(f, "bout", bout, constant=True)
        emit_memref_global(f, "Xtest", Xtest, constant=True)
        emit_memref_global(f, "Ytest", Ytest, constant=True)
        emit_memref_global(f, "x0", x0, constant=True)
        emit_memref_global(f, "pred", np.zeros(
            (Ttest, Dout), np.float32), constant=False, init_splat_zero=True)
        WT = z["WT"].astype(np.float32)  # ★追加
        emit_memref_global(f, "WT", WT, constant=True)

        emit_infer(f, store_pred=False, fn_name="bench")
        emit_infer(f, store_pred=True,  fn_name="infer_store")
        emit_rmse_r2(f)

        f.write("}\n")

# -----------------------
# main
# -----------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["dense_rc", "dense_linalg", "esn_mg_rc", "esn_mg_linalg"])
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--Din", type=int, default=64)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--assets", type=str, default=None,
                    help="(esn_mg_*) path to .npz")
    args = ap.parse_args()

    if args.mode in ("dense_rc", "dense_linalg"):
        if args.N is None or args.B is None:
            raise SystemExit("dense_* modes require --N and --B")
        if args.mode == "dense_rc":
            text = gen_case_dense_rc(
                args.N, args.B, args.Din, args.T, args.leak)
        else:
            text = gen_case_dense_linalg(
                args.N, args.B, args.Din, args.T, args.leak)
        Path(args.out).write_text(text)
        return

    # esn_mg modes
    if not args.assets:
        raise SystemExit(
            "esn_mg_* modes require --assets path/to/esn_mg_assets.npz")
    use_rc = (args.mode == "esn_mg_rc")
    gen_esn_mg_mlir(args.out, args.assets, use_rc=use_rc)


if __name__ == "__main__":
    main()
