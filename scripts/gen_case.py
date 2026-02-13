#!/usr/bin/env python3
import argparse
from pathlib import Path

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
      // IMPORTANT: linalg.matmul is C += A*B, so outputs must be zeroed each step.
      linalg.fill ins(%c0f : f32) outs(%tmp1 : memref<{B}x{N}xf32>)
      linalg.fill ins(%c0f : f32) outs(%tmp2 : memref<{B}x{N}xf32>)

      // tmp1 += x*W
      linalg.matmul ins(%x, %W : memref<{B}x{N}xf32>, memref<{N}x{N}xf32>)
                   outs(%tmp1 : memref<{B}x{N}xf32>)

      // tmp2 += u*Win
      linalg.matmul ins(%u, %Win : memref<{B}x{Din}xf32>, memref<{Din}x{N}xf32>)
                   outs(%tmp2 : memref<{B}x{N}xf32>)

      // pre = tmp1 + tmp2
      linalg.generic
        {{ indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
           iterator_types = ["parallel", "parallel"] }}
        ins(%tmp1, %tmp2 : memref<{B}x{N}xf32>, memref<{B}x{N}xf32>)
        outs(%pre : memref<{B}x{N}xf32>) {{
          ^bb0(%a: f32, %b: f32, %out: f32):
            %s = arith.addf %a, %b : f32
            linalg.yield %s : f32
        }}

      // pre = tanh(pre) in-place
      linalg.generic
        {{ indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,j)>],
           iterator_types = ["parallel", "parallel"] }}
        ins(%pre : memref<{B}x{N}xf32>)
        outs(%pre : memref<{B}x{N}xf32>) {{
          ^bb0(%a: f32, %out: f32):
            %th = math.tanh %a : f32
            linalg.yield %th : f32
        }}

      // x2 = (1-leak)*x + leak*pre
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["dense_rc", "dense_linalg"])
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--Din", type=int, default=64)
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--leak", type=float, default=0.3)
    args = ap.parse_args()

    if args.mode == "dense_rc":
        text = gen_case_dense_rc(args.N, args.B, args.Din, args.T, args.leak)
    else:
        text = gen_case_dense_linalg(args.N, args.B, args.Din, args.T, args.leak)

    Path(args.out).write_text(text)

if __name__ == "__main__":
    main()
