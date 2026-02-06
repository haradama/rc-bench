#!/usr/bin/env python3
import argparse
from pathlib import Path

def gen_dense_call(N, B, Din, T, leak, mode):
    # フェーズ1: dense_rc / dense_linalg ともに runtime 呼び出しベースで同一IR
    # 重要:
    # - MLIRでは memref<*xf32> を引数にする（i64 rank を手で渡さない）
    # - lowering後に自動で (i64 rank, ptr desc) に展開される
    # - C側は (int64_t rank, void* desc, ...) を受ける想定
    # - rc_dense_step は tmp1/tmp2/pre を引数で受ける（per-step malloc/freeなし）版を想定

    return f"""
module {{
  // C側: void rc_fill2d(int64_t rank, void* desc, int d0, int d1, float scale);
  func.func private @rc_fill2d(memref<*xf32>, i32, i32, f32) -> ()

  // C側: void rc_dense_step(
  //   (rW,dW), (rWin,dWin), (ru,du), (rx,dx), (rx2,dx2),
  //   (rtmp1,dtmp1), (rtmp2,dtmp2), (rpre,dpre),
  //   B,N,Din,leak);
  //
  // ※ MLIRでは memref<*xf32> のみ宣言し、loweringが (i64,ptr) へ展開する
  func.func private @rc_dense_step(
    memref<*xf32>, memref<*xf32>, memref<*xf32>,
    memref<*xf32>, memref<*xf32>,
    memref<*xf32>, memref<*xf32>, memref<*xf32>,
    i32, i32, i32, f32) -> ()

  func.func @bench() -> i32 {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %T_idx = arith.constant {T} : index
    %B_idx = arith.constant {B} : index
    %N_idx = arith.constant {N} : index

    %N_i32   = arith.constant {N} : i32
    %B_i32   = arith.constant {B} : i32
    %Din_i32 = arith.constant {Din} : i32
    %leak_f32 = arith.constant {leak:.6f} : f32

    // memrefs (row-major) allocation
    %W    = memref.alloc() : memref<{N}x{N}xf32>
    %Win  = memref.alloc() : memref<{Din}x{N}xf32>
    %u    = memref.alloc() : memref<{B}x{Din}xf32>
    %x    = memref.alloc() : memref<{B}x{N}xf32>
    %x2   = memref.alloc() : memref<{B}x{N}xf32>

    // scratch buffers (allocated once; reused each step)
    %tmp1 = memref.alloc() : memref<{B}x{N}xf32>
    %tmp2 = memref.alloc() : memref<{B}x{N}xf32>
    %pre  = memref.alloc() : memref<{B}x{N}xf32>

    // cast to memref<*xf32> for ABI-unification (unranked memref)
    %W_u    = memref.cast %W    : memref<{N}x{N}xf32>   to memref<*xf32>
    %Win_u  = memref.cast %Win  : memref<{Din}x{N}xf32> to memref<*xf32>
    %u_u    = memref.cast %u    : memref<{B}x{Din}xf32> to memref<*xf32>
    %x_u    = memref.cast %x    : memref<{B}x{N}xf32>   to memref<*xf32>
    %x2_u   = memref.cast %x2   : memref<{B}x{N}xf32>   to memref<*xf32>
    %tmp1_u = memref.cast %tmp1 : memref<{B}x{N}xf32>   to memref<*xf32>
    %tmp2_u = memref.cast %tmp2 : memref<{B}x{N}xf32>   to memref<*xf32>
    %pre_u  = memref.cast %pre  : memref<{B}x{N}xf32>   to memref<*xf32>

    // fill scales
    %sW   = arith.constant 0.05 : f32
    %sWin = arith.constant 0.10 : f32
    %sU   = arith.constant 1.00 : f32
    %sX   = arith.constant 1.00 : f32

    // init
    func.call @rc_fill2d(%W_u,   %N_i32,   %N_i32,   %sW)
      : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%Win_u, %Din_i32, %N_i32,   %sWin)
      : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%u_u,   %B_i32,   %Din_i32, %sU)
      : (memref<*xf32>, i32, i32, f32) -> ()
    func.call @rc_fill2d(%x_u,   %B_i32,   %N_i32,   %sX)
      : (memref<*xf32>, i32, i32, f32) -> ()

    // Loop T steps
    scf.for %t = %c0 to %T_idx step %c1 {{
      func.call @rc_dense_step(
        %W_u, %Win_u, %u_u,
        %x_u, %x2_u,
        %tmp1_u, %tmp2_u, %pre_u,
        %B_i32, %N_i32, %Din_i32, %leak_f32
      ) : (
        memref<*xf32>, memref<*xf32>, memref<*xf32>,
        memref<*xf32>, memref<*xf32>,
        memref<*xf32>, memref<*xf32>, memref<*xf32>,
        i32, i32, i32, f32
      ) -> ()

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

    text = gen_dense_call(args.N, args.B, args.Din, args.T, args.leak, args.mode)
    Path(args.out).write_text(text)

if __name__ == "__main__":
    main()
