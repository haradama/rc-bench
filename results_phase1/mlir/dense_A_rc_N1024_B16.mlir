
module {
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

  func.func @bench() -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %T_idx = arith.constant 10000 : index
    %B_idx = arith.constant 16 : index
    %N_idx = arith.constant 1024 : index

    %N_i32   = arith.constant 1024 : i32
    %B_i32   = arith.constant 16 : i32
    %Din_i32 = arith.constant 64 : i32
    %leak_f32 = arith.constant 0.300000 : f32

    // memrefs (row-major) allocation
    %W    = memref.alloc() : memref<1024x1024xf32>
    %Win  = memref.alloc() : memref<64x1024xf32>
    %u    = memref.alloc() : memref<16x64xf32>
    %x    = memref.alloc() : memref<16x1024xf32>
    %x2   = memref.alloc() : memref<16x1024xf32>

    // scratch buffers (allocated once; reused each step)
    %tmp1 = memref.alloc() : memref<16x1024xf32>
    %tmp2 = memref.alloc() : memref<16x1024xf32>
    %pre  = memref.alloc() : memref<16x1024xf32>

    // cast to memref<*xf32> for ABI-unification (unranked memref)
    %W_u    = memref.cast %W    : memref<1024x1024xf32>   to memref<*xf32>
    %Win_u  = memref.cast %Win  : memref<64x1024xf32> to memref<*xf32>
    %u_u    = memref.cast %u    : memref<16x64xf32> to memref<*xf32>
    %x_u    = memref.cast %x    : memref<16x1024xf32>   to memref<*xf32>
    %x2_u   = memref.cast %x2   : memref<16x1024xf32>   to memref<*xf32>
    %tmp1_u = memref.cast %tmp1 : memref<16x1024xf32>   to memref<*xf32>
    %tmp2_u = memref.cast %tmp2 : memref<16x1024xf32>   to memref<*xf32>
    %pre_u  = memref.cast %pre  : memref<16x1024xf32>   to memref<*xf32>

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
    scf.for %t = %c0 to %T_idx step %c1 {
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
      scf.for %i = %c0 to %B_idx step %c1 {
        scf.for %j = %c0 to %N_idx step %c1 {
          %v = memref.load %x2[%i, %j] : memref<16x1024xf32>
          memref.store %v, %x[%i, %j] : memref<16x1024xf32>
        }
      }
    }

    %e = memref.load %x[%c0, %c0] : memref<16x1024xf32>
    %k = arith.fptosi %e : f32 to i32

    memref.dealloc %W    : memref<1024x1024xf32>
    memref.dealloc %Win  : memref<64x1024xf32>
    memref.dealloc %u    : memref<16x64xf32>
    memref.dealloc %x    : memref<16x1024xf32>
    memref.dealloc %x2   : memref<16x1024xf32>
    memref.dealloc %tmp1 : memref<16x1024xf32>
    memref.dealloc %tmp2 : memref<16x1024xf32>
    memref.dealloc %pre  : memref<16x1024xf32>

    return %k : i32
  }
}
