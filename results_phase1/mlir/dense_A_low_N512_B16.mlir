module {
  func.func private @rc_fill2d(memref<*xf32>, i32, i32, f32)
  func.func private @rc_dense_step(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, f32)
  func.func @bench() -> i32 {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %cst_1 = arith.constant 5.000000e-02 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10000 = arith.constant 10000 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %c512_i32 = arith.constant 512 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant 3.000000e-01 : f32
    %alloc = memref.alloc() : memref<512x512xf32>
    %alloc_3 = memref.alloc() : memref<64x512xf32>
    %alloc_4 = memref.alloc() : memref<16x64xf32>
    %alloc_5 = memref.alloc() : memref<16x512xf32>
    %alloc_6 = memref.alloc() : memref<16x512xf32>
    %alloc_7 = memref.alloc() : memref<16x512xf32>
    %alloc_8 = memref.alloc() : memref<16x512xf32>
    %alloc_9 = memref.alloc() : memref<16x512xf32>
    %cast = memref.cast %alloc : memref<512x512xf32> to memref<*xf32>
    %cast_10 = memref.cast %alloc_3 : memref<64x512xf32> to memref<*xf32>
    %cast_11 = memref.cast %alloc_4 : memref<16x64xf32> to memref<*xf32>
    %cast_12 = memref.cast %alloc_5 : memref<16x512xf32> to memref<*xf32>
    %cast_13 = memref.cast %alloc_6 : memref<16x512xf32> to memref<*xf32>
    %cast_14 = memref.cast %alloc_7 : memref<16x512xf32> to memref<*xf32>
    %cast_15 = memref.cast %alloc_8 : memref<16x512xf32> to memref<*xf32>
    %cast_16 = memref.cast %alloc_9 : memref<16x512xf32> to memref<*xf32>
    call @rc_fill2d(%cast, %c512_i32, %c512_i32, %cst_1) : (memref<*xf32>, i32, i32, f32) -> ()
    call @rc_fill2d(%cast_10, %c64_i32, %c512_i32, %cst_0) : (memref<*xf32>, i32, i32, f32) -> ()
    call @rc_fill2d(%cast_11, %c16_i32, %c64_i32, %cst) : (memref<*xf32>, i32, i32, f32) -> ()
    call @rc_fill2d(%cast_12, %c16_i32, %c512_i32, %cst) : (memref<*xf32>, i32, i32, f32) -> ()
    scf.for %arg0 = %c0 to %c10000 step %c1 {
      func.call @rc_dense_step(%cast, %cast_10, %cast_11, %cast_12, %cast_13, %cast_14, %cast_15, %cast_16, %c16_i32, %c512_i32, %c64_i32, %cst_2) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>, i32, i32, i32, f32) -> ()
      scf.for %arg1 = %c0 to %c16 step %c1 {
        scf.for %arg2 = %c0 to %c512 step %c1 {
          %2 = memref.load %alloc_6[%arg1, %arg2] : memref<16x512xf32>
          memref.store %2, %alloc_5[%arg1, %arg2] : memref<16x512xf32>
        }
      }
    }
    %0 = memref.load %alloc_5[%c0, %c0] : memref<16x512xf32>
    %1 = arith.fptosi %0 : f32 to i32
    memref.dealloc %alloc : memref<512x512xf32>
    memref.dealloc %alloc_3 : memref<64x512xf32>
    memref.dealloc %alloc_4 : memref<16x64xf32>
    memref.dealloc %alloc_5 : memref<16x512xf32>
    memref.dealloc %alloc_6 : memref<16x512xf32>
    memref.dealloc %alloc_7 : memref<16x512xf32>
    memref.dealloc %alloc_8 : memref<16x512xf32>
    memref.dealloc %alloc_9 : memref<16x512xf32>
    return %1 : i32
  }
}

