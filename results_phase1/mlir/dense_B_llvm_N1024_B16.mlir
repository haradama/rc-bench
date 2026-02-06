module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @rc_fill2d(i64, !llvm.ptr, i32, i32, f32) attributes {sym_visibility = "private"}
  llvm.func @rc_dense_step(i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i32, i32, i32, f32) attributes {sym_visibility = "private"}
  llvm.func @bench() -> i32 {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(1.000000e-01 : f32) : f32
    %2 = llvm.mlir.constant(5.000000e-02 : f32) : f32
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(10000 : index) : i64
    %6 = llvm.mlir.constant(16 : index) : i64
    %7 = llvm.mlir.constant(1024 : index) : i64
    %8 = llvm.mlir.constant(1024 : i32) : i32
    %9 = llvm.mlir.constant(16 : i32) : i32
    %10 = llvm.mlir.constant(64 : i32) : i32
    %11 = llvm.mlir.constant(3.000000e-01 : f32) : f32
    %12 = llvm.mlir.constant(1024 : index) : i64
    %13 = llvm.mlir.constant(1024 : index) : i64
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(1048576 : index) : i64
    %16 = llvm.mlir.zero : !llvm.ptr
    %17 = llvm.getelementptr %16[%15] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.call @malloc(%18) : (i64) -> !llvm.ptr
    %20 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %19, %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(0 : index) : i64
    %24 = llvm.insertvalue %23, %22[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %12, %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %13, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %13, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %14, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(64 : index) : i64
    %30 = llvm.mlir.constant(1024 : index) : i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.mlir.constant(65536 : index) : i64
    %33 = llvm.mlir.zero : !llvm.ptr
    %34 = llvm.getelementptr %33[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %35 = llvm.ptrtoint %34 : !llvm.ptr to i64
    %36 = llvm.call @malloc(%35) : (i64) -> !llvm.ptr
    %37 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.insertvalue %36, %37[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.insertvalue %36, %38[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.insertvalue %40, %39[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %29, %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %30, %42[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %30, %43[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %31, %44[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.mlir.constant(16 : index) : i64
    %47 = llvm.mlir.constant(64 : index) : i64
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.constant(1024 : index) : i64
    %50 = llvm.mlir.zero : !llvm.ptr
    %51 = llvm.getelementptr %50[%49] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %52 = llvm.ptrtoint %51 : !llvm.ptr to i64
    %53 = llvm.call @malloc(%52) : (i64) -> !llvm.ptr
    %54 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %55 = llvm.insertvalue %53, %54[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %53, %55[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mlir.constant(0 : index) : i64
    %58 = llvm.insertvalue %57, %56[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %46, %58[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %47, %59[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %47, %60[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.insertvalue %48, %61[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.mlir.constant(16 : index) : i64
    %64 = llvm.mlir.constant(1024 : index) : i64
    %65 = llvm.mlir.constant(1 : index) : i64
    %66 = llvm.mlir.constant(16384 : index) : i64
    %67 = llvm.mlir.zero : !llvm.ptr
    %68 = llvm.getelementptr %67[%66] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.call @malloc(%69) : (i64) -> !llvm.ptr
    %71 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.insertvalue %70, %72[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mlir.constant(0 : index) : i64
    %75 = llvm.insertvalue %74, %73[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.insertvalue %63, %75[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %64, %76[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %64, %77[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.insertvalue %65, %78[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mlir.constant(16 : index) : i64
    %81 = llvm.mlir.constant(1024 : index) : i64
    %82 = llvm.mlir.constant(1 : index) : i64
    %83 = llvm.mlir.constant(16384 : index) : i64
    %84 = llvm.mlir.zero : !llvm.ptr
    %85 = llvm.getelementptr %84[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.call @malloc(%86) : (i64) -> !llvm.ptr
    %88 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %89 = llvm.insertvalue %87, %88[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.insertvalue %87, %89[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.mlir.constant(0 : index) : i64
    %92 = llvm.insertvalue %91, %90[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.insertvalue %80, %92[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %81, %93[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.insertvalue %81, %94[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.insertvalue %82, %95[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.mlir.constant(16 : index) : i64
    %98 = llvm.mlir.constant(1024 : index) : i64
    %99 = llvm.mlir.constant(1 : index) : i64
    %100 = llvm.mlir.constant(16384 : index) : i64
    %101 = llvm.mlir.zero : !llvm.ptr
    %102 = llvm.getelementptr %101[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %103 = llvm.ptrtoint %102 : !llvm.ptr to i64
    %104 = llvm.call @malloc(%103) : (i64) -> !llvm.ptr
    %105 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %106 = llvm.insertvalue %104, %105[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.insertvalue %104, %106[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %108 = llvm.mlir.constant(0 : index) : i64
    %109 = llvm.insertvalue %108, %107[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %97, %109[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %98, %110[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %98, %111[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %99, %112[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.mlir.constant(16 : index) : i64
    %115 = llvm.mlir.constant(1024 : index) : i64
    %116 = llvm.mlir.constant(1 : index) : i64
    %117 = llvm.mlir.constant(16384 : index) : i64
    %118 = llvm.mlir.zero : !llvm.ptr
    %119 = llvm.getelementptr %118[%117] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %120 = llvm.ptrtoint %119 : !llvm.ptr to i64
    %121 = llvm.call @malloc(%120) : (i64) -> !llvm.ptr
    %122 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %123 = llvm.insertvalue %121, %122[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.insertvalue %121, %123[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %125 = llvm.mlir.constant(0 : index) : i64
    %126 = llvm.insertvalue %125, %124[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.insertvalue %114, %126[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.insertvalue %115, %127[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %115, %128[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.insertvalue %116, %129[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.mlir.constant(16 : index) : i64
    %132 = llvm.mlir.constant(1024 : index) : i64
    %133 = llvm.mlir.constant(1 : index) : i64
    %134 = llvm.mlir.constant(16384 : index) : i64
    %135 = llvm.mlir.zero : !llvm.ptr
    %136 = llvm.getelementptr %135[%134] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %137 = llvm.ptrtoint %136 : !llvm.ptr to i64
    %138 = llvm.call @malloc(%137) : (i64) -> !llvm.ptr
    %139 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %140 = llvm.insertvalue %138, %139[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.insertvalue %138, %140[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mlir.constant(0 : index) : i64
    %143 = llvm.insertvalue %142, %141[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.insertvalue %131, %143[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %132, %144[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.insertvalue %132, %145[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.insertvalue %133, %146[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.mlir.constant(1 : index) : i64
    %149 = llvm.alloca %148 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %28, %149 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %150 = llvm.mlir.constant(2 : index) : i64
    %151 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %152 = llvm.insertvalue %150, %151[0] : !llvm.struct<(i64, ptr)> 
    %153 = llvm.insertvalue %149, %152[1] : !llvm.struct<(i64, ptr)> 
    %154 = llvm.mlir.constant(1 : index) : i64
    %155 = llvm.alloca %154 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %45, %155 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %156 = llvm.mlir.constant(2 : index) : i64
    %157 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %158 = llvm.insertvalue %156, %157[0] : !llvm.struct<(i64, ptr)> 
    %159 = llvm.insertvalue %155, %158[1] : !llvm.struct<(i64, ptr)> 
    %160 = llvm.mlir.constant(1 : index) : i64
    %161 = llvm.alloca %160 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %62, %161 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %162 = llvm.mlir.constant(2 : index) : i64
    %163 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %164 = llvm.insertvalue %162, %163[0] : !llvm.struct<(i64, ptr)> 
    %165 = llvm.insertvalue %161, %164[1] : !llvm.struct<(i64, ptr)> 
    %166 = llvm.mlir.constant(1 : index) : i64
    %167 = llvm.alloca %166 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %79, %167 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %168 = llvm.mlir.constant(2 : index) : i64
    %169 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %170 = llvm.insertvalue %168, %169[0] : !llvm.struct<(i64, ptr)> 
    %171 = llvm.insertvalue %167, %170[1] : !llvm.struct<(i64, ptr)> 
    %172 = llvm.mlir.constant(1 : index) : i64
    %173 = llvm.alloca %172 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %96, %173 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %174 = llvm.mlir.constant(2 : index) : i64
    %175 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %176 = llvm.insertvalue %174, %175[0] : !llvm.struct<(i64, ptr)> 
    %177 = llvm.insertvalue %173, %176[1] : !llvm.struct<(i64, ptr)> 
    %178 = llvm.mlir.constant(1 : index) : i64
    %179 = llvm.alloca %178 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %113, %179 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %180 = llvm.mlir.constant(2 : index) : i64
    %181 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %182 = llvm.insertvalue %180, %181[0] : !llvm.struct<(i64, ptr)> 
    %183 = llvm.insertvalue %179, %182[1] : !llvm.struct<(i64, ptr)> 
    %184 = llvm.mlir.constant(1 : index) : i64
    %185 = llvm.alloca %184 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %130, %185 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %186 = llvm.mlir.constant(2 : index) : i64
    %187 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %188 = llvm.insertvalue %186, %187[0] : !llvm.struct<(i64, ptr)> 
    %189 = llvm.insertvalue %185, %188[1] : !llvm.struct<(i64, ptr)> 
    %190 = llvm.mlir.constant(1 : index) : i64
    %191 = llvm.alloca %190 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %147, %191 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %192 = llvm.mlir.constant(2 : index) : i64
    %193 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %194 = llvm.insertvalue %192, %193[0] : !llvm.struct<(i64, ptr)> 
    %195 = llvm.insertvalue %191, %194[1] : !llvm.struct<(i64, ptr)> 
    %196 = llvm.extractvalue %153[0] : !llvm.struct<(i64, ptr)> 
    %197 = llvm.extractvalue %153[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%196, %197, %8, %8, %2) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    %198 = llvm.extractvalue %159[0] : !llvm.struct<(i64, ptr)> 
    %199 = llvm.extractvalue %159[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%198, %199, %10, %8, %1) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    %200 = llvm.extractvalue %165[0] : !llvm.struct<(i64, ptr)> 
    %201 = llvm.extractvalue %165[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%200, %201, %9, %10, %0) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    %202 = llvm.extractvalue %171[0] : !llvm.struct<(i64, ptr)> 
    %203 = llvm.extractvalue %171[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%202, %203, %9, %8, %0) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    llvm.br ^bb1(%3 : i64)
  ^bb1(%204: i64):  // 2 preds: ^bb0, ^bb8
    %205 = llvm.icmp "slt" %204, %5 : i64
    llvm.cond_br %205, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %206 = llvm.extractvalue %153[0] : !llvm.struct<(i64, ptr)> 
    %207 = llvm.extractvalue %153[1] : !llvm.struct<(i64, ptr)> 
    %208 = llvm.extractvalue %159[0] : !llvm.struct<(i64, ptr)> 
    %209 = llvm.extractvalue %159[1] : !llvm.struct<(i64, ptr)> 
    %210 = llvm.extractvalue %165[0] : !llvm.struct<(i64, ptr)> 
    %211 = llvm.extractvalue %165[1] : !llvm.struct<(i64, ptr)> 
    %212 = llvm.extractvalue %171[0] : !llvm.struct<(i64, ptr)> 
    %213 = llvm.extractvalue %171[1] : !llvm.struct<(i64, ptr)> 
    %214 = llvm.extractvalue %177[0] : !llvm.struct<(i64, ptr)> 
    %215 = llvm.extractvalue %177[1] : !llvm.struct<(i64, ptr)> 
    %216 = llvm.extractvalue %183[0] : !llvm.struct<(i64, ptr)> 
    %217 = llvm.extractvalue %183[1] : !llvm.struct<(i64, ptr)> 
    %218 = llvm.extractvalue %189[0] : !llvm.struct<(i64, ptr)> 
    %219 = llvm.extractvalue %189[1] : !llvm.struct<(i64, ptr)> 
    %220 = llvm.extractvalue %195[0] : !llvm.struct<(i64, ptr)> 
    %221 = llvm.extractvalue %195[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_dense_step(%206, %207, %208, %209, %210, %211, %212, %213, %214, %215, %216, %217, %218, %219, %220, %221, %9, %8, %10, %11) : (i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i32, i32, i32, f32) -> ()
    llvm.br ^bb3(%3 : i64)
  ^bb3(%222: i64):  // 2 preds: ^bb2, ^bb7
    %223 = llvm.icmp "slt" %222, %6 : i64
    llvm.cond_br %223, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%3 : i64)
  ^bb5(%224: i64):  // 2 preds: ^bb4, ^bb6
    %225 = llvm.icmp "slt" %224, %7 : i64
    llvm.cond_br %225, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %226 = llvm.extractvalue %96[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %227 = llvm.mlir.constant(1024 : index) : i64
    %228 = llvm.mul %222, %227 overflow<nsw, nuw> : i64
    %229 = llvm.add %228, %224 overflow<nsw, nuw> : i64
    %230 = llvm.getelementptr inbounds|nuw %226[%229] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %231 = llvm.load %230 : !llvm.ptr -> f32
    %232 = llvm.extractvalue %79[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %233 = llvm.mlir.constant(1024 : index) : i64
    %234 = llvm.mul %222, %233 overflow<nsw, nuw> : i64
    %235 = llvm.add %234, %224 overflow<nsw, nuw> : i64
    %236 = llvm.getelementptr inbounds|nuw %232[%235] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %231, %236 : f32, !llvm.ptr
    %237 = llvm.add %224, %4 : i64
    llvm.br ^bb5(%237 : i64)
  ^bb7:  // pred: ^bb5
    %238 = llvm.add %222, %4 : i64
    llvm.br ^bb3(%238 : i64)
  ^bb8:  // pred: ^bb3
    %239 = llvm.add %204, %4 : i64
    llvm.br ^bb1(%239 : i64)
  ^bb9:  // pred: ^bb1
    %240 = llvm.extractvalue %79[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %241 = llvm.mlir.constant(1024 : index) : i64
    %242 = llvm.mul %3, %241 overflow<nsw, nuw> : i64
    %243 = llvm.add %242, %3 overflow<nsw, nuw> : i64
    %244 = llvm.getelementptr inbounds|nuw %240[%243] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %245 = llvm.load %244 : !llvm.ptr -> f32
    %246 = llvm.fptosi %245 : f32 to i32
    %247 = llvm.extractvalue %28[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%247) : (!llvm.ptr) -> ()
    %248 = llvm.extractvalue %45[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%248) : (!llvm.ptr) -> ()
    %249 = llvm.extractvalue %62[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%249) : (!llvm.ptr) -> ()
    %250 = llvm.extractvalue %79[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%250) : (!llvm.ptr) -> ()
    %251 = llvm.extractvalue %96[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%251) : (!llvm.ptr) -> ()
    %252 = llvm.extractvalue %113[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%252) : (!llvm.ptr) -> ()
    %253 = llvm.extractvalue %130[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%253) : (!llvm.ptr) -> ()
    %254 = llvm.extractvalue %147[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%254) : (!llvm.ptr) -> ()
    llvm.return %246 : i32
  }
}

