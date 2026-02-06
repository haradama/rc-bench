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
    %6 = llvm.mlir.constant(1024 : index) : i64
    %7 = llvm.mlir.constant(1024 : i32) : i32
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.mlir.constant(64 : i32) : i32
    %10 = llvm.mlir.constant(3.000000e-01 : f32) : f32
    %11 = llvm.mlir.constant(1024 : index) : i64
    %12 = llvm.mlir.constant(1024 : index) : i64
    %13 = llvm.mlir.constant(1 : index) : i64
    %14 = llvm.mlir.constant(1048576 : index) : i64
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.getelementptr %15[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %19 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(0 : index) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.insertvalue %11, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %12, %24[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %12, %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %13, %26[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.mlir.constant(64 : index) : i64
    %29 = llvm.mlir.constant(1024 : index) : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.constant(65536 : index) : i64
    %32 = llvm.mlir.zero : !llvm.ptr
    %33 = llvm.getelementptr %32[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.call @malloc(%34) : (i64) -> !llvm.ptr
    %36 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %37 = llvm.insertvalue %35, %36[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %38 = llvm.insertvalue %35, %37[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.insertvalue %39, %38[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %28, %40[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.insertvalue %29, %41[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %29, %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %30, %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.mlir.constant(64 : index) : i64
    %47 = llvm.mlir.constant(1 : index) : i64
    %48 = llvm.mlir.constant(64 : index) : i64
    %49 = llvm.mlir.zero : !llvm.ptr
    %50 = llvm.getelementptr %49[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.call @malloc(%51) : (i64) -> !llvm.ptr
    %53 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %54 = llvm.insertvalue %52, %53[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.mlir.constant(0 : index) : i64
    %57 = llvm.insertvalue %56, %55[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %45, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %46, %58[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %46, %59[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %47, %60[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mlir.constant(1 : index) : i64
    %63 = llvm.mlir.constant(1024 : index) : i64
    %64 = llvm.mlir.constant(1 : index) : i64
    %65 = llvm.mlir.constant(1024 : index) : i64
    %66 = llvm.mlir.zero : !llvm.ptr
    %67 = llvm.getelementptr %66[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %68 = llvm.ptrtoint %67 : !llvm.ptr to i64
    %69 = llvm.call @malloc(%68) : (i64) -> !llvm.ptr
    %70 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %71 = llvm.insertvalue %69, %70[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.mlir.constant(0 : index) : i64
    %74 = llvm.insertvalue %73, %72[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %75 = llvm.insertvalue %62, %74[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.insertvalue %63, %75[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %77 = llvm.insertvalue %63, %76[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %78 = llvm.insertvalue %64, %77[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %79 = llvm.mlir.constant(1 : index) : i64
    %80 = llvm.mlir.constant(1024 : index) : i64
    %81 = llvm.mlir.constant(1 : index) : i64
    %82 = llvm.mlir.constant(1024 : index) : i64
    %83 = llvm.mlir.zero : !llvm.ptr
    %84 = llvm.getelementptr %83[%82] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %86 = llvm.call @malloc(%85) : (i64) -> !llvm.ptr
    %87 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %88 = llvm.insertvalue %86, %87[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.insertvalue %86, %88[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.mlir.constant(0 : index) : i64
    %91 = llvm.insertvalue %90, %89[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.insertvalue %79, %91[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.insertvalue %80, %92[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %80, %93[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.insertvalue %81, %94[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mlir.constant(1 : index) : i64
    %97 = llvm.mlir.constant(1024 : index) : i64
    %98 = llvm.mlir.constant(1 : index) : i64
    %99 = llvm.mlir.constant(1024 : index) : i64
    %100 = llvm.mlir.zero : !llvm.ptr
    %101 = llvm.getelementptr %100[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %102 = llvm.ptrtoint %101 : !llvm.ptr to i64
    %103 = llvm.call @malloc(%102) : (i64) -> !llvm.ptr
    %104 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %105 = llvm.insertvalue %103, %104[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.insertvalue %103, %105[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %107 = llvm.mlir.constant(0 : index) : i64
    %108 = llvm.insertvalue %107, %106[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %109 = llvm.insertvalue %96, %108[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %110 = llvm.insertvalue %97, %109[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.insertvalue %97, %110[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %112 = llvm.insertvalue %98, %111[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.mlir.constant(1024 : index) : i64
    %115 = llvm.mlir.constant(1 : index) : i64
    %116 = llvm.mlir.constant(1024 : index) : i64
    %117 = llvm.mlir.zero : !llvm.ptr
    %118 = llvm.getelementptr %117[%116] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %119 = llvm.ptrtoint %118 : !llvm.ptr to i64
    %120 = llvm.call @malloc(%119) : (i64) -> !llvm.ptr
    %121 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %122 = llvm.insertvalue %120, %121[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %123 = llvm.insertvalue %120, %122[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %124 = llvm.mlir.constant(0 : index) : i64
    %125 = llvm.insertvalue %124, %123[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.insertvalue %113, %125[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %127 = llvm.insertvalue %114, %126[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %128 = llvm.insertvalue %114, %127[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %129 = llvm.insertvalue %115, %128[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.mlir.constant(1 : index) : i64
    %131 = llvm.mlir.constant(1024 : index) : i64
    %132 = llvm.mlir.constant(1 : index) : i64
    %133 = llvm.mlir.constant(1024 : index) : i64
    %134 = llvm.mlir.zero : !llvm.ptr
    %135 = llvm.getelementptr %134[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %136 = llvm.ptrtoint %135 : !llvm.ptr to i64
    %137 = llvm.call @malloc(%136) : (i64) -> !llvm.ptr
    %138 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %139 = llvm.insertvalue %137, %138[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.insertvalue %137, %139[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.mlir.constant(0 : index) : i64
    %142 = llvm.insertvalue %141, %140[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %143 = llvm.insertvalue %130, %142[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.insertvalue %131, %143[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %145 = llvm.insertvalue %131, %144[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.insertvalue %132, %145[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.mlir.constant(1 : index) : i64
    %148 = llvm.alloca %147 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %27, %148 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %149 = llvm.mlir.constant(2 : index) : i64
    %150 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %151 = llvm.insertvalue %149, %150[0] : !llvm.struct<(i64, ptr)> 
    %152 = llvm.insertvalue %148, %151[1] : !llvm.struct<(i64, ptr)> 
    %153 = llvm.mlir.constant(1 : index) : i64
    %154 = llvm.alloca %153 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %44, %154 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %155 = llvm.mlir.constant(2 : index) : i64
    %156 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %157 = llvm.insertvalue %155, %156[0] : !llvm.struct<(i64, ptr)> 
    %158 = llvm.insertvalue %154, %157[1] : !llvm.struct<(i64, ptr)> 
    %159 = llvm.mlir.constant(1 : index) : i64
    %160 = llvm.alloca %159 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %61, %160 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %161 = llvm.mlir.constant(2 : index) : i64
    %162 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %163 = llvm.insertvalue %161, %162[0] : !llvm.struct<(i64, ptr)> 
    %164 = llvm.insertvalue %160, %163[1] : !llvm.struct<(i64, ptr)> 
    %165 = llvm.mlir.constant(1 : index) : i64
    %166 = llvm.alloca %165 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %78, %166 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %167 = llvm.mlir.constant(2 : index) : i64
    %168 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %169 = llvm.insertvalue %167, %168[0] : !llvm.struct<(i64, ptr)> 
    %170 = llvm.insertvalue %166, %169[1] : !llvm.struct<(i64, ptr)> 
    %171 = llvm.mlir.constant(1 : index) : i64
    %172 = llvm.alloca %171 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %95, %172 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %173 = llvm.mlir.constant(2 : index) : i64
    %174 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %175 = llvm.insertvalue %173, %174[0] : !llvm.struct<(i64, ptr)> 
    %176 = llvm.insertvalue %172, %175[1] : !llvm.struct<(i64, ptr)> 
    %177 = llvm.mlir.constant(1 : index) : i64
    %178 = llvm.alloca %177 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %112, %178 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %179 = llvm.mlir.constant(2 : index) : i64
    %180 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %181 = llvm.insertvalue %179, %180[0] : !llvm.struct<(i64, ptr)> 
    %182 = llvm.insertvalue %178, %181[1] : !llvm.struct<(i64, ptr)> 
    %183 = llvm.mlir.constant(1 : index) : i64
    %184 = llvm.alloca %183 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %129, %184 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %185 = llvm.mlir.constant(2 : index) : i64
    %186 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %187 = llvm.insertvalue %185, %186[0] : !llvm.struct<(i64, ptr)> 
    %188 = llvm.insertvalue %184, %187[1] : !llvm.struct<(i64, ptr)> 
    %189 = llvm.mlir.constant(1 : index) : i64
    %190 = llvm.alloca %189 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %146, %190 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    %191 = llvm.mlir.constant(2 : index) : i64
    %192 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %193 = llvm.insertvalue %191, %192[0] : !llvm.struct<(i64, ptr)> 
    %194 = llvm.insertvalue %190, %193[1] : !llvm.struct<(i64, ptr)> 
    %195 = llvm.extractvalue %152[0] : !llvm.struct<(i64, ptr)> 
    %196 = llvm.extractvalue %152[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%195, %196, %7, %7, %2) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    %197 = llvm.extractvalue %158[0] : !llvm.struct<(i64, ptr)> 
    %198 = llvm.extractvalue %158[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%197, %198, %9, %7, %1) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    %199 = llvm.extractvalue %164[0] : !llvm.struct<(i64, ptr)> 
    %200 = llvm.extractvalue %164[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%199, %200, %8, %9, %0) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    %201 = llvm.extractvalue %170[0] : !llvm.struct<(i64, ptr)> 
    %202 = llvm.extractvalue %170[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_fill2d(%201, %202, %8, %7, %0) : (i64, !llvm.ptr, i32, i32, f32) -> ()
    llvm.br ^bb1(%3 : i64)
  ^bb1(%203: i64):  // 2 preds: ^bb0, ^bb5
    %204 = llvm.icmp "slt" %203, %5 : i64
    llvm.cond_br %204, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %205 = llvm.extractvalue %152[0] : !llvm.struct<(i64, ptr)> 
    %206 = llvm.extractvalue %152[1] : !llvm.struct<(i64, ptr)> 
    %207 = llvm.extractvalue %158[0] : !llvm.struct<(i64, ptr)> 
    %208 = llvm.extractvalue %158[1] : !llvm.struct<(i64, ptr)> 
    %209 = llvm.extractvalue %164[0] : !llvm.struct<(i64, ptr)> 
    %210 = llvm.extractvalue %164[1] : !llvm.struct<(i64, ptr)> 
    %211 = llvm.extractvalue %170[0] : !llvm.struct<(i64, ptr)> 
    %212 = llvm.extractvalue %170[1] : !llvm.struct<(i64, ptr)> 
    %213 = llvm.extractvalue %176[0] : !llvm.struct<(i64, ptr)> 
    %214 = llvm.extractvalue %176[1] : !llvm.struct<(i64, ptr)> 
    %215 = llvm.extractvalue %182[0] : !llvm.struct<(i64, ptr)> 
    %216 = llvm.extractvalue %182[1] : !llvm.struct<(i64, ptr)> 
    %217 = llvm.extractvalue %188[0] : !llvm.struct<(i64, ptr)> 
    %218 = llvm.extractvalue %188[1] : !llvm.struct<(i64, ptr)> 
    %219 = llvm.extractvalue %194[0] : !llvm.struct<(i64, ptr)> 
    %220 = llvm.extractvalue %194[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @rc_dense_step(%205, %206, %207, %208, %209, %210, %211, %212, %213, %214, %215, %216, %217, %218, %219, %220, %8, %7, %9, %10) : (i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i32, i32, i32, f32) -> ()
    llvm.br ^bb3(%3 : i64)
  ^bb3(%221: i64):  // 2 preds: ^bb2, ^bb4
    %222 = llvm.icmp "slt" %221, %6 : i64
    llvm.cond_br %222, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %223 = llvm.extractvalue %95[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %224 = llvm.mlir.constant(1024 : index) : i64
    %225 = llvm.mul %3, %224 overflow<nsw, nuw> : i64
    %226 = llvm.add %225, %221 overflow<nsw, nuw> : i64
    %227 = llvm.getelementptr inbounds|nuw %223[%226] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %228 = llvm.load %227 : !llvm.ptr -> f32
    %229 = llvm.extractvalue %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.mlir.constant(1024 : index) : i64
    %231 = llvm.mul %3, %230 overflow<nsw, nuw> : i64
    %232 = llvm.add %231, %221 overflow<nsw, nuw> : i64
    %233 = llvm.getelementptr inbounds|nuw %229[%232] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %228, %233 : f32, !llvm.ptr
    %234 = llvm.add %221, %4 : i64
    llvm.br ^bb3(%234 : i64)
  ^bb5:  // pred: ^bb3
    %235 = llvm.add %203, %4 : i64
    llvm.br ^bb1(%235 : i64)
  ^bb6:  // pred: ^bb1
    %236 = llvm.extractvalue %78[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %237 = llvm.mlir.constant(1024 : index) : i64
    %238 = llvm.mul %3, %237 overflow<nsw, nuw> : i64
    %239 = llvm.add %238, %3 overflow<nsw, nuw> : i64
    %240 = llvm.getelementptr inbounds|nuw %236[%239] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %241 = llvm.load %240 : !llvm.ptr -> f32
    %242 = llvm.fptosi %241 : f32 to i32
    %243 = llvm.extractvalue %27[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%243) : (!llvm.ptr) -> ()
    %244 = llvm.extractvalue %44[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%244) : (!llvm.ptr) -> ()
    %245 = llvm.extractvalue %61[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%245) : (!llvm.ptr) -> ()
    %246 = llvm.extractvalue %78[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%246) : (!llvm.ptr) -> ()
    %247 = llvm.extractvalue %95[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%247) : (!llvm.ptr) -> ()
    %248 = llvm.extractvalue %112[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%248) : (!llvm.ptr) -> ()
    %249 = llvm.extractvalue %129[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%249) : (!llvm.ptr) -> ()
    %250 = llvm.extractvalue %146[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%250) : (!llvm.ptr) -> ()
    llvm.return %242 : i32
  }
}

