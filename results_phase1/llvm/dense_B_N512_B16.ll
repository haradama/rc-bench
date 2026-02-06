; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare void @free(ptr)

declare ptr @malloc(i64)

declare void @rc_fill2d(i64, ptr, i32, i32, float)

declare void @rc_dense_step(i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i32, i32, i32, float)

define i32 @bench() {
  %1 = call ptr @malloc(i64 1048576)
  %2 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, i64 512, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %5, i64 512, 3, 1
  %7 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %6, i64 512, 4, 0
  %8 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %7, i64 1, 4, 1
  %9 = call ptr @malloc(i64 131072)
  %10 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %9, 0
  %11 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %10, ptr %9, 1
  %12 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %11, i64 0, 2
  %13 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, i64 64, 3, 0
  %14 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %13, i64 512, 3, 1
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %14, i64 512, 4, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, i64 1, 4, 1
  %17 = call ptr @malloc(i64 4096)
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %17, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, ptr %17, 1
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 0, 2
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 16, 3, 0
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, i64 64, 3, 1
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, i64 64, 4, 0
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 1, 4, 1
  %25 = call ptr @malloc(i64 32768)
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %25, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, ptr %25, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 0, 2
  %29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, i64 16, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %29, i64 512, 3, 1
  %31 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %30, i64 512, 4, 0
  %32 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %31, i64 1, 4, 1
  %33 = call ptr @malloc(i64 32768)
  %34 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %33, 0
  %35 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %34, ptr %33, 1
  %36 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %35, i64 0, 2
  %37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %36, i64 16, 3, 0
  %38 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %37, i64 512, 3, 1
  %39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %38, i64 512, 4, 0
  %40 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %39, i64 1, 4, 1
  %41 = call ptr @malloc(i64 32768)
  %42 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %41, 0
  %43 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %42, ptr %41, 1
  %44 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %43, i64 0, 2
  %45 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %44, i64 16, 3, 0
  %46 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %45, i64 512, 3, 1
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %46, i64 512, 4, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, i64 1, 4, 1
  %49 = call ptr @malloc(i64 32768)
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %49, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, ptr %49, 1
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 0, 2
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 16, 3, 0
  %54 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, i64 512, 3, 1
  %55 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %54, i64 512, 4, 0
  %56 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %55, i64 1, 4, 1
  %57 = call ptr @malloc(i64 32768)
  %58 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %57, 0
  %59 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %58, ptr %57, 1
  %60 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %59, i64 0, 2
  %61 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %60, i64 16, 3, 0
  %62 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %61, i64 512, 3, 1
  %63 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %62, i64 512, 4, 0
  %64 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %63, i64 1, 4, 1
  %65 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, ptr %65, align 8
  %66 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %65, 1
  %67 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, ptr %67, align 8
  %68 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %67, 1
  %69 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, ptr %69, align 8
  %70 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %69, 1
  %71 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, ptr %71, align 8
  %72 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %71, 1
  %73 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, ptr %73, align 8
  %74 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %73, 1
  %75 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, ptr %75, align 8
  %76 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %75, 1
  %77 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, ptr %77, align 8
  %78 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %77, 1
  %79 = alloca { ptr, ptr, i64, [2 x i64], [2 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, ptr %79, align 8
  %80 = insertvalue { i64, ptr } { i64 2, ptr poison }, ptr %79, 1
  %81 = extractvalue { i64, ptr } %66, 0
  %82 = extractvalue { i64, ptr } %66, 1
  call void @rc_fill2d(i64 %81, ptr %82, i32 512, i32 512, float 0x3FA99999A0000000)
  %83 = extractvalue { i64, ptr } %68, 0
  %84 = extractvalue { i64, ptr } %68, 1
  call void @rc_fill2d(i64 %83, ptr %84, i32 64, i32 512, float 0x3FB99999A0000000)
  %85 = extractvalue { i64, ptr } %70, 0
  %86 = extractvalue { i64, ptr } %70, 1
  call void @rc_fill2d(i64 %85, ptr %86, i32 16, i32 64, float 1.000000e+00)
  %87 = extractvalue { i64, ptr } %72, 0
  %88 = extractvalue { i64, ptr } %72, 1
  call void @rc_fill2d(i64 %87, ptr %88, i32 16, i32 512, float 1.000000e+00)
  br label %89

89:                                               ; preds = %129, %0
  %90 = phi i64 [ %130, %129 ], [ 0, %0 ]
  %91 = icmp slt i64 %90, 10000
  br i1 %91, label %92, label %131

92:                                               ; preds = %89
  %93 = extractvalue { i64, ptr } %66, 0
  %94 = extractvalue { i64, ptr } %66, 1
  %95 = extractvalue { i64, ptr } %68, 0
  %96 = extractvalue { i64, ptr } %68, 1
  %97 = extractvalue { i64, ptr } %70, 0
  %98 = extractvalue { i64, ptr } %70, 1
  %99 = extractvalue { i64, ptr } %72, 0
  %100 = extractvalue { i64, ptr } %72, 1
  %101 = extractvalue { i64, ptr } %74, 0
  %102 = extractvalue { i64, ptr } %74, 1
  %103 = extractvalue { i64, ptr } %76, 0
  %104 = extractvalue { i64, ptr } %76, 1
  %105 = extractvalue { i64, ptr } %78, 0
  %106 = extractvalue { i64, ptr } %78, 1
  %107 = extractvalue { i64, ptr } %80, 0
  %108 = extractvalue { i64, ptr } %80, 1
  call void @rc_dense_step(i64 %93, ptr %94, i64 %95, ptr %96, i64 %97, ptr %98, i64 %99, ptr %100, i64 %101, ptr %102, i64 %103, ptr %104, i64 %105, ptr %106, i64 %107, ptr %108, i32 16, i32 512, i32 64, float 0x3FD3333340000000)
  br label %109

109:                                              ; preds = %127, %92
  %110 = phi i64 [ %128, %127 ], [ 0, %92 ]
  %111 = icmp slt i64 %110, 16
  br i1 %111, label %112, label %129

112:                                              ; preds = %109
  br label %113

113:                                              ; preds = %116, %112
  %114 = phi i64 [ %126, %116 ], [ 0, %112 ]
  %115 = icmp slt i64 %114, 512
  br i1 %115, label %116, label %127

116:                                              ; preds = %113
  %117 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 1
  %118 = mul nuw nsw i64 %110, 512
  %119 = add nuw nsw i64 %118, %114
  %120 = getelementptr inbounds nuw float, ptr %117, i64 %119
  %121 = load float, ptr %120, align 4
  %122 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %123 = mul nuw nsw i64 %110, 512
  %124 = add nuw nsw i64 %123, %114
  %125 = getelementptr inbounds nuw float, ptr %122, i64 %124
  store float %121, ptr %125, align 4
  %126 = add i64 %114, 1
  br label %113

127:                                              ; preds = %113
  %128 = add i64 %110, 1
  br label %109

129:                                              ; preds = %109
  %130 = add i64 %90, 1
  br label %89

131:                                              ; preds = %89
  %132 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 1
  %133 = getelementptr inbounds nuw float, ptr %132, i64 0
  %134 = load float, ptr %133, align 4
  %135 = fptosi float %134 to i32
  %136 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0
  call void @free(ptr %136)
  %137 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 0
  call void @free(ptr %137)
  %138 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 0
  call void @free(ptr %138)
  %139 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %32, 0
  call void @free(ptr %139)
  %140 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %40, 0
  call void @free(ptr %140)
  %141 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, 0
  call void @free(ptr %141)
  %142 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %56, 0
  call void @free(ptr %142)
  %143 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %64, 0
  call void @free(ptr %143)
  ret i32 %135
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
