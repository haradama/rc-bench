include "mlir/Pass/PassBase.td"

def ConvertRcToLinalg : Pass<"convert-rc-to-linalg", "mlir::func::FuncOp"> {
  let summary = "Lower rc dialect ops to linalg (memref-based, no allocations)";
  let constructor = "rc::createConvertRcToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [];
}
