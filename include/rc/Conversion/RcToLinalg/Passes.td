include "mlir/Pass/PassBase.td"

def ConvertRcToLinalg : Pass<"convert-rc-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Lower rc dialect ops to linalg/sparse_tensor + arith";
  let constructor = "rc::createConvertRcToLinalgPass()";
  // 追加: これがないとベースクラスが rc::impl ではなくグローバルの impl に生成されてしまう
  // let cppNamespace = "::rc"; 
}