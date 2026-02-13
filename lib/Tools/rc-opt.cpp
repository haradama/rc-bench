#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/PassRegistry.h"

// Generic pass registrations
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

// LLVM lowering registrations
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"

// ★追加：linalg の transforms / lowering pass 登録
#include "mlir/Dialect/Linalg/Passes.h"

#include "rc/Dialect/Rc/RcDialect.h"
#include "rc/Conversion/RcToLinalg/Passes.h"

namespace rc {
#define GEN_PASS_REGISTRATION
#include "rc/Conversion/RcToLinalg/Passes.h.inc"
} // namespace rc

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<rc::RcDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();

  // Transforms
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  // Conversions
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();

  rc::registerRcToLinalgPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "rc-opt\n", registry));
}
