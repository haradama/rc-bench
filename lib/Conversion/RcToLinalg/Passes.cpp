#include "rc/Conversion/RcToLinalg/Passes.h"

// Passes.h.inc が dependentDialects を使って registry.insert<mlir::xxx::...>() を生成するので、
// それらの Dialect クラス定義が見えるようにヘッダを必ず include する必要があります。
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace rc {
#define GEN_PASS_DEF_CONVERTRCTOLINALG
#include "rc/Conversion/RcToLinalg/Passes.h.inc"
} // namespace rc

#include "rc/Conversion/RcToLinalg/RcToLinalg.h"

using namespace mlir;

namespace {
struct ConvertRcToLinalgPass
    : public rc::impl::ConvertRcToLinalgBase<ConvertRcToLinalgPass> {
  void runOnOperation() override {
    auto f = getOperation();
    auto name = f.getName();
    if (name != "bench" && name != "infer_store") return;
    rc::runConvertRcToLinalg(f);
  }
};
}

std::unique_ptr<Pass> rc::createConvertRcToLinalgPass() {
  return std::make_unique<ConvertRcToLinalgPass>();
}
