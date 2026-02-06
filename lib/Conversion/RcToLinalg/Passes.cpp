#include "rc/Conversion/RcToLinalg/Passes.h"

// 修正点1: TableGenが生成したベースクラスを有効にするマクロを追加
// (Passes.td内のdef名が "ConvertRcToLinalg" であると仮定しています)
namespace rc {
#define GEN_PASS_DEF_CONVERTRCTOLINALG
#include "rc/Conversion/RcToLinalg/Passes.h.inc"
}
#include "rc/Conversion/RcToLinalg/RcToLinalg.h"

using namespace mlir;

namespace {
// 修正点2: impl名前空間は GEN_PASS_DEF_... によって可視化されますが
// 通常は `impl::ConvertRcToLinalgBase` のようにアクセスします。
// rc::impl::... でエラーになる場合、単に impl::... を試してください。
// ここでは、Passes.tdで `cppNamespace = "::rc"` が指定されている前提で `rc::impl` のままとします。
struct ConvertRcToLinalgPass
    : public rc::impl::ConvertRcToLinalgBase<ConvertRcToLinalgPass> {
  void runOnOperation() override {
    rc::runConvertRcToLinalg(getOperation());
  }
};
} // namespace

std::unique_ptr<Pass> rc::createConvertRcToLinalgPass() {
  return std::make_unique<ConvertRcToLinalgPass>();
}