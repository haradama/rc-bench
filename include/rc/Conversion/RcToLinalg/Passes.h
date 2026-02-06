#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace rc {
std::unique_ptr<mlir::Pass> createConvertRcToLinalgPass();
} // namespace rc

