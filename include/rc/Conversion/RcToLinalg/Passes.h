#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
namespace rc {
std::unique_ptr<mlir::Pass> createConvertRcToLinalgPass();
}
