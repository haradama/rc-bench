#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
namespace rc {
void runConvertRcToLinalg(mlir::func::FuncOp func);
}

