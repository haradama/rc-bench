#pragma once
#include "mlir/IR/Dialect.h"

namespace rc {
class RcDialect : public mlir::Dialect {
public:
  explicit RcDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "rc"; }
};
} // namespace rc

