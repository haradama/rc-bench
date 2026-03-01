#include "rc/Dialect/Rc/RcDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "rc/Dialect/Rc/RcOps.h"

using namespace mlir;
using namespace rc;

RcDialect::RcDialect(MLIRContext* ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<RcDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "rc/Dialect/Rc/RcOps.cpp.inc"
        >();
}
