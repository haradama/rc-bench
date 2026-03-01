#include "rc/Dialect/Rc/RcOps.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "rc/Dialect/Rc/RcDialect.h"

#define GET_OP_CLASSES
#include "rc/Dialect/Rc/RcOps.cpp.inc"

using namespace mlir;

namespace rc {

void ReservoirStepDenseOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance>& effects) {
    using namespace mlir;

    // Reads
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(0));  // W
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(1));  // Win
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(2));  // u
    effects.emplace_back(MemoryEffects::Read::get(),
                         &getOperation()->getOpOperand(3));  // x

    // Writes
    effects.emplace_back(MemoryEffects::Write::get(),
                         &getOperation()->getOpOperand(4));  // x2
    effects.emplace_back(MemoryEffects::Write::get(),
                         &getOperation()->getOpOperand(5));  // tmp1
    effects.emplace_back(MemoryEffects::Write::get(),
                         &getOperation()->getOpOperand(6));  // tmp2
    effects.emplace_back(MemoryEffects::Write::get(),
                         &getOperation()->getOpOperand(7));  // pre
}

}  // namespace rc
