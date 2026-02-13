#pragma once

// Core op definitions
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

// ★追加：生成コードが Builder / OpBuilder / ImplicitLocOpBuilder を使う
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

// Needed by generated traits / interfaces
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Bytecode interface/types referenced by generated op classes (LLVM/MLIR >= around 18+)
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"

#define GET_OP_CLASSES
#include "rc/Dialect/Rc/RcOps.h.inc"
