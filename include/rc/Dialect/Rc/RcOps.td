#ifndef RC_OPS
#define RC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Rc_Dialect : Dialect {
  let name = "rc";
  let cppNamespace = "::rc";
}

def ReservoirStepDenseOp : Op<Rc_Dialect, "reservoir_step_dense",
  [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {

  let summary = "Reservoir step (dense) updating memrefs";
  let description = [{
    Performs one reservoir step and writes results/scratch buffers.
  }];

  let arguments = (ins
    AnyMemRef:$W,
    AnyMemRef:$Win,
    AnyMemRef:$u,
    AnyMemRef:$x,
    AnyMemRef:$x2,
    AnyMemRef:$tmp1,
    AnyMemRef:$tmp2,
    AnyMemRef:$pre,
    F32Attr:$leak
  );

  let results = (outs);

  let assemblyFormat = [{
    $W `,` $Win `,` $u `,` $x `,` $x2 `,` $tmp1 `,` $tmp2 `,` $pre
    attr-dict `:` functional-type(operands, results)
  }];
}

#endif // RC_OPS
