include "mlir/IR/OpBase.td"

def Rc_Dialect : Dialect {
  let name = "rc";
  let cppNamespace = "::rc";
}

def Rc_ReservoirUpdateDenseOp : Op<Rc_Dialect, "reservoir_update_dense"> {
  let summary = "Dense ESN reservoir update (one step)";
  let arguments = (ins
    AnyRankedTensor:$W,
    AnyRankedTensor:$x,
    AnyRankedTensor:$u,
    AnyRankedTensor:$Win
  );
  let results = (outs AnyRankedTensor:$x_next);

  // leak is a normal attribute (NOT a property)
  let extraClassDeclaration = [{
    ::mlir::FloatAttr getLeakAttr() { return (*this)->getAttrOfType<::mlir::FloatAttr>("leak"); }
  }];
}

def Rc_ReservoirUpdateCsrOp : Op<Rc_Dialect, "reservoir_update_csr"> {
  let summary = "CSR sparse ESN reservoir update (one step)";
  let arguments = (ins
    AnyRankedTensor:$rowptr,
    AnyRankedTensor:$colind,
    AnyRankedTensor:$values,
    AnyRankedTensor:$x,
    AnyRankedTensor:$u,
    AnyRankedTensor:$Win
  );
  let results = (outs AnyRankedTensor:$x_next);

  let extraClassDeclaration = [{
    ::mlir::FloatAttr getLeakAttr() { return (*this)->getAttrOfType<::mlir::FloatAttr>("leak"); }
  }];
}
