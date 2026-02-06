# Ensure tablegen output directories exist (mlir_tablegen does not create them)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rc/Dialect/Rc)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rc/Conversion/RcToLinalg)

# --- RcOps TableGen ---
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/rc/Dialect/Rc/RcOps.td)

mlir_tablegen(rc/Dialect/Rc/RcOps.h.inc  -gen-op-decls)
mlir_tablegen(rc/Dialect/Rc/RcOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIRRcOpsIncGen)
add_dependencies(mlir-headers MLIRRcOpsIncGen)

# --- Passes TableGen ---
set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/rc/Conversion/RcToLinalg/Passes.td)

mlir_tablegen(rc/Conversion/RcToLinalg/Passes.h.inc -gen-pass-decls -name RcToLinalg)

add_public_tablegen_target(MLIRRcToLinalgPassesIncGen)
add_dependencies(mlir-headers MLIRRcToLinalgPassesIncGen)


add_mlir_dialect_library(MLIRRc
  Dialect/Rc/RcDialect.cpp
  Dialect/Rc/RcOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  MLIRRcOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_library(MLIRRcToLinalg
  Conversion/RcToLinalg/Passes.cpp
  Conversion/RcToLinalg/RcToLinalg.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  MLIRRcToLinalgPassesIncGen
  MLIRRcOpsIncGen

  LINK_LIBS PUBLIC
  MLIRRc
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRMathDialect
  MLIRPass
  MLIRTransforms
)

target_include_directories(MLIRRc PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_include_directories(MLIRRcToLinalg PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

# ★修正点: add_mlir_tool を add_executable に変更し、確実にビルドされるようにする
add_executable(rc-opt
  Tools/rc-opt.cpp
)

# LLVM/MLIRのコンパイルフラグを適用
llvm_update_compile_flags(rc-opt)

target_link_libraries(rc-opt PRIVATE
  MLIRRc
  MLIRRcToLinalg
  MLIROptLib
  MLIRMemRefDialect
  MLIRSCFDialect
  
  MLIRTransforms
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  # ★追加: CF -> LLVM 変換用ライブラリ
  MLIRControlFlowToLLVM 
  MLIRReconcileUnrealizedCasts
)