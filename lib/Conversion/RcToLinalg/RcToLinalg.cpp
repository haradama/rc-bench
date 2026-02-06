#include "rc/Conversion/RcToLinalg/RcToLinalg.h"
#include "rc/Dialect/Rc/RcDialect.h"
#include "rc/Dialect/Rc/RcOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Dense: x_next = (1-a)*x + a*tanh( x*W + u*Win )
struct LowerDenseUpdate : OpRewritePattern<rc::ReservoirUpdateDenseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(rc::ReservoirUpdateDenseOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value W   = op.getW();   // [N,N]
    Value x   = op.getX();   // [B,N]
    Value u   = op.getU();   // [B,Din]
    Value Win = op.getWin(); // [Din,N]
    float a = op.getLeakAttr().getValueAsDouble();

    // mat1 = linalg.matmul(x, W) : [B,N]
    // mat2 = linalg.matmul(u, Win) : [B,N]
    // pre = mat1 + mat2
    // act = tanh(pre) (フェーズ1では近似無し：後でmath.tanhに置き換え可。ここでは外部関数呼びに逃がす)
    // x_next = (1-a)*x + a*act

    // まず add + matmul は linalg.generic/matmul を使う。テンプレ簡略のため matmul named op を使用。
    auto xTy = cast<RankedTensorType>(x.getType());
    // init tensors
    Value init1 = rewriter.create<tensor::EmptyOp>(loc, xTy.getShape(), xTy.getElementType());
    Value init2 = rewriter.create<tensor::EmptyOp>(loc, xTy.getShape(), xTy.getElementType());

    Value mat1 = rewriter.create<linalg::MatmulOp>(loc, ValueRange{ x, W }, ValueRange{ init1 }).getResult(0);
    Value mat2 = rewriter.create<linalg::MatmulOp>(loc, ValueRange{ u, Win }, ValueRange{ init2 }).getResult(0);

    // pre = mat1 + mat2 (linalg.generic)
    Value preInit = rewriter.create<tensor::EmptyOp>(loc, xTy.getShape(), xTy.getElementType());
    auto addMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    auto addOp = rewriter.create<linalg::GenericOp>(
      loc, xTy, ValueRange{mat1, mat2}, ValueRange{preInit},
      ArrayRef<AffineMap>{addMap, addMap, addMap},
      ArrayRef<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel},
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value sum = b.create<arith::AddFOp>(l, args[0], args[1]);
        b.create<linalg::YieldOp>(l, sum);
      });
    Value pre = addOp.getResult(0);

    // act = tanh(pre): フェーズ1は runtime の tanh を呼ぶ（後で math.tanh にする）
    // ここでは単純化のため、外部関数 "rc_tanh" に tensor全体を渡すのは大変なので、
    // いったん B側テンプレでは runtime C を呼ぶ方式に揃えるのが簡単。
    // → フェーズ1では「IR比較の土台」優先で、act生成は linalg.generic 内で llvm.tanh 相当を使うより、
    //    math dialect を入れるのが楽です（環境により math.tanh がある）。
    // ここでは math.tanh がある前提で書く：
    // act = linalg.generic( pre ) { tanh }
    rewriter.getContext()->getOrLoadDialect<mlir::math::MathDialect>(); 
    Value actInit = rewriter.create<tensor::EmptyOp>(loc, xTy.getShape(), xTy.getElementType());
    auto actOp = rewriter.create<linalg::GenericOp>(
      loc, xTy, ValueRange{pre}, ValueRange{actInit},
      ArrayRef<AffineMap>{addMap, addMap},
      ArrayRef<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel},
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value t = b.create<mlir::math::TanhOp>(l, args[0]);
        b.create<linalg::YieldOp>(l, t);
      });
    Value act = actOp.getResult(0);

    // x_next = (1-a)*x + a*act
    Value outInit = rewriter.create<tensor::EmptyOp>(loc, xTy.getShape(), xTy.getElementType());
    auto mixOp = rewriter.create<linalg::GenericOp>(
      loc, xTy, ValueRange{x, act}, ValueRange{outInit},
      ArrayRef<AffineMap>{addMap, addMap, addMap},
      ArrayRef<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel},
      [&](OpBuilder &b, Location l, ValueRange args) {
        Value one = b.create<arith::ConstantOp>(l, b.getF32Type(), b.getF32FloatAttr(1.0f));
        Value alpha = b.create<arith::ConstantOp>(l, b.getF32Type(), b.getF32FloatAttr(a));
        Value oneMinus = b.create<arith::SubFOp>(l, one, alpha);
        Value term1 = b.create<arith::MulFOp>(l, oneMinus, args[0]);
        Value term2 = b.create<arith::MulFOp>(l, alpha, args[1]);
        Value y = b.create<arith::AddFOp>(l, term1, term2);
        b.create<linalg::YieldOp>(l, y);
      });

    rewriter.replaceOp(op, mixOp.getResult(0));
    return success();
  }
};

/// Sparse CSR: フェーズ1は「動く」優先で runtime C の csr_matmul を呼ぶ方式が最短
/// ただし “汎用IR比較” を崩さないため、B側も同じ runtime を使うようにする。
/// → フェーズ1は sparse_tensor を無理に使わず、両者とも runtime 呼びで揃えるのが最も早い。
struct LowerCsrUpdateToCall : OpRewritePattern<rc::ReservoirUpdateCsrOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(rc::ReservoirUpdateCsrOp op, PatternRewriter &rewriter) const override {
    // ここでは「rc_runtime.c にある関数」を呼ぶ形に落とす方が2-3日で確実です。
    // 実装は後述の “runtime方式” に委ね、ここは骨組みのみ。
    return failure(); // フェーズ1ではまず Dense 完成→次にここをcall化
  }
};

} // namespace

namespace rc {
void runConvertRcToLinalg(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<LowerDenseUpdate>(ctx);
  // CSRは後で追加：patterns.add<LowerCsrUpdateToCall>(ctx);

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyPatternsGreedily(func, frozenPatterns);
}
} // namespace rc

