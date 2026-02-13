#include "rc/Conversion/RcToLinalg/RcToLinalg.h"
#include "rc/Dialect/Rc/RcDialect.h"
#include "rc/Dialect/Rc/RcOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Lower rc.reservoir_step_dense to:
/// tmp1 = linalg.matmul(x, W) -> tmp1
/// tmp2 = linalg.matmul(u, Win) -> tmp2
/// pre  = tmp1 + tmp2 -> pre
/// pre  = tanh(pre) -> pre (in-place)
/// x2   = (1-leak)*x + leak*pre -> x2

/// Lower rc.reservoir_step_dense to:
/// tmp1 = 0; tmp1 += x*W
/// tmp2 = 0; tmp2 += u*Win
/// pre  = tmp1 + tmp2
/// pre  = tanh(pre) (in-place)
/// x2   = (1-leak)*x + leak*pre
struct LowerDenseStep : OpRewritePattern<rc::ReservoirStepDenseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(rc::ReservoirStepDenseOp op,
                                PatternRewriter &rewriter) const override {
    op.emitRemark() << "LowerDenseStep fired";
    Location loc = op.getLoc();

    Value W    = op.getW();
    Value Win  = op.getWin();
    Value u    = op.getU();
    Value x    = op.getX();
    Value x2   = op.getX2();
    Value tmp1 = op.getTmp1();
    Value tmp2 = op.getTmp2();
    Value pre  = op.getPre();

    // leak is an Attribute (FloatAttr)
    float leak = static_cast<float>(op.getLeak().convertToDouble());

    rewriter.getContext()->getOrLoadDialect<mlir::math::MathDialect>();

    // IMPORTANT: linalg.matmul is C += A*B, so we must zero tmp1/tmp2 each step.
    Value c0f = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0f));
    rewriter.create<linalg::FillOp>(loc, ValueRange{c0f}, ValueRange{tmp1});
    rewriter.create<linalg::FillOp>(loc, ValueRange{c0f}, ValueRange{tmp2});

    // tmp1 += x*W
    rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{x, W}, ValueRange{tmp1});

    // tmp2 += u*Win
    rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{u, Win}, ValueRange{tmp2});

    // Common affine maps for 2D elementwise
    auto id2 = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    SmallVector<utils::IteratorType, 2> iters = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel
    };

    // pre = tmp1 + tmp2
    rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange{tmp1, tmp2},
        /*outputs=*/ValueRange{pre},
        /*indexingMaps=*/ArrayRef<AffineMap>{id2, id2, id2},
        /*iteratorTypes=*/iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(l, args[0], args[1]);
          b.create<linalg::YieldOp>(l, sum);
        });

    // pre = tanh(pre) in-place
    rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{},
        ValueRange{pre},
        ValueRange{pre},
        ArrayRef<AffineMap>{id2, id2},
        iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value t = b.create<mlir::math::TanhOp>(l, args[0]);
          b.create<linalg::YieldOp>(l, t);
        });

    // x2 = (1-leak)*x + leak*pre
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(1.0f));
    Value alpha = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(leak));
    Value oneMinus = rewriter.create<arith::SubFOp>(loc, one, alpha);

    rewriter.create<linalg::GenericOp>(
        loc,
        TypeRange{},
        ValueRange{x, pre},
        ValueRange{x2},
        ArrayRef<AffineMap>{id2, id2, id2},
        iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value term1 = b.create<arith::MulFOp>(l, oneMinus, args[0]);
          Value term2 = b.create<arith::MulFOp>(l, alpha, args[1]);
          Value y = b.create<arith::AddFOp>(l, term1, term2);
          b.create<linalg::YieldOp>(l, y);
        });

    rewriter.eraseOp(op);
    return success();
  }
};


} // namespace

namespace rc {
void runConvertRcToLinalg(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<LowerDenseStep>(ctx);

  // (void)applyPatternsGreedily(func, std::move(patterns));
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
} // namespace rc
