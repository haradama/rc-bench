#include "rc/Conversion/RcToLinalg/RcToLinalg.h"
#include "rc/Dialect/Rc/RcDialect.h"
#include "rc/Dialect/Rc/RcOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct PingPongElideCopy : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isCopyLoop(scf::ForOp iLoop, Value src, Value dst) {
    // Expect: scf.for %i ... { scf.for %j ... { %v=load src[%i,%j]; store %v, dst[%i,%j] } }
    auto *iBody = iLoop.getBody();
    if (!iBody) return false;

    scf::ForOp jLoop;
    for (Operation &op : iBody->getOperations()) {
      if (auto f = dyn_cast<scf::ForOp>(op)) {
        jLoop = f;
        break;
      }
      if (isa<scf::YieldOp>(op)) break;
      // Any other op before jLoop -> not our template
      if (!isa<scf::ForOp>(op)) return false;
    }
    if (!jLoop) return false;

    Block *jBody = jLoop.getBody();
    if (!jBody) return false;

    // Find load/store in the inner loop body (ignore terminator).
    memref::LoadOp load;
    memref::StoreOp store;
    for (Operation &op : jBody->getOperations()) {
      if (isa<scf::YieldOp>(op)) break;
      if (auto l = dyn_cast<memref::LoadOp>(op)) load = l;
      else if (auto s = dyn_cast<memref::StoreOp>(op)) store = s;
      else return false; // template is strict: only load/store
    }
    if (!load || !store) return false;

    if (load.getMemRef() != src) return false;
    if (store.getMemRef() != dst) return false;
    if (store.getValue() != load.getResult()) return false;

    // Check indices are [%i, %j] in both.
    if (load.getIndices().size() != 2 || store.getIndices().size() != 2) return false;

    Value iIv = iLoop.getInductionVar();
    Value jIv = jLoop.getInductionVar();

    if (load.getIndices()[0] != iIv || load.getIndices()[1] != jIv) return false;
    if (store.getIndices()[0] != iIv || store.getIndices()[1] != jIv) return false;

    return true;
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // We only handle single-block scf.for in the template.
    Block *body = forOp.getBody();
    if (!body) return failure();

    // Find first meaningful op: rc.reservoir_step_dense
    rc::ReservoirStepDenseOp step;
    scf::ForOp copyILoop;

    for (Operation &op : body->getOperations()) {
      if (isa<scf::YieldOp>(op)) break;
      if (!step) {
        step = dyn_cast<rc::ReservoirStepDenseOp>(op);
        if (!step) return failure(); // template expects step first
        continue;
      }
      // After step, we expect the copy i-loop.
      copyILoop = dyn_cast<scf::ForOp>(op);
      if (!copyILoop) return failure();
      break;
    }

    if (!step || !copyILoop) return failure();

    Value x  = step.getX();
    Value x2 = step.getX2();

    // Verify copy is exactly: x2 -> x
    if (!isCopyLoop(copyILoop, /*src=*/x2, /*dst=*/x))
      return failure();

    // Rewrite to ping-pong: scf.for ... iter_args(%x_cur=%x, %x_next=%x2)
    Location loc = forOp.getLoc();
    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value st = forOp.getStep();

    // Create new loop right before the old one.
    auto newFor = rewriter.create<scf::ForOp>(
        loc, lb, ub, st, ValueRange{x, x2},
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          Value xCur  = iterArgs[0];
          Value xNext = iterArgs[1];

          // Recreate rc.reservoir_step_dense but with xCur/xNext swapped in.
          // operands: W, Win, u, x, x2, tmp1, tmp2, pre
          b.create<rc::ReservoirStepDenseOp>(
              l,
              /*W*/    step.getW(),
              /*Win*/  step.getWin(),
              /*u*/    step.getU(),
              /*x*/    xCur,
              /*x2*/   xNext,
              /*tmp1*/ step.getTmp1(),
              /*tmp2*/ step.getTmp2(),
              /*pre*/  step.getPre(),
              /*leak*/ step.getLeakAttr());

          // Swap for next iteration: xCur <- xNext, xNext <- xCur
          b.create<scf::YieldOp>(l, ValueRange{xNext, xCur});
        });

    // Replace uses of original %x AFTER the loop with newFor.getResult(0).
    // (Inside the loop, we already rebuilt the step using iter args.)
    Value xFinal = newFor.getResult(0);

    // loop 後にある "読み" だけ置換（dealloc は触らない）
    for (Operation *user : llvm::make_early_inc_range(x.getUsers())) {
      // old loop / new loop 内はスキップ
      if (forOp->isAncestor(user) || newFor->isAncestor(user)) continue;

      // 同一 block で old loop より後にあるものだけ
      if (user->getBlock() != forOp->getBlock()) continue;
      if (!forOp->isBeforeInBlock(user)) continue;

      if (auto load = dyn_cast<memref::LoadOp>(user)) {
        load.getMemrefMutable().assign(xFinal);
      }

      // memref.dealloc は絶対に触らない
      // if (isa<memref::DeallocOp>(user)) continue;
    }

    // Erase the old loop (removes the copy loop too).
    rewriter.eraseOp(forOp);
    return success();
  }
};

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

  // --- Stage 1: ping-pong 化（rc op + copy を消す）---
  {
    RewritePatternSet pp(ctx);
    pp.add<PingPongElideCopy>(ctx);
    (void)applyPatternsGreedily(func, std::move(pp));
  }

  // --- Stage 2: rc op を linalg に lower ---
  {
    RewritePatternSet lower(ctx);
    lower.add<LowerDenseStep>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(lower));
  }
}
} // namespace rc
