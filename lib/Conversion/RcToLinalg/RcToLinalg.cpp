#include "rc/Conversion/RcToLinalg/RcToLinalg.h"

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "rc/Dialect/Rc/RcDialect.h"
#include "rc/Dialect/Rc/RcOps.h"

using namespace mlir;

namespace {

static bool isConstantIndex(Value v, int64_t& out) {
    if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
        out = c.value();
        return true;
    }
    if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
        if (!llvm::isa<mlir::IndexType>(c.getType())) return false;
        if (auto ia = llvm::dyn_cast<IntegerAttr>(c.getValue())) {
            out = ia.getInt();
            return true;
        }
    }
    return false;
}

static bool isCopyLoop(scf::ForOp iLoop, Value src, Value dst) {
    auto* iBody = iLoop.getBody();
    if (!iBody) return false;

    scf::ForOp jLoop;
    for (Operation& op : iBody->getOperations()) {
        if (auto f = dyn_cast<scf::ForOp>(op)) {
            jLoop = f;
            break;
        }
        if (isa<scf::YieldOp>(op)) break;
        if (!isa<scf::ForOp>(op)) return false;
    }
    if (!jLoop) return false;

    Block* jBody = jLoop.getBody();
    if (!jBody) return false;

    memref::LoadOp load;
    memref::StoreOp store;
    for (Operation& op : jBody->getOperations()) {
        if (isa<scf::YieldOp>(op)) break;
        if (auto l = dyn_cast<memref::LoadOp>(op))
            load = l;
        else if (auto s = dyn_cast<memref::StoreOp>(op))
            store = s;
        else
            return false;
    }
    if (!load || !store) return false;

    if (load.getMemRef() != src) return false;
    if (store.getMemRef() != dst) return false;
    if (store.getValue() != load.getResult()) return false;

    if (load.getIndices().size() != 2 || store.getIndices().size() != 2)
        return false;

    Value iIv = iLoop.getInductionVar();
    Value jIv = jLoop.getInductionVar();
    if (load.getIndices()[0] != iIv || load.getIndices()[1] != jIv)
        return false;
    if (store.getIndices()[0] != iIv || store.getIndices()[1] != jIv)
        return false;

    return true;
}

static bool findStepAndCopy(scf::ForOp forOp, rc::ReservoirStepDenseOp& step,
                            scf::ForOp& copyILoop) {
    Block* body = forOp.getBody();
    if (!body) return false;

    step = nullptr;
    copyILoop = nullptr;

    for (Operation& op : body->getOperations()) {
        if (isa<scf::YieldOp>(op)) break;

        if (!step) {
            step = dyn_cast<rc::ReservoirStepDenseOp>(op);
            if (!step) return false;
            continue;
        }

        copyILoop = dyn_cast<scf::ForOp>(op);
        if (!copyILoop) return false;
        break;
    }
    return (step && copyILoop);
}

// ---- linalg helpers ----

// C(i,j) += A(i,k) * B(k,j) with loop order i,k,j (better for row-major B)
static void emitMatmulAccumulateIKJ(PatternRewriter& rewriter, Location loc,
                                    Value A, Value B, Value C) {
    MLIRContext* ctx = rewriter.getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx);
    AffineExpr d1 = getAffineDimExpr(1, ctx);
    AffineExpr d2 = getAffineDimExpr(2, ctx);

    AffineMap aMap = AffineMap::get(3, 0, {d0, d1}, ctx);
    AffineMap bMap = AffineMap::get(3, 0, {d1, d2}, ctx);
    AffineMap cMap = AffineMap::get(3, 0, {d0, d2}, ctx);

    SmallVector<utils::IteratorType, 3> iters = {
        utils::IteratorType::parallel,
        utils::IteratorType::reduction,
        utils::IteratorType::parallel,
    };

    rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange{A, B},
        /*outputs=*/ValueRange{C},
        /*indexingMaps=*/ArrayRef<AffineMap>{aMap, bMap, cMap},
        /*iteratorTypes=*/iters,
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value mul = b.create<arith::MulFOp>(l, args[0], args[1]);
            Value sum = b.create<arith::AddFOp>(l, args[2], mul);
            b.create<linalg::YieldOp>(l, sum);
        });
}

static void emitTanhInPlace2D(PatternRewriter& rewriter, Location loc,
                              Value X) {
    MLIRContext* ctx = rewriter.getContext();
    auto id2 = AffineMap::getMultiDimIdentityMap(2, ctx);
    SmallVector<utils::IteratorType, 2> iters = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel,
    };

    rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{X}, ValueRange{X},
        ArrayRef<AffineMap>{id2, id2}, iters,
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value t = b.create<mlir::math::TanhOp>(l, args[0]);
            b.create<linalg::YieldOp>(l, t);
        });
}

static void emitLerp2D(PatternRewriter& rewriter, Location loc, Value X,
                       Value Act, Value Out, Value oneMinus, Value alpha) {
    MLIRContext* ctx = rewriter.getContext();
    auto id2 = AffineMap::getMultiDimIdentityMap(2, ctx);
    SmallVector<utils::IteratorType, 2> iters = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel,
    };

    rewriter.create<linalg::GenericOp>(
        loc, TypeRange{}, ValueRange{X, Act}, ValueRange{Out},
        ArrayRef<AffineMap>{id2, id2, id2}, iters,
        [&](OpBuilder& b, Location l, ValueRange args) {
            Value term1 = b.create<arith::MulFOp>(l, oneMinus, args[0]);
            Value term2 = b.create<arith::MulFOp>(l, alpha, args[1]);
            Value y = b.create<arith::AddFOp>(l, term1, term2);
            b.create<linalg::YieldOp>(l, y);
        });
}

struct Unroll2ElideCopyConstT : OpRewritePattern<scf::ForOp> {
    explicit Unroll2ElideCopyConstT(MLIRContext* ctx)
        : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/2) {}

    LogicalResult matchAndRewrite(scf::ForOp forOp,
                                  PatternRewriter& rewriter) const override {
        rc::ReservoirStepDenseOp step;
        scf::ForOp copyILoop;
        if (!findStepAndCopy(forOp, step, copyILoop)) return failure();

        Value x = step.getX();
        Value x2 = step.getX2();
        if (!isCopyLoop(copyILoop, /*src=*/x2, /*dst=*/x)) return failure();

        int64_t lbC = 0, stC = 0, ubC = 0;
        if (!isConstantIndex(forOp.getLowerBound(), lbC)) return failure();
        if (!isConstantIndex(forOp.getStep(), stC)) return failure();
        if (!isConstantIndex(forOp.getUpperBound(), ubC)) return failure();
        if (stC != 1) return failure();
        if (lbC != 0) return failure();
        if (ubC < 0) return failure();

        int64_t T = ubC;
        int64_t ubEven = (T / 2) * 2;
        bool hasTail = (T % 2) != 0;

        Location loc = forOp.getLoc();
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
        Value ubEvenV = rewriter.create<arith::ConstantIndexOp>(loc, ubEven);

        auto newFor = rewriter.create<scf::ForOp>(
            loc, c0, ubEvenV, c2, ValueRange{},
            [&](OpBuilder& b, Location l, Value /*iv*/,
                ValueRange /*iterArgs*/) {
                b.create<rc::ReservoirStepDenseOp>(
                    l, step.getW(), step.getWin(), step.getU(),
                    /*x*/ x,
                    /*x2*/ x2, step.getTmp1(), step.getTmp2(), step.getPre(),
                    step.getLeakAttr());

                b.create<rc::ReservoirStepDenseOp>(
                    l, step.getW(), step.getWin(), step.getU(),
                    /*x*/ x2,
                    /*x2*/ x, step.getTmp1(), step.getTmp2(), step.getPre(),
                    step.getLeakAttr());

                b.create<scf::YieldOp>(l);
            });

        Value finalX = x;
        if (hasTail) {
            rewriter.setInsertionPointAfter(newFor);
            rewriter.create<rc::ReservoirStepDenseOp>(
                loc, step.getW(), step.getWin(), step.getU(),
                /*x*/ x,
                /*x2*/ x2, step.getTmp1(), step.getTmp2(), step.getPre(),
                step.getLeakAttr());
            finalX = x2;
        }

        for (Operation* user : llvm::make_early_inc_range(x.getUsers())) {
            if (forOp->isAncestor(user) || newFor->isAncestor(user)) continue;
            if (user->getBlock() != forOp->getBlock()) continue;
            if (!forOp->isBeforeInBlock(user)) continue;

            if (auto load = dyn_cast<memref::LoadOp>(user)) {
                load.getMemrefMutable().assign(finalX);
            }
        }

        rewriter.eraseOp(forOp);
        return success();
    }
};

struct PingPongElideCopyFallback : OpRewritePattern<scf::ForOp> {
    explicit PingPongElideCopyFallback(MLIRContext* ctx)
        : OpRewritePattern<scf::ForOp>(ctx, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(scf::ForOp forOp,
                                  PatternRewriter& rewriter) const override {
        rc::ReservoirStepDenseOp step;
        scf::ForOp copyILoop;
        if (!findStepAndCopy(forOp, step, copyILoop)) return failure();

        Value x = step.getX();
        Value x2 = step.getX2();
        if (!isCopyLoop(copyILoop, /*src=*/x2, /*dst=*/x)) return failure();

        Location loc = forOp.getLoc();
        Value lb = forOp.getLowerBound();
        Value ub = forOp.getUpperBound();
        Value st = forOp.getStep();

        auto newFor = rewriter.create<scf::ForOp>(
            loc, lb, ub, st, ValueRange{x, x2},
            [&](OpBuilder& b, Location l, Value /*iv*/, ValueRange iterArgs) {
                Value xCur = iterArgs[0];
                Value xNext = iterArgs[1];

                b.create<rc::ReservoirStepDenseOp>(
                    l, step.getW(), step.getWin(), step.getU(),
                    /*x*/ xCur,
                    /*x2*/ xNext, step.getTmp1(), step.getTmp2(), step.getPre(),
                    step.getLeakAttr());

                b.create<scf::YieldOp>(l, ValueRange{xNext, xCur});
            });

        Value xFinal = newFor.getResult(0);

        for (Operation* user : llvm::make_early_inc_range(x.getUsers())) {
            if (forOp->isAncestor(user) || newFor->isAncestor(user)) continue;
            if (user->getBlock() != forOp->getBlock()) continue;
            if (!forOp->isBeforeInBlock(user)) continue;

            if (auto load = dyn_cast<memref::LoadOp>(user)) {
                load.getMemrefMutable().assign(xFinal);
            }
        }

        rewriter.eraseOp(forOp);
        return success();
    }
};

struct LowerDenseStep : OpRewritePattern<rc::ReservoirStepDenseOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(rc::ReservoirStepDenseOp op,
                                  PatternRewriter& rewriter) const override {
        Location loc = op.getLoc();

        Value W = op.getW();
        Value Win = op.getWin();
        Value u = op.getU();
        Value x = op.getX();
        Value x2 = op.getX2();

        float leak = static_cast<float>(op.getLeak().convertToDouble());
        rewriter.getContext()->getOrLoadDialect<mlir::math::MathDialect>();

        Value c0f = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(0.0f));
        Value one = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(1.0f));
        Value alpha = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32Type(), rewriter.getF32FloatAttr(leak));
        Value oneMinus = rewriter.create<arith::SubFOp>(loc, one, alpha);

        rewriter.create<linalg::FillOp>(loc, ValueRange{c0f}, ValueRange{x2});

        emitMatmulAccumulateIKJ(rewriter, loc, x, W, x2);

        emitMatmulAccumulateIKJ(rewriter, loc, u, Win, x2);

        emitTanhInPlace2D(rewriter, loc, x2);

        emitLerp2D(rewriter, loc, x, x2, x2, oneMinus, alpha);

        rewriter.eraseOp(op);
        return success();
    }
};

struct EraseAllocDeallocPair : OpRewritePattern<memref::AllocOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::AllocOp op,
                                  PatternRewriter& rewriter) const override {
        SmallVector<memref::DeallocOp, 4> deallocs;
        for (Operation* u : op->getUsers()) {
            auto d = dyn_cast<memref::DeallocOp>(u);
            if (!d) return failure();
            deallocs.push_back(d);
        }
        if (deallocs.empty()) return failure();

        for (auto d : deallocs) rewriter.eraseOp(d);
        rewriter.eraseOp(op);
        return success();
    }
};

}  // namespace

namespace rc {

void runConvertRcToLinalg(func::FuncOp func) {
    MLIRContext* ctx = func.getContext();

    {
        RewritePatternSet pp(ctx);
        pp.add<Unroll2ElideCopyConstT>(ctx);
        pp.add<PingPongElideCopyFallback>(ctx);
        (void)applyPatternsGreedily(func, std::move(pp));
    }

    {
        RewritePatternSet lower(ctx);
        lower.add<LowerDenseStep>(ctx);
        (void)applyPatternsGreedily(func, std::move(lower));
    }

    {
        RewritePatternSet dce(ctx);
        dce.add<EraseAllocDeallocPair>(ctx);
        (void)applyPatternsGreedily(func, std::move(dce));
    }
}

}  // namespace rc
