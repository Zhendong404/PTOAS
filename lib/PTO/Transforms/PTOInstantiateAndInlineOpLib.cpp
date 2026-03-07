#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINLINELIBCALL
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kOpLibAttrInstVariantId =
    "pto.oplib.instance.variant_id";
static constexpr llvm::StringLiteral kOpLibAttrInstOp = "pto.oplib.instance.op";
static constexpr llvm::StringLiteral kOpLibAttrInstDType =
    "pto.oplib.instance.dtype";

static bool isSupportedOpName(StringRef opName) {
  return opName == "tmul" || opName == "tdiv" || opName == "tadd" ||
         opName == "tsub" || opName == "tmax" || opName == "tmin";
}

static bool isInstanceFunc(func::FuncOp fn) {
  return fn->hasAttr(kOpLibAttrInstVariantId);
}

static FailureOr<MemRefType> convertTileBufToMemRefType(pto::TileBufType tileTy,
                                                         MLIRContext *ctx) {
  SmallVector<int64_t, 4> shape(tileTy.getShape().begin(), tileTy.getShape().end());
  SmallVector<int64_t, 4> dynStrides(tileTy.getRank(), ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dynStrides);
  return MemRefType::get(shape, tileTy.getElementType(), layout,
                         tileTy.getMemorySpace());
}

static Value maybeUnwrapCastToExpected(Value operand, Type expectedType) {
  if (operand.getType() == expectedType)
    return operand;

  auto cast = operand.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return operand;

  if (cast.getOperand(0).getType() == expectedType)
    return cast.getOperand(0);
  return operand;
}

static void eraseDeadBridgeCasts(func::FuncOp func) {
  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<UnrealizedConversionCastOp, 8> deadUnrealized;
    func.walk([&](UnrealizedConversionCastOp cast) {
      if (cast->use_empty())
        deadUnrealized.push_back(cast);
    });

    SmallVector<memref::CastOp, 8> deadMemrefCasts;
    func.walk([&](memref::CastOp cast) {
      if (cast->use_empty())
        deadMemrefCasts.push_back(cast);
    });

    if (deadUnrealized.empty() && deadMemrefCasts.empty())
      break;

    for (UnrealizedConversionCastOp cast : llvm::reverse(deadUnrealized))
      cast.erase();
    for (memref::CastOp cast : llvm::reverse(deadMemrefCasts))
      cast.erase();
    changed = true;
  }
}

static LogicalResult inlineCall(func::CallOp call, func::FuncOp callee) {
  if (call.getNumResults() != 0)
    return call.emitOpError("OP-Lib inline expects call without results");
  if (callee.isExternal())
    return call.emitOpError("callee must have a body before inlining");

  Block &entry = callee.getBody().front();
  if (entry.getNumArguments() != call.getNumOperands())
    return call.emitOpError("callee argument count mismatch during inlining");

  OpBuilder builder(call);
  IRMapping mapping;
  for (auto [arg, operand] : llvm::zip(entry.getArguments(), call.getOperands()))
    mapping.map(arg, operand);

  for (Operation &op : entry.without_terminator()) {
    Operation *newOp = builder.clone(op, mapping);
    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), newOp->getResults()))
      mapping.map(oldRes, newRes);
  }

  call.erase();
  return success();
}

static LogicalResult emitFakeBinaryBody(func::FuncOp instance) {
  if (!instance.isExternal())
    return success();

  auto opAttr = instance->getAttrOfType<StringAttr>(kOpLibAttrInstOp);
  if (!opAttr)
    return instance.emitOpError("missing attr pto.oplib.instance.op");

  StringRef opName = opAttr.getValue();
  if (!isSupportedOpName(opName))
    return instance.emitOpError("unsupported pto.oplib.instance.op");

  FunctionType fnTy = instance.getFunctionType();
  if (fnTy.getNumInputs() != 3 || fnTy.getNumResults() != 0)
    return instance.emitOpError("instance signature must be 3-input and void-return");

  auto src0Ty = dyn_cast<pto::TileBufType>(fnTy.getInput(0));
  auto src1Ty = dyn_cast<pto::TileBufType>(fnTy.getInput(1));
  auto dstTy = dyn_cast<pto::TileBufType>(fnTy.getInput(2));
  if (!src0Ty || !src1Ty || !dstTy)
    return instance.emitOpError("instance inputs must be !pto.tile_buf");
  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return instance.emitOpError("instance tile_buf rank must be 2");
  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return instance.emitOpError("instance input/output element type mismatch");

  Type elemTy = src0Ty.getElementType();
  if (!elemTy.isF16() && !elemTy.isF32())
    return instance.emitOpError("only f16/f32 instance dtype is supported");

  FailureOr<MemRefType> src0MemTyOr =
      convertTileBufToMemRefType(src0Ty, instance.getContext());
  FailureOr<MemRefType> src1MemTyOr =
      convertTileBufToMemRefType(src1Ty, instance.getContext());
  FailureOr<MemRefType> dstMemTyOr =
      convertTileBufToMemRefType(dstTy, instance.getContext());
  if (failed(src0MemTyOr) || failed(src1MemTyOr) || failed(dstMemTyOr)) {
    return instance.emitOpError(
        "failed to derive memref bridge type from instance !pto.tile_buf signature");
  }

  Block *entry = instance.addEntryBlock();
  OpBuilder b = OpBuilder::atBlockBegin(entry);
  Location loc = instance.getLoc();

  Value src0 = entry->getArgument(0);
  Value src1 = entry->getArgument(1);
  Value dst = entry->getArgument(2);

  // Keep public OP-Lib interface as !pto.tile_buf and bridge internally.
  Value src0Mem = b
                      .create<UnrealizedConversionCastOp>(
                          loc, TypeRange{*src0MemTyOr}, ValueRange{src0})
                      .getResult(0);
  Value src1Mem = b
                      .create<UnrealizedConversionCastOp>(
                          loc, TypeRange{*src1MemTyOr}, ValueRange{src1})
                      .getResult(0);
  Value dstMem = b
                     .create<UnrealizedConversionCastOp>(
                         loc, TypeRange{*dstMemTyOr}, ValueRange{dst})
                     .getResult(0);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
  Value m = b.create<memref::DimOp>(loc, dstMem, c0);
  Value n = b.create<memref::DimOp>(loc, dstMem, c1);

  auto outer = b.create<scf::ForOp>(loc, c0, m, c1);
  {
    OpBuilder bOuter = OpBuilder::atBlockBegin(outer.getBody());
    Value i = outer.getInductionVar();
    auto inner = bOuter.create<scf::ForOp>(loc, c0, n, c1);

    OpBuilder bInner = OpBuilder::atBlockBegin(inner.getBody());
    Value j = inner.getInductionVar();

    Value a = bInner.create<memref::LoadOp>(loc, src0Mem, ValueRange{i, j});
    Value c = bInner.create<memref::LoadOp>(loc, src1Mem, ValueRange{i, j});

    Value out;
    if (opName == "tadd") {
      out = bInner.create<arith::AddFOp>(loc, a, c);
    } else if (opName == "tsub") {
      out = bInner.create<arith::SubFOp>(loc, a, c);
    } else if (opName == "tmul") {
      out = bInner.create<arith::MulFOp>(loc, a, c);
    } else if (opName == "tdiv") {
      out = bInner.create<arith::DivFOp>(loc, a, c);
    } else if (opName == "tmax") {
      out = bInner.create<arith::MaximumFOp>(loc, a, c);
    } else {
      out = bInner.create<arith::MinimumFOp>(loc, a, c);
    }

    bInner.create<memref::StoreOp>(loc, out, dstMem, ValueRange{i, j});
  }

  b.create<func::ReturnOp>(loc);
  return success();
}

struct PTOInlineLibCallPass
    : public pto::impl::PTOInlineLibCallBase<PTOInlineLibCallPass> {
  using pto::impl::PTOInlineLibCallBase<
      PTOInlineLibCallPass>::PTOInlineLibCallBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int inlinedCalls = 0;
    int touchedFuncs = 0;

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func.getSymName().starts_with("__pto_oplib_"))
        continue;
      if (func.empty())
        continue;

      SmallVector<func::CallOp, 16> calls;
      func.walk([&](func::CallOp call) { calls.push_back(call); });

      bool changedThisFunc = false;
      for (func::CallOp oldCall : calls) {
        if (!oldCall || !oldCall->getBlock())
          continue;

        auto calleeAttr = oldCall.getCalleeAttr();
        if (!calleeAttr)
          continue;

        func::FuncOp callee = module.lookupSymbol<func::FuncOp>(calleeAttr.getValue());
        if (!callee || !isInstanceFunc(callee))
          continue;

        if (callee.isExternal()) {
          if (failed(emitFakeBinaryBody(callee))) {
            signalPassFailure();
            return;
          }
          if (debug) {
            auto op = callee->getAttrOfType<StringAttr>(kOpLibAttrInstOp);
            auto dtype = callee->getAttrOfType<StringAttr>(kOpLibAttrInstDType);
            llvm::errs() << "[op-fusion] instantiate-libcall: materialized @"
                         << callee.getSymName();
            if (op)
              llvm::errs() << " op=" << op.getValue();
            if (dtype)
              llvm::errs() << " dtype=" << dtype.getValue();
            llvm::errs() << "\n";
          }
        }

        func::CallOp call = oldCall;
        SmallVector<Value, 4> concreteOperands;
        concreteOperands.reserve(call.getNumOperands());
        for (auto [operand, expectedTy] :
             llvm::zip(call.getOperands(), callee.getFunctionType().getInputs())) {
          concreteOperands.push_back(maybeUnwrapCastToExpected(operand, expectedTy));
        }

        OpBuilder builder(call);
        auto newCall =
            builder.create<func::CallOp>(call.getLoc(), callee, concreteOperands);
        call.erase();

        if (failed(inlineCall(newCall, callee))) {
          signalPassFailure();
          return;
        }

        ++inlinedCalls;
        changedThisFunc = true;
        if (debug) {
          llvm::errs() << "[op-fusion] inline-libcall: inlined @" << callee.getSymName()
                       << " into @" << func.getSymName() << "\n";
        }
      }

      if (changedThisFunc) {
        eraseDeadBridgeCasts(func);
        ++touchedFuncs;
      }
    }

    if (debug) {
      llvm::errs() << "[op-fusion] inline-libcall touched " << touchedFuncs
                   << " function(s), inlined " << inlinedCalls << " call(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOInlineLibCallPass(
    const PTOInlineLibCallOptions &options) {
  return std::make_unique<PTOInlineLibCallPass>(options);
}
