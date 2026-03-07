#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

static bool isSupportedFusionOp(Operation *op) {
  return isa<pto::TMulOp, pto::TDivOp, pto::TAddOp, pto::TSubOp, pto::TMaxOp,
             pto::TMinOp>(op);
}

static bool isSupportedElemType(Type ty) { return ty.isF16() || ty.isF32(); }

static FailureOr<pto::TileBufType>
convertMemRefToTileBufType(MemRefType memTy, MLIRContext *ctx) {
  if (memTy.getRank() != 2)
    return failure();
  if (!isSupportedElemType(memTy.getElementType()))
    return failure();

  SmallVector<int64_t, 2> shape(memTy.getShape().begin(), memTy.getShape().end());
  SmallVector<int64_t, 2> validShape(shape.begin(), shape.end());
  auto cfg = pto::TileBufConfigAttr::getDefault(ctx);
  return pto::TileBufType::get(ctx, shape, memTy.getElementType(),
                               memTy.getMemorySpace(), validShape, cfg);
}

static FailureOr<MemRefType> convertTileBufToMemRefType(pto::TileBufType tileTy,
                                                         MLIRContext *ctx) {
  SmallVector<int64_t, 4> shape(tileTy.getShape().begin(), tileTy.getShape().end());
  SmallVector<int64_t, 4> dynStrides(tileTy.getRank(), ShapedType::kDynamic);
  auto layout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, dynStrides);
  return MemRefType::get(shape, tileTy.getElementType(), layout,
                         tileTy.getMemorySpace());
}

class PTOMemrefToTileBufPass
    : public PassWrapper<PTOMemrefToTileBufPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOMemrefToTileBufPass)

  StringRef getArgument() const override { return "pto-memref-to-tilebuf"; }
  StringRef getDescription() const override {
    return "Cast memref operands of fusible PTO ops back to tile_buf";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect, func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override { processRegion(getOperation().getBody()); }

private:
  using CastCache = llvm::DenseMap<Value, Value>;

  void processRegion(Region &region) {
    for (Block &block : region.getBlocks()) {
      CastCache castCache;
      for (Operation &op : llvm::make_early_inc_range(block.getOperations())) {
        for (Region &nested : op.getRegions())
          processRegion(nested);

        if (!isSupportedFusionOp(&op))
          continue;
        if (op.getNumOperands() < 3)
          continue;

        for (unsigned i = 0; i < 3; ++i)
          rewriteOperandToTileBuf(&op, i, castCache);
      }
    }
  }

  void rewriteOperandToTileBuf(Operation *op, unsigned operandIndex,
                               CastCache &castCache) {
    Value operand = op->getOperand(operandIndex);
    auto memTy = dyn_cast<MemRefType>(operand.getType());
    if (!memTy)
      return;

    FailureOr<pto::TileBufType> tileTyOr =
        convertMemRefToTileBufType(memTy, &getContext());
    if (failed(tileTyOr))
      return;

    auto cached = castCache.find(operand);
    if (cached != castCache.end() && cached->second.getType() == *tileTyOr) {
      op->setOperand(operandIndex, cached->second);
      return;
    }

    OpBuilder builder(op);
    auto cast = builder.create<UnrealizedConversionCastOp>(
        op->getLoc(), TypeRange{*tileTyOr}, ValueRange{operand});
    Value tileVal = cast.getResult(0);
    castCache[operand] = tileVal;
    op->setOperand(operandIndex, tileVal);
  }
};

class PTOTileBufToMemrefPass
    : public PassWrapper<PTOTileBufToMemrefPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOTileBufToMemrefPass)

  StringRef getArgument() const override { return "pto-tilebuf-to-memref"; }
  StringRef getDescription() const override {
    return "Cast tile_buf values back to memref at OP-Lib call boundaries";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect, func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(rewriteFunctionSignatures(module))) {
      signalPassFailure();
      return;
    }
    if (failed(rewriteCallsAndReturns(module))) {
      signalPassFailure();
      return;
    }
  }

private:
  Type convertType(Type ty) {
    auto tileTy = dyn_cast<pto::TileBufType>(ty);
    if (!tileTy)
      return ty;
    FailureOr<MemRefType> memTyOr = convertTileBufToMemRefType(tileTy, &getContext());
    if (failed(memTyOr))
      return Type();
    return *memTyOr;
  }

  LogicalResult rewriteFunctionSignatures(ModuleOp module) {
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      FunctionType fnTy = func.getFunctionType();
      SmallVector<Type, 8> newInputs;
      SmallVector<Type, 4> newResults;
      newInputs.reserve(fnTy.getNumInputs());
      newResults.reserve(fnTy.getNumResults());

      bool changed = false;
      for (Type inTy : fnTy.getInputs()) {
        Type converted = convertType(inTy);
        if (!converted) {
          func.emitError("failed to convert function input tile_buf type to memref");
          return failure();
        }
        changed |= (converted != inTy);
        newInputs.push_back(converted);
      }

      for (Type resTy : fnTy.getResults()) {
        Type converted = convertType(resTy);
        if (!converted) {
          func.emitError("failed to convert function result tile_buf type to memref");
          return failure();
        }
        changed |= (converted != resTy);
        newResults.push_back(converted);
      }

      if (!changed)
        continue;

      func.setType(FunctionType::get(func.getContext(), newInputs, newResults));

      if (!func.isExternal() && !func.empty()) {
        Block &entry = func.front();
        for (auto [idx, argTy] : llvm::enumerate(newInputs))
          entry.getArgument(idx).setType(argTy);
      }
    }
    return success();
  }

  FailureOr<Value> castOperandToExpected(OpBuilder &builder, Location loc,
                                         Value operand, Type expectedType) {
    if (operand.getType() == expectedType)
      return operand;

    auto toMem = dyn_cast<MemRefType>(expectedType);
    if (!toMem)
      return failure();

    // Use UnrealizedConversionCast as a bridge for both tile_buf->memref and
    // memref->memref signature adaptation at call boundaries.
    if (!isa<pto::TileBufType, MemRefType>(operand.getType()))
      return failure();

    auto cast = builder.create<UnrealizedConversionCastOp>(
        loc, TypeRange{expectedType}, ValueRange{operand});
    return cast.getResult(0);
  }

  LogicalResult rewriteOneCall(func::CallOp call, ModuleOp module) {
    auto calleeAttr = call.getCalleeAttr();
    if (!calleeAttr)
      return success();

    func::FuncOp callee = module.lookupSymbol<func::FuncOp>(calleeAttr.getValue());
    if (!callee)
      return success();

    FunctionType calleeTy = callee.getFunctionType();
    if (calleeTy.getNumInputs() != call.getNumOperands()) {
      call.emitOpError("call operand count mismatch with callee signature");
      return failure();
    }

    OpBuilder builder(call);
    SmallVector<Value, 8> newOperands;
    newOperands.reserve(call.getNumOperands());

    bool changed = false;
    for (auto [operand, expectedTy] :
         llvm::zip(call.getOperands(), calleeTy.getInputs())) {
      FailureOr<Value> converted =
          castOperandToExpected(builder, call.getLoc(), operand, expectedTy);
      if (failed(converted)) {
        call.emitOpError("failed to cast call operand to callee argument type");
        return failure();
      }
      changed |= (*converted != operand);
      newOperands.push_back(*converted);
    }

    bool resultTypeChanged =
        !llvm::equal(call.getResultTypes(), calleeTy.getResults());
    if (!changed && !resultTypeChanged)
      return success();

    auto newCall = builder.create<func::CallOp>(call.getLoc(), callee, newOperands);
    for (NamedAttribute attr : call->getAttrs()) {
      if (attr.getName().getValue() == "callee")
        continue;
      newCall->setAttr(attr.getName(), attr.getValue());
    }
    call.replaceAllUsesWith(newCall.getResults());
    call.erase();
    return success();
  }

  LogicalResult rewriteOneReturn(func::ReturnOp ret, func::FuncOp parentFunc) {
    FunctionType fnTy = parentFunc.getFunctionType();
    if (fnTy.getNumResults() != ret.getNumOperands()) {
      ret.emitOpError("return operand count mismatch with function signature");
      return failure();
    }

    if (ret.getNumOperands() == 0)
      return success();

    OpBuilder builder(ret);
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(ret.getNumOperands());

    bool changed = false;
    for (auto [operand, expectedTy] :
         llvm::zip(ret.getOperands(), fnTy.getResults())) {
      FailureOr<Value> converted =
          castOperandToExpected(builder, ret.getLoc(), operand, expectedTy);
      if (failed(converted)) {
        ret.emitOpError("failed to cast return operand to function result type");
        return failure();
      }
      changed |= (*converted != operand);
      newOperands.push_back(*converted);
    }

    if (!changed)
      return success();

    builder.create<func::ReturnOp>(ret.getLoc(), newOperands);
    ret.erase();
    return success();
  }

  LogicalResult rewriteCallsAndReturns(ModuleOp module) {
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;

      SmallVector<Operation *, 16> worklist;
      func.walk([&](Operation *op) { worklist.push_back(op); });
      for (Operation *op : worklist) {
        if (!op->getBlock())
          continue;
        if (auto call = dyn_cast<func::CallOp>(op)) {
          if (failed(rewriteOneCall(call, module)))
            return failure();
          continue;
        }
        if (auto ret = dyn_cast<func::ReturnOp>(op)) {
          if (failed(rewriteOneReturn(ret, func)))
            return failure();
          continue;
        }
      }
    }
    return success();
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createPTOMemrefToTileBufPass() {
  return std::make_unique<PTOMemrefToTileBufPass>();
}

std::unique_ptr<Pass> createPTOTileBufToMemrefPass() {
  return std::make_unique<PTOTileBufToMemrefPass>();
}

} // namespace pto
} // namespace mlir
