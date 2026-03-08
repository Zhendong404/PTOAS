#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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

static bool isTileWorldPTOp(Operation *op) { return isa<pto::OpPipeInterface>(op); }

static bool isSupportedElemType(Type ty) { return ty.isF16() || ty.isF32(); }

static int64_t getElemBytes(Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16())
      return 2;
    if (ft.isF32())
      return 4;
    if (ft.isF64())
      return 8;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bytes = it.getWidth() / 8;
    return bytes > 0 ? bytes : 1;
  }
  return -1;
}

static bool readBLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<pto::BLayoutAttr>(attr)) {
    out = static_cast<int32_t>(a.getValue());
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(a.getInt());
    return true;
  }
  return false;
}

static bool readSLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<pto::SLayoutAttr>(attr)) {
    out = static_cast<int32_t>(a.getValue());
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(a.getInt());
    return true;
  }
  return false;
}

static bool inferTileBufStaticStrides(pto::TileBufType tileTy,
                                      SmallVectorImpl<int64_t> &strides) {
  if (tileTy.getRank() != 2)
    return false;

  ArrayRef<int64_t> shape = tileTy.getShape();
  if (shape[0] == ShapedType::kDynamic || shape[1] == ShapedType::kDynamic)
    return false;

  int64_t rows = shape[0];
  int64_t cols = shape[1];

  int32_t bl = 0; // row_major
  int32_t sl = 0; // none_box
  int32_t fr = 512;
  auto cfg = tileTy.getConfigAttr();
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    fr = static_cast<int32_t>(attr.getInt());

  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (sl != 0) {
    int64_t elemBytes = getElemBytes(tileTy.getElementType());
    if (elemBytes <= 0)
      return false;
    if (fr == 1024) {
      innerRows = 16;
      innerCols = 16;
    } else if (fr == 32) {
      innerRows = 16;
      innerCols = 2;
    } else if (fr == 512) {
      if (sl == 1) {
        innerRows = 16;
        innerCols = 32 / elemBytes;
      } else if (sl == 2) {
        innerRows = 32 / elemBytes;
        innerCols = 16;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  if (sl == 0) {
    if (bl == 1) {
      strides.clear();
      strides.push_back(1);
      strides.push_back(rows);
    } else {
      strides.clear();
      strides.push_back(cols);
      strides.push_back(1);
    }
    return true;
  }

  if (bl == 1) {
    if (sl != 1)
      return false;
    strides.clear();
    strides.push_back(innerCols);
    strides.push_back(rows);
    return true;
  }

  strides.clear();
  strides.push_back(cols);
  strides.push_back(innerRows);
  return true;
}

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

static bool getConstIndexLike(Value v, int64_t &out) {
  if (!v)
    return false;
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = c.value();
    return true;
  }
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = c.value();
    return true;
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexLike(cast.getIn(), out);
  return false;
}

struct MemRefTileMetadata {
  pto::TileBufConfigAttr config;
  Value validRow;
  Value validCol;
  Value addr;
};

static bool enrichFromPointerCast(pto::PointerCastOp pc, MemRefTileMetadata &meta) {
  if (!meta.validRow)
    meta.validRow = pc.getValidRow();
  if (!meta.validCol)
    meta.validCol = pc.getValidCol();
  if (!meta.config) {
    if (auto cfg = pc.getConfig())
      meta.config = *cfg;
  }
  if (!meta.addr && pc.getAddrs().size() == 1)
    meta.addr = pc.getAddrs().front();
  return true;
}

static bool collectTileMetadata(Value v, MemRefTileMetadata &meta) {
  if (!v)
    return false;
  if (auto bind = v.getDefiningOp<pto::BindTileOp>()) {
    meta.config = bind.getConfig();
    if (!meta.validRow)
      meta.validRow = bind.getValidRow();
    if (!meta.validCol)
      meta.validCol = bind.getValidCol();
    if (auto pc = bind.getSource().getDefiningOp<pto::PointerCastOp>())
      (void)enrichFromPointerCast(pc, meta);
    return true;
  }
  if (auto pc = v.getDefiningOp<pto::PointerCastOp>())
    return enrichFromPointerCast(pc, meta);
  if (auto subview = v.getDefiningOp<memref::SubViewOp>())
    return collectTileMetadata(subview.getSource(), meta);
  if (auto recast = v.getDefiningOp<memref::ReinterpretCastOp>())
    return collectTileMetadata(recast.getSource(), meta);
  if (auto cast = v.getDefiningOp<memref::CastOp>())
    return collectTileMetadata(cast.getSource(), meta);
  return false;
}

static FailureOr<pto::TileBufType>
buildTileTypeFromMemRefAndMetadata(MemRefType memTy, const MemRefTileMetadata &meta,
                                   Value &dynamicVRow, Value &dynamicVCol,
                                   MLIRContext *ctx) {
  if (memTy.getRank() != 2)
    return failure();
  if (!isSupportedElemType(memTy.getElementType()))
    return failure();

  constexpr int64_t kDynamicValid = -1;
  auto resolveValidDim = [&](int64_t shapeDim, Value v, int64_t &outStatic,
                             Value &outDynamic) -> LogicalResult {
    outStatic = shapeDim;
    outDynamic = Value();
    if (!v) {
      if (shapeDim == ShapedType::kDynamic)
        return failure();
      return success();
    }

    int64_t constVal = 0;
    if (getConstIndexLike(v, constVal)) {
      if (constVal < 0)
        return failure();
      outStatic = constVal;
      return success();
    }

    outStatic = kDynamicValid;
    outDynamic = v;
    return success();
  };

  SmallVector<int64_t, 2> shape(memTy.getShape().begin(), memTy.getShape().end());
  int64_t validRow = ShapedType::kDynamic;
  int64_t validCol = ShapedType::kDynamic;
  if (failed(resolveValidDim(shape[0], meta.validRow, validRow, dynamicVRow)))
    return failure();
  if (failed(resolveValidDim(shape[1], meta.validCol, validCol, dynamicVCol)))
    return failure();

  SmallVector<int64_t, 2> validShape{validRow, validCol};
  auto cfg = meta.config ? meta.config : pto::TileBufConfigAttr::getDefault(ctx);
  return pto::TileBufType::get(ctx, shape, memTy.getElementType(),
                               memTy.getMemorySpace(), validShape, cfg);
}

static FailureOr<MemRefType> convertTileBufToMemRefType(pto::TileBufType tileTy,
                                                         MLIRContext *ctx) {
  SmallVector<int64_t, 4> shape(tileTy.getShape().begin(), tileTy.getShape().end());
  SmallVector<int64_t, 4> strides(tileTy.getRank(), ShapedType::kDynamic);
  (void)inferTileBufStaticStrides(tileTy, strides);
  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/0, strides);
  return MemRefType::get(shape, tileTy.getElementType(), layout,
                         tileTy.getMemorySpace());
}

class PTOMemrefToTileBufPass
    : public PassWrapper<PTOMemrefToTileBufPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOMemrefToTileBufPass)

  StringRef getArgument() const override { return "pto-memref-to-tilebuf"; }
  StringRef getDescription() const override {
    return "Recover tile_buf operands from bind_tile metadata for tile-world PTO ops";
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

        const bool isFusionOp = isSupportedFusionOp(&op);
        if (!isTileWorldPTOp(&op) && !isFusionOp)
          continue;

        for (unsigned i = 0, e = op.getNumOperands(); i < e; ++i) {
          // Keep legacy fallback cast only for OP-Lib V1 fusible binary ops.
          bool allowLegacyFallback = isFusionOp && i < 3;
          rewriteOperandToTileBuf(&op, i, castCache, allowLegacyFallback);
        }
      }
    }
  }

  void rewriteOperandToTileBuf(Operation *op, unsigned operandIndex,
                               CastCache &castCache,
                               bool allowLegacyFallback) {
    Value operand = op->getOperand(operandIndex);
    auto memTy = dyn_cast<MemRefType>(operand.getType());
    if (!memTy)
      return;

    auto cached = castCache.find(operand);
    if (cached != castCache.end()) {
      op->setOperand(operandIndex, cached->second);
      return;
    }

    OpBuilder builder(op);
    Value tileVal;

    MemRefTileMetadata meta;
    if (collectTileMetadata(operand, meta)) {
      Value dynamicVRow;
      Value dynamicVCol;
      FailureOr<pto::TileBufType> tileTyOr = buildTileTypeFromMemRefAndMetadata(
          memTy, meta, dynamicVRow, dynamicVCol, &getContext());
      if (succeeded(tileTyOr)) {
        auto alloc = builder.create<pto::AllocTileOp>(
            op->getLoc(), *tileTyOr, meta.addr, dynamicVRow, dynamicVCol);
        tileVal = alloc.getResult();
      }
    }

    if (!tileVal && !allowLegacyFallback)
      return;

    // Fallback: keep legacy cast behavior for non-bind_tile memref values.
    if (!tileVal) {
      FailureOr<pto::TileBufType> tileTyOr =
          convertMemRefToTileBufType(memTy, &getContext());
      if (failed(tileTyOr))
        return;
      auto cast = builder.create<UnrealizedConversionCastOp>(
          op->getLoc(), TypeRange{*tileTyOr}, ValueRange{operand});
      tileVal = cast.getResult(0);
    }

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
    if (failed(lowerSimdTileToMemrefOps(module))) {
      signalPassFailure();
      return;
    }
    if (failed(rewriteFunctionSignatures(module))) {
      signalPassFailure();
      return;
    }
    if (failed(rewriteCallsAndReturns(module))) {
      signalPassFailure();
      return;
    }
    if (failed(lowerSimdTileToMemrefOps(module))) {
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

    if (isa<pto::TileBufType>(operand.getType())) {
      auto cast =
          builder.create<pto::SimdTileToMemrefOp>(loc, toMem, operand);
      return cast.getDst();
    }

    // Keep memref->memref signature adaptation permissive.
    if (!isa<MemRefType>(operand.getType()))
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

  LogicalResult lowerSimdTileToMemrefOps(ModuleOp module) {
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;

      SmallVector<pto::SimdTileToMemrefOp, 16> bridgeOps;
      func.walk([&](pto::SimdTileToMemrefOp op) { bridgeOps.push_back(op); });

      for (pto::SimdTileToMemrefOp op : bridgeOps) {
        if (!op || !op->getBlock())
          continue;

        OpBuilder builder(op);
        Value src = op->getOperand(0);
        Value dst = op->getResult(0);
        Type dstTy = dst.getType();
        Type loweredTy = dstTy;

        // Prefer a shape-specialized memref when source is tile_buf so
        // downstream memref.dim can fold for static tile shapes.
        if (auto tileTy = dyn_cast<pto::TileBufType>(src.getType())) {
          FailureOr<MemRefType> inferredTyOr =
              convertTileBufToMemRefType(tileTy, &getContext());
          if (succeeded(inferredTyOr))
            loweredTy = *inferredTyOr;
        }

        if (src.getType() == loweredTy) {
          dst.replaceAllUsesWith(src);
          op.erase();
          continue;
        }

        auto cast = builder.create<UnrealizedConversionCastOp>(
            op.getLoc(), TypeRange{loweredTy}, ValueRange{src});
        dst.replaceAllUsesWith(cast.getResult(0));
        op.erase();
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
