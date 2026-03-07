#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVALIDATESIMDIR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kOpLibAttrKind = "pto.oplib.kind";
static constexpr llvm::StringLiteral kOpLibKindL3BinaryTemplate =
    "l3_binary_elementwise_template";
static constexpr llvm::StringLiteral kSimdLevelAttr = "pto.simd.level";
static constexpr llvm::StringLiteral kSimdLanesAttr = "pto.simd.lanes";
static constexpr llvm::StringLiteral kSimdCoreSlotAttr = "pto.simd.core_slot";
static constexpr llvm::StringLiteral kSimdLevelBinaryEwiseV1 =
    "binary_ewise_v1";
static constexpr llvm::StringLiteral kSimdCoreSlotBinaryEwise =
    "binary_ewise_core";

static constexpr llvm::StringLiteral kErrEmptyBody =
    "E_OPLIB_EMPTY_BODY_FOR_SIMD";
static constexpr llvm::StringLiteral kErrLanesMismatch =
    "E_OPLIB_SIMD_LANES_MISMATCH";
static constexpr llvm::StringLiteral kErrCoreSlot =
    "E_OPLIB_SIMD_INVALID_CORE_SLOT";
static constexpr llvm::StringLiteral kErrDType =
    "E_OPLIB_SIMD_UNSUPPORTED_DTYPE";
static constexpr llvm::StringLiteral kErrLayout =
    "E_OPLIB_SIMD_UNSUPPORTED_LAYOUT";
static constexpr llvm::StringLiteral kErrSimdAttrRequired =
    "E_OPLIB_SIMD_ATTR_REQUIRED";

static bool isBinaryFloatCore(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}

static FailureOr<int64_t> getFixedVectorLanes(Type ty) {
  auto vecTy = dyn_cast<VectorType>(ty);
  if (!vecTy || vecTy.isScalable())
    return failure();
  return vecTy.getNumElements();
}

static bool hasSimdBridgeOps(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::SimdPredicateOp, pto::SimdLoadOp, pto::SimdStoreOp,
            pto::SimdLoadPUOp, pto::SimdStorePUOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult emitCodeError(Operation *op, StringRef code,
                                   const Twine &msg) {
  op->emitError() << code << ": " << msg;
  return failure();
}

static LogicalResult emitCodeError(Location loc, StringRef code,
                                   const Twine &msg) {
  emitError(loc) << code << ": " << msg;
  return failure();
}

static LogicalResult validateFuncSignature(func::FuncOp func) {
  FunctionType fnTy = func.getFunctionType();
  if (fnTy.getNumInputs() != 3 || fnTy.getNumResults() != 0) {
    return emitCodeError(func.getLoc(), kErrEmptyBody,
                         "SIMD template signature must be 3-input void-return");
  }

  for (Type inTy : fnTy.getInputs()) {
    auto tileTy = dyn_cast<pto::TileBufType>(inTy);
    if (!tileTy)
      return emitCodeError(func.getLoc(), kErrEmptyBody,
                           "SIMD template inputs must be !pto.tile_buf");

    Type elemTy = tileTy.getElementType();
    if (!elemTy.isF16() && !elemTy.isF32()) {
      return emitCodeError(func.getLoc(), kErrDType,
                           "SIMD V1 supports f16/f32 only");
    }

    if (tileTy.getBLayoutValueI32() != 0) {
      return emitCodeError(func.getLoc(), kErrLayout,
                           "SIMD V1 supports row_major only");
    }
  }

  return success();
}

static LogicalResult validateSimdBody(func::FuncOp func, int64_t lanes) {
  if (func.isExternal() || func.empty() ||
      func.front().without_terminator().empty()) {
    return emitCodeError(func.getLoc(), kErrEmptyBody,
                         "SIMD template body must not be empty");
  }

  llvm::DenseMap<Operation *, int64_t> preorder;
  int64_t seq = 0;
  func.walk([&](Operation *op) { preorder[op] = seq++; });

  int64_t firstLoad = std::numeric_limits<int64_t>::max();
  int64_t firstStore = std::numeric_limits<int64_t>::max();
  int64_t coreSeq = -1;
  int coreCount = 0;

  for (Operation &op : func.getBody().front().getOperations()) {
    (void)op;
  }

  func.walk([&](Operation *op) {
    if (isa<pto::SimdLoadOp, pto::SimdLoadPUOp>(op)) {
      firstLoad = std::min(firstLoad, preorder[op]);
    }
    if (isa<pto::SimdStoreOp, pto::SimdStorePUOp>(op)) {
      firstStore = std::min(firstStore, preorder[op]);
    }

    if (auto slotAttr = op->getAttrOfType<StringAttr>(kSimdCoreSlotAttr)) {
      if (slotAttr.getValue() != kSimdCoreSlotBinaryEwise) {
        (void)emitCodeError(op, kErrCoreSlot,
                            "unsupported core_slot, expected 'binary_ewise_core'");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      if (!isBinaryFloatCore(op)) {
        (void)emitCodeError(op, kErrCoreSlot,
                            "core_slot op must be binary floating-point arith op");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      if (op->getNumResults() != 1) {
        (void)emitCodeError(op, kErrCoreSlot,
                            "core_slot op must have exactly one result");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      FailureOr<int64_t> vLanes = getFixedVectorLanes(op->getResult(0).getType());
      if (failed(vLanes) || *vLanes != lanes) {
        (void)emitCodeError(op, kErrLanesMismatch,
                            "core_slot vector lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      ++coreCount;
      coreSeq = preorder[op];
    }

    if (auto pred = dyn_cast<pto::SimdPredicateOp>(op)) {
      FailureOr<int64_t> predLanes = getFixedVectorLanes(pred.getMask().getType());
      if (failed(predLanes) || *predLanes != lanes) {
        (void)emitCodeError(pred, kErrLanesMismatch,
                            "simd.predicate lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto load = dyn_cast<pto::SimdLoadOp>(op)) {
      FailureOr<int64_t> loadLanes = getFixedVectorLanes(load.getValue().getType());
      if (failed(loadLanes) || *loadLanes != lanes) {
        (void)emitCodeError(load, kErrLanesMismatch,
                            "simd.load lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto loadPU = dyn_cast<pto::SimdLoadPUOp>(op)) {
      FailureOr<int64_t> loadLanes =
          getFixedVectorLanes(loadPU.getValue().getType());
      if (failed(loadLanes) || *loadLanes != lanes) {
        (void)emitCodeError(loadPU, kErrLanesMismatch,
                            "simd.load_pu lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto store = dyn_cast<pto::SimdStoreOp>(op)) {
      FailureOr<int64_t> storeLanes = getFixedVectorLanes(store.getValue().getType());
      if (failed(storeLanes) || *storeLanes != lanes) {
        (void)emitCodeError(store, kErrLanesMismatch,
                            "simd.store lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto storePU = dyn_cast<pto::SimdStorePUOp>(op)) {
      FailureOr<int64_t> storeLanes =
          getFixedVectorLanes(storePU.getValue().getType());
      if (failed(storeLanes) || *storeLanes != lanes) {
        (void)emitCodeError(storePU, kErrLanesMismatch,
                            "simd.store_pu lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }
  });

  if (coreSeq == std::numeric_limits<int64_t>::min())
    return failure();

  if (coreCount != 1) {
    return emitCodeError(func.getLoc(), kErrCoreSlot,
                         "SIMD template must contain exactly one 'binary_ewise_core' op");
  }

  if (firstLoad == std::numeric_limits<int64_t>::max()) {
    return emitCodeError(func.getLoc(), kErrCoreSlot,
                         "SIMD template must contain at least one simd.load/simd.load_pu");
  }

  if (firstStore == std::numeric_limits<int64_t>::max()) {
    return emitCodeError(func.getLoc(), kErrCoreSlot,
                         "SIMD template must contain at least one simd.store/simd.store_pu");
  }

  if (!(firstLoad < coreSeq && coreSeq < firstStore)) {
    return emitCodeError(func.getLoc(), kErrCoreSlot,
                         "SIMD template ordering must satisfy load -> core -> store");
  }

  return success();
}

struct PTOValidateSimdIRPass
    : public pto::impl::PTOValidateSimdIRBase<PTOValidateSimdIRPass> {
  using pto::impl::PTOValidateSimdIRBase<
      PTOValidateSimdIRPass>::PTOValidateSimdIRBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!hasSimdBridgeOps(func))
        continue;

      auto levelAttr = func->getAttrOfType<StringAttr>(kSimdLevelAttr);
      if (!levelAttr) {
        func.emitError() << kErrSimdAttrRequired
                         << ": using pto.simd.* requires attr pto.simd.level";
        signalPassFailure();
        return;
      }

      if (levelAttr.getValue() != kSimdLevelBinaryEwiseV1) {
        func.emitError() << kErrCoreSlot
                         << ": unsupported pto.simd.level, expected '"
                         << kSimdLevelBinaryEwiseV1 << "'";
        signalPassFailure();
        return;
      }

      auto lanesAttr = func->getAttrOfType<IntegerAttr>(kSimdLanesAttr);
      if (!lanesAttr || lanesAttr.getInt() <= 0) {
        func.emitError() << kErrSimdAttrRequired
                         << ": using pto.simd.* requires positive pto.simd.lanes";
        signalPassFailure();
        return;
      }
      int64_t lanes = lanesAttr.getInt();

      // Only enforce OP-Lib binary template shape constraints on OP-Lib entries.
      auto kindAttr = func->getAttrOfType<StringAttr>(kOpLibAttrKind);
      if (kindAttr && kindAttr.getValue() == kOpLibKindL3BinaryTemplate) {
        if (failed(validateFuncSignature(func))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(validateSimdBody(func, lanes))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOValidateSimdIRPass() {
  return std::make_unique<PTOValidateSimdIRPass>();
}
