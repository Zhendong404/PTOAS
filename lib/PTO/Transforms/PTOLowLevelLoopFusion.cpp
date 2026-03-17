#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWLEVELLOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct StageInfo {
  pto::SimdVecScopeOp scope;
  SmallVector<Operation *, 4> setupOps;
  SmallVector<Operation *, 4> scopePreludeOps;
  scf::ForOp loop;
  SmallVector<Operation *, 8> loopOps;
  vector::MaskedStoreOp store;
};

struct ForwardedStore {
  Value base;
  SmallVector<Value, 2> indices;
  Value mask;
  Value value;
};

static bool sameValues(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  return lhs.size() == rhs.size() && llvm::equal(lhs, rhs);
}

static SmallVector<Value, 4> mapValues(ValueRange values, IRMapping &mapping) {
  SmallVector<Value, 4> mapped;
  mapped.reserve(values.size());
  for (Value value : values)
    mapped.push_back(mapping.lookupOrDefault(value));
  return mapped;
}

static bool sameForHeader(scf::ForOp lhs, scf::ForOp rhs) {
  return lhs.getLowerBound() == rhs.getLowerBound() &&
         lhs.getUpperBound() == rhs.getUpperBound() &&
         lhs.getStep() == rhs.getStep() && lhs.getInitArgs().empty() &&
         rhs.getInitArgs().empty() && lhs->getAttrs() == rhs->getAttrs();
}

static bool isPureNoRegionOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static bool isSupportedLoopOp(Operation *op) {
  if (isa<vector::MaskedLoadOp, vector::MaskedStoreOp>(op))
    return true;
  return isPureNoRegionOp(op);
}

static bool isInterstageSetupOp(Operation *op) {
  if (isa<pto::SimdTileToMemrefOp>(op))
    return true;
  return isPureNoRegionOp(op);
}

static bool areEquivalentMaskValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsMask = lhs.getDefiningOp<vector::ConstantMaskOp>();
  auto rhsMask = rhs.getDefiningOp<vector::ConstantMaskOp>();
  if (!lhsMask || !rhsMask)
    return false;
  return lhs.getType() == rhs.getType() &&
         lhsMask.getMaskDimSizesAttr() == rhsMask.getMaskDimSizesAttr();
}

static LogicalResult analyzeStage(pto::SimdVecScopeOp scope, StageInfo &stage) {
  stage.scope = scope;

  Block &scopeBody = scope.getBody().front();
  bool seenLoop = false;
  for (Operation &op : scopeBody) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (seenLoop)
        return failure();
      seenLoop = true;
      stage.loop = loop;
      continue;
    }
    if (seenLoop || !isPureNoRegionOp(&op))
      return failure();
    stage.scopePreludeOps.push_back(&op);
  }

  if (!stage.loop || !stage.loop.getInitArgs().empty())
    return failure();

  int storesSeen = 0;
  for (Operation &op : stage.loop.getBody()->without_terminator()) {
    if (!isSupportedLoopOp(&op))
      return failure();
    if (auto store = dyn_cast<vector::MaskedStoreOp>(op)) {
      if (++storesSeen > 1)
        return failure();
      stage.store = store;
    }
    stage.loopOps.push_back(&op);
  }

  if (storesSeen != 1)
    return failure();

  for (Operation *op : stage.loopOps)
    if (auto load = dyn_cast<vector::MaskedLoadOp>(op))
      if (!areEquivalentMaskValues(load.getMask(), stage.store.getMask()))
        return failure();

  return success();
}

static SmallVector<StageInfo, 8>
collectStageRunFrom(pto::SimdVecScopeOp firstScope) {
  SmallVector<StageInfo, 8> stages;

  StageInfo firstStage;
  if (failed(analyzeStage(firstScope, firstStage)))
    return stages;
  stages.push_back(std::move(firstStage));

  SmallVector<Operation *, 4> pendingSetup;
  for (Operation *op = firstScope->getNextNode(); op; op = op->getNextNode()) {
    if (auto nextScope = dyn_cast<pto::SimdVecScopeOp>(op)) {
      StageInfo nextStage;
      nextStage.setupOps = pendingSetup;
      pendingSetup.clear();
      if (failed(analyzeStage(nextScope, nextStage)))
        break;
      stages.push_back(std::move(nextStage));
      continue;
    }

    if (!isInterstageSetupOp(op))
      break;
    pendingSetup.push_back(op);
  }

  return stages;
}

static const ForwardedStore *findForwardedStore(ArrayRef<ForwardedStore> stores,
                                                Value base,
                                                ArrayRef<Value> indices,
                                                Value mask) {
  for (const ForwardedStore &store : llvm::reverse(stores))
    if (store.base == base && sameValues(store.indices, indices) &&
        areEquivalentMaskValues(store.mask, mask))
      return &store;
  return nullptr;
}

static bool fuseStageRun(SmallVectorImpl<StageInfo> &stages) {
  if (stages.size() < 2)
    return false;

  StageInfo &first = stages.front();
  for (StageInfo &stage : llvm::drop_begin(stages)) {
    if (stage.scope->getAttrs() != first.scope->getAttrs())
      return false;
    if (!sameForHeader(first.loop, stage.loop))
      return false;
  }

  DenseMap<Value, unsigned> lastStoreStage;
  for (auto [index, stage] : llvm::enumerate(stages))
    lastStoreStage[stage.store.getBase()] = index;

  OpBuilder blockBuilder(first.scope);
  auto fusedScope = blockBuilder.create<pto::SimdVecScopeOp>(first.scope.getLoc());
  fusedScope->setAttrs(first.scope->getAttrs());

  for (StageInfo &stage : llvm::drop_begin(stages))
    for (Operation *setupOp : stage.setupOps)
      setupOp->moveBefore(fusedScope);

  Block *scopeBody = new Block();
  fusedScope.getBody().push_back(scopeBody);
  OpBuilder scopeBuilder = OpBuilder::atBlockBegin(scopeBody);

  SmallVector<IRMapping, 8> stageMappings(stages.size());
  for (auto [index, stage] : llvm::enumerate(stages)) {
    for (Operation *op : stage.scopePreludeOps) {
      Operation *cloned = scopeBuilder.clone(*op, stageMappings[index]);
      for (auto [oldRes, newRes] :
           llvm::zip(op->getResults(), cloned->getResults()))
        stageMappings[index].map(oldRes, newRes);
    }
  }

  auto fusedLoop = scopeBuilder.create<scf::ForOp>(
      first.loop.getLoc(), first.loop.getLowerBound(), first.loop.getUpperBound(),
      first.loop.getStep());
  fusedLoop->setAttrs(first.loop->getAttrs());

  OpBuilder loopBuilder = OpBuilder::atBlockBegin(fusedLoop.getBody());
  SmallVector<ForwardedStore, 8> forwardedStores;

  for (auto [index, stage] : llvm::enumerate(stages)) {
    IRMapping &mapping = stageMappings[index];
    mapping.map(stage.loop.getInductionVar(), fusedLoop.getInductionVar());

    for (Operation *op : stage.loopOps) {
      if (auto load = dyn_cast<vector::MaskedLoadOp>(op)) {
        Value base = mapping.lookupOrDefault(load.getBase());
        SmallVector<Value, 4> indices = mapValues(load.getIndices(), mapping);
        Value mask = mapping.lookupOrDefault(load.getMask());
        Value passThru = mapping.lookupOrDefault(load.getPassThru());

        if (const ForwardedStore *forwarded =
                findForwardedStore(forwardedStores, base, indices, mask)) {
          mapping.map(load.getResult(), forwarded->value);
          continue;
        }

        auto cloned = loopBuilder.create<vector::MaskedLoadOp>(
            load.getLoc(), load.getResult().getType(), base, indices, mask,
            passThru);
        cloned->setAttrs(load->getAttrs());
        mapping.map(load.getResult(), cloned.getResult());
        continue;
      }

      if (auto store = dyn_cast<vector::MaskedStoreOp>(op)) {
        Value base = mapping.lookupOrDefault(store.getBase());
        SmallVector<Value, 4> indices = mapValues(store.getIndices(), mapping);
        Value mask = mapping.lookupOrDefault(store.getMask());
        Value value = mapping.lookupOrDefault(store.getValueToStore());

        if (lastStoreStage[stage.store.getBase()] == index) {
          auto cloned = loopBuilder.create<vector::MaskedStoreOp>(
              store.getLoc(), base, indices, mask, value);
          cloned->setAttrs(store->getAttrs());
        }

        forwardedStores.push_back(
            ForwardedStore{base, SmallVector<Value, 2>(indices.begin(),
                                                       indices.end()),
                           mask,
                           value});
        continue;
      }

      Operation *cloned = loopBuilder.clone(*op, mapping);
      for (auto [oldRes, newRes] :
           llvm::zip(op->getResults(), cloned->getResults()))
        mapping.map(oldRes, newRes);
    }
  }

  for (StageInfo &stage : llvm::reverse(stages))
    stage.scope.erase();

  return true;
}

static bool fuseStageRunsInBlock(Block &block) {
  bool changed = false;
  bool localChange = true;

  while (localChange) {
    localChange = false;
    for (Operation &op : block) {
      auto firstScope = dyn_cast<pto::SimdVecScopeOp>(op);
      if (!firstScope)
        continue;

      SmallVector<StageInfo, 8> stages = collectStageRunFrom(firstScope);
      if (!fuseStageRun(stages))
        continue;

      changed = true;
      localChange = true;
      break;
    }
  }

  return changed;
}

struct PTOLowLevelLoopFusionPass
    : public pto::impl::PTOLowLevelLoopFusionBase<PTOLowLevelLoopFusionPass> {
  using pto::impl::PTOLowLevelLoopFusionBase<
      PTOLowLevelLoopFusionPass>::PTOLowLevelLoopFusionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int fusedFuncs = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func.getSymName().starts_with("__pto_fused_group_"))
        continue;
      if (func.empty())
        continue;

      bool changed = false;
      for (Block &block : func.getBody())
        if (fuseStageRunsInBlock(block))
          changed = true;

      if (changed)
        ++fusedFuncs;
    }

    if (debug) {
      llvm::errs() << "[op-fusion] low-level loop fusion changed "
                   << fusedFuncs << " fused function(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOLowLevelLoopFusionPass(
    const PTOLowLevelLoopFusionOptions &options) {
  return std::make_unique<PTOLowLevelLoopFusionPass>(options);
}
