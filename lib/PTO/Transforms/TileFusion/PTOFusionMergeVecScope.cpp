#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONMERGEVECSCOPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct VecScopeStage {
  pto::VecScopeOp scope;
  SmallVector<Operation *, 4> setupOps;
};

static bool isMergeableInterScopeSetupOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static SmallVector<VecScopeStage, 8>
collectVecScopeRunFrom(pto::VecScopeOp firstScope) {
  SmallVector<VecScopeStage, 8> stages;
  stages.push_back(VecScopeStage{firstScope, {}});

  SmallVector<Operation *, 4> pendingSetup;
  for (Operation *op = firstScope->getNextNode(); op; op = op->getNextNode()) {
    if (auto nextScope = dyn_cast<pto::VecScopeOp>(op)) {
      stages.push_back(VecScopeStage{nextScope, pendingSetup});
      pendingSetup.clear();
      continue;
    }

    if (!isMergeableInterScopeSetupOp(op))
      break;
    pendingSetup.push_back(op);
  }

  return stages;
}

static void moveOpToBlockEnd(Operation *op, Block &block) {
  op->moveBefore(&block, block.end());
}

static bool mergeVecScopeRun(SmallVectorImpl<VecScopeStage> &stages) {
  if (stages.size() < 2)
    return false;

  if (stages.front().scope.getBody().empty())
    return false;
  Block &mergedBody = stages.front().scope.getBody().front();

  for (VecScopeStage &stage : llvm::drop_begin(stages)) {
    for (Operation *setupOp : stage.setupOps)
      moveOpToBlockEnd(setupOp, mergedBody);

    if (stage.scope.getBody().empty()) {
      stage.scope.erase();
      continue;
    }

    SmallVector<Operation *, 16> bodyOps;
    for (Operation &op : stage.scope.getBody().front())
      bodyOps.push_back(&op);
    for (Operation *op : bodyOps)
      moveOpToBlockEnd(op, mergedBody);

    stage.scope.erase();
  }

  return true;
}

static bool mergeFusionRegionVecScopes(pto::FusionRegionOp fusionRegion) {
  if (fusionRegion.getBody().empty())
    return false;

  Block &body = fusionRegion.getBody().front();
  bool changed = false;
  bool localChange = true;

  while (localChange) {
    localChange = false;
    for (Operation &op : body) {
      auto firstScope = dyn_cast<pto::VecScopeOp>(op);
      if (!firstScope)
        continue;

      SmallVector<VecScopeStage, 8> stages = collectVecScopeRunFrom(firstScope);
      if (!mergeVecScopeRun(stages))
        continue;

      changed = true;
      localChange = true;
      break;
    }
  }

  return changed;
}

struct PTOFusionMergeVecScopePass
    : public pto::impl::PTOFusionMergeVecScopeBase<
          PTOFusionMergeVecScopePass> {
  using pto::impl::PTOFusionMergeVecScopeBase<
      PTOFusionMergeVecScopePass>::PTOFusionMergeVecScopeBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    func.walk([&](pto::FusionRegionOp fusionRegion) {
      (void)mergeFusionRegionVecScopes(fusionRegion);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionMergeVecScopePass() {
  return std::make_unique<PTOFusionMergeVecScopePass>();
}
