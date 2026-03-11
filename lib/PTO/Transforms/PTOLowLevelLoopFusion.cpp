#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <functional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWLEVELLOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static bool sameForHeader(scf::ForOp a, scf::ForOp b) {
  return a.getLowerBound() == b.getLowerBound() &&
         a.getUpperBound() == b.getUpperBound() && a.getStep() == b.getStep() &&
         a.getInitArgs().empty() && b.getInitArgs().empty();
}

static scf::ForOp getSingleInnerFor(scf::ForOp outer) {
  Block &body = outer.getRegion().front();
  scf::ForOp inner;
  for (Operation &op : body.without_terminator()) {
    if (isa<scf::ForOp>(op)) {
      if (inner)
        return {};
      inner = cast<scf::ForOp>(op);
      continue;
    }
    return {};
  }
  return inner;
}

static bool fuseAdjacentDoubleForNest(Block &block) {
  bool changed = false;
  bool localChange = true;

  while (localChange) {
    localChange = false;
    for (auto it = block.begin(), e = block.end(); it != e;) {
      auto first = dyn_cast<scf::ForOp>(*it);
      if (!first) {
        ++it;
        continue;
      }

      auto nextIt = std::next(it);
      if (nextIt == e)
        break;
      auto second = dyn_cast<scf::ForOp>(*nextIt);
      if (!second) {
        ++it;
        continue;
      }

      if (!sameForHeader(first, second)) {
        ++it;
        continue;
      }

      scf::ForOp firstInner = getSingleInnerFor(first);
      scf::ForOp secondInner = getSingleInnerFor(second);
      if (!firstInner || !secondInner) {
        ++it;
        continue;
      }
      if (!sameForHeader(firstInner, secondInner)) {
        ++it;
        continue;
      }

      // Clone second inner body ops into first inner body, remapping IVs.
      OpBuilder builder(firstInner.getBody()->getTerminator());
      IRMapping mapping;
      mapping.map(second.getInductionVar(), first.getInductionVar());
      mapping.map(secondInner.getInductionVar(), firstInner.getInductionVar());

      for (Operation &op : secondInner.getBody()->without_terminator()) {
        Operation *newOp = builder.clone(op, mapping);
        for (auto [oldRes, newRes] : llvm::zip(op.getResults(), newOp->getResults()))
          mapping.map(oldRes, newRes);
      }

      second.erase();
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

    int fusedNests = 0;
    std::function<bool(Region &)> processRegion = [&](Region &region) -> bool {
      bool changedAny = false;
      for (Block &block : region.getBlocks()) {
        if (fuseAdjacentDoubleForNest(block))
          changedAny = true;
        for (Operation &op : block.getOperations()) {
          for (Region &nested : op.getRegions()) {
            if (processRegion(nested))
              changedAny = true;
          }
        }
      }
      return changedAny;
    };

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func.getSymName().starts_with("__pto_fused_group_"))
        continue;
      if (func.empty())
        continue;

      bool changed = processRegion(func.getRegion());
      if (changed)
        ++fusedNests;
    }

    if (debug) {
      llvm::errs() << "[op-fusion] low-level loop fusion changed " << fusedNests
                   << " fused function(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOLowLevelLoopFusionPass(
    const PTOLowLevelLoopFusionOptions &options) {
  return std::make_unique<PTOLowLevelLoopFusionPass>(options);
}
