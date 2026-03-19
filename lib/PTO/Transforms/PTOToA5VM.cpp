//===- PTOToA5VM.cpp - PTO to A5VM pass wiring ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"
#include "PTO/Transforms/Passes.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOTOA5VM
#include "PTO/Transforms/Passes.h.inc"

namespace {

struct PTOToA5VMPass : public impl::PTOToA5VMBase<PTOToA5VMPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOToA5VMPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      if (isa<TLoadOp, TAbsOp, TStoreOp>(op))
        worklist.push_back(op);
    });

    PatternRewriter rewriter(&getContext());
    for (Operation *op : worklist) {
      if (!op->getBlock())
        continue;

      rewriter.setInsertionPoint(op);
      LogicalResult status = TypeSwitch<Operation *, LogicalResult>(op)
                                 .Case<TLoadOp>([&](TLoadOp loadOp) {
                                   return lowerTLOAD(loadOp, rewriter);
                                 })
                                 .Case<TAbsOp>([&](TAbsOp absOp) {
                                   return lowerTABS(absOp, rewriter);
                                 })
                                 .Case<TStoreOp>([&](TStoreOp storeOp) {
                                   return lowerTSTORE(storeOp, rewriter);
                                 })
                                 .Default([](Operation *) { return failure(); });
      if (succeeded(status))
        rewriter.eraseOp(op);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerPTOToA5VMPass() {
  return std::make_unique<PTOToA5VMPass>();
}

} // namespace pto
} // namespace mlir
