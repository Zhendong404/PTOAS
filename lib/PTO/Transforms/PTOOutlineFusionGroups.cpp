#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <string>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOOUTLINEFUSIONGROUPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr = "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

struct GroupInfo {
  int64_t groupId = -1;
  SmallVector<func::CallOp, 8> calls;
};

struct GroupInterface {
  SmallVector<Value, 8> producedInOrder;
  DenseSet<Value> producedSet;
  SmallVector<Value, 8> externalInputs;
  SmallVector<Value, 8> callArgs;
};

static bool hasFusionGroupAttrs(Operation *op) {
  return op->hasAttr(kFusionGroupIdAttr) && op->hasAttr(kFusionOrderAttr);
}

static bool valueInList(ArrayRef<Value> vals, Value v) {
  return llvm::is_contained(vals, v);
}

static void appendUniqueValue(SmallVectorImpl<Value> &vals, Value v) {
  if (!valueInList(vals, v))
    vals.push_back(v);
}

static FailureOr<Value> getCallSrc0(func::CallOp call) {
  if (call.getNumOperands() < 3)
    return failure();
  return call.getOperand(0);
}

static FailureOr<Value> getCallSrc1(func::CallOp call) {
  if (call.getNumOperands() < 3)
    return failure();
  return call.getOperand(1);
}

static FailureOr<Value> getCallDst(func::CallOp call) {
  if (call.getNumOperands() < 3)
    return failure();
  return call.getOperand(2);
}

static GroupInterface buildGroupInterface(GroupInfo &group) {
  GroupInterface iface;

  for (func::CallOp call : group.calls) {
    FailureOr<Value> dstOr = getCallDst(call);
    if (failed(dstOr))
      continue;
    appendUniqueValue(iface.producedInOrder, *dstOr);
    iface.producedSet.insert(*dstOr);
  }

  for (func::CallOp call : group.calls) {
    FailureOr<Value> src0Or = getCallSrc0(call);
    FailureOr<Value> src1Or = getCallSrc1(call);
    if (failed(src0Or) || failed(src1Or))
      continue;

    for (Value src : {*src0Or, *src1Or}) {
      if (!iface.producedSet.contains(src))
        appendUniqueValue(iface.externalInputs, src);
    }
  }

  for (Value v : iface.externalInputs)
    appendUniqueValue(iface.callArgs, v);
  for (Value v : iface.producedInOrder)
    appendUniqueValue(iface.callArgs, v);

  return iface;
}

static std::string makeUniqueFusedName(SymbolTable &symbolTable, int64_t groupId,
                                       int &fusedCounter) {
  std::string fusedName = "__pto_fused_group_" + std::to_string(groupId) +
                          "_" + std::to_string(fusedCounter++);
  while (symbolTable.lookup(fusedName))
    fusedName += "_x";
  return fusedName;
}

static func::FuncOp createFusedFunc(ModuleOp module, Location loc,
                                    StringRef name, ArrayRef<Value> callArgs,
                                    int64_t groupId) {
  SmallVector<Type, 8> argTypes;
  for (Value v : callArgs)
    argTypes.push_back(v.getType());

  OpBuilder moduleBuilder(module.getContext());
  moduleBuilder.setInsertionPointToStart(module.getBody());
  auto fusedFunc = moduleBuilder.create<func::FuncOp>(
      loc, name, FunctionType::get(module.getContext(), argTypes, {}));
  fusedFunc.setPrivate();
  fusedFunc->setAttr(kFusionGroupIdAttr,
                     IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                      groupId));
  return fusedFunc;
}

static DenseMap<Value, Value> mapCallArgsToEntry(Block *entry,
                                                 ArrayRef<Value> callArgs) {
  DenseMap<Value, Value> valueMap;
  for (auto [orig, arg] : llvm::zip(callArgs, entry->getArguments()))
    valueMap[orig] = arg;
  return valueMap;
}

static FailureOr<Value> mapOrFail(DenseMap<Value, Value> &valueMap, Value key) {
  auto it = valueMap.find(key);
  if (it == valueMap.end())
    return failure();
  return it->second;
}

struct PTOOutlineFusionGroupsPass
    : public pto::impl::PTOOutlineFusionGroupsBase<PTOOutlineFusionGroupsPass> {
  using pto::impl::PTOOutlineFusionGroupsBase<
      PTOOutlineFusionGroupsPass>::PTOOutlineFusionGroupsBase;

  void collectGroupsInRegion(Region &region,
                             SmallVectorImpl<GroupInfo> &groups) {
    for (Block &block : region.getBlocks()) {
      GroupInfo current;
      int64_t expectedOrder = 0;

      auto flush = [&]() {
        if (current.calls.size() >= 2)
          groups.push_back(current);
        current = GroupInfo();
        expectedOrder = 0;
      };

      for (Operation &op : block.getOperations()) {
        auto call = dyn_cast<func::CallOp>(op);
        if (call && hasFusionGroupAttrs(call.getOperation())) {
          auto gidAttr = call->getAttrOfType<IntegerAttr>(kFusionGroupIdAttr);
          auto orderAttr = call->getAttrOfType<IntegerAttr>(kFusionOrderAttr);

          if (!gidAttr || !orderAttr) {
            flush();
          } else {
            int64_t gid = gidAttr.getInt();
            int64_t order = orderAttr.getInt();
            if (current.calls.empty()) {
              current.groupId = gid;
              current.calls.push_back(call);
              expectedOrder = order + 1;
            } else if (current.groupId == gid && order == expectedOrder) {
              current.calls.push_back(call);
              ++expectedOrder;
            } else {
              flush();
              current.groupId = gid;
              current.calls.push_back(call);
              expectedOrder = order + 1;
            }
          }
        } else {
          flush();
        }

        for (Region &nested : op.getRegions())
          collectGroupsInRegion(nested, groups);
      }

      flush();
    }
  }

  LogicalResult outlineOneGroup(ModuleOp module, SymbolTable &symbolTable,
                                GroupInfo &group, int &fusedCounter) {
    if (group.calls.empty())
      return success();

    func::CallOp firstCall = group.calls.front();
    Location loc = firstCall.getLoc();

    for (func::CallOp call : group.calls) {
      if (call.getNumResults() != 0 || call.getNumOperands() != 3) {
        firstCall.emitWarning()
            << "fusion-group fallback: only 3-input/void OP-Lib calls are "
               "supported for outlining";
        return success();
      }

      if (!call.getCalleeAttr()) {
        firstCall.emitWarning()
            << "fusion-group fallback: call without callee attr, keep original "
               "group";
        return success();
      }
    }

    GroupInterface iface = buildGroupInterface(group);
    if (iface.callArgs.empty()) {
      firstCall.emitWarning()
          << "fusion-group fallback: group has no external interface values, "
             "keep original ops";
      return success();
    }

    std::string fusedName =
        makeUniqueFusedName(symbolTable, group.groupId, fusedCounter);
    auto fusedFunc =
        createFusedFunc(module, loc, fusedName, iface.callArgs, group.groupId);

    Block *entry = fusedFunc.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(entry);
    DenseMap<Value, Value> valueMap = mapCallArgsToEntry(entry, iface.callArgs);

    for (func::CallOp call : group.calls) {
      FailureOr<Value> src0Or = mapOrFail(valueMap, call.getOperand(0));
      FailureOr<Value> src1Or = mapOrFail(valueMap, call.getOperand(1));
      FailureOr<Value> dstOr = mapOrFail(valueMap, call.getOperand(2));
      if (failed(src0Or) || failed(src1Or) || failed(dstOr)) {
        firstCall.emitWarning()
            << "fusion-group fallback: failed to map group call operands, keep "
               "original group";
        fusedFunc.erase();
        return success();
      }

      func::FuncOp callee =
          module.lookupSymbol<func::FuncOp>(call.getCalleeAttr().getValue());
      if (!callee) {
        firstCall.emitWarning()
            << "fusion-group fallback: missing OP-Lib callee symbol, keep "
               "original group";
        fusedFunc.erase();
        return success();
      }

      bodyBuilder.create<func::CallOp>(loc, callee,
                                       ValueRange{*src0Or, *src1Or, *dstOr});
    }

    bodyBuilder.create<func::ReturnOp>(loc);

    OpBuilder callerBuilder(firstCall);
    callerBuilder.create<func::CallOp>(loc, fusedFunc, iface.callArgs);

    for (func::CallOp call : llvm::reverse(group.calls))
      call.erase();

    if (debug) {
      llvm::errs() << "[op-fusion] materialized group_id=" << group.groupId
                   << " into @" << fusedFunc.getSymName() << "\n";
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);

    int fusedCounter = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func.getSymName().starts_with("__pto_oplib_"))
        continue;
      if (func.getSymName().starts_with("__pto_fused_group_"))
        continue;

      SmallVector<GroupInfo, 8> groups;
      collectGroupsInRegion(func.getRegion(), groups);
      if (debug && !groups.empty()) {
        llvm::errs() << "[op-fusion] found " << groups.size() << " group(s) in @"
                     << func.getSymName() << "\n";
      }

      for (GroupInfo &group : groups) {
        if (failed(outlineOneGroup(module, symbolTable, group, fusedCounter))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOOutlineFusionGroupsPass(
    const PTOOutlineFusionGroupsOptions &options) {
  return std::make_unique<PTOOutlineFusionGroupsPass>(options);
}
