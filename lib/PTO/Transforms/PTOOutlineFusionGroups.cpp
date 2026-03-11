#include "PTO/IR/PTO.h"
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
  SmallVector<Operation *, 8> ops;
};

struct BinaryOpInterface {
  Value src0;
  Value src1;
  Value dst;
};

struct GroupInterface {
  SmallVector<Value, 8> producedInOrder;
  DenseSet<Value> producedSet;
  SmallVector<Value, 8> externalInputs;
  SmallVector<Value, 8> callArgs;
};

static bool isSupportedFusionOp(Operation *op) {
  return isa<pto::TMulOp, pto::TDivOp, pto::TAddOp, pto::TSubOp, pto::TMaxOp,
             pto::TMinOp>(op);
}

static FailureOr<BinaryOpInterface> getBinaryOpInterface(Operation *op) {
  if (auto mul = dyn_cast<pto::TMulOp>(op))
    return BinaryOpInterface{mul.getSrc0(), mul.getSrc1(), mul.getDst()};
  if (auto div = dyn_cast<pto::TDivOp>(op))
    return BinaryOpInterface{div.getSrc0(), div.getSrc1(), div.getDst()};
  if (auto add = dyn_cast<pto::TAddOp>(op))
    return BinaryOpInterface{add.getSrc0(), add.getSrc1(), add.getDst()};
  if (auto sub = dyn_cast<pto::TSubOp>(op))
    return BinaryOpInterface{sub.getSrc0(), sub.getSrc1(), sub.getDst()};
  if (auto max = dyn_cast<pto::TMaxOp>(op))
    return BinaryOpInterface{max.getSrc0(), max.getSrc1(), max.getDst()};
  if (auto min = dyn_cast<pto::TMinOp>(op))
    return BinaryOpInterface{min.getSrc0(), min.getSrc1(), min.getDst()};
  return failure();
}

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

static GroupInterface buildGroupInterface(GroupInfo &group) {
  GroupInterface iface;

  for (Operation *op : group.ops) {
    FailureOr<BinaryOpInterface> opIfaceOr = getBinaryOpInterface(op);
    if (failed(opIfaceOr))
      continue;
    appendUniqueValue(iface.producedInOrder, opIfaceOr->dst);
    iface.producedSet.insert(opIfaceOr->dst);
  }

  for (Operation *op : group.ops) {
    FailureOr<BinaryOpInterface> opIfaceOr = getBinaryOpInterface(op);
    if (failed(opIfaceOr))
      continue;

    for (Value src : {opIfaceOr->src0, opIfaceOr->src1}) {
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

static LogicalResult createClonedBinaryOp(Operation *op, OpBuilder &builder,
                                          Location loc, Value src0, Value src1,
                                          Value dst) {
  if (isa<pto::TMulOp>(op)) {
    builder.create<pto::TMulOp>(loc, TypeRange{}, src0, src1, dst);
    return success();
  }
  if (isa<pto::TDivOp>(op)) {
    builder.create<pto::TDivOp>(loc, TypeRange{}, src0, src1, dst);
    return success();
  }
  if (isa<pto::TAddOp>(op)) {
    builder.create<pto::TAddOp>(loc, TypeRange{}, src0, src1, dst);
    return success();
  }
  if (isa<pto::TSubOp>(op)) {
    builder.create<pto::TSubOp>(loc, TypeRange{}, src0, src1, dst);
    return success();
  }
  if (isa<pto::TMaxOp>(op)) {
    builder.create<pto::TMaxOp>(loc, TypeRange{}, src0, src1, dst);
    return success();
  }
  if (isa<pto::TMinOp>(op)) {
    builder.create<pto::TMinOp>(loc, TypeRange{}, src0, src1, dst);
    return success();
  }
  return failure();
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
        if (current.ops.size() >= 2)
          groups.push_back(current);
        current = GroupInfo();
        expectedOrder = 0;
      };

      for (Operation &op : block.getOperations()) {
        Operation *curOp = &op;

        if (isSupportedFusionOp(curOp) && hasFusionGroupAttrs(curOp)) {
          auto gidAttr = curOp->getAttrOfType<IntegerAttr>(kFusionGroupIdAttr);
          auto orderAttr = curOp->getAttrOfType<IntegerAttr>(kFusionOrderAttr);

          if (!gidAttr || !orderAttr) {
            flush();
          } else {
            int64_t gid = gidAttr.getInt();
            int64_t order = orderAttr.getInt();
            if (current.ops.empty()) {
              current.groupId = gid;
              current.ops.push_back(curOp);
              expectedOrder = order + 1;
            } else if (current.groupId == gid && order == expectedOrder) {
              current.ops.push_back(curOp);
              ++expectedOrder;
            } else {
              flush();
              current.groupId = gid;
              current.ops.push_back(curOp);
              expectedOrder = order + 1;
            }
          }
        } else {
          flush();
        }

        for (Region &nested : curOp->getRegions())
          collectGroupsInRegion(nested, groups);
      }

      flush();
    }
  }

  LogicalResult outlineOneGroup(ModuleOp module, SymbolTable &symbolTable,
                                GroupInfo &group, int &fusedCounter) {
    if (group.ops.empty())
      return success();

    Operation *firstOp = group.ops.front();
    Location loc = firstOp->getLoc();

    for (Operation *op : group.ops) {
      FailureOr<BinaryOpInterface> opIfaceOr = getBinaryOpInterface(op);
      if (failed(opIfaceOr)) {
        firstOp->emitWarning()
            << "fusion-group fallback: unsupported grouped PTO op for outlining";
        return success();
      }
    }

    GroupInterface iface = buildGroupInterface(group);
    if (iface.callArgs.empty()) {
      firstOp->emitWarning()
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

    for (Operation *op : group.ops) {
      FailureOr<BinaryOpInterface> opIfaceOr = getBinaryOpInterface(op);
      if (failed(opIfaceOr)) {
        fusedFunc.erase();
        return success();
      }

      FailureOr<Value> src0Or = mapOrFail(valueMap, opIfaceOr->src0);
      FailureOr<Value> src1Or = mapOrFail(valueMap, opIfaceOr->src1);
      FailureOr<Value> dstOr = mapOrFail(valueMap, opIfaceOr->dst);
      if (failed(src0Or) || failed(src1Or) || failed(dstOr)) {
        firstOp->emitWarning()
            << "fusion-group fallback: failed to map group op operands, keep "
               "original group";
        fusedFunc.erase();
        return success();
      }

      if (failed(createClonedBinaryOp(op, bodyBuilder, loc, *src0Or, *src1Or,
                                      *dstOr))) {
        firstOp->emitWarning()
            << "fusion-group fallback: failed to clone group op into fused helper";
        fusedFunc.erase();
        return success();
      }
    }

    bodyBuilder.create<func::ReturnOp>(loc);

    OpBuilder callerBuilder(firstOp);
    callerBuilder.create<func::CallOp>(loc, fusedFunc, iface.callArgs);

    for (Operation *op : llvm::reverse(group.ops))
      op->erase();

    if (debug) {
      llvm::errs() << "[op-fusion] outlined group_id=" << group.groupId
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
