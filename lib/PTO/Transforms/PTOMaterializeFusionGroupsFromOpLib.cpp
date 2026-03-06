#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#include <algorithm>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMATERIALIZEFUSIONGROUPSFROMOPLIB
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr = "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

static constexpr llvm::StringLiteral kOpLibAttrKind = "pto.oplib.kind";
static constexpr llvm::StringLiteral kOpLibAttrEntryRole = "pto.oplib.entry_role";
static constexpr llvm::StringLiteral kOpLibAttrOp = "pto.oplib.op";
static constexpr llvm::StringLiteral kOpLibAttrVariantId = "pto.oplib.variant_id";
static constexpr llvm::StringLiteral kOpLibAttrMatchDType = "pto.oplib.match.dtype";
static constexpr llvm::StringLiteral kOpLibAttrMatchRows = "pto.oplib.match.rows";
static constexpr llvm::StringLiteral kOpLibAttrMatchCols = "pto.oplib.match.cols";
static constexpr llvm::StringLiteral kOpLibAttrMatchBLayout =
    "pto.oplib.match.blayout";
static constexpr llvm::StringLiteral kOpLibAttrMatchSLayout =
    "pto.oplib.match.slayout";
static constexpr llvm::StringLiteral kOpLibAttrMatchFractal =
    "pto.oplib.match.fractal";
static constexpr llvm::StringLiteral kOpLibAttrCost = "pto.oplib.cost";
static constexpr llvm::StringLiteral kOpLibAttrPriority = "pto.oplib.priority";
static constexpr llvm::StringLiteral kOpLibAttrSync = "pto.oplib.sync";

static constexpr llvm::StringLiteral kOpLibAttrSeedId = "pto.oplib.seed_id";
static constexpr llvm::StringLiteral kOpLibAttrSeedDType = "pto.oplib.seed_dtype";
static constexpr llvm::StringLiteral kOpLibAttrSeedSupportDTypes =
    "pto.oplib.seed.support_dtypes";
static constexpr llvm::StringLiteral kOpLibAttrSeedSupportOps =
    "pto.oplib.seed.support_ops";
static constexpr llvm::StringLiteral kOpLibAttrSeedCoreSlot =
    "pto.oplib.seed.core_slot";

static constexpr llvm::StringLiteral kOpLibAttrInstVariantId =
    "pto.oplib.instance.variant_id";
static constexpr llvm::StringLiteral kOpLibAttrInstOp = "pto.oplib.instance.op";
static constexpr llvm::StringLiteral kOpLibAttrInstDType =
    "pto.oplib.instance.dtype";
static constexpr llvm::StringLiteral kOpLibAttrInstSource =
    "pto.oplib.instance.source";
static constexpr llvm::StringLiteral kOpLibAttrInstFromSeed =
    "pto.oplib.instance.from_seed";
static constexpr llvm::StringLiteral kOpLibAttrInstSeedId =
    "pto.oplib.instance.seed_id";
static constexpr llvm::StringLiteral kOpLibAttrInstCoreSlot =
    "pto.oplib.instance.core_slot";

static constexpr llvm::StringLiteral kOpLibKindL3BinaryTemplate =
    "l3_binary_elementwise_template";

static constexpr llvm::StringLiteral kEntryRoleVariant = "variant";
static constexpr llvm::StringLiteral kEntryRoleSeed = "seed";

static constexpr llvm::StringLiteral kDefaultAny = "any";
static constexpr llvm::StringLiteral kDefaultCoreSlot = "binary_ewise_core";

struct GroupInfo {
  int64_t groupId = -1;
  SmallVector<Operation *, 8> ops;
};

struct GroupInterface {
  DenseSet<Operation *> groupSet;
  SmallVector<Value, 8> producedInOrder;
  DenseSet<Value> producedSet;
  SmallVector<Value, 8> externalInputs;
  SmallVector<Value, 8> callArgs;
};

enum class EntryRole {
  Variant,
  Seed,
};

struct MatchKey {
  int64_t rows = -1;
  int64_t cols = -1;
  std::string blayout = kDefaultAny.str();
  std::string slayout = kDefaultAny.str();
  int64_t fractal = -1;
};

struct TemplateEntry {
  func::FuncOp symbol;
  EntryRole role = EntryRole::Variant;
  std::string op;
  std::string variantId;
  std::string matchDType;

  std::string seedId;
  std::string seedDType;
  SmallVector<std::string, 8> supportDTypes;
  SmallVector<std::string, 8> supportOps;
  std::string coreSlot = kDefaultCoreSlot.str();

  MatchKey match;
  int64_t cost = 0;
  int64_t priority = 0;
  bool sync = false;
};

struct SelectedVariant {
  TemplateEntry *entry = nullptr;
  std::string variantId;
  std::string op;
  std::string dtype;
  bool fromSeed = false;
  std::string seedId;
  std::string coreSlot;
  int64_t cost = 0;
  int64_t priority = 0;
};

struct PlannedOpLowering {
  Operation *op = nullptr;
  SelectedVariant selected;
  func::FuncOp instance;
};

struct BinaryOpInterface {
  Value src0;
  Value src1;
  Value dst;
  StringRef opName;
};

static bool isSupportedFusionOp(Operation *op) {
  return isa<pto::TMulOp, pto::TDivOp, pto::TAddOp, pto::TSubOp, pto::TMaxOp,
             pto::TMinOp>(op);
}

static BinaryOpInterface getBinaryOpInterface(Operation *op) {
  if (auto mul = dyn_cast<pto::TMulOp>(op))
    return {mul.getSrc0(), mul.getSrc1(), mul.getDst(), "tmul"};
  if (auto div = dyn_cast<pto::TDivOp>(op))
    return {div.getSrc0(), div.getSrc1(), div.getDst(), "tdiv"};
  if (auto add = dyn_cast<pto::TAddOp>(op))
    return {add.getSrc0(), add.getSrc1(), add.getDst(), "tadd"};
  if (auto sub = dyn_cast<pto::TSubOp>(op))
    return {sub.getSrc0(), sub.getSrc1(), sub.getDst(), "tsub"};
  if (auto max = dyn_cast<pto::TMaxOp>(op))
    return {max.getSrc0(), max.getSrc1(), max.getDst(), "tmax"};
  if (auto min = dyn_cast<pto::TMinOp>(op))
    return {min.getSrc0(), min.getSrc1(), min.getDst(), "tmin"};
  return {{}, {}, {}, ""};
}

static bool isSupportedOpName(StringRef opName) {
  return opName == "tmul" || opName == "tdiv" || opName == "tadd" ||
         opName == "tsub" || opName == "tmax" || opName == "tmin";
}

static bool hasFusionGroupAttrs(Operation *op) {
  return op->hasAttr(kFusionGroupIdAttr) && op->hasAttr(kFusionOrderAttr);
}

static bool isFloatDTypeSupported(Type ty) { return ty.isF16() || ty.isF32(); }

static std::string dtypeToString(Type ty) {
  if (ty.isF16())
    return "f16";
  if (ty.isF32())
    return "f32";
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  return os.str();
}

static std::string memRefTypeToString(MemRefType ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  return os.str();
}

static std::string sanitizeSymbolComponent(StringRef text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty())
    out = "unnamed";
  return out;
}

static Value castToMemRefTypeIfNeeded(OpBuilder &builder, Location loc, Value v,
                                      MemRefType dstTy) {
  if (v.getType() == dstTy)
    return v;

  auto srcTy = dyn_cast<MemRefType>(v.getType());
  if (!srcTy)
    return {};
  if (srcTy.getRank() != dstTy.getRank())
    return {};
  if (srcTy.getElementType() != dstTy.getElementType())
    return {};
  if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
    return {};

  return builder.create<memref::CastOp>(loc, dstTy, v);
}

static bool valueInList(ArrayRef<Value> vals, Value v) {
  return llvm::is_contained(vals, v);
}

static void appendUniqueValue(SmallVectorImpl<Value> &vals, Value v) {
  if (!valueInList(vals, v))
    vals.push_back(v);
}

static void eraseDeadValueDefChain(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || !def->use_empty())
    return;

  if (!isa<memref::AllocOp, pto::BindTileOp, memref::CastOp,
           memref::ReinterpretCastOp, memref::SubViewOp>(def))
    return;

  SmallVector<Value, 4> operands(def->getOperands().begin(),
                                 def->getOperands().end());
  def->erase();
  for (Value operand : operands)
    eraseDeadValueDefChain(operand);
}

static GroupInterface buildGroupInterface(GroupInfo &group) {
  GroupInterface iface;
  iface.groupSet.insert(group.ops.begin(), group.ops.end());

  for (Operation *op : group.ops) {
    Value dst = getBinaryOpInterface(op).dst;
    appendUniqueValue(iface.producedInOrder, dst);
    iface.producedSet.insert(dst);
  }

  for (Operation *op : group.ops) {
    BinaryOpInterface ifaceOp = getBinaryOpInterface(op);
    for (Value src : {ifaceOp.src0, ifaceOp.src1}) {
      if (!iface.producedSet.contains(src))
        appendUniqueValue(iface.externalInputs, src);
    }
  }

  for (Value dst : iface.producedInOrder) {
    bool usedOutside = false;
    for (Operation *user : dst.getUsers()) {
      if (!iface.groupSet.contains(user)) {
        usedOutside = true;
        break;
      }
    }
    (void)usedOutside;
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

static bool isAllowedDTypeName(StringRef dtypeName) {
  return dtypeName == "f16" || dtypeName == "f32";
}

static bool isAllowedLayoutName(StringRef layoutName) {
  return layoutName == "row_major" || layoutName == "col_major" ||
         layoutName == "none_box" || layoutName == "any";
}

static bool containsString(ArrayRef<std::string> values, StringRef key) {
  return llvm::any_of(values, [&](const std::string &s) { return s == key; });
}

struct TemplateRegistry {
  ModuleOp module;
  SymbolTable &symbolTable;
  bool debug;

  SmallVector<std::unique_ptr<TemplateEntry>, 32> entries;
  llvm::StringMap<func::FuncOp> instanceByKey;
  llvm::StringSet<> seedIds;
  llvm::StringSet<> variantIds;

  LogicalResult emitFailure(Operation *anchor, const Twine &msg) {
    anchor->emitError(msg);
    return failure();
  }

  LogicalResult emitFailure(Location loc, const Twine &msg) {
    emitError(loc) << msg;
    return failure();
  }

  static FailureOr<int64_t> parseI64Attr(Operation *op, StringRef attrName,
                                         bool allowWildcard) {
    auto attr = op->getAttrOfType<IntegerAttr>(attrName);
    if (!attr)
      return failure();
    int64_t value = attr.getInt();
    if (value == -1) {
      if (!allowWildcard)
        return failure();
      return value;
    }
    if (value <= 0)
      return failure();
    return value;
  }

  static FailureOr<std::string> parseStringAttr(Operation *op,
                                                StringRef attrName) {
    auto attr = op->getAttrOfType<StringAttr>(attrName);
    if (!attr)
      return failure();
    return attr.getValue().str();
  }

  static FailureOr<SmallVector<std::string, 8>>
  parseStringArrayAttr(Operation *op, StringRef attrName) {
    auto attr = op->getAttrOfType<ArrayAttr>(attrName);
    if (!attr)
      return failure();

    SmallVector<std::string, 8> out;
    out.reserve(attr.size());
    for (Attribute item : attr) {
      auto s = dyn_cast<StringAttr>(item);
      if (!s)
        return failure();
      out.push_back(s.getValue().str());
    }
    return out;
  }

  bool validateTemplateSignature(func::FuncOp fn) {
    FunctionType fnTy = fn.getFunctionType();
    if (fnTy.getNumInputs() != 3 || fnTy.getNumResults() != 0)
      return false;
    for (Type inTy : fnTy.getInputs()) {
      auto memTy = dyn_cast<MemRefType>(inTy);
      if (!memTy)
        return false;
      if (memTy.getRank() != 2)
        return false;
      if (!isFloatDTypeSupported(memTy.getElementType()))
        return false;
    }
    return true;
  }

  FailureOr<func::FuncOp> cloneIntoModule(func::FuncOp libFunc) {
    OpBuilder modBuilder(module.getContext());
    modBuilder.setInsertionPointToStart(module.getBody());

    std::string sym =
        "__pto_oplib_entry_" + sanitizeSymbolComponent(libFunc.getSymName());
    int suffix = 0;
    while (symbolTable.lookup(sym)) {
      sym = "__pto_oplib_entry_" +
            sanitizeSymbolComponent(libFunc.getSymName()) + "_" +
            std::to_string(++suffix);
    }

    auto imported = cast<func::FuncOp>(modBuilder.clone(*libFunc.getOperation()));
    imported.setSymName(sym);
    imported.setPrivate();
    return imported;
  }

  LogicalResult parseCommonAttrs(func::FuncOp imported,
                                 std::unique_ptr<TemplateEntry> &entry) {
    Operation *op = imported.getOperation();

    auto rowsOr = parseI64Attr(op, kOpLibAttrMatchRows, /*allowWildcard=*/true);
    auto colsOr = parseI64Attr(op, kOpLibAttrMatchCols, /*allowWildcard=*/true);
    auto fractalOr =
        parseI64Attr(op, kOpLibAttrMatchFractal, /*allowWildcard=*/true);
    if (failed(rowsOr) || failed(colsOr) || failed(fractalOr)) {
      imported.emitError("missing or invalid match rows/cols/fractal attrs");
      return failure();
    }

    auto blayoutOr = parseStringAttr(op, kOpLibAttrMatchBLayout);
    auto slayoutOr = parseStringAttr(op, kOpLibAttrMatchSLayout);
    if (failed(blayoutOr) || failed(slayoutOr)) {
      imported.emitError("missing match layout attrs: pto.oplib.match.blayout / "
                         "pto.oplib.match.slayout");
      return failure();
    }
    if (!isAllowedLayoutName(*blayoutOr) || !isAllowedLayoutName(*slayoutOr)) {
      imported.emitError("invalid layout value in match.blayout/match.slayout");
      return failure();
    }

    auto costAttr = op->getAttrOfType<IntegerAttr>(kOpLibAttrCost);
    auto priorityAttr = op->getAttrOfType<IntegerAttr>(kOpLibAttrPriority);
    if (!costAttr || !priorityAttr) {
      imported.emitError("missing required attrs: pto.oplib.cost / "
                         "pto.oplib.priority");
      return failure();
    }

    entry->match.rows = *rowsOr;
    entry->match.cols = *colsOr;
    entry->match.fractal = *fractalOr;
    entry->match.blayout = *blayoutOr;
    entry->match.slayout = *slayoutOr;
    entry->cost = costAttr.getInt();
    entry->priority = priorityAttr.getInt();

    if (auto syncAttr = op->getAttrOfType<BoolAttr>(kOpLibAttrSync))
      entry->sync = syncAttr.getValue();

    return success();
  }

  LogicalResult parseVariantAttrs(func::FuncOp imported,
                                  std::unique_ptr<TemplateEntry> &entry) {
    Operation *op = imported.getOperation();

    auto opOr = parseStringAttr(op, kOpLibAttrOp);
    auto variantIdOr = parseStringAttr(op, kOpLibAttrVariantId);
    auto dtypeOr = parseStringAttr(op, kOpLibAttrMatchDType);
    if (failed(opOr) || failed(variantIdOr) || failed(dtypeOr)) {
      imported.emitError("variant entry missing attrs: pto.oplib.op / "
                         "pto.oplib.variant_id / pto.oplib.match.dtype");
      return failure();
    }

    if (!isSupportedOpName(*opOr)) {
      imported.emitError() << "unsupported variant op: " << *opOr;
      return failure();
    }
    if (!isAllowedDTypeName(*dtypeOr)) {
      imported.emitError() << "unsupported variant dtype: " << *dtypeOr;
      return failure();
    }

    std::string uniqueKey = *opOr + "|" + *variantIdOr;
    if (!variantIds.insert(uniqueKey).second) {
      imported.emitError() << "duplicate variant_id under op=" << *opOr
                           << ": " << *variantIdOr;
      return failure();
    }

    entry->role = EntryRole::Variant;
    entry->op = *opOr;
    entry->variantId = *variantIdOr;
    entry->matchDType = *dtypeOr;

    return success();
  }

  LogicalResult parseSeedAttrs(func::FuncOp imported,
                               std::unique_ptr<TemplateEntry> &entry) {
    Operation *op = imported.getOperation();

    auto seedIdOr = parseStringAttr(op, kOpLibAttrSeedId);
    auto seedDTypeOr = parseStringAttr(op, kOpLibAttrSeedDType);
    auto supportDTypesOr = parseStringArrayAttr(op, kOpLibAttrSeedSupportDTypes);
    auto supportOpsOr = parseStringArrayAttr(op, kOpLibAttrSeedSupportOps);
    if (failed(seedIdOr) || failed(seedDTypeOr) || failed(supportDTypesOr) ||
        failed(supportOpsOr)) {
      imported.emitError("seed entry missing attrs: pto.oplib.seed_id / "
                         "pto.oplib.seed_dtype / pto.oplib.seed.support_dtypes / "
                         "pto.oplib.seed.support_ops");
      return failure();
    }

    if (!seedIds.insert(*seedIdOr).second) {
      imported.emitError() << "duplicate seed_id: " << *seedIdOr;
      return failure();
    }

    if (!isAllowedDTypeName(*seedDTypeOr)) {
      imported.emitError() << "unsupported seed_dtype: " << *seedDTypeOr;
      return failure();
    }

    for (const std::string &dtype : *supportDTypesOr) {
      if (!isAllowedDTypeName(dtype)) {
        imported.emitError() << "unsupported support_dtypes item: " << dtype;
        return failure();
      }
    }

    for (const std::string &opName : *supportOpsOr) {
      if (!isSupportedOpName(opName)) {
        imported.emitError() << "unsupported support_ops item: " << opName;
        return failure();
      }
    }

    entry->role = EntryRole::Seed;
    entry->seedId = *seedIdOr;
    entry->seedDType = *seedDTypeOr;
    entry->supportDTypes = *supportDTypesOr;
    entry->supportOps = *supportOpsOr;

    if (auto slotAttr = op->getAttrOfType<StringAttr>(kOpLibAttrSeedCoreSlot))
      entry->coreSlot = slotAttr.getValue().str();

    return success();
  }

  LogicalResult loadFromDir(StringRef opLibDir, MLIRContext *ctx, Location loc) {
    if (opLibDir.empty()) {
      return emitFailure(loc,
                         "--op-lib-dir is required when OP-LIB lowering is enabled");
    }

    llvm::SmallVector<std::string, 16> mlirFiles;
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(opLibDir, ec), end;
         it != end && !ec; it.increment(ec)) {
      if (!llvm::sys::fs::is_regular_file(it->path()))
        continue;
      if (llvm::sys::path::extension(it->path()) == ".mlir")
        mlirFiles.push_back(it->path());
    }
    if (ec) {
      return emitFailure(loc, Twine("failed to iterate --op-lib-dir '") + opLibDir +
                                  "': " + ec.message());
    }

    llvm::sort(mlirFiles);
    if (mlirFiles.empty()) {
      return emitFailure(loc, Twine("no .mlir files found in --op-lib-dir '") +
                                  opLibDir + "'");
    }

    int importedCount = 0;
    for (const std::string &path : mlirFiles) {
      auto fileOrErr = llvm::MemoryBuffer::getFile(path);
      if (!fileOrErr) {
        return emitFailure(loc, Twine("failed to read OP-Lib file '") + path +
                                    "': " + fileOrErr.getError().message());
      }

      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
      OwningOpRef<ModuleOp> libModule = parseSourceFile<ModuleOp>(sourceMgr, ctx);
      if (!libModule)
        return emitFailure(loc, Twine("failed to parse OP-Lib file '") + path + "'");

      for (func::FuncOp libFunc : libModule->getOps<func::FuncOp>()) {
        auto kindAttr = libFunc->getAttrOfType<StringAttr>(kOpLibAttrKind);
        if (!kindAttr || kindAttr.getValue() != kOpLibKindL3BinaryTemplate)
          continue;

        if (!validateTemplateSignature(libFunc)) {
          libFunc.emitError("invalid OP-Lib signature; expected "
                            "(memref<*x*xf16/f32>, memref<*x*xf16/f32>, "
                            "memref<*x*xf16/f32>) -> () with rank-2 memref args");
          return failure();
        }

        auto roleAttr = libFunc->getAttrOfType<StringAttr>(kOpLibAttrEntryRole);
        if (!roleAttr) {
          libFunc.emitError("missing required attr: pto.oplib.entry_role");
          return failure();
        }

        FailureOr<func::FuncOp> importedOr = cloneIntoModule(libFunc);
        if (failed(importedOr)) {
          libFunc.emitError("failed to import template function into module");
          return failure();
        }
        func::FuncOp imported = *importedOr;

        auto entry = std::make_unique<TemplateEntry>();
        entry->symbol = imported;

        if (failed(parseCommonAttrs(imported, entry)))
          return failure();

        if (roleAttr.getValue() == kEntryRoleVariant) {
          if (failed(parseVariantAttrs(imported, entry)))
            return failure();
        } else if (roleAttr.getValue() == kEntryRoleSeed) {
          if (failed(parseSeedAttrs(imported, entry)))
            return failure();
        } else {
          imported.emitError("invalid pto.oplib.entry_role, expected variant|seed");
          return failure();
        }

        if (debug) {
          llvm::errs() << "[op-fusion] imported OP-Lib entry: role="
                       << (entry->role == EntryRole::Variant ? "variant" : "seed");
          if (entry->role == EntryRole::Variant) {
            llvm::errs() << " op=" << entry->op
                         << " variant_id=" << entry->variantId;
          } else {
            llvm::errs() << " seed_id=" << entry->seedId;
          }
          llvm::errs() << " symbol=@" << imported.getSymName() << " file=" << path
                       << "\n";
        }

        entries.push_back(std::move(entry));
        ++importedCount;
      }
    }

    if (importedCount == 0)
      return emitFailure(loc, "no valid OP-Lib entries imported from --op-lib-dir");

    return success();
  }

  static bool matchDim(int64_t pattern, int64_t target) {
    if (pattern == -1)
      return true;
    if (target == -1)
      return false;
    return pattern == target;
  }

  static bool matchLayout(StringRef pattern, StringRef target) {
    if (pattern == kDefaultAny)
      return true;
    return pattern == target;
  }

  static bool matchCommon(const TemplateEntry &entry, const MatchKey &target) {
    if (!matchDim(entry.match.rows, target.rows))
      return false;
    if (!matchDim(entry.match.cols, target.cols))
      return false;
    if (!matchDim(entry.match.fractal, target.fractal))
      return false;
    if (!matchLayout(entry.match.blayout, target.blayout))
      return false;
    if (!matchLayout(entry.match.slayout, target.slayout))
      return false;
    return true;
  }

  FailureOr<SelectedVariant> selectVariantFor(StringRef targetOp,
                                              StringRef targetDType,
                                              const MatchKey &targetMatch,
                                              Location loc) {
    SmallVector<SelectedVariant, 16> candidates;
    for (std::unique_ptr<TemplateEntry> &entryPtr : entries) {
      TemplateEntry &entry = *entryPtr;
      if (!matchCommon(entry, targetMatch))
        continue;

      if (entry.role == EntryRole::Variant) {
        if (entry.op != targetOp)
          continue;
        if (entry.matchDType != targetDType)
          continue;

        candidates.push_back(SelectedVariant{
            &entry,
            entry.variantId,
            entry.op,
            entry.matchDType,
            /*fromSeed=*/false,
            /*seedId=*/"",
            entry.coreSlot,
            entry.cost,
            entry.priority,
        });
        continue;
      }

      if (!containsString(entry.supportOps, targetOp))
        continue;
      if (!containsString(entry.supportDTypes, targetDType))
        continue;

      std::string variantId = "__seed__" + entry.seedId + "__" +
                              targetOp.str() + "__" + targetDType.str();
      candidates.push_back(SelectedVariant{
          &entry,
          variantId,
          targetOp.str(),
          targetDType.str(),
          /*fromSeed=*/true,
          entry.seedId,
          entry.coreSlot,
          entry.cost,
          entry.priority,
      });
    }

    if (candidates.empty()) {
      (void)emitFailure(loc, Twine("no matching OP-Lib entry for op=") + targetOp +
                                 " dtype=" + targetDType);
      return failure();
    }

    llvm::sort(candidates, [](const SelectedVariant &lhs,
                              const SelectedVariant &rhs) {
      if (lhs.cost != rhs.cost)
        return lhs.cost < rhs.cost;
      if (lhs.priority != rhs.priority)
        return lhs.priority > rhs.priority;
      return lhs.variantId < rhs.variantId;
    });

    if (debug) {
      const SelectedVariant &best = candidates.front();
      llvm::errs() << "[op-fusion] selected variant: op=" << best.op
                   << " dtype=" << best.dtype
                   << " variant_id=" << best.variantId
                   << " cost=" << best.cost
                   << " priority=" << best.priority
                   << (best.fromSeed ? " source=seed" : " source=variant")
                   << "\n";
    }

    return candidates.front();
  }

  FailureOr<func::FuncOp>
  getOrCreateInstance(const SelectedVariant &selected,
                      ArrayRef<MemRefType> concreteArgTypes, Location loc) {
    std::string key = selected.variantId;
    for (MemRefType ty : concreteArgTypes) {
      key += "|";
      key += memRefTypeToString(ty);
    }

    auto cached = instanceByKey.find(key);
    if (cached != instanceByKey.end())
      return cached->second;

    SmallVector<Type, 3> argTypes;
    for (MemRefType memTy : concreteArgTypes)
      argTypes.push_back(memTy);

    std::string symBase =
        "__pto_oplib_inst_" + sanitizeSymbolComponent(selected.variantId);
    std::string sym = symBase;
    int suffix = 0;
    while (symbolTable.lookup(sym))
      sym = symBase + "_" + std::to_string(++suffix);

    OpBuilder modBuilder(module.getContext());
    modBuilder.setInsertionPointToStart(module.getBody());
    auto inst = modBuilder.create<func::FuncOp>(
        loc, sym, FunctionType::get(module.getContext(), argTypes, {}));
    inst.setPrivate();
    inst->setAttr(kOpLibAttrInstVariantId,
                  StringAttr::get(module.getContext(), selected.variantId));
    inst->setAttr(kOpLibAttrInstOp,
                  StringAttr::get(module.getContext(), selected.op));
    inst->setAttr(kOpLibAttrInstDType,
                  StringAttr::get(module.getContext(), selected.dtype));
    inst->setAttr(kOpLibAttrInstSource,
                  FlatSymbolRefAttr::get(module.getContext(),
                                         selected.entry->symbol.getSymName()));
    inst->setAttr(kOpLibAttrInstFromSeed,
                  BoolAttr::get(module.getContext(), selected.fromSeed));
    if (selected.fromSeed) {
      inst->setAttr(kOpLibAttrInstSeedId,
                    StringAttr::get(module.getContext(), selected.seedId));
      inst->setAttr(kOpLibAttrInstCoreSlot,
                    StringAttr::get(module.getContext(), selected.coreSlot));
    }
    inst->setAttr(kOpLibAttrSync,
                  BoolAttr::get(module.getContext(), selected.entry->sync));

    instanceByKey[key] = inst;

    if (debug) {
      llvm::errs() << "[op-fusion] instantiate candidate: variant_id="
                   << selected.variantId << " -> @" << inst.getSymName() << "\n";
    }

    return inst;
  }
};

static FailureOr<MatchKey> buildMatchKeyFromMemRef(MemRefType dstTy) {
  if (dstTy.getRank() != 2)
    return failure();

  MatchKey key;
  key.rows = dstTy.isDynamicDim(0) ? -1 : dstTy.getDimSize(0);
  key.cols = dstTy.isDynamicDim(1) ? -1 : dstTy.getDimSize(1);
  key.blayout = kDefaultAny.str();
  key.slayout = kDefaultAny.str();
  key.fractal = -1;
  return key;
}

static FailureOr<SmallVector<MemRefType, 3>>
getConcreteMemRefTypes(Operation *op) {
  BinaryOpInterface iface = getBinaryOpInterface(op);
  if (iface.opName.empty())
    return failure();

  auto src0Ty = dyn_cast<MemRefType>(iface.src0.getType());
  auto src1Ty = dyn_cast<MemRefType>(iface.src1.getType());
  auto dstTy = dyn_cast<MemRefType>(iface.dst.getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return failure();

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return failure();

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return failure();

  if (!isFloatDTypeSupported(src0Ty.getElementType()))
    return failure();

  return SmallVector<MemRefType, 3>{src0Ty, src1Ty, dstTy};
}

static FailureOr<PlannedOpLowering>
planOneOpLowering(Operation *op, TemplateRegistry &registry,
                  StringRef warningPrefix) {
  BinaryOpInterface iface = getBinaryOpInterface(op);
  if (iface.opName.empty())
    return failure();

  FailureOr<SmallVector<MemRefType, 3>> concreteTypesOr =
      getConcreteMemRefTypes(op);
  if (failed(concreteTypesOr)) {
    op->emitWarning() << warningPrefix
                      << ": unsupported operand signature for op '" << iface.opName
                      << "' (requires rank-2 memref<f16/f32>)";
    return failure();
  }

  SmallVector<MemRefType, 3> concreteTypes = *concreteTypesOr;
  Type elemTy = concreteTypes.front().getElementType();

  FailureOr<MatchKey> matchKeyOr = buildMatchKeyFromMemRef(concreteTypes[2]);
  if (failed(matchKeyOr)) {
    op->emitWarning() << warningPrefix << ": unsupported shape rank for op '"
                      << iface.opName << "'";
    return failure();
  }

  std::string dtype = dtypeToString(elemTy);
  FailureOr<SelectedVariant> selectedOr =
      registry.selectVariantFor(iface.opName, dtype, *matchKeyOr, op->getLoc());
  if (failed(selectedOr)) {
    op->emitWarning() << warningPrefix << ": no OP-Lib candidate for op="
                      << iface.opName << " dtype=" << dtype;
    return failure();
  }

  FailureOr<func::FuncOp> instanceOr =
      registry.getOrCreateInstance(*selectedOr, concreteTypes, op->getLoc());
  if (failed(instanceOr)) {
    op->emitWarning() << warningPrefix
                      << ": failed to instantiate OP-Lib candidate for op="
                      << iface.opName << " dtype=" << dtype;
    return failure();
  }

  return PlannedOpLowering{op, *selectedOr, *instanceOr};
}

static FailureOr<Value> mapOrFail(DenseMap<Value, Value> &valueMap, Value key) {
  auto it = valueMap.find(key);
  if (it == valueMap.end())
    return failure();
  return it->second;
}

static LogicalResult materializePlannedOpInGroup(
    const PlannedOpLowering &planned, DenseMap<Value, Value> &valueMap,
    OpBuilder &bodyBuilder) {
  Operation *op = planned.op;
  BinaryOpInterface iface = getBinaryOpInterface(op);

  FailureOr<Value> src0Or = mapOrFail(valueMap, iface.src0);
  FailureOr<Value> src1Or = mapOrFail(valueMap, iface.src1);
  if (failed(src0Or) || failed(src1Or))
    return emitError(op->getLoc(), "failed to map group source values"), failure();

  Value mappedSrc0 = *src0Or;
  Value mappedSrc1 = *src1Or;

  Value mappedDst;
  auto dstIt = valueMap.find(iface.dst);
  if (dstIt == valueMap.end())
    return emitError(op->getLoc(), "group dst missing mapped value"), failure();
  mappedDst = dstIt->second;

  func::FuncOp instance = planned.instance;
  auto instTy = instance.getFunctionType();
  auto instSrc0Ty = dyn_cast<MemRefType>(instTy.getInput(0));
  auto instSrc1Ty = dyn_cast<MemRefType>(instTy.getInput(1));
  auto instDstTy = dyn_cast<MemRefType>(instTy.getInput(2));
  if (!instSrc0Ty || !instSrc1Ty || !instDstTy)
    return emitError(op->getLoc(), "instance signature is invalid"), failure();

  Value callSrc0 =
      castToMemRefTypeIfNeeded(bodyBuilder, op->getLoc(), mappedSrc0, instSrc0Ty);
  Value callSrc1 =
      castToMemRefTypeIfNeeded(bodyBuilder, op->getLoc(), mappedSrc1, instSrc1Ty);
  Value callDst =
      castToMemRefTypeIfNeeded(bodyBuilder, op->getLoc(), mappedDst, instDstTy);
  if (!callSrc0 || !callSrc1 || !callDst)
    return emitError(op->getLoc(), "failed to adapt operands to instance type"),
           failure();

  bodyBuilder.create<func::CallOp>(op->getLoc(), instance,
                                   ValueRange{callSrc0, callSrc1, callDst});

  return success();
}

struct PTOMaterializeFusionGroupsFromOpLibPass
    : public pto::impl::PTOMaterializeFusionGroupsFromOpLibBase<
          PTOMaterializeFusionGroupsFromOpLibPass> {
  using pto::impl::PTOMaterializeFusionGroupsFromOpLibBase<
      PTOMaterializeFusionGroupsFromOpLibPass>::PTOMaterializeFusionGroupsFromOpLibBase;

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

        if (hasFusionGroupAttrs(curOp) && isSupportedFusionOp(curOp)) {
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

  LogicalResult planGroupLowerings(GroupInfo &group, TemplateRegistry &registry,
                                   SmallVectorImpl<PlannedOpLowering> &planned) {
    for (Operation *op : group.ops) {
      FailureOr<PlannedOpLowering> oneOr =
          planOneOpLowering(op, registry, "fusion-group fallback");
      if (failed(oneOr))
        return failure();
      planned.push_back(*oneOr);
    }
    return success();
  }

  LogicalResult materializeOneGroup(ModuleOp module, SymbolTable &symbolTable,
                                    TemplateRegistry &registry, GroupInfo &group,
                                    int &fusedCounter) {
    if (group.ops.empty())
      return success();

    SmallVector<PlannedOpLowering, 8> planned;
    if (failed(planGroupLowerings(group, registry, planned))) {
      group.ops.front()->emitWarning()
          << "fusion-group fallback: keep original group_id=" << group.groupId;
      return success();
    }

    Operation *firstOp = group.ops.front();
    Location loc = firstOp->getLoc();

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

    for (const PlannedOpLowering &plan : planned) {
      if (failed(materializePlannedOpInGroup(plan, valueMap, bodyBuilder))) {
        firstOp->emitWarning() << "fusion-group fallback: failed to materialize "
                                  "selected OP-Lib calls, keep original group";
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
      llvm::errs() << "[op-fusion] materialized group_id=" << group.groupId
                   << " into @" << fusedFunc.getSymName() << "\n";
    }

    return success();
  }

  LogicalResult materializeSingleOps(func::FuncOp func, TemplateRegistry &registry) {
    SmallVector<Operation *, 32> toRewrite;

    func.walk([&](Operation *op) {
      if (!isSupportedFusionOp(op))
        return;
      if (hasFusionGroupAttrs(op))
        return;
      toRewrite.push_back(op);
    });

    for (Operation *op : toRewrite) {
      if (!op || !op->getBlock())
        continue;

      FailureOr<PlannedOpLowering> plannedOr =
          planOneOpLowering(op, registry, "single-op fallback");
      if (failed(plannedOr))
        continue;

      BinaryOpInterface iface = getBinaryOpInterface(op);
      OpBuilder builder(op);
      builder.create<func::CallOp>(op->getLoc(), plannedOr->instance,
                                   ValueRange{iface.src0, iface.src1, iface.dst});
      op->erase();

      if (debug) {
        llvm::errs() << "[op-fusion] materialized single op=" << iface.opName
                     << " in @" << func.getSymName() << "\n";
      }
    }

    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SymbolTable symbolTable(module);

    TemplateRegistry registry{module, symbolTable, debug};
    if (failed(registry.loadFromDir(opLibDir, ctx, module.getLoc()))) {
      signalPassFailure();
      return;
    }

    int fusedCounter = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func.getSymName().starts_with("__pto_oplib_"))
        continue;

      SmallVector<GroupInfo, 8> groups;
      collectGroupsInRegion(func.getRegion(), groups);
      if (debug && !groups.empty()) {
        llvm::errs() << "[op-fusion] found " << groups.size() << " group(s) in @"
                     << func.getSymName() << "\n";
      }

      for (GroupInfo &group : groups) {
        if (failed(
                materializeOneGroup(module, symbolTable, registry, group, fusedCounter))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(materializeSingleOps(func, registry))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOMaterializeFusionGroupsFromOpLibPass(
    const PTOMaterializeFusionGroupsFromOpLibOptions &options) {
  return std::make_unique<PTOMaterializeFusionGroupsFromOpLibPass>(options);
}
