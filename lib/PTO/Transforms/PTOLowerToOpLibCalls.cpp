#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINSTANTIATEANDLOWERTOLIBCALL
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

static constexpr llvm::StringLiteral kSimdLevelAttr = "pto.simd.level";
static constexpr llvm::StringLiteral kSimdLanesAttr = "pto.simd.lanes";
static constexpr llvm::StringLiteral kSimdCoreSlotAttr = "pto.simd.core_slot";
static constexpr llvm::StringLiteral kSimdLevelBinaryEwiseV1 =
    "binary_ewise_v1";
static constexpr llvm::StringLiteral kSimdVldDistAttr = "pto.simd.vld_dist";
static constexpr llvm::StringLiteral kSimdVstDistAttr = "pto.simd.vst_dist";
static constexpr llvm::StringLiteral kSimdExecModeAttr = "pto.simd.exec_mode";

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
static constexpr llvm::StringLiteral kErrBodyDisallowedIR =
    "E_OPLIB_BODY_DISALLOWED_IR";
static constexpr llvm::StringLiteral kErrSimdAttrRequired =
    "E_OPLIB_SIMD_ATTR_REQUIRED";

struct GroupInfo {
  int64_t groupId = -1;
  SmallVector<Operation *, 8> ops;
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
  std::string simdLevel;
  int64_t simdLanes = -1;
  bool hasSimdBridgeOps = false;

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

static FailureOr<MemRefType> inferSimdBridgeMemRefType(pto::TileBufType tileTy,
                                                        MLIRContext *ctx) {
  if (tileTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> physicalShape = tileTy.getShape();
  if (physicalShape.size() != 2)
    return failure();
  if (physicalShape[0] == ShapedType::kDynamic ||
      physicalShape[1] == ShapedType::kDynamic)
    return failure();

  SmallVector<int64_t, 2> memShape(physicalShape.begin(), physicalShape.end());
  ArrayRef<int64_t> validShape = tileTy.getValidShape();
  if (validShape.size() == memShape.size()) {
    for (unsigned i = 0; i < validShape.size(); ++i) {
      memShape[i] =
          validShape[i] < 0 ? ShapedType::kDynamic : validShape[i];
    }
  }

  auto cfg = tileTy.getConfigAttr();
  if (!cfg)
    cfg = pto::TileBufConfigAttr::getDefault(ctx);

  int32_t bl = 0; // row_major
  int32_t sl = 0; // none_box
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    fr = static_cast<int32_t>(attr.getInt());

  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (sl != 0) {
    int64_t elemBytes = getElemBytes(tileTy.getElementType());
    if (elemBytes <= 0)
      return failure();
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
        return failure();
      }
    } else {
      return failure();
    }
  }

  SmallVector<int64_t, 2> strides;
  if (sl == 0) {
    if (bl == 1) {
      strides.push_back(1);
      strides.push_back(physicalShape[0]);
    } else {
      strides.push_back(physicalShape[1]);
      strides.push_back(1);
    }
  } else if (bl == 1) {
    if (sl != 1)
      return failure();
    strides.push_back(innerCols);
    strides.push_back(physicalShape[0]);
  } else {
    strides.push_back(physicalShape[1]);
    strides.push_back(innerRows);
  }

  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/0, strides);
  return MemRefType::get(memShape, tileTy.getElementType(), layout,
                         tileTy.getMemorySpace());
}

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

static std::string typeToString(Type ty) {
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

static bool isBinaryFloatCore(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}

static bool isVectorFloatBinaryArith(Operation *op) {
  if (!isBinaryFloatCore(op))
    return false;
  if (op->getNumResults() != 1)
    return false;
  return isa<VectorType>(op->getResult(0).getType());
}

static FailureOr<int64_t> getFixedVectorLanes(Type ty) {
  auto vecTy = dyn_cast<VectorType>(ty);
  if (!vecTy || vecTy.isScalable())
    return failure();
  return vecTy.getNumElements();
}

static bool isSimdBridgeOp(Operation *op) {
  return isa<pto::SimdPredicateOp, pto::SimdLoadOp, pto::SimdStoreOp,
             pto::SimdLoadPUOp, pto::SimdStorePUOp>(op);
}

static bool isAllowedTemplateBodyOp(Operation *op) {
  if (isa<func::ReturnOp>(op))
    return true;
  if (isa<pto::SimdVecScopeOp>(op))
    return true;
  if (isa<pto::SimdTileToMemrefOp>(op))
    return true;
  if (isSimdBridgeOp(op))
    return true;

  // V1.2 strict mode: reject legacy bridge in OP-Lib templates.
  if (isa<UnrealizedConversionCastOp>(op))
    return false;

  // OP-Lib template bodies must use vector memory ops, not scalar memref
  // load/store.
  if (isa<memref::LoadOp, memref::StoreOp>(op))
    return false;

  StringRef ns = op->getName().getDialectNamespace();
  return ns == "arith" || ns == "vector" || ns == "memref" || ns == "scf";
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

  LogicalResult emitFailureWithCode(Operation *anchor, StringRef code,
                                    const Twine &msg) {
    anchor->emitError() << code << ": " << msg;
    return failure();
  }

  LogicalResult emitFailureWithCode(Location loc, StringRef code,
                                    const Twine &msg) {
    emitError(loc) << code << ": " << msg;
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
      auto tileTy = dyn_cast<pto::TileBufType>(inTy);
      if (!tileTy)
        return false;
      if (tileTy.getRank() != 2)
        return false;
      if (!isFloatDTypeSupported(tileTy.getElementType()))
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

  LogicalResult parseSimdAttrs(func::FuncOp imported,
                               std::unique_ptr<TemplateEntry> &entry,
                               bool hasSimdOps) {
    Operation *op = imported.getOperation();
    auto levelAttr = op->getAttrOfType<StringAttr>(kSimdLevelAttr);
    auto lanesAttr = op->getAttrOfType<IntegerAttr>(kSimdLanesAttr);
    bool hasLevelAttr = static_cast<bool>(levelAttr);
    bool hasLanesAttr = static_cast<bool>(lanesAttr);

    if (!hasSimdOps && !hasLevelAttr && !hasLanesAttr)
      return success();

    if (!hasLevelAttr || !hasLanesAttr) {
      return emitFailureWithCode(
          op, kErrSimdAttrRequired,
          "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
    }

    auto levelOr = parseStringAttr(op, kSimdLevelAttr);
    auto lanesOr = parseI64Attr(op, kSimdLanesAttr, /*allowWildcard=*/false);
    if (failed(levelOr) || failed(lanesOr)) {
      return emitFailureWithCode(
          op, kErrLanesMismatch,
          "missing required attrs: pto.simd.level / pto.simd.lanes");
    }
    if (*levelOr != kSimdLevelBinaryEwiseV1) {
      return emitFailureWithCode(op, kErrCoreSlot,
                                 Twine("unsupported pto.simd.level: ") +
                                     *levelOr);
    }
    entry->simdLevel = *levelOr;
    entry->simdLanes = *lanesOr;
    entry->hasSimdBridgeOps = hasSimdOps;
    return success();
  }

  LogicalResult validateTemplateBody(func::FuncOp imported,
                                     const TemplateEntry &entry) {
    if (imported.isExternal() || imported.empty() ||
        imported.front().without_terminator().empty()) {
      return emitFailureWithCode(imported.getLoc(), kErrEmptyBody,
                                 "template body must not be empty");
    }

    FunctionType fnTy = imported.getFunctionType();
    for (Type inTy : fnTy.getInputs()) {
      auto tileTy = dyn_cast<pto::TileBufType>(inTy);
      if (!tileTy) {
        return emitFailureWithCode(imported.getLoc(), kErrEmptyBody,
                                   "template inputs must be !pto.tile_buf");
      }
      Type elemTy = tileTy.getElementType();
      if (!elemTy.isF16() && !elemTy.isF32()) {
        return emitFailureWithCode(imported.getLoc(), kErrDType,
                                   "SIMD template supports f16/f32 only");
      }
      if (tileTy.getBLayoutValueI32() != 0) {
        return emitFailureWithCode(imported.getLoc(), kErrLayout,
                                   "SIMD template supports row_major only");
      }
    }

    int64_t firstLoad = std::numeric_limits<int64_t>::max();
    int64_t firstStore = std::numeric_limits<int64_t>::max();
    int64_t coreSeq = -1;
    int coreCount = 0;
    std::string inferredVldDist;
    std::string inferredVstDist;

    llvm::DenseMap<Operation *, int64_t> preorder;
    int64_t seq = 0;
    imported.walk([&](Operation *op) { preorder[op] = seq++; });

    LogicalResult status = success();
    auto requireNonEmptyTokenAttr = [&](Operation *targetOp, StringRef attrName,
                                        StringRef usage,
                                        StringRef requiredPrefix) -> bool {
      auto tokenAttr = targetOp->getAttrOfType<StringAttr>(attrName);
      if (!tokenAttr) {
        status = emitFailureWithCode(
            targetOp, kErrSimdAttrRequired,
            Twine(usage) + " requires string attr '" + attrName + "'");
        return false;
      }

      StringRef token = tokenAttr.getValue();
      if (token.empty()) {
        status = emitFailureWithCode(
            targetOp, kErrSimdAttrRequired,
            Twine(usage) + " attr '" + attrName + "' must be non-empty");
        return false;
      }

      if (!requiredPrefix.empty() && !token.starts_with(requiredPrefix)) {
        status = emitFailureWithCode(
            targetOp, kErrSimdAttrRequired,
            Twine(usage) + " attr '" + attrName +
                "' must start with '" + requiredPrefix + "', got '" + token +
                "'");
        return false;
      }

      return true;
    };

    imported.walk([&](Operation *op) {
      if (failed(status))
        return;

      if (op == imported.getOperation())
        return;
      if (!isAllowedTemplateBodyOp(op)) {
        status = emitFailureWithCode(
            op, kErrBodyDisallowedIR,
            Twine("unsupported template body op: ") + op->getName().getStringRef());
        return;
      }

      if (isa<vector::LoadOp, vector::MaskedLoadOp>(op)) {
        if (!requireNonEmptyTokenAttr(op, kSimdVldDistAttr, "vector load",
                                      /*requiredPrefix=*/""))
          return;
        if (inferredVldDist.empty()) {
          auto vld = op->getAttrOfType<StringAttr>(kSimdVldDistAttr);
          if (vld)
            inferredVldDist = vld.getValue().str();
        }
      }

      if (isa<vector::StoreOp, vector::MaskedStoreOp>(op)) {
        if (!requireNonEmptyTokenAttr(op, kSimdVstDistAttr, "vector store",
                                      /*requiredPrefix=*/"DIST_"))
          return;
        if (inferredVstDist.empty()) {
          auto vst = op->getAttrOfType<StringAttr>(kSimdVstDistAttr);
          if (vst)
            inferredVstDist = vst.getValue().str();
        }
      }

      if (isVectorFloatBinaryArith(op)) {
        if (!requireNonEmptyTokenAttr(op, kSimdExecModeAttr,
                                      "vector float binary arith op",
                                      /*requiredPrefix=*/"MODE_"))
          return;
      }

      if (isa<pto::SimdLoadOp, pto::SimdLoadPUOp>(op))
        firstLoad = std::min(firstLoad, preorder[op]);
      if (isa<pto::SimdStoreOp, pto::SimdStorePUOp>(op))
        firstStore = std::min(firstStore, preorder[op]);

      if (auto pred = dyn_cast<pto::SimdPredicateOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(
              pred, kErrSimdAttrRequired,
              "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes = getFixedVectorLanes(pred.getMask().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(pred, kErrLanesMismatch,
                                       "simd.predicate lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto load = dyn_cast<pto::SimdLoadOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(
              load, kErrSimdAttrRequired,
              "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes = getFixedVectorLanes(load.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(load, kErrLanesMismatch,
                                       "simd.load lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto loadPU = dyn_cast<pto::SimdLoadPUOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(
              loadPU, kErrSimdAttrRequired,
              "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(loadPU.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(loadPU, kErrLanesMismatch,
                                       "simd.load_pu lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto store = dyn_cast<pto::SimdStoreOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(
              store, kErrSimdAttrRequired,
              "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes = getFixedVectorLanes(store.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(store, kErrLanesMismatch,
                                       "simd.store lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto storePU = dyn_cast<pto::SimdStorePUOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(
              storePU, kErrSimdAttrRequired,
              "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(storePU.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(storePU, kErrLanesMismatch,
                                       "simd.store_pu lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      auto slotAttr = op->getAttrOfType<StringAttr>(kSimdCoreSlotAttr);
      if (!slotAttr)
        return;

      if (slotAttr.getValue() != entry.coreSlot) {
        status = emitFailureWithCode(op, kErrCoreSlot,
                                     "core slot value mismatches seed.core_slot");
        return;
      }
      if (!isBinaryFloatCore(op)) {
        status = emitFailureWithCode(
            op, kErrCoreSlot,
            "core slot op must be one of arith.addf/subf/mulf/divf/maximumf/minimumf");
        return;
      }
      if (op->getNumResults() != 1) {
        status = emitFailureWithCode(op, kErrCoreSlot,
                                     "core slot op must have exactly one result");
        return;
      }
      if (entry.simdLanes > 0) {
        FailureOr<int64_t> lanes = getFixedVectorLanes(op->getResult(0).getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(op, kErrLanesMismatch,
                                       "core slot vector lanes mismatch with pto.simd.lanes");
          return;
        }
      }
      coreSeq = preorder[op];
      ++coreCount;
    });

    if (failed(status))
      return failure();

    // Canonicalization may rewrite vector.maskedload/store to vector.load/store
    // and drop op attrs. Keep function-local fallback tokens on vector float
    // arithmetic ops so EmitC lowering can still recover vld/vst tokens.
    if (!inferredVldDist.empty() || !inferredVstDist.empty()) {
      auto *ctx = imported.getContext();
      imported.walk([&](Operation *op) {
        if (!isVectorFloatBinaryArith(op))
          return;
        if (!inferredVldDist.empty() && !op->getAttr(kSimdVldDistAttr)) {
          op->setAttr(kSimdVldDistAttr,
                      StringAttr::get(ctx, inferredVldDist));
        }
        if (!inferredVstDist.empty() && !op->getAttr(kSimdVstDistAttr)) {
          op->setAttr(kSimdVstDistAttr,
                      StringAttr::get(ctx, inferredVstDist));
        }
      });
    }

    if (entry.role == EntryRole::Seed && coreCount != 1) {
      return emitFailureWithCode(imported.getLoc(), kErrCoreSlot,
                                 "seed template must contain exactly one core slot op");
    }
    if (entry.hasSimdBridgeOps && coreCount != 1) {
      return emitFailureWithCode(imported.getLoc(), kErrCoreSlot,
                                 "template using pto.simd.* must contain exactly one core slot op");
    }
    if (coreCount > 1) {
      return emitFailureWithCode(imported.getLoc(), kErrCoreSlot,
                                 "template must not contain multiple core slot ops");
    }
    if (entry.hasSimdBridgeOps && firstLoad == std::numeric_limits<int64_t>::max()) {
      return emitFailureWithCode(imported.getLoc(), kErrCoreSlot,
                                 "template must contain simd.load/simd.load_pu");
    }
    if (entry.hasSimdBridgeOps && firstStore == std::numeric_limits<int64_t>::max()) {
      return emitFailureWithCode(imported.getLoc(), kErrCoreSlot,
                                 "template must contain simd.store/simd.store_pu");
    }
    if (entry.hasSimdBridgeOps && !(firstLoad < coreSeq && coreSeq < firstStore)) {
      return emitFailureWithCode(imported.getLoc(), kErrCoreSlot,
                                 "template ordering must satisfy load -> core -> store");
    }

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
                            "(!pto.tile_buf<...>, !pto.tile_buf<...>, "
                            "!pto.tile_buf<...>) -> () with rank-2 f16/f32 tile_buf args");
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

        bool hasSimdOps = false;
        imported.walk([&](Operation *bodyOp) {
          if (isSimdBridgeOp(bodyOp))
            hasSimdOps = true;
        });

        if (failed(parseSimdAttrs(imported, entry, hasSimdOps)))
          return failure();

        if (failed(validateTemplateBody(imported, *entry)))
          return failure();

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
                      ArrayRef<Type> concreteArgTypes, Location loc) {
    std::string key = selected.variantId;
    for (Type ty : concreteArgTypes) {
      key += "|";
      key += typeToString(ty);
    }

    auto cached = instanceByKey.find(key);
    if (cached != instanceByKey.end())
      return cached->second;

    SmallVector<Type, 3> argTypes;
    for (Type ty : concreteArgTypes)
      argTypes.push_back(ty);

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
    if (auto levelAttr =
            selected.entry->symbol->getAttrOfType<StringAttr>(kSimdLevelAttr)) {
      inst->setAttr(kSimdLevelAttr, levelAttr);
    }
    if (auto lanesAttr =
            selected.entry->symbol->getAttrOfType<IntegerAttr>(kSimdLanesAttr)) {
      inst->setAttr(kSimdLanesAttr, lanesAttr);
    }

    func::FuncOp source = selected.entry->symbol;
    if (source.isExternal() || source.empty() ||
        source.front().without_terminator().empty()) {
      (void)emitFailureWithCode(
          loc, kErrEmptyBody,
          Twine("source template has empty body: @") + source.getSymName());
      return failure();
    }
    if (!source.getBody().hasOneBlock()) {
      (void)emitFailureWithCode(
          loc, kErrCoreSlot,
          Twine("source template must have a single entry block: @") +
              source.getSymName());
      return failure();
    }

    Block &srcEntry = source.front();
    Block *dstEntry = inst.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(dstEntry);
    IRMapping mapping;
    for (auto [srcArg, dstArg] :
         llvm::zip(srcEntry.getArguments(), dstEntry->getArguments())) {
      mapping.map(srcArg, dstArg);
    }

    for (Operation &op : srcEntry.without_terminator()) {
      if (auto bridge = dyn_cast<pto::SimdTileToMemrefOp>(&op)) {
        Value mappedSrc = mapping.lookupOrNull(bridge.getSrc());
        auto mappedTileTy = dyn_cast<pto::TileBufType>(mappedSrc.getType());
        if (!mappedSrc || !mappedTileTy) {
          (void)emitFailureWithCode(
              loc, kErrCoreSlot,
              Twine("failed to remap simd.tile_to_memref source in instance @") +
                  inst.getSymName());
          return failure();
        }
        FailureOr<MemRefType> inferredTyOr =
            inferSimdBridgeMemRefType(mappedTileTy, module.getContext());
        if (failed(inferredTyOr)) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("failed to infer simd.tile_to_memref memref type for instance @") +
                  inst.getSymName());
          return failure();
        }

        auto newBridge = bodyBuilder.create<pto::SimdTileToMemrefOp>(
            bridge.getLoc(), *inferredTyOr, mappedSrc);
        mapping.map(bridge.getDst(), newBridge.getDst());
        continue;
      }

      Operation *cloned = bodyBuilder.clone(op, mapping);
      for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults()))
        mapping.map(oldRes, newRes);
    }
    bodyBuilder.create<func::ReturnOp>(loc);

    if (selected.fromSeed) {
      SmallVector<Operation *, 4> coreOps;
      inst.walk([&](Operation *op) {
        auto slotAttr = op->getAttrOfType<StringAttr>(kSimdCoreSlotAttr);
        if (!slotAttr)
          return;
        if (slotAttr.getValue() == selected.coreSlot)
          coreOps.push_back(op);
      });
      if (coreOps.size() != 1) {
        (void)emitFailureWithCode(
            loc, kErrCoreSlot,
            Twine("seed instance expects exactly one core slot op in @") +
                inst.getSymName());
        return failure();
      }

      Operation *core = coreOps.front();
      if (core->getNumOperands() != 2 || core->getNumResults() != 1) {
        (void)emitFailureWithCode(core, kErrCoreSlot,
                                  "core slot op must be binary op with one result");
        return failure();
      }

      OpBuilder b(core);
      Value lhs = core->getOperand(0);
      Value rhs = core->getOperand(1);
      Operation *newCore = nullptr;
      if (selected.op == "tadd") {
        newCore = b.create<arith::AddFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tsub") {
        newCore = b.create<arith::SubFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmul") {
        newCore = b.create<arith::MulFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tdiv") {
        newCore = b.create<arith::DivFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmax") {
        newCore = b.create<arith::MaximumFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmin") {
        newCore = b.create<arith::MinimumFOp>(core->getLoc(), lhs, rhs);
      } else {
        (void)emitFailureWithCode(core, kErrCoreSlot,
                                  Twine("unsupported seed target op: ") +
                                      selected.op);
        return failure();
      }
      newCore->setAttrs(core->getAttrs());
      core->replaceAllUsesWith(newCore->getResults());
      core->erase();
    }

    instanceByKey[key] = inst;

    if (debug) {
      llvm::errs() << "[op-fusion] instantiate candidate: variant_id="
                   << selected.variantId << " -> @" << inst.getSymName() << "\n";
    }

    return inst;
  }
};

static FailureOr<MatchKey> buildMatchKeyFromTileBuf(pto::TileBufType dstTy) {
  if (dstTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = dstTy.getShape();
  MatchKey key;
  key.rows = shape[0] == ShapedType::kDynamic ? -1 : shape[0];
  key.cols = shape[1] == ShapedType::kDynamic ? -1 : shape[1];

  switch (dstTy.getBLayoutValueI32()) {
  case 0:
    key.blayout = "row_major";
    break;
  case 1:
    key.blayout = "col_major";
    break;
  default:
    key.blayout = kDefaultAny.str();
    break;
  }

  switch (dstTy.getSLayoutValueI32()) {
  case 0:
    key.slayout = "none_box";
    break;
  case 1:
    key.slayout = "row_major";
    break;
  case 2:
    key.slayout = "col_major";
    break;
  default:
    key.slayout = kDefaultAny.str();
    break;
  }

  key.fractal = -1;
  return key;
}

static FailureOr<SmallVector<pto::TileBufType, 3>>
getConcreteTileBufTypes(Operation *op) {
  BinaryOpInterface iface = getBinaryOpInterface(op);
  if (iface.opName.empty())
    return failure();

  auto src0Ty = dyn_cast<pto::TileBufType>(iface.src0.getType());
  auto src1Ty = dyn_cast<pto::TileBufType>(iface.src1.getType());
  auto dstTy = dyn_cast<pto::TileBufType>(iface.dst.getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return failure();

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return failure();

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return failure();

  if (!isFloatDTypeSupported(src0Ty.getElementType()))
    return failure();

  return SmallVector<pto::TileBufType, 3>{src0Ty, src1Ty, dstTy};
}

static FailureOr<PlannedOpLowering>
planOneOpLowering(Operation *op, TemplateRegistry &registry,
                  StringRef warningPrefix) {
  BinaryOpInterface iface = getBinaryOpInterface(op);
  if (iface.opName.empty())
    return failure();

  FailureOr<SmallVector<pto::TileBufType, 3>> concreteTypesOr =
      getConcreteTileBufTypes(op);
  if (failed(concreteTypesOr)) {
    op->emitWarning() << warningPrefix
                      << ": unsupported operand signature for op '" << iface.opName
                      << "' (requires rank-2 !pto.tile_buf<f16/f32>)";
    return failure();
  }

  SmallVector<pto::TileBufType, 3> concreteTypes = *concreteTypesOr;
  Type elemTy = concreteTypes.front().getElementType();

  FailureOr<MatchKey> matchKeyOr = buildMatchKeyFromTileBuf(concreteTypes[2]);
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

  SmallVector<Type, 3> concreteArgTypes;
  concreteArgTypes.append(concreteTypes.begin(), concreteTypes.end());
  FailureOr<func::FuncOp> instanceOr =
      registry.getOrCreateInstance(*selectedOr, concreteArgTypes, op->getLoc());
  if (failed(instanceOr)) {
    op->emitWarning() << warningPrefix
                      << ": failed to instantiate OP-Lib candidate for op="
                      << iface.opName << " dtype=" << dtype;
    return failure();
  }

  return PlannedOpLowering{op, *selectedOr, *instanceOr};
}

static LogicalResult rewriteOneGroupedOpAsCall(const PlannedOpLowering &planned) {
  Operation *op = planned.op;
  BinaryOpInterface iface = getBinaryOpInterface(op);
  if (iface.opName.empty())
    return op->emitOpError("unsupported grouped op during OP-Lib lowering");

  OpBuilder builder(op);
  auto call = builder.create<func::CallOp>(
      op->getLoc(), planned.instance, ValueRange{iface.src0, iface.src1, iface.dst});

  if (auto gid = op->getAttrOfType<IntegerAttr>(kFusionGroupIdAttr))
    call->setAttr(kFusionGroupIdAttr, gid);
  if (auto order = op->getAttrOfType<IntegerAttr>(kFusionOrderAttr))
    call->setAttr(kFusionOrderAttr, order);

  op->erase();
  return success();
}

struct PTOInstantiateAndLowerToLibCallPass
    : public pto::impl::PTOInstantiateAndLowerToLibCallBase<
          PTOInstantiateAndLowerToLibCallPass> {
  using pto::impl::PTOInstantiateAndLowerToLibCallBase<
      PTOInstantiateAndLowerToLibCallPass>::PTOInstantiateAndLowerToLibCallBase;

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

  LogicalResult lowerOneGroup(TemplateRegistry &registry, GroupInfo &group) {
    if (group.ops.empty())
      return success();

    SmallVector<PlannedOpLowering, 8> planned;
    if (failed(planGroupLowerings(group, registry, planned))) {
      group.ops.front()->emitWarning()
          << "fusion-group fallback: keep original group_id=" << group.groupId;
      return success();
    }

    for (const PlannedOpLowering &plan : planned) {
      if (failed(rewriteOneGroupedOpAsCall(plan))) {
        group.ops.front()->emitWarning()
            << "fusion-group fallback: failed to rewrite selected OP-Lib calls, "
               "keep original group";
        return success();
      }
    }

    if (debug) {
      llvm::errs() << "[op-fusion] lowered group_id=" << group.groupId
                   << " to OP-Lib calls (" << planned.size() << " op(s))\n";
    }

    return success();
  }

  LogicalResult lowerSingleOps(func::FuncOp func, TemplateRegistry &registry) {
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
        if (failed(lowerOneGroup(registry, group))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(lowerSingleOps(func, registry))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOInstantiateAndLowerToLibCallPass(
    const PTOInstantiateAndLowerToLibCallOptions &options) {
  return std::make_unique<PTOInstantiateAndLowerToLibCallPass>(options);
}
