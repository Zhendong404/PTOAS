#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#include <algorithm>
#include <memory>
#include <limits>
#include <optional>
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
static constexpr llvm::StringLiteral kOpLibAttrMatchScalarPos =
    "pto.oplib.match.scalar_pos";
static constexpr llvm::StringLiteral kOpLibAttrMatchIsBinary =
    "pto.oplib.match.is_binary";
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
static constexpr llvm::StringLiteral kOpLibAttrInstKind =
    "pto.oplib.instance.kind";
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

enum class TemplateArgRole {
  Tile,
  Scalar,
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
  std::string kind;
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

  SmallVector<TemplateArgRole, 4> argRoles;
  SmallVector<std::optional<MatchKey>, 4> argMatches;
  std::optional<int64_t> scalarPos;
  std::optional<bool> isBinary;
  int64_t cost = 0;
  int64_t priority = 0;
  bool sync = false;
};

struct SelectedVariant {
  TemplateEntry *entry = nullptr;
  std::string kind;
  std::string variantId;
  std::string op;
  std::string dtype;
  bool fromSeed = false;
  std::string seedId;
  std::string coreSlot;
  SmallVector<std::string, 4> instanceKeyAttrs;
  int64_t cost = 0;
  int64_t priority = 0;
};

struct PlannedOpLowering {
  Operation *op = nullptr;
  SelectedVariant selected;
  SmallVector<Value, 4> operands;
  func::FuncOp instance;
};

struct MatchRequest {
  std::string kind;
  std::string op;
  std::string dtype;
  SmallVector<Type, 4> argTypes;
  SmallVector<Value, 4> operands;
  SmallVector<std::optional<MatchKey>, 4> argMatches;
  std::optional<int64_t> scalarPos;
  std::optional<bool> isBinary;
  std::optional<std::string> requiredVariantId;
};

static bool isSupportedFusionOp(Operation *op) {
  return isa<pto::TMulOp, pto::TDivOp, pto::TAddOp, pto::TSubOp, pto::TMaxOp,
             pto::TMinOp, pto::TPartAddOp, pto::TPartMaxOp, pto::TPartMinOp,
             pto::TAddSOp, pto::TSubSOp, pto::TMulSOp, pto::TDivSOp,
             pto::TMaxSOp, pto::TMinSOp, pto::TAddCOp, pto::TSubCOp,
             pto::TAddSCOp, pto::TSubSCOp, pto::TAbsOp, pto::TNegOp,
             pto::TExpOp, pto::TLogOp, pto::TSqrtOp, pto::TRsqrtOp,
             pto::TRecipOp, pto::TReluOp>(op);
}

static bool hasFusionGroupAttrs(Operation *op) {
  return op->hasAttr(kFusionGroupIdAttr) && op->hasAttr(kFusionOrderAttr);
}

static bool isOplibScalarType(Type ty) {
  return isa<FloatType, IntegerType, IndexType>(ty);
}

static bool isOplibTileElementType(Type ty) {
  return isa<FloatType, IntegerType>(ty);
}

static bool isOplibTileLikeType(Type ty, bool allowLoweredMemRefAbi) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    return tileTy.getRank() == 2 &&
           isOplibTileElementType(tileTy.getElementType());
  }
  if (!allowLoweredMemRefAbi)
    return false;
  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    return memTy.getRank() == 2 &&
           isOplibTileElementType(memTy.getElementType());
  }
  return false;
}

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

struct ValueTileMetadata {
  pto::TileBufConfigAttr config;
};

static void collectValueTileMetadata(Value v, ValueTileMetadata &meta) {
  if (!v || meta.config)
    return;

  Operation *def = v.getDefiningOp();
  if (!def)
    return;

  if (auto bind = dyn_cast<pto::BindTileOp>(def)) {
    meta.config = bind.getConfig();
    collectValueTileMetadata(bind.getSource(), meta);
    return;
  }

  if (auto pc = dyn_cast<pto::PointerCastOp>(def)) {
    if (auto cfg = pc.getConfig())
      meta.config = *cfg;
    return;
  }

  if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
    collectValueTileMetadata(subview.getSource(), meta);
    return;
  }

  if (auto recast = dyn_cast<memref::ReinterpretCastOp>(def)) {
    collectValueTileMetadata(recast.getSource(), meta);
    return;
  }

  if (auto cast = dyn_cast<memref::CastOp>(def)) {
    collectValueTileMetadata(cast.getSource(), meta);
    return;
  }

  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() == 1)
      collectValueTileMetadata(cast.getOperand(0), meta);
    return;
  }
}

static pto::TileBufConfigAttr lookupTileConfigForValue(Value v, MLIRContext *ctx) {
  ValueTileMetadata meta;
  collectValueTileMetadata(v, meta);
  if (meta.config)
    return meta.config;
  return pto::TileBufConfigAttr::getDefault(ctx);
}

static std::string toBLayoutName(int32_t bLayout) {
  switch (bLayout) {
  case 0:
    return "row_major";
  case 1:
    return "col_major";
  default:
    return kDefaultAny.str();
  }
}

static std::string toSLayoutName(int32_t sLayout) {
  switch (sLayout) {
  case 0:
    return "none_box";
  case 1:
    return "row_major";
  case 2:
    return "col_major";
  default:
    return kDefaultAny.str();
  }
}

static int64_t getFractalOrWildcard(pto::TileBufConfigAttr cfg) {
  if (!cfg)
    return -1;
  if (auto fractalAttr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    return fractalAttr.getInt();
  return -1;
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
  return !dtypeName.empty();
}

static bool isAllowedLayoutName(StringRef layoutName) {
  return layoutName == "row_major" || layoutName == "col_major" ||
         layoutName == "none_box" || layoutName == "any";
}

static bool containsString(ArrayRef<std::string> values, StringRef key) {
  return llvm::any_of(values, [&](const std::string &s) { return s == key; });
}

static bool canUseSeedRewriteFor(StringRef kind, StringRef op) {
  if (kind == "l3_float_binary_elementwise_template")
    return op == "tadd" || op == "tsub" || op == "tmul" || op == "tdiv" ||
           op == "tmax" || op == "tmin";
  if (kind == "l3_float_partial_binary_template")
    return op == "tpartadd" || op == "tpartmax" || op == "tpartmin";
  if (kind == "l3_float_tile_scalar_template")
    return op == "tadds" || op == "tsubs" || op == "tmuls" || op == "tmaxs" ||
           op == "tmins";
  return false;
}

static FailureOr<SmallVector<TemplateArgRole, 4>>
getTemplateArgRolesForKind(StringRef kind) {
  auto build = [&](std::initializer_list<TemplateArgRole> roles)
      -> FailureOr<SmallVector<TemplateArgRole, 4>> {
    return SmallVector<TemplateArgRole, 4>(roles);
  };

  if (kind == kOpLibKindL3BinaryTemplate)
    return build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                  TemplateArgRole::Tile});

  return llvm::StringSwitch<FailureOr<SmallVector<TemplateArgRole, 4>>>(kind)
      .Case("l3_float_binary_elementwise_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_float_partial_binary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_float_binary_special_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_float_tile_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile}))
      .Case("l3_float_ternary_tile_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_ternary_tile_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_unary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_unary_math_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_unary_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile}))
      .Case("l3_reduce_row_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_reduce_col_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_reduce_colsum_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_broadcast_row_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_broadcast_col_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_broadcast_row_binary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_scalar_expand_template",
            build({TemplateArgRole::Scalar, TemplateArgRole::Tile}))
      .Case("l3_select_mask_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_select_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Scalar, TemplateArgRole::Tile}))
      .Case("l3_int_binary_elementwise_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_int_tile_scalar_elementwise_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile}))
      .Case("l3_int_unary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Default(failure());
}

static bool kindRequiresIsBinary(StringRef kind) {
  return kind == "l3_reduce_colsum_template";
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
  if (ns == "arith" || ns == "vector" || ns == "memref" || ns == "scf")
    return true;
  if (isa<math::ExpOp, math::LogOp, math::SqrtOp, math::RsqrtOp>(op))
    return true;
  return false;
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

  static std::string getArgMatchAttrName(unsigned argIndex, StringRef suffix) {
    return (Twine("pto.oplib.match.arg") + Twine(argIndex) + "." + suffix).str();
  }

  bool validateTemplateSignature(func::FuncOp fn, StringRef kind,
                                 SmallVectorImpl<TemplateArgRole> &argRoles,
                                 bool allowLoweredMemRefAbi = false) {
    FailureOr<SmallVector<TemplateArgRole, 4>> rolesOr =
        getTemplateArgRolesForKind(kind);
    if (failed(rolesOr))
      return false;

    argRoles.assign(rolesOr->begin(), rolesOr->end());
    FunctionType fnTy = fn.getFunctionType();
    if (fnTy.getNumInputs() != argRoles.size() || fnTy.getNumResults() != 0)
      return false;

    for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), argRoles)) {
      if (role == TemplateArgRole::Tile) {
        if (!isOplibTileLikeType(inTy, allowLoweredMemRefAbi))
          return false;
        continue;
      }
      if (!isOplibScalarType(inTy))
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
    if (entry->kind == kOpLibKindL3BinaryTemplate) {
      auto rowsOr =
          parseI64Attr(op, kOpLibAttrMatchRows, /*allowWildcard=*/true);
      auto colsOr =
          parseI64Attr(op, kOpLibAttrMatchCols, /*allowWildcard=*/true);
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

      MatchKey legacyMatch;
      legacyMatch.rows = *rowsOr;
      legacyMatch.cols = *colsOr;
      legacyMatch.fractal = *fractalOr;
      legacyMatch.blayout = *blayoutOr;
      legacyMatch.slayout = *slayoutOr;
      entry->argMatches.resize(entry->argRoles.size());
      for (unsigned i = 0; i < entry->argRoles.size(); ++i) {
        if (entry->argRoles[i] == TemplateArgRole::Tile)
          entry->argMatches[i] = legacyMatch;
      }
    } else {
      entry->argMatches.resize(entry->argRoles.size());
      for (unsigned i = 0; i < entry->argRoles.size(); ++i) {
        if (entry->argRoles[i] != TemplateArgRole::Tile) {
          bool hasUnexpectedAttr = op->hasAttr(getArgMatchAttrName(i, "rows")) ||
                                   op->hasAttr(getArgMatchAttrName(i, "cols")) ||
                                   op->hasAttr(getArgMatchAttrName(i, "blayout")) ||
                                   op->hasAttr(getArgMatchAttrName(i, "slayout")) ||
                                   op->hasAttr(getArgMatchAttrName(i, "fractal"));
          if (hasUnexpectedAttr) {
            imported.emitError() << "arg" << i
                                 << " is scalar in kind=" << entry->kind
                                 << ", but carries tile match attrs";
            return failure();
          }
          continue;
        }

        auto rowsOr =
            parseI64Attr(op, getArgMatchAttrName(i, "rows"),
                         /*allowWildcard=*/true);
        auto colsOr =
            parseI64Attr(op, getArgMatchAttrName(i, "cols"),
                         /*allowWildcard=*/true);
        auto fractalOr =
            parseI64Attr(op, getArgMatchAttrName(i, "fractal"),
                         /*allowWildcard=*/true);
        auto blayoutOr = parseStringAttr(op, getArgMatchAttrName(i, "blayout"));
        auto slayoutOr = parseStringAttr(op, getArgMatchAttrName(i, "slayout"));
        if (failed(rowsOr) || failed(colsOr) || failed(fractalOr) ||
            failed(blayoutOr) || failed(slayoutOr)) {
          imported.emitError() << "missing or invalid arg" << i
                               << " match attrs: rows/cols/blayout/slayout/fractal";
          return failure();
        }
        if (!isAllowedLayoutName(*blayoutOr) || !isAllowedLayoutName(*slayoutOr)) {
          imported.emitError() << "invalid layout value in arg" << i
                               << ".blayout/.slayout";
          return failure();
        }

        MatchKey argMatch;
        argMatch.rows = *rowsOr;
        argMatch.cols = *colsOr;
        argMatch.fractal = *fractalOr;
        argMatch.blayout = *blayoutOr;
        argMatch.slayout = *slayoutOr;
        entry->argMatches[i] = argMatch;
      }
    }

    auto costAttr = op->getAttrOfType<IntegerAttr>(kOpLibAttrCost);
    auto priorityAttr = op->getAttrOfType<IntegerAttr>(kOpLibAttrPriority);
    if (!costAttr || !priorityAttr) {
      imported.emitError("missing required attrs: pto.oplib.cost / "
                         "pto.oplib.priority");
      return failure();
    }

    entry->cost = costAttr.getInt();
    entry->priority = priorityAttr.getInt();

    if (auto syncAttr = op->getAttrOfType<BoolAttr>(kOpLibAttrSync))
      entry->sync = syncAttr.getValue();

    if (auto scalarPosAttr =
            op->getAttrOfType<IntegerAttr>(kOpLibAttrMatchScalarPos)) {
      int64_t scalarPos = scalarPosAttr.getInt();
      if (scalarPos < 0 || scalarPos >= static_cast<int64_t>(entry->argRoles.size()) ||
          entry->argRoles[scalarPos] != TemplateArgRole::Scalar) {
        imported.emitError("invalid pto.oplib.match.scalar_pos");
        return failure();
      }
      entry->scalarPos = scalarPos;
    }

    if (auto isBinaryAttr = op->getAttrOfType<BoolAttr>(kOpLibAttrMatchIsBinary)) {
      entry->isBinary = isBinaryAttr.getValue();
    } else if (kindRequiresIsBinary(entry->kind)) {
      imported.emitError("missing required attr: pto.oplib.match.is_binary");
      return failure();
    }

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

    if (!isAllowedDTypeName(*dtypeOr)) {
      imported.emitError() << "unsupported variant dtype: " << *dtypeOr;
      return failure();
    }

    std::string uniqueKey = entry->kind + "|" + *opOr + "|" + *variantIdOr;
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

    std::string uniqueSeedKey = entry->kind + "|" + *seedIdOr;
    if (!seedIds.insert(uniqueSeedKey).second) {
      imported.emitError() << "duplicate seed_id under kind=" << entry->kind
                           << ": " << *seedIdOr;
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
    for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), entry.argRoles)) {
      if (role == TemplateArgRole::Scalar) {
        if (!isOplibScalarType(inTy)) {
          return emitFailureWithCode(imported.getLoc(), kErrDType,
                                     "template scalar inputs must use builtin scalar types");
        }
        continue;
      }

      auto tileTy = dyn_cast<pto::TileBufType>(inTy);
      if (!tileTy) {
        return emitFailureWithCode(imported.getLoc(), kErrEmptyBody,
                                   "template tile inputs must be !pto.tile_buf");
      }
      Type elemTy = tileTy.getElementType();
      if (!isOplibTileElementType(elemTy)) {
        return emitFailureWithCode(imported.getLoc(), kErrDType,
                                   "SIMD template supports float/integer tile inputs only");
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
    std::string inferredExecMode;

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
        if (inferredExecMode.empty()) {
          auto execMode = op->getAttrOfType<StringAttr>(kSimdExecModeAttr);
          if (execMode)
            inferredExecMode = execMode.getValue().str();
        }
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
    // and drop op attrs. Keep function-local and enclosing-op fallback tokens
    // on imported OP-Lib regions so EmitC lowering can still recover vld/vst
    // tokens from ancestors after later rewrites.
    if (!inferredVldDist.empty() || !inferredVstDist.empty() ||
        !inferredExecMode.empty()) {
      auto *ctx = imported.getContext();
      if (!inferredVldDist.empty() && !imported->getAttr(kSimdVldDistAttr)) {
        imported->setAttr(kSimdVldDistAttr,
                          StringAttr::get(ctx, inferredVldDist));
      }
      if (!inferredVstDist.empty() && !imported->getAttr(kSimdVstDistAttr)) {
        imported->setAttr(kSimdVstDistAttr,
                          StringAttr::get(ctx, inferredVstDist));
      }
      if (!inferredExecMode.empty() && !imported->getAttr(kSimdExecModeAttr)) {
        imported->setAttr(kSimdExecModeAttr,
                          StringAttr::get(ctx, inferredExecMode));
      }
      imported.walk([&](Operation *op) {
        if (!inferredVldDist.empty() && !op->getAttr(kSimdVldDistAttr)) {
          op->setAttr(kSimdVldDistAttr,
                      StringAttr::get(ctx, inferredVldDist));
        }
        if (!inferredVstDist.empty() && !op->getAttr(kSimdVstDistAttr)) {
          op->setAttr(kSimdVstDistAttr,
                      StringAttr::get(ctx, inferredVstDist));
        }
        if (!inferredExecMode.empty() && !op->getAttr(kSimdExecModeAttr)) {
          op->setAttr(kSimdExecModeAttr,
                      StringAttr::get(ctx, inferredExecMode));
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

  LogicalResult registerImportedEntry(func::FuncOp imported, StringRef sourceTag,
                                      bool validateBody,
                                      bool allowLoweredMemRefAbi = false) {
    auto kindAttr = imported->getAttrOfType<StringAttr>(kOpLibAttrKind);
    if (!kindAttr) {
      imported.emitError("missing required attr: pto.oplib.kind");
      return failure();
    }

    auto roleAttr = imported->getAttrOfType<StringAttr>(kOpLibAttrEntryRole);
    if (!roleAttr) {
      imported.emitError("missing required attr: pto.oplib.entry_role");
      return failure();
    }

    auto entry = std::make_unique<TemplateEntry>();
    entry->symbol = imported;
    entry->kind = kindAttr.getValue().str();

    if (!validateTemplateSignature(imported, entry->kind, entry->argRoles,
                                   allowLoweredMemRefAbi)) {
      imported.emitError()
          << "invalid OP-Lib signature for kind=" << entry->kind;
      return failure();
    }

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

    if (validateBody && failed(validateTemplateBody(imported, *entry)))
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
      llvm::errs() << " symbol=@" << imported.getSymName();
      if (!sourceTag.empty())
        llvm::errs() << " source=" << sourceTag;
      llvm::errs() << "\n";
    }

    entries.push_back(std::move(entry));
    return success();
  }

  LogicalResult loadFromModuleEntries(Location loc, int &importedCount) {
    importedCount = 0;
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      auto kindAttr = fn->getAttrOfType<StringAttr>(kOpLibAttrKind);
      if (!kindAttr)
        continue;
      if (failed(getTemplateArgRolesForKind(kindAttr.getValue()))) {
        fn.emitError() << "unsupported pto.oplib.kind: " << kindAttr.getValue();
        return failure();
      }
      if (failed(registerImportedEntry(fn, "module", /*validateBody=*/false,
                                       /*allowLoweredMemRefAbi=*/true)))
        return failure();
      ++importedCount;
    }

    if (importedCount > 0)
      return success();

    (void)loc;
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
        if (!kindAttr)
          continue;
        SmallVector<TemplateArgRole, 4> argRoles;
        if (failed(getTemplateArgRolesForKind(kindAttr.getValue()))) {
          libFunc.emitError() << "unsupported pto.oplib.kind: "
                              << kindAttr.getValue();
          return failure();
        }
        if (!validateTemplateSignature(libFunc, kindAttr.getValue(), argRoles,
                                       /*allowLoweredMemRefAbi=*/false)) {
          libFunc.emitError()
              << "invalid OP-Lib signature for kind=" << kindAttr.getValue();
          return failure();
        }

        FailureOr<func::FuncOp> importedOr = cloneIntoModule(libFunc);
        if (failed(importedOr)) {
          libFunc.emitError("failed to import template function into module");
          return failure();
        }
        func::FuncOp imported = *importedOr;
        if (failed(registerImportedEntry(imported, path, /*validateBody=*/true,
                                         /*allowLoweredMemRefAbi=*/false)))
          return failure();
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

  static bool matchCommon(const TemplateEntry &entry, const MatchRequest &target) {
    if (entry.kind != target.kind)
      return false;
    if (entry.argRoles.size() != target.argTypes.size() ||
        entry.argMatches.size() != target.argTypes.size())
      return false;

    for (unsigned i = 0; i < entry.argRoles.size(); ++i) {
      if (entry.argRoles[i] == TemplateArgRole::Scalar)
        continue;
      if (!entry.argMatches[i] || !target.argMatches[i])
        return false;
      const MatchKey &pattern = *entry.argMatches[i];
      const MatchKey &request = *target.argMatches[i];
      if (!matchDim(pattern.rows, request.rows))
        return false;
      if (!matchDim(pattern.cols, request.cols))
        return false;
      if (!matchDim(pattern.fractal, request.fractal))
        return false;
      if (!matchLayout(pattern.blayout, request.blayout))
        return false;
      if (!matchLayout(pattern.slayout, request.slayout))
        return false;
    }

    if (entry.scalarPos &&
        (!target.scalarPos || *entry.scalarPos != *target.scalarPos))
      return false;
    if (entry.isBinary &&
        (!target.isBinary || *entry.isBinary != *target.isBinary))
      return false;
    return true;
  }

  FailureOr<SelectedVariant> selectVariantFor(const MatchRequest &request,
                                              Location loc) {
    SmallVector<SelectedVariant, 16> candidates;
    for (std::unique_ptr<TemplateEntry> &entryPtr : entries) {
      TemplateEntry &entry = *entryPtr;
      if (!matchCommon(entry, request))
        continue;

      if (entry.role == EntryRole::Variant) {
        if (entry.op != request.op)
          continue;
        if (entry.matchDType != request.dtype)
          continue;
        if (request.requiredVariantId &&
            entry.variantId != *request.requiredVariantId)
          continue;

        SelectedVariant selected;
        selected.entry = &entry;
        selected.kind = entry.kind;
        selected.variantId = entry.variantId;
        selected.op = entry.op;
        selected.dtype = entry.matchDType;
        selected.fromSeed = false;
        selected.coreSlot = entry.coreSlot;
        selected.cost = entry.cost;
        selected.priority = entry.priority;
        if (request.scalarPos)
          selected.instanceKeyAttrs.push_back("scalar_pos=" +
                                             std::to_string(*request.scalarPos));
        if (request.isBinary)
          selected.instanceKeyAttrs.push_back("is_binary=" +
                                             std::string(*request.isBinary ? "true"
                                                                           : "false"));
        candidates.push_back(std::move(selected));
        continue;
      }

      if (request.requiredVariantId)
        continue;
      if (!canUseSeedRewriteFor(entry.kind, request.op))
        continue;
      if (!containsString(entry.supportOps, request.op))
        continue;
      if (!containsString(entry.supportDTypes, request.dtype))
        continue;

      std::string variantId =
          "__seed__" + entry.seedId + "__" + request.op + "__" + request.dtype;
      SelectedVariant selected;
      selected.entry = &entry;
      selected.kind = entry.kind;
      selected.variantId = variantId;
      selected.op = request.op;
      selected.dtype = request.dtype;
      selected.fromSeed = true;
      selected.seedId = entry.seedId;
      selected.coreSlot = entry.coreSlot;
      selected.cost = entry.cost;
      selected.priority = entry.priority;
      if (request.scalarPos)
        selected.instanceKeyAttrs.push_back("scalar_pos=" +
                                           std::to_string(*request.scalarPos));
      if (request.isBinary)
        selected.instanceKeyAttrs.push_back("is_binary=" +
                                           std::string(*request.isBinary ? "true"
                                                                         : "false"));
      candidates.push_back(std::move(selected));
    }

    if (candidates.empty()) {
      (void)emitFailure(loc, Twine("no matching OP-Lib entry for kind=") +
                                 request.kind + " op=" + request.op +
                                 " dtype=" + request.dtype);
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
      llvm::errs() << "[op-fusion] selected variant: kind=" << best.kind
                   << " op=" << best.op
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
    std::string key = selected.kind;
    key += "|";
    key += selected.op;
    key += "|";
    key += selected.variantId;
    for (const std::string &piece : selected.instanceKeyAttrs) {
      key += "|";
      key += piece;
    }
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

    std::string symBase = "__pto_oplib_inst_" +
                          sanitizeSymbolComponent(selected.kind) + "_" +
                          sanitizeSymbolComponent(selected.op) + "_" +
                          sanitizeSymbolComponent(selected.variantId);
    std::string sym = symBase;
    int suffix = 0;
    while (symbolTable.lookup(sym))
      sym = symBase + "_" + std::to_string(++suffix);

    OpBuilder modBuilder(module.getContext());
    modBuilder.setInsertionPointToStart(module.getBody());
    auto inst = modBuilder.create<func::FuncOp>(
        loc, sym, FunctionType::get(module.getContext(), argTypes, {}));
    inst.setPrivate();
    inst->setAttr(kOpLibAttrInstKind,
                  StringAttr::get(module.getContext(), selected.kind));
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
    if (auto vldAttr =
            selected.entry->symbol->getAttrOfType<StringAttr>(kSimdVldDistAttr)) {
      inst->setAttr(kSimdVldDistAttr, vldAttr);
    }
    if (auto vstAttr =
            selected.entry->symbol->getAttrOfType<StringAttr>(kSimdVstDistAttr)) {
      inst->setAttr(kSimdVstDistAttr, vstAttr);
    }
    if (auto execModeAttr = selected.entry->symbol->getAttrOfType<StringAttr>(
            kSimdExecModeAttr)) {
      inst->setAttr(kSimdExecModeAttr, execModeAttr);
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
        if (!mappedSrc) {
          (void)emitFailureWithCode(
              loc, kErrCoreSlot,
              Twine("failed to remap simd.tile_to_memref source in instance @") +
                  inst.getSymName());
          return failure();
        }

        if (auto mappedTileTy = dyn_cast<pto::TileBufType>(mappedSrc.getType())) {
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

        auto mappedMemTy = dyn_cast<MemRefType>(mappedSrc.getType());
        auto dstMemTy = dyn_cast<MemRefType>(bridge.getDst().getType());
        if (!mappedMemTy || !dstMemTy) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("unsupported simd.tile_to_memref remap in instance @") +
                  inst.getSymName());
          return failure();
        }

        if (mappedMemTy.getRank() != dstMemTy.getRank() ||
            mappedMemTy.getElementType() != dstMemTy.getElementType()) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("incompatible memref remap for simd.tile_to_memref in instance @") +
                  inst.getSymName());
          return failure();
        }

        // Keep simd.tile_to_memref as a backend marker even in memref-world.
        // Use the concrete mapped memref type as the bridge result to avoid
        // re-introducing unrealized casts for shape-only differences.
        auto newBridge = bodyBuilder.create<pto::SimdTileToMemrefOp>(
            bridge.getLoc(), mappedMemTy, mappedSrc);
        mapping.map(bridge.getDst(), newBridge.getDst());
        continue;
      }

      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(&op)) {
        if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("unsupported unrealized_conversion_cast in instance source @") +
                  source.getSymName());
          return failure();
        }

        Value mappedSrc = mapping.lookupOrNull(cast.getOperand(0));
        if (!mappedSrc) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("failed to remap unrealized_conversion_cast source in instance @") +
                  inst.getSymName());
          return failure();
        }

        // Source templates may contain transient conversion casts introduced by
        // earlier legalization. Treat them as transparent while cloning
        // instance bodies so concrete memref signatures do not reintroduce
        // unresolved cast chains.
        mapping.map(cast.getResult(0), mappedSrc);
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
      if (selected.op == "tadd" || selected.op == "tpartadd" ||
          selected.op == "tadds") {
        newCore = b.create<arith::AddFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tsub" || selected.op == "tsubs") {
        newCore = b.create<arith::SubFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmul" || selected.op == "tmuls") {
        newCore = b.create<arith::MulFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tdiv") {
        newCore = b.create<arith::DivFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmax" || selected.op == "tpartmax" ||
                 selected.op == "tmaxs") {
        newCore = b.create<arith::MaximumFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmin" || selected.op == "tpartmin" ||
                 selected.op == "tmins") {
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
  auto cfg = dstTy.getConfigAttr();
  if (!cfg)
    cfg = pto::TileBufConfigAttr::getDefault(dstTy.getContext());
  key.blayout = toBLayoutName(dstTy.getBLayoutValueI32());
  key.slayout = toSLayoutName(dstTy.getSLayoutValueI32());
  key.fractal = getFractalOrWildcard(cfg);
  return key;
}

static FailureOr<MatchKey> buildMatchKeyFromMemRef(MemRefType dstTy, Value dst) {
  if (dstTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = dstTy.getShape();
  MatchKey key;
  key.rows = shape[0] == ShapedType::kDynamic ? -1 : shape[0];
  key.cols = shape[1] == ShapedType::kDynamic ? -1 : shape[1];

  auto cfg = lookupTileConfigForValue(dst, dstTy.getContext());
  int32_t bl = 0;
  int32_t sl = 0;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  key.blayout = toBLayoutName(bl);
  key.slayout = toSLayoutName(sl);
  key.fractal = getFractalOrWildcard(cfg);
  return key;
}

static FailureOr<std::pair<Type, MatchKey>>
getTileOperandInfo(Value operand) {
  Type ty = operand.getType();
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    if (tileTy.getRank() != 2 || !isOplibTileElementType(tileTy.getElementType()))
      return failure();
    FailureOr<MatchKey> keyOr = buildMatchKeyFromTileBuf(tileTy);
    if (failed(keyOr))
      return failure();
    return std::make_pair(tileTy.getElementType(), *keyOr);
  }
  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    if (memTy.getRank() != 2 || !isOplibTileElementType(memTy.getElementType()))
      return failure();
    FailureOr<MatchKey> keyOr = buildMatchKeyFromMemRef(memTy, operand);
    if (failed(keyOr))
      return failure();
    return std::make_pair(memTy.getElementType(), *keyOr);
  }
  return failure();
}

static LogicalResult addTileOperandToRequest(MatchRequest &request, Value operand,
                                             std::optional<Type> &elemTy) {
  FailureOr<std::pair<Type, MatchKey>> infoOr = getTileOperandInfo(operand);
  if (failed(infoOr))
    return failure();
  if (elemTy && *elemTy != infoOr->first)
    return failure();
  elemTy = infoOr->first;
  request.argTypes.push_back(operand.getType());
  request.operands.push_back(operand);
  request.argMatches.push_back(infoOr->second);
  return success();
}

static LogicalResult addScalarOperandToRequest(MatchRequest &request, Value operand,
                                               std::optional<Type> elemTy) {
  Type ty = operand.getType();
  if (!isOplibScalarType(ty))
    return failure();
  if (elemTy && ty != *elemTy)
    return failure();
  request.argTypes.push_back(ty);
  request.operands.push_back(operand);
  request.argMatches.push_back(std::nullopt);
  return success();
}

static FailureOr<MatchRequest>
buildBinaryTileMatchRequest(StringRef kind, StringRef opName, Value src0,
                            Value src1, Value dst) {
  MatchRequest request;
  request.kind = kind.str();
  request.op = opName.str();

  std::optional<Type> elemTy;
  if (failed(addTileOperandToRequest(request, src0, elemTy)) ||
      failed(addTileOperandToRequest(request, src1, elemTy)) ||
      failed(addTileOperandToRequest(request, dst, elemTy)) || !elemTy) {
    return failure();
  }

  request.dtype = dtypeToString(*elemTy);
  return request;
}

static FailureOr<MatchRequest>
buildUnaryTileMatchRequest(StringRef kind, StringRef opName, Value src,
                           Value dst) {
  MatchRequest request;
  request.kind = kind.str();
  request.op = opName.str();

  std::optional<Type> elemTy;
  if (failed(addTileOperandToRequest(request, src, elemTy)) ||
      failed(addTileOperandToRequest(request, dst, elemTy)) || !elemTy) {
    return failure();
  }

  request.dtype = dtypeToString(*elemTy);
  return request;
}

static FailureOr<MatchRequest>
buildTileScalarMatchRequest(StringRef kind, StringRef opName, Value src,
                            Value scalar, Value dst,
                            std::optional<StringRef> requiredVariantId = std::nullopt) {
  MatchRequest request;
  request.kind = kind.str();
  request.op = opName.str();
  request.scalarPos = 1;
  if (requiredVariantId)
    request.requiredVariantId = requiredVariantId->str();

  std::optional<Type> elemTy;
  if (failed(addTileOperandToRequest(request, src, elemTy)) ||
      failed(addScalarOperandToRequest(request, scalar, elemTy)) ||
      failed(addTileOperandToRequest(request, dst, elemTy)) || !elemTy) {
    return failure();
  }

  request.dtype = dtypeToString(*elemTy);
  return request;
}

static FailureOr<MatchRequest>
buildTernaryTileMatchRequest(StringRef kind, StringRef opName, Value src0,
                             Value src1, Value src2, Value dst) {
  MatchRequest request;
  request.kind = kind.str();
  request.op = opName.str();

  std::optional<Type> elemTy;
  if (failed(addTileOperandToRequest(request, src0, elemTy)) ||
      failed(addTileOperandToRequest(request, src1, elemTy)) ||
      failed(addTileOperandToRequest(request, src2, elemTy)) ||
      failed(addTileOperandToRequest(request, dst, elemTy)) || !elemTy) {
    return failure();
  }

  request.dtype = dtypeToString(*elemTy);
  return request;
}

static FailureOr<MatchRequest>
buildTernaryTileScalarMatchRequest(StringRef kind, StringRef opName, Value src0,
                                   Value scalar, Value src1, Value dst) {
  MatchRequest request;
  request.kind = kind.str();
  request.op = opName.str();
  request.scalarPos = 1;

  std::optional<Type> elemTy;
  if (failed(addTileOperandToRequest(request, src0, elemTy)) ||
      failed(addScalarOperandToRequest(request, scalar, elemTy)) ||
      failed(addTileOperandToRequest(request, src1, elemTy)) ||
      failed(addTileOperandToRequest(request, dst, elemTy)) || !elemTy) {
    return failure();
  }

  request.dtype = dtypeToString(*elemTy);
  return request;
}

static FailureOr<MatchRequest>
buildMatchRequest(Operation *op) {
  if (auto mul = dyn_cast<pto::TMulOp>(op))
    return buildBinaryTileMatchRequest("l3_float_binary_elementwise_template",
                                       "tmul", mul.getSrc0(), mul.getSrc1(),
                                       mul.getDst());
  if (auto div = dyn_cast<pto::TDivOp>(op))
    return buildBinaryTileMatchRequest("l3_float_binary_elementwise_template",
                                       "tdiv", div.getSrc0(), div.getSrc1(),
                                       div.getDst());
  if (auto add = dyn_cast<pto::TAddOp>(op))
    return buildBinaryTileMatchRequest("l3_float_binary_elementwise_template",
                                       "tadd", add.getSrc0(), add.getSrc1(),
                                       add.getDst());
  if (auto sub = dyn_cast<pto::TSubOp>(op))
    return buildBinaryTileMatchRequest("l3_float_binary_elementwise_template",
                                       "tsub", sub.getSrc0(), sub.getSrc1(),
                                       sub.getDst());
  if (auto max = dyn_cast<pto::TMaxOp>(op))
    return buildBinaryTileMatchRequest("l3_float_binary_elementwise_template",
                                       "tmax", max.getSrc0(), max.getSrc1(),
                                       max.getDst());
  if (auto min = dyn_cast<pto::TMinOp>(op))
    return buildBinaryTileMatchRequest("l3_float_binary_elementwise_template",
                                       "tmin", min.getSrc0(), min.getSrc1(),
                                       min.getDst());
  if (auto partAdd = dyn_cast<pto::TPartAddOp>(op))
    return buildBinaryTileMatchRequest("l3_float_partial_binary_template",
                                       "tpartadd", partAdd.getSrc0(),
                                       partAdd.getSrc1(), partAdd.getDst());
  if (auto partMax = dyn_cast<pto::TPartMaxOp>(op))
    return buildBinaryTileMatchRequest("l3_float_partial_binary_template",
                                       "tpartmax", partMax.getSrc0(),
                                       partMax.getSrc1(), partMax.getDst());
  if (auto partMin = dyn_cast<pto::TPartMinOp>(op))
    return buildBinaryTileMatchRequest("l3_float_partial_binary_template",
                                       "tpartmin", partMin.getSrc0(),
                                       partMin.getSrc1(), partMin.getDst());

  if (auto adds = dyn_cast<pto::TAddSOp>(op))
    return buildTileScalarMatchRequest("l3_float_tile_scalar_template",
                                       "tadds", adds.getSrc(),
                                       adds.getScalar(), adds.getDst());
  if (auto subs = dyn_cast<pto::TSubSOp>(op))
    return buildTileScalarMatchRequest("l3_float_tile_scalar_template",
                                       "tsubs", subs.getSrc(),
                                       subs.getScalar(), subs.getDst());
  if (auto muls = dyn_cast<pto::TMulSOp>(op))
    return buildTileScalarMatchRequest("l3_float_tile_scalar_template",
                                       "tmuls", muls.getSrc0(),
                                       muls.getScalar(), muls.getDst());
  if (auto maxs = dyn_cast<pto::TMaxSOp>(op))
    return buildTileScalarMatchRequest("l3_float_tile_scalar_template",
                                       "tmaxs", maxs.getSrc(),
                                       maxs.getScalar(), maxs.getDst());
  if (auto mins = dyn_cast<pto::TMinSOp>(op))
    return buildTileScalarMatchRequest("l3_float_tile_scalar_template",
                                       "tmins", mins.getSrc(),
                                       mins.getScalar(), mins.getDst());
  if (auto divs = dyn_cast<pto::TDivSOp>(op)) {
    auto orderAttr =
        divs->getAttrOfType<StringAttr>("pto.tdivs.order");
    if (!orderAttr)
      return failure();
    StringRef order = orderAttr.getValue();
    if (order != "tile_scalar" && order != "scalar_tile")
      return failure();
    return buildTileScalarMatchRequest("l3_float_tile_scalar_template",
                                       "tdivs", divs.getSrc(),
                                       divs.getScalar(), divs.getDst(), order);
  }

  if (auto addc = dyn_cast<pto::TAddCOp>(op))
    return buildTernaryTileMatchRequest("l3_float_ternary_tile_template",
                                        "taddc", addc.getSrc0(),
                                        addc.getSrc1(), addc.getSrc2(),
                                        addc.getDst());
  if (auto subc = dyn_cast<pto::TSubCOp>(op))
    return buildTernaryTileMatchRequest("l3_float_ternary_tile_template",
                                        "tsubc", subc.getSrc0(),
                                        subc.getSrc1(), subc.getSrc2(),
                                        subc.getDst());

  if (auto addsc = dyn_cast<pto::TAddSCOp>(op))
    return buildTernaryTileScalarMatchRequest(
        "l3_float_ternary_tile_scalar_template", "taddsc", addsc.getSrc0(),
        addsc.getScalar(), addsc.getSrc1(), addsc.getDst());
  if (auto subsc = dyn_cast<pto::TSubSCOp>(op))
    return buildTernaryTileScalarMatchRequest(
        "l3_float_ternary_tile_scalar_template", "tsubsc", subsc.getSrc0(),
        subsc.getScalar(), subsc.getSrc1(), subsc.getDst());

  if (auto abs = dyn_cast<pto::TAbsOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_template", "tabs",
                                      abs.getSrc(), abs.getDst());
  if (auto neg = dyn_cast<pto::TNegOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_template", "tneg",
                                      neg.getSrc(), neg.getDst());
  if (auto recip = dyn_cast<pto::TRecipOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_template", "trecip",
                                      recip.getSrc(), recip.getDst());
  if (auto relu = dyn_cast<pto::TReluOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_template", "trelu",
                                      relu.getSrc(), relu.getDst());

  if (auto exp = dyn_cast<pto::TExpOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_math_template", "texp",
                                      exp.getSrc(), exp.getDst());
  if (auto log = dyn_cast<pto::TLogOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_math_template", "tlog",
                                      log.getSrc(), log.getDst());
  if (auto sqrt = dyn_cast<pto::TSqrtOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_math_template", "tsqrt",
                                      sqrt.getSrc(), sqrt.getDst());
  if (auto rsqrt = dyn_cast<pto::TRsqrtOp>(op))
    return buildUnaryTileMatchRequest("l3_float_unary_math_template",
                                      "trsqrt", rsqrt.getSrc(),
                                      rsqrt.getDst());

  return failure();
}

static FailureOr<PlannedOpLowering>
planOneOpLowering(Operation *op, TemplateRegistry &registry,
                  StringRef warningPrefix) {
  FailureOr<MatchRequest> requestOr = buildMatchRequest(op);
  if (failed(requestOr)) {
    op->emitWarning() << warningPrefix
                      << ": unsupported operand signature or missing match metadata";
    return failure();
  }

  const MatchRequest &request = *requestOr;
  FailureOr<SelectedVariant> selectedOr =
      registry.selectVariantFor(request, op->getLoc());
  if (failed(selectedOr)) {
    op->emitWarning() << warningPrefix << ": no OP-Lib candidate for op="
                      << request.op << " dtype=" << request.dtype;
    return failure();
  }

  FailureOr<func::FuncOp> instanceOr =
      registry.getOrCreateInstance(*selectedOr, request.argTypes, op->getLoc());
  if (failed(instanceOr)) {
    op->emitWarning() << warningPrefix
                      << ": failed to instantiate OP-Lib candidate for op="
                      << request.op << " dtype=" << request.dtype;
    return failure();
  }

  PlannedOpLowering planned;
  planned.op = op;
  planned.selected = *selectedOr;
  planned.operands.assign(request.operands.begin(), request.operands.end());
  planned.instance = *instanceOr;
  return planned;
}

static LogicalResult rewriteOneGroupedOpAsCall(const PlannedOpLowering &planned) {
  Operation *op = planned.op;
  OpBuilder builder(op);
  auto call =
      builder.create<func::CallOp>(op->getLoc(), planned.instance, planned.operands);

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

      OpBuilder builder(op);
      builder.create<func::CallOp>(op->getLoc(), plannedOr->instance,
                                   plannedOr->operands);
      op->erase();

      if (debug) {
        llvm::errs() << "[op-fusion] materialized single op="
                     << plannedOr->selected.op
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
    int importedFromModule = 0;
    if (failed(registry.loadFromModuleEntries(module.getLoc(), importedFromModule))) {
      signalPassFailure();
      return;
    }
    if (importedFromModule == 0) {
      if (failed(registry.loadFromDir(opLibDir, ctx, module.getLoc()))) {
        signalPassFailure();
        return;
      }
    } else if (debug) {
      llvm::errs() << "[op-fusion] using " << importedFromModule
                   << " pre-imported OP-Lib entries from module\n";
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func->hasAttr(kOpLibAttrKind) || func->hasAttr(kOpLibAttrInstVariantId))
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

LogicalResult mlir::pto::importPTOOpLibTemplates(ModuleOp module,
                                                 StringRef opLibDir,
                                                 bool debug) {
  SymbolTable symbolTable(module);
  TemplateRegistry registry{module, symbolTable, debug};
  return registry.loadFromDir(opLibDir, module.getContext(), module.getLoc());
}
