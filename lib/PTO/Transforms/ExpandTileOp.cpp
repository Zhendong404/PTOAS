// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- ExpandTileOp.cpp ---------------------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Expand tile-level ops (pto.tadd, pto.tsub, ...) by invoking the TileLang
// Python DSL to instantiate template libraries.  This pass is the PTO -> VPTO
// boundary: local tile operands must arrive as native !pto.tile_buf values.
// Memref values must not cross into this pass; even GM tensor views should
// still be represented as tensor_view / partition_tensor_view descriptors here.
//
// The generated template functions use tile_buf parameters. After this pass,
// the Inline pass inlines the template body, and FoldTileBufIntrinsics
// resolves tile_buf_addr / tile_valid_rows / tile_valid_cols.
//
// Workflow per tile op:
//   1. Extract SpecKey from ALL operands' tile_buf types.
//   2. Invoke Python DSL helper to generate a specialized MLIR function
//      (with tile_buf parameters).
//   3. Parse the generated MLIR and clone the function into the module.
//   4. Replace the original tile op with func.call, passing tile_buf operands
//      directly; only tensor-view operands may need helper-call type bridging.
//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <unistd.h>

extern "C" {
extern char **environ;
}

using namespace mlir;

namespace mlir {
namespace pto {
  namespace func = ::mlir::func;

  #define GEN_PASS_DEF_EXPANDTILEOP
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

// ============================================================================
// OperandTypeInfo: describes one operand for template specialization.
//
// Three kinds of operands:
//   Tile   — from TileBufType.  dtype + shape + memorySpace + config
//            all participate in the specialization key (SpecKey).
//   View   — from TensorViewType / PartitionTensorViewType. Only dtype
//            participates in SpecKey — the template is fully dynamic so
//            shape/strides don't affect code generation. They are
//            carried here solely for JSON serialization to the Python DSL for
//            constraint checking.
//   Scalar — from a scalar element type.  Only dtype participates in SpecKey.
// ============================================================================
enum class OperandKind { Tile, View, Scalar };

struct OperandTypeInfo {
  OperandKind kind = OperandKind::Tile;
  std::string dtype; // all kinds: element type string (e.g. "f32")

  // --- Tile-only (TileBufType) ---
  SmallVector<int64_t, 2> tileShape;
  SmallVector<int64_t, 2> tileValidShape;
  std::string tileMemorySpace; // "ub" or "gm"
  int32_t blayout = 0;
  int32_t slayout = 0;
  int32_t fractal = 0;
  uint64_t pad = 0;

  // --- View-only (TensorViewLike) — for JSON / constraint checking only ---
  SmallVector<int64_t> viewShape;

  /// Equality for SpecKey caching — only compares fields relevant to each kind.
  bool operator==(const OperandTypeInfo &rhs) const {
    if (kind != rhs.kind || dtype != rhs.dtype)
      return false;
    if (kind == OperandKind::Tile)
      return tileShape == rhs.tileShape &&
             tileValidShape == rhs.tileValidShape &&
             tileMemorySpace == rhs.tileMemorySpace &&
             blayout == rhs.blayout && slayout == rhs.slayout &&
             fractal == rhs.fractal && pad == rhs.pad;
    // View and Scalar: dtype alone is sufficient for template caching.
    return true;
  }
};

// ============================================================================
// SpecKey: identifies a specialized template instance using ALL operands.
// ============================================================================
struct SpecKey {
  std::string opName;
  std::string targetArch;
  SmallVector<OperandTypeInfo, 4> operands;
  SmallVector<std::pair<std::string, std::string>, 4> contextAttrs;

  bool operator==(const SpecKey &rhs) const {
    return opName == rhs.opName && targetArch == rhs.targetArch &&
           operands == rhs.operands && contextAttrs == rhs.contextAttrs;
  }
};

struct SpecKeyInfo : public llvm::DenseMapInfo<SpecKey> {
  static inline SpecKey getEmptyKey() { return {"", "", {}}; }
  static inline SpecKey getTombstoneKey() { return {"__tombstone__", "", {}}; }
  static unsigned getHashValue(const SpecKey &key) {
    unsigned h = llvm::hash_combine(key.opName, key.targetArch);
    for (const auto &op : key.operands) {
      h = llvm::hash_combine(h, static_cast<int>(op.kind), op.dtype);
      if (op.kind == OperandKind::Tile) {
        h = llvm::hash_combine(h, op.tileMemorySpace, op.blayout,
                               op.slayout, op.fractal, op.pad);
        for (int64_t d : op.tileShape)
          h = llvm::hash_combine(h, d);
        for (int64_t d : op.tileValidShape)
          h = llvm::hash_combine(h, d);
      }
      // View/Scalar: only kind + dtype contribute to hash.
    }
    for (const auto &[attrName, attrValue] : key.contextAttrs)
      h = llvm::hash_combine(h, attrName, attrValue);
    return h;
  }
  static bool isEqual(const SpecKey &lhs, const SpecKey &rhs) {
    return lhs == rhs;
  }
};

// ============================================================================
// Helpers
// ============================================================================
static std::string getDtypeString(Type elemTy) {
  if (elemTy.isInteger(1)) return "i1";
  if (elemTy.isF32()) return "f32";
  if (elemTy.isF16()) return "f16";
  if (elemTy.isBF16()) return "bf16";
  if (elemTy.isUnsignedInteger(64)) return "ui64";
  if (elemTy.isUnsignedInteger(32)) return "ui32";
  if (elemTy.isUnsignedInteger(16)) return "ui16";
  if (elemTy.isUnsignedInteger(8)) return "ui8";
  if (elemTy.isSignedInteger(64)) return "si64";
  if (elemTy.isSignedInteger(32)) return "si32";
  if (elemTy.isSignedInteger(16)) return "si16";
  if (elemTy.isSignedInteger(8)) return "si8";
  if (elemTy.isSignlessInteger(64)) return "i64";
  if (elemTy.isSignlessInteger(32)) return "i32";
  if (elemTy.isSignlessInteger(16)) return "i16";
  if (elemTy.isSignlessInteger(8)) return "i8";
  return "";
}

static StringRef getTileOpName(Operation *op) {
  return op->getName().stripDialect();
}

static std::string getTargetArchString(ModuleOp mod) {
  if (!mod)
    return "";
  auto targetAttr = mod->getAttrOfType<StringAttr>("pto.target_arch");
  if (!targetAttr)
    return "";
  return targetAttr.getValue().str();
}

static std::string getMemorySpaceString(pto::TileBufType tbTy) {
  auto msAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
  if (!msAttr) return "ub";
  if (msAttr.getAddressSpace() == pto::AddressSpace::GM) return "gm";
  return "ub";
}

static LogicalResult validateBoundaryOperandTypes(Operation *op) {
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    auto memrefTy = dyn_cast<MemRefType>(operand.getType());
    if (!memrefTy)
      continue;

    return op->emitError()
           << "ExpandTileOp: memref operand #" << index
           << " violates the PTO -> VPTO boundary contract; operands must "
              "reach ExpandTileOp as native PTO descriptors such as "
              "!pto.tile_buf, !pto.tensor_view, or "
              "!pto.partition_tensor_view, got "
           << memrefTy
           << ". Memref values are only allowed after ExpandTileOp has entered "
              "VPTO authoring IR.";
  }
  return success();
}

static std::string getBLayoutString(int32_t blayout) {
  if (blayout == static_cast<int32_t>(pto::BLayout::ColMajor))
    return "col_major";
  return "row_major";
}

static std::string getSLayoutString(int32_t slayout) {
  if (slayout == static_cast<int32_t>(pto::SLayout::RowMajor))
    return "row_major";
  if (slayout == static_cast<int32_t>(pto::SLayout::ColMajor))
    return "col_major";
  return "none_box";
}

static std::optional<std::string> getTCvtRoundModeString(pto::TCvtOp op) {
  switch (op.getRmode()) {
  case pto::RoundMode::NONE:
  case pto::RoundMode::RINT:
  case pto::RoundMode::CAST_RINT:
    return "RINT";
  case pto::RoundMode::ROUND:
    return "ROUND";
  case pto::RoundMode::FLOOR:
    return "FLOOR";
  case pto::RoundMode::CEIL:
    return "CEIL";
  case pto::RoundMode::TRUNC:
    return "TRUNC";
  case pto::RoundMode::ODD:
    return "ODD";
  }
  return std::nullopt;
}

static std::string getTRandomRoundsString(pto::TRandomOp op) {
  return std::to_string(op.getRounds());
}

static void appendOpContextAttrs(
    Operation *op,
    SmallVectorImpl<std::pair<std::string, std::string>> &attrs) {
  if (auto tcvt = dyn_cast<pto::TCvtOp>(op)) {
    std::optional<std::string> roundMode = getTCvtRoundModeString(tcvt);
    if (roundMode)
      attrs.emplace_back("round_mode", *roundMode);
  }
  if (auto trandom = dyn_cast<pto::TRandomOp>(op))
    attrs.emplace_back("rounds", getTRandomRoundsString(trandom));
  if (auto tcmp = dyn_cast<pto::TCmpOp>(op)) {
    if (auto cmpModeAttr = tcmp.getCmpModeAttr()) {
      attrs.emplace_back("cmp_mode",
                         stringifyCmpMode(cmpModeAttr.getValue()).str());
    }
  }
  if (auto tcmps = dyn_cast<pto::TCmpSOp>(op)) {
    if (auto cmpModeAttr = tcmps.getCmpModeAttr()) {
      attrs.emplace_back("cmp_mode",
                         stringifyCmpMode(cmpModeAttr.getValue()).str());
    }
  }
}

static std::optional<OperandTypeInfo> buildOperandTypeInfo(Value value) {
  Type ty = value.getType();
  // Tile operand — from TileBufType.
  if (auto tbTy = dyn_cast<pto::TileBufType>(ty)) {
    OperandTypeInfo info;
    info.kind = OperandKind::Tile;
    info.dtype = getDtypeString(tbTy.getElementType());
    if (info.dtype.empty())
      return std::nullopt;
    info.tileShape.assign(tbTy.getShape().begin(), tbTy.getShape().end());
    auto validShape = tbTy.getValidShape();
    if (validShape.empty())
      info.tileValidShape.assign(tbTy.getShape().begin(), tbTy.getShape().end());
    else
      info.tileValidShape.assign(validShape.begin(), validShape.end());
    info.tileMemorySpace = getMemorySpaceString(tbTy);
    if (auto config = tbTy.getConfigAttr()) {
      info.blayout = static_cast<int32_t>(config.getBLayout().getValue());
      info.slayout = static_cast<int32_t>(config.getSLayout().getValue());
      info.fractal = config.getSFractalSize()
                         ? static_cast<int32_t>(config.getSFractalSize().getInt())
                         : 0;
      info.pad = static_cast<uint64_t>(config.getPad().getValue());
    }
    return info;
  }

  // View operand — native tensor_view / partition_tensor_view before the
  // PTO -> VPTO boundary.
  if (auto tvTy = dyn_cast<pto::TensorViewType>(ty)) {
    OperandTypeInfo info;
    info.kind = OperandKind::View;
    info.dtype = getDtypeString(tvTy.getElementType());
    if (info.dtype.empty())
      return std::nullopt;
    info.viewShape.assign(tvTy.getShape().begin(), tvTy.getShape().end());
    return info;
  }

  if (auto partTy = dyn_cast<pto::PartitionTensorViewType>(ty)) {
    OperandTypeInfo info;
    info.kind = OperandKind::View;
    info.dtype = getDtypeString(partTy.getElementType());
    if (info.dtype.empty())
      return std::nullopt;
    info.viewShape.assign(partTy.getShape().begin(), partTy.getShape().end());
    return info;
  }

  // Scalar operand — from a scalar element type.
  OperandTypeInfo info;
  info.kind = OperandKind::Scalar;
  info.dtype = getDtypeString(ty);
  if (info.dtype.empty())
    return std::nullopt;
  return info;
}

static std::optional<SpecKey> buildSpecKey(Operation *op) {
  SpecKey key;
  key.opName = getTileOpName(op).str();
  key.targetArch = getTargetArchString(op->getParentOfType<ModuleOp>());

  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    auto info = buildOperandTypeInfo(op->getOperand(i));
    if (!info)
      return std::nullopt;
    key.operands.push_back(*info);
  }
  if (key.operands.empty())
    return std::nullopt;

  appendOpContextAttrs(op, key.contextAttrs);
  return key;
}

// ============================================================================
// ExpandState: runtime state for a single pass invocation.
// ============================================================================
struct ExpandState {
  std::vector<OwningOpRef<ModuleOp>> parsedModules;
  llvm::DenseMap<SpecKey, func::FuncOp, SpecKeyInfo> specCache;

  std::string tilelangPath;
  std::string tilelangPkgPath;
  std::string pythonExe;

  func::FuncOp invokeTilelangDSL(const SpecKey &key, Operation *tileOp,
                                  ModuleOp mod, MLIRContext *ctx);

  LogicalResult expandTileOpsInFunction(func::FuncOp func, ModuleOp mod,
                                        MLIRContext *ctx);
};

// ============================================================================
// The Pass
// ============================================================================
struct ExpandTileOpPass
    : public mlir::pto::impl::ExpandTileOpBase<ExpandTileOpPass> {
  using ExpandTileOpBase::ExpandTileOpBase;

  void runOnOperation() override;
};

/// Serialize a JSON array of integers.
static void appendJsonIntArray(std::string &json, ArrayRef<int64_t> arr) {
  json += "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i > 0)
      json += ",";
    json += std::to_string(arr[i]);
  }
  json += "]";
}

/// Serialize a JSON array where dynamic dimensions become `null`.
static void appendJsonDimArray(std::string &json, ArrayRef<int64_t> arr,
                               bool negativeIsDynamic = false) {
  json += "[";
  for (size_t i = 0; i < arr.size(); ++i) {
    if (i > 0)
      json += ",";
    int64_t dim = arr[i];
    if (ShapedType::isDynamic(dim) || (negativeIsDynamic && dim < 0)) {
      json += "null";
      continue;
    }
    json += std::to_string(dim);
  }
  json += "]";
}

static std::string buildOperandSpecsJson(const SpecKey &key) {
  std::string json = "[";
  for (size_t i = 0; i < key.operands.size(); ++i) {
    const auto &op = key.operands[i];
    if (i > 0)
      json += ",";

    if (op.kind == OperandKind::Tile) {
      json += "{\"kind\":\"tile\",\"dtype\":\"" + op.dtype + "\",\"shape\":";
      appendJsonIntArray(json, op.tileShape);
      json += ",\"valid_shape\":";
      appendJsonDimArray(json, op.tileValidShape, /*negativeIsDynamic=*/true);
      json += ",\"memory_space\":\"";
      json += op.tileMemorySpace;
      json += "\",\"config\":{";
      json += "\"b_layout\":\"";
      json += getBLayoutString(op.blayout);
      json += "\",\"s_layout\":\"";
      json += getSLayoutString(op.slayout);
      json += "\",\"s_fractal_size\":";
      json += std::to_string(op.fractal);
      json += ",\"pad_value\":\"0x";
      json += llvm::utohexstr(op.pad, /*LowerCase=*/false);
      json += "\"}}";
      continue;
    }

    if (op.kind == OperandKind::View) {
      json += "{\"kind\":\"view\",\"dtype\":\"" + op.dtype + "\",\"shape\":";
      appendJsonDimArray(json, op.viewShape);
      json += ",\"memory_space\":\"gm\"}";
      continue;
    }

    // Scalar
    json += "{\"kind\":\"scalar\",\"dtype\":\"" + op.dtype + "\"}";
  }
  json += "]";
  return json;
}

static std::string buildContextAttrsJson(const SpecKey &key) {
  std::string json = "{";
  for (size_t i = 0; i < key.contextAttrs.size(); ++i) {
    const auto &[attrName, attrValue] = key.contextAttrs[i];
    if (i > 0)
      json += ",";
    json += "\"";
    json += attrName;
    json += "\":\"";
    json += attrValue;
    json += "\"";
  }
  json += "}";
  return json;
}

// ============================================================================
// Invoke Python DSL helper to generate a specialized template function.
// ============================================================================
func::FuncOp ExpandState::invokeTilelangDSL(const SpecKey &key,
                                              Operation *tileOp,
                                              ModuleOp mod, MLIRContext *ctx) {
  // Check cache first.
  auto cacheIt = specCache.find(key);
  if (cacheIt != specCache.end())
    return cacheIt->second;

  // 1. Locate the Python executable.
  auto pythonPath = llvm::sys::findProgramByName(pythonExe);
  if (!pythonPath) {
    llvm::errs() << "ExpandTileOp: cannot find '" << pythonExe << "'\n";
    return nullptr;
  }

  // 2. Build operand schema JSON for mixed tile/scalar specialization.
  std::string operandSpecsJson = buildOperandSpecsJson(key);
  std::string contextAttrsJson = buildContextAttrsJson(key);
  if (key.targetArch.empty()) {
    llvm::errs() << "ExpandTileOp: missing pto.target_arch module attribute\n";
    return nullptr;
  }

  // 3. Create temp file for stdout redirect.
  SmallString<128> tmpPath;
  int tmpFD;
  if (auto ec = llvm::sys::fs::createTemporaryFile("tilelang_expand", "mlir",
                                                     tmpFD, tmpPath)) {
    llvm::errs() << "ExpandTileOp: cannot create temp file: "
                 << ec.message() << "\n";
    return nullptr;
  }
  ::close(tmpFD);

  // 4. Build command args.
  std::string opName = "pto." + key.opName;
  SmallVector<StringRef> args = {
      *pythonPath, "-m", "tilelang_dsl.expand_helper",
      "--template-dir", tilelangPath,
      "--target",       key.targetArch,
      "--op",           opName,
      "--operand-specs", operandSpecsJson,
  };
  if (!key.contextAttrs.empty()) {
    args.push_back("--context-attrs");
    args.push_back(contextAttrsJson);
  }

  // 5. Set up environment with PYTHONPATH.
  std::optional<StringRef> redirects[] = {std::nullopt, StringRef(tmpPath),
                                          std::nullopt};

  SmallVector<StringRef> envp;
  std::string pythonPathEnv;
  std::vector<std::string> envStorage;
  bool hasPythonPath = !tilelangPkgPath.empty();
  if (hasPythonPath) {
    const char *existingPath = ::getenv("PYTHONPATH");
    pythonPathEnv = "PYTHONPATH=" + tilelangPkgPath;
    if (existingPath && existingPath[0] != '\0') {
      pythonPathEnv += ":";
      pythonPathEnv += existingPath;
    }
    for (char **e = environ; *e; ++e) {
      StringRef entry(*e);
      if (entry.starts_with("PYTHONPATH="))
        continue;
      envStorage.push_back(std::string(entry));
    }
    envStorage.push_back(pythonPathEnv);
    for (auto &s : envStorage)
      envp.push_back(s);
  }

  // 6. Execute.
  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(
      *pythonPath, args,
      hasPythonPath ? std::optional<ArrayRef<StringRef>>(envp) : std::nullopt,
      redirects, /*secondsToWait=*/30, /*memoryLimit=*/0, &errMsg);

  if (rc != 0) {
    std::string cmd;
    llvm::raw_string_ostream os(cmd);
    bool first = true;
    auto appendToken = [&](StringRef token) {
      if (!first)
        os << ' ';
      first = false;
      llvm::sys::printArg(os, token, /*Quote=*/true);
    };
    if (hasPythonPath) {
      appendToken("env");
      appendToken(pythonPathEnv);
    }
    for (StringRef arg : args)
      appendToken(arg);
    os.flush();

    llvm::errs() << "ExpandTileOp: tilelang DSL helper failed (rc=" << rc
                 << "): " << errMsg << "\n";
    llvm::errs() << "ExpandTileOp: run: " << cmd << "\n";
    llvm::sys::fs::remove(tmpPath);
    return nullptr;
  }

  // 7. Read the generated MLIR.
  auto bufOrErr = llvm::MemoryBuffer::getFile(tmpPath);
  llvm::sys::fs::remove(tmpPath);
  if (!bufOrErr) {
    llvm::errs() << "ExpandTileOp: cannot read DSL output\n";
    return nullptr;
  }
  StringRef mlirText = (*bufOrErr)->getBuffer();
  if (mlirText.empty()) {
    llvm::errs() << "ExpandTileOp: empty DSL output\n";
    return nullptr;
  }

  // 8. Parse the MLIR text.
  auto parsedMod = parseSourceString<ModuleOp>(mlirText, ctx);
  if (!parsedMod) {
    llvm::errs() << "ExpandTileOp: failed to parse DSL output\n";
    return nullptr;
  }

  // 9. Clone the generated function set into the target module. The TileLang
  // output may include private inline helper funcs referenced by the entry.
  SmallVector<func::FuncOp, 4> parsedFuncs;
  for (auto fn : parsedMod->getOps<func::FuncOp>())
    parsedFuncs.push_back(fn);
  if (parsedFuncs.empty()) {
    llvm::errs() << "ExpandTileOp: no func.func in DSL output\n";
    return nullptr;
  }
  func::FuncOp srcFn = parsedFuncs.front();

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(mod.getBody());
  SmallVector<func::FuncOp, 4> clonedFuncs;
  llvm::StringMap<std::string> renamedSymbols;

  // Build a unique name from the spec-key-relevant operand fields.
  std::string uniqueName = "__pto_tilelang_" + key.targetArch + "_" + key.opName;
  for (const auto &op : key.operands) {
    uniqueName += op.kind == OperandKind::Tile   ? "_tile"
                 : op.kind == OperandKind::View ? "_view"
                                                : "_scalar";
    uniqueName += "_" + op.dtype;
    if (op.kind == OperandKind::Tile) {
      for (int64_t d : op.tileShape)
        uniqueName += "_" + std::to_string(d);
      for (int64_t d : op.tileValidShape)
        uniqueName += "_v" + std::to_string(d);
      uniqueName += "_bl" + std::to_string(op.blayout);
      uniqueName += "_sl" + std::to_string(op.slayout);
      uniqueName += "_fr" + std::to_string(op.fractal);
      uniqueName += "_pd" + llvm::utohexstr(op.pad, /*LowerCase=*/false);
    }
  }
  for (const auto &[attrName, attrValue] : key.contextAttrs)
    uniqueName += "_ctx_" + attrName + "_" + attrValue;

  for (auto [index, fn] : llvm::enumerate(parsedFuncs)) {
    IRMapping mapping;
    auto cloned = cast<func::FuncOp>(builder.clone(*fn, mapping));
    std::string newName;
    if (index == 0) {
      newName = uniqueName;
      cloned.setVisibility(SymbolTable::Visibility::Private);
    } else {
      newName = uniqueName + "__" + std::string(fn.getSymName());
    }
    renamedSymbols[fn.getSymName()] = newName;
    cloned.setName(newName);
    clonedFuncs.push_back(cloned);
  }

  for (func::FuncOp fn : clonedFuncs) {
    fn.walk([&](func::CallOp call) {
      StringRef callee = call.getCallee();
      if (callee.empty())
        return;
      auto renameIt = renamedSymbols.find(callee);
      if (renameIt == renamedSymbols.end())
        return;
      call.setCallee(renameIt->second);
    });
  }

  auto cloned = clonedFuncs.front();
  // The pto.tilelang.instance attribute should already be set by the
  // TileLang DSL frontend in the generated MLIR. Verify it exists.
  if (!cloned->hasAttr("pto.tilelang.instance")) {
    llvm::errs() << "ExpandTileOp: warning: DSL output function @"
                 << cloned.getSymName()
                 << " missing pto.tilelang.instance attribute\n";
  }

  // Keep the parsed module alive.
  parsedModules.push_back(std::move(parsedMod));

  specCache[key] = cloned;
  return cloned;
}

// ============================================================================
// Expand tile ops in a single function.
// ============================================================================
LogicalResult ExpandState::expandTileOpsInFunction(func::FuncOp func,
                                                   ModuleOp mod,
                                                   MLIRContext *ctx) {
  OpBuilder builder(ctx);

  // Collect tile ops first (avoid modifying while iterating).
  SmallVector<Operation *, 16> tileOps;
  func.walk([&](Operation *op) {
    if (isa<pto::OpPipeInterface>(op))
      tileOps.push_back(op);
  });

  for (auto *op : tileOps) {
    if (failed(validateBoundaryOperandTypes(op)))
      return failure();

    auto specKeyOpt = buildSpecKey(op);
    if (!specKeyOpt) {
      op->emitError(
          "ExpandTileOp: cannot build specialization key for this operand schema");
      return failure();
    }

    // Invoke tilelang DSL (with caching).
    func::FuncOp dslFn = invokeTilelangDSL(*specKeyOpt, op, mod, ctx);
    if (!dslFn) {
      StringRef opName = getTileOpName(op);
      op->emitError("ExpandTileOp: failed to instantiate tilelang template for " +
                    opName);
      return failure();
    }

    // Replace tile op with func.call.  For view operands whose caller type
    // (memref) differs from the template parameter type (tensor_view /
    // partition_tensor_view), insert an unrealized_conversion_cast bridge.
    // FoldTileBufIntrinsics will later resolve these casts.
    builder.setInsertionPoint(op);
    SmallVector<Value> operands;
    auto fnArgTypes = dslFn.getArgumentTypes();
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      Value operand = op->getOperand(i);
      if (i < fnArgTypes.size() && operand.getType() != fnArgTypes[i]) {
        operand = builder.create<UnrealizedConversionCastOp>(
            op->getLoc(), fnArgTypes[i], operand).getResult(0);
      }
      operands.push_back(operand);
    }
    builder.create<func::CallOp>(op->getLoc(), dslFn, operands);
    op->erase();
  }

  return success();
}

// ============================================================================
// Main entry point.
// ============================================================================
void ExpandTileOpPass::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = &getContext();

  if (tilelangPath.empty()) {
    mod.emitError(
        "ExpandTileOp requires a non-empty tilelang-path on the VPTO backend");
    signalPassFailure();
    return;
  }

  ExpandState state;
  state.tilelangPath = std::string(tilelangPath);
  state.tilelangPkgPath = std::string(tilelangPkgPath);
  state.pythonExe = std::string(pythonExe);

  for (auto func : mod.getOps<func::FuncOp>()) {
    if (func.isExternal())
      continue;
    if (failed(state.expandTileOpsInFunction(func, mod, ctx)))
      return signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createExpandTileOpPass() {
  return std::make_unique<ExpandTileOpPass>();
}

std::unique_ptr<Pass>
createExpandTileOpPass(const ExpandTileOpOptions &options) {
  return std::make_unique<ExpandTileOpPass>(options);
}

} // namespace pto
} // namespace mlir
