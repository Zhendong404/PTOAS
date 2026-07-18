// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTOASCppEmission.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/CppPostprocess.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;
using namespace pto;

namespace {
constexpr unsigned kSeenCalleeInlineCapacity = 8;
constexpr unsigned kStringRefInlineCapacity = 4;
constexpr unsigned kEmptyExpressionInlineCapacity = 8;
constexpr unsigned kBranchInlineCapacity = 16;
constexpr size_t kMarkerCallReserveExtra = 16;
constexpr size_t kRewriteOutputReserveExtra = 64;
constexpr size_t kMarkerRewriteMinArgCount = 2;
constexpr size_t kMarkerRewriteTernaryArgCount = 3;

using StringRefVector =
    llvm::SmallVector<llvm::StringRef, kStringRefInlineCapacity>;

static void narrowUnusedMultiResultProvenanceLocs(Operation *root);
static void splitDerivedSingleResultProvenanceLocs(Operation *root);
static void dropEmptyEmitCExpressions(Operation *rootOp);
static void materializeControlFlowOperands(Operation *rootOp);
static void normalizeEmitCIntegerAttrsForCppEmission(Operation *rootOp);
static LogicalResult reorderEmitCFunctions(ModuleOp module);
static void annotateEmitCProvenanceHints(ModuleOp module);
static bool shouldDeclareVariablesAtTop(ModuleOp module);
static void rewriteTileGetSetValueMarkers(std::string &cpp);
static void rewriteAsyncEventMarkers(std::string &cpp);
static void rewritePtrScalarMarkers(std::string &cpp);
static void rewriteScalarGMStoreFlushMarkers(std::string &cpp);
static void rewriteEventIdArrayMarkers(std::string &cpp);
static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace);
static void rewriteMalformedVerbatimSemicolons(std::string &cpp);
static void rewriteScalarConstantDecls(std::string &cpp);
static void rewriteHoistedGlobalTensorDecls(std::string &cpp);
static void rewriteNameHintMarkers(std::string &cpp);

struct ApplySIMTEntryNoInlinePass final
    : public PassWrapper<ApplySIMTEntryNoInlinePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplySIMTEntryNoInlinePass)

  void runOnOperation() final {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>())
      if (func->hasAttr(pto::kPTOSimtEntryAttrName))
        func.setNoInline(true);
  }
};

struct ParsedMarkerCall {
  size_t markerPos = std::string::npos;
  size_t rparenPos = std::string::npos;
  StringRefVector args;
};

struct MarkerRewriteSpec {
  llvm::StringRef marker;
  llvm::StringRef memberName;
  unsigned expectedNumArgs = 0;
};

struct MarkerSubscriptRewriteSpec {
  llvm::StringRef marker;
  unsigned expectedNumArgs = 0;
  bool isStore = false;
};

static std::optional<ParsedMarkerCall>
findNextMarkerCall(const std::string &cpp, llvm::StringRef marker,
                   size_t searchPos);
static void splitDerivedSingleResultProvenanceLocsInRegion(Region &region);
static void appendEmitCIntegerAttrLiteral(std::string &storage,
                                          const APInt &value, bool isUnsigned);
static bool shouldPrintEmitCIntegerAttrAsUnsigned(IntegerAttr attr);
static std::string getEmitCIntegerAttrLiteral(IntegerAttr attr);
static std::optional<std::string>
getEmitCDenseIntElementsAttrLiteral(DenseIntElementsAttr attr);
static Attribute normalizeEmitCPrintedAttrForCppEmission(MLIRContext *ctx,
                                                         Attribute attr);
static IntegerAttr normalizeEmitCIndexPlaceholderAttr(MLIRContext *ctx,
                                                      IntegerAttr attr);
static ArrayAttr normalizeEmitCCallArgsForCppEmission(MLIRContext *ctx,
                                                      ArrayAttr args);
static ArrayAttr normalizeEmitCTemplateArgsForCppEmission(MLIRContext *ctx,
                                                          ArrayAttr args);
static Attribute getDefaultEmitCVariableInitAttr(OpBuilder &builder, Type type);
static Type getEmitCVariableStorageType(Type valueType);
static std::string getLineIndent(llvm::StringRef line);
static bool isAICOREFunctionStart(llvm::StringRef trimmed);
static int countBraceDelta(llvm::StringRef line);
static void appendScalarGMFlush(std::string &out, llvm::StringRef indent);
static bool stripScalarGMFlushMarkersFromLine(std::string &line);
static bool previousSignificantLineIsTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index);
static bool previousSignificantLineIsExitOrTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index);
static std::string rewriteScalarGMStoreFlushMarkersInFunction(
    llvm::ArrayRef<std::string> functionLines, bool hasTrailingNewline);
static bool isPreprocessorDirectiveLine(llvm::StringRef trimmedLine);
static bool isGeneratedGlobalTensorDecl(llvm::StringRef trimmed,
                                        llvm::StringRef &decl,
                                        llvm::StringRef &varName);
static std::optional<llvm::SmallVector<std::string, 4>>
parseNameHintMarker(llvm::StringRef markerBody);
static void stripHintMarkersWithPrefix(std::string &cpp,
                                       llvm::StringRef markerPrefix);
static void stripAllHintMarkers(std::string &cpp);
static std::string sanitizeCommentText(llvm::StringRef text);
static std::string buildHintMarker(llvm::StringRef prefix,
                                   llvm::ArrayRef<std::string> hints);
static void emitProvenanceComments(std::string &segment);
static void appendRawLocationProvenance(Location loc,
                                        llvm::SmallVectorImpl<std::string> &hints);
static llvm::SmallVector<std::string, 4> getRawLocationProvenance(Location loc);
static Location getIndexedRawProvenanceLoc(Location fallbackLoc,
                                           unsigned index);
static Location attachLocationNameHints(Location baseLoc,
                                        llvm::ArrayRef<std::string> hints,
                                        MLIRContext *context);
static llvm::SmallVector<std::string, 4> getRawResultProvenance(Operation *op);

static bool isGeneratedValueName(llvm::StringRef name) {
  if (!name.consume_front("v") || name.empty())
    return false;
  return llvm::all_of(name, [](char c) { return std::isdigit(c); });
}

static void appendRawLocationProvenance(Location loc,
                                        llvm::SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    std::string raw = nameLoc.getName().getValue().str();
    if (!raw.empty())
      hints.push_back(std::move(raw));
    return;
  }

  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata()) {
      if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
        std::string raw = strAttr.getValue().str();
        if (!raw.empty())
          hints.push_back(std::move(raw));
        return;
      }
      if (auto arrayAttr = dyn_cast<ArrayAttr>(metadata)) {
        for (Attribute attr : arrayAttr) {
          auto strAttr = dyn_cast<StringAttr>(attr);
          if (!strAttr)
            continue;
          std::string raw = strAttr.getValue().str();
          if (!raw.empty())
            hints.push_back(std::move(raw));
        }
        if (!hints.empty())
          return;
      }
    }
    return;
  }

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendRawLocationProvenance(callSiteLoc.getCallee(), hints);
    if (hints.empty())
      appendRawLocationProvenance(callSiteLoc.getCaller(), hints);
  }
}

static llvm::SmallVector<std::string, 4> getRawLocationProvenance(Location loc) {
  llvm::SmallVector<std::string, 4> hints;
  appendRawLocationProvenance(loc, hints);
  hints.erase(std::remove_if(hints.begin(), hints.end(),
                             [](const std::string &hint) {
                               return hint.empty();
                             }),
              hints.end());
  return hints;
}

static Location getIndexedRawProvenanceLoc(Location fallbackLoc,
                                           unsigned index) {
  llvm::SmallVector<std::string, 4> hints = getRawLocationProvenance(fallbackLoc);
  if (index >= hints.size())
    return fallbackLoc;
  return NameLoc::get(StringAttr::get(fallbackLoc.getContext(), hints[index]),
                      fallbackLoc);
}

static Location attachLocationNameHints(Location baseLoc,
                                        llvm::ArrayRef<std::string> hints,
                                        MLIRContext *context) {
  llvm::SmallVector<Attribute, 4> attrs;
  attrs.reserve(hints.size());
  for (llvm::StringRef hint : hints) {
    if (!hint.empty())
      attrs.push_back(StringAttr::get(context, hint));
  }
  if (attrs.empty())
    return baseLoc;
  if (attrs.size() == 1)
    return NameLoc::get(cast<StringAttr>(attrs.front()), baseLoc);
  return FusedLoc::get(ArrayRef<Location>{baseLoc}, ArrayAttr::get(context, attrs),
                       context);
}

static llvm::SmallVector<std::string, 4> getRawResultProvenance(Operation *op) {
  llvm::SmallVector<std::string, 4> hints;
  if (!op || op->getNumResults() == 0)
    return hints;
  appendRawLocationProvenance(op->getLoc(), hints);
  if (hints.empty())
    return hints;
  hints.erase(std::remove_if(hints.begin(), hints.end(),
                             [](const std::string &name) {
                               return name.empty();
                             }),
              hints.end());
  if (hints.empty())
    return hints;
  if (op->getNumResults() == 1) {
    if (hints.size() > 1)
      hints.resize(1);
    return hints;
  }
  if (hints.size() > op->getNumResults())
    hints.resize(op->getNumResults());
  return hints;
}

static void splitDerivedSingleResultProvenanceLocsInRegion(Region &region);

static void splitDerivedSingleResultProvenanceLocsInBlock(Block &block) {
  llvm::SmallVector<Operation *, 16> ops;
  ops.reserve(block.getOperations().size());
  for (Operation &op : block)
    ops.push_back(&op);

  for (size_t i = 0; i < ops.size();) {
    Operation *op = ops[i];
    if (op->getNumResults() != 1) {
      ++i;
      continue;
    }

    llvm::SmallVector<std::string, 4> hints = getRawLocationProvenance(op->getLoc());
    if (hints.size() <= 1) {
      ++i;
      continue;
    }

    size_t runEnd = i + 1;
    while (runEnd < ops.size() && ops[runEnd]->getNumResults() == 1 &&
           ops[runEnd]->getLoc() == op->getLoc()) {
      ++runEnd;
    }

    size_t runSize = runEnd - i;
    if (runSize == hints.size()) {
      Location sharedLoc = op->getLoc();
      for (size_t j = 0; j < runSize; ++j)
        ops[i + j]->setLoc(getIndexedRawProvenanceLoc(sharedLoc, j));
    }

    i = runEnd;
  }

  for (Operation &op : block) {
    for (Region &region : op.getRegions())
      splitDerivedSingleResultProvenanceLocsInRegion(region);
  }
}

static void splitDerivedSingleResultProvenanceLocsInRegion(Region &region) {
  for (Block &block : region)
    splitDerivedSingleResultProvenanceLocsInBlock(block);
}

static void splitDerivedSingleResultProvenanceLocs(Operation *root) {
  if (!root)
    return;
  for (Region &region : root->getRegions())
    splitDerivedSingleResultProvenanceLocsInRegion(region);
}

static void narrowUnusedMultiResultProvenanceLocs(Operation *root) {
  if (!root)
    return;

  root->walk([&](Operation *op) {
    if (op->getNumResults() <= 1)
      return;

    llvm::SmallVector<std::string, 4> hints = getRawLocationProvenance(op->getLoc());
    if (hints.size() != op->getNumResults())
      return;

    llvm::SmallVector<std::string, 4> liveHints;
    liveHints.reserve(hints.size());
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      if (!result.use_empty())
        liveHints.push_back(hints[index]);
    }

    if (liveHints.empty() || liveHints.size() == hints.size())
      return;

    op->setLoc(attachLocationNameHints(op->getLoc(), liveHints,
                                       op->getContext()));
  });
}

struct ConstantDeclCandidate {
  size_t declLine = 0;
  std::string indent;
  std::string type;
  bool hasInitializer = false;
  std::string initializer;
  size_t assignmentCount = 0;
  size_t assignmentLine = 0;
  std::string assignmentRhs;
};

static bool isConstFoldableScalarType(llvm::StringRef type) {
  type = type.trim();
  if (type.starts_with("const ") || type.starts_with("constexpr "))
    return false;
  return llvm::StringSwitch<bool>(type)
      .Cases("bool", "float", "double", "half", "bfloat16_t", true)
      .Cases("int8_t", "uint8_t", "int16_t", "uint16_t", true)
      .Cases("int32_t", "uint32_t", "int64_t", "uint64_t", true)
      .Default(false);
}

static bool isLiteralInitializer(llvm::StringRef rhs) {
  rhs = rhs.trim();
  if (rhs.empty())
    return false;
  if (rhs == "true" || rhs == "false" || rhs == "nullptr")
    return true;

  static const llvm::Regex kIntLiteral(
      R"(^[+-]?(0[xX][0-9A-Fa-f]+|[0-9]+)[uUlL]*$)");
  static const llvm::Regex kFloatLiteral(
      R"(^[+-]?(([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)[fF]?$)");
  static const llvm::Regex kHexFloatLiteral(
      R"(^[+-]?0[xX]([0-9A-Fa-f]+\.[0-9A-Fa-f]*|[0-9A-Fa-f]+|\.[0-9A-Fa-f]+)[pP][+-]?[0-9]+[fF]?$)");
  static const llvm::Regex kSpecialFloatLiteral(
      R"(^[+-]?(nan|inf)[fF]?$)");

  return kIntLiteral.match(rhs) || kFloatLiteral.match(rhs) ||
         kHexFloatLiteral.match(rhs) || kSpecialFloatLiteral.match(rhs);
}

static std::string normalizeConstInitializer(llvm::StringRef type,
                                             llvm::StringRef rhs) {
  type = type.trim();
  rhs = rhs.trim();
  if (type == "bool") {
    if (rhs == "0" || rhs == "false")
      return "false";
    if (rhs == "1" || rhs == "-1" || rhs == "true")
      return "true";
  }
  return rhs.str();
}

static bool parseConstantDeclarationLine(llvm::StringRef line,
                                         ConstantDeclCandidate &candidate,
                                         std::string &valueName) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  if (body.starts_with("return") || body.starts_with("goto ") ||
      body.starts_with("if ") || body.starts_with("if(") ||
      body.starts_with("switch ") || body.starts_with("switch(") ||
      body.starts_with("for ") || body.starts_with("for(") ||
      body.starts_with("while ") || body.starts_with("while(") ||
      body.starts_with("case ") || body == "default")
    return false;

  llvm::StringRef lhs = body;
  llvm::StringRef rhs;
  if (size_t eqPos = body.find('='); eqPos != llvm::StringRef::npos) {
    lhs = body.take_front(eqPos).rtrim();
    rhs = body.drop_front(eqPos + 1).trim();
  }

  size_t lastWs = lhs.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;

  llvm::StringRef type = lhs.take_front(lastWs).rtrim();
  llvm::StringRef name = lhs.drop_front(lastWs + 1).trim();
  if (!isGeneratedValueName(name) || !isConstFoldableScalarType(type))
    return false;

  size_t indentLen = line.find_first_not_of(" \t");
  if (indentLen == llvm::StringRef::npos)
    indentLen = 0;
  candidate.indent = line.take_front(indentLen).str();
  candidate.type = type.str();
  valueName = name.str();

  if (!rhs.empty()) {
    if (!isLiteralInitializer(rhs))
      return false;
    candidate.hasInitializer = true;
    candidate.initializer = normalizeConstInitializer(type, rhs);
  }

  return true;
}

static bool parseGeneratedValueAssignment(llvm::StringRef line,
                                          llvm::StringRef &valueName,
                                          llvm::StringRef &rhs) {
  llvm::StringRef trimmed = line.trim();
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//") ||
      !trimmed.ends_with(";"))
    return false;

  llvm::StringRef body = trimmed.drop_back().rtrim();
  size_t eqPos = body.find('=');
  if (eqPos == llvm::StringRef::npos)
    return false;

  llvm::StringRef lhs = body.take_front(eqPos).rtrim();
  rhs = body.drop_front(eqPos + 1).trim();
  if (!isGeneratedValueName(lhs))
    return false;
  valueName = lhs;
  return true;
}

static void rewriteScalarConstantDecls(std::string &cpp) {
  llvm::SmallVector<std::string, 0> lines;
  for (llvm::StringRef ref(cpp); !ref.empty(); ref = ref.split('\n').second) {
    auto split = ref.split('\n');
    lines.push_back(split.first.str());
  }

  llvm::SmallVector<bool, 0> eraseLine(lines.size(), false);
  auto rewriteSegment = [&](size_t beginLine, size_t endLine) {
    llvm::StringMap<ConstantDeclCandidate> candidates;

    for (size_t i = beginLine; i <= endLine; ++i) {
      ConstantDeclCandidate candidate;
      std::string valueName;
      if (parseConstantDeclarationLine(lines[i], candidate, valueName)) {
        candidate.declLine = i;
        candidates[valueName] = std::move(candidate);
        continue;
      }

      llvm::StringRef assignedName;
      llvm::StringRef rhs;
      if (!parseGeneratedValueAssignment(lines[i], assignedName, rhs))
        continue;

      auto it = candidates.find(assignedName);
      if (it == candidates.end())
        continue;

      ConstantDeclCandidate &info = it->second;
      ++info.assignmentCount;
      info.assignmentLine = i;
      info.assignmentRhs = rhs.str();
    }

    for (auto &entry : candidates) {
      llvm::StringRef valueName = entry.getKey();
      ConstantDeclCandidate &info = entry.getValue();

      std::string initializer;
      if (info.hasInitializer) {
        if (info.assignmentCount != 0)
          continue;
        initializer = info.initializer;
      } else {
        if (info.assignmentCount != 1)
          continue;
        if (!isLiteralInitializer(info.assignmentRhs))
          continue;
        initializer = normalizeConstInitializer(
            info.type, llvm::StringRef(info.assignmentRhs));
        eraseLine[info.assignmentLine] = true;
      }

      lines[info.declLine] = (info.indent + "const " + info.type + " " +
                              valueName.str() + " = " + initializer + ";");
    }
  };

  int braceDepth = 0;
  size_t segmentStart = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    int depthBefore = braceDepth;
    for (char c : lines[i]) {
      if (c == '{')
        ++braceDepth;
      else if (c == '}')
        --braceDepth;
    }

    if (depthBefore == 0 && braceDepth > 0)
      segmentStart = i;
    if (depthBefore > 0 && braceDepth == 0)
      rewriteSegment(segmentStart, i);
  }

  std::string out;
  out.reserve(cpp.size());
  for (size_t i = 0; i < lines.size(); ++i) {
    if (eraseLine[i])
      continue;
    out.append(lines[i]);
    if (i + 1 != lines.size())
      out.push_back('\n');
  }
  cpp.swap(out);
}

static bool shouldPrintEmitCIntegerAttrAsUnsigned(IntegerAttr attr) {
  auto intTy = dyn_cast<IntegerType>(attr.getType());
  return intTy && intTy.getSignedness() == IntegerType::Unsigned;
}

static void appendEmitCIntegerAttrLiteral(std::string &storage,
                                          const APInt &value, bool isUnsigned) {
  if (value.getBitWidth() == 0) {
    storage.append("0");
    return;
  }
  if (value.getBitWidth() == 1) {
    storage.append(value.getBoolValue() ? "true" : "false");
    return;
  }

  SmallString<128> strValue;
  value.toString(strValue, 10, !isUnsigned, false);
  storage.append(strValue.data(), strValue.size());
}

static std::string getEmitCIntegerAttrLiteral(IntegerAttr attr) {
  std::string literal;
  appendEmitCIntegerAttrLiteral(literal, attr.getValue(),
                                shouldPrintEmitCIntegerAttrAsUnsigned(attr));
  return literal;
}

static std::optional<std::string>
getEmitCDenseIntElementsAttrLiteral(DenseIntElementsAttr attr) {
  auto tensorTy = dyn_cast<TensorType>(attr.getType());
  if (!tensorTy)
    return std::nullopt;

  Type elementType = tensorTy.getElementType();
  bool isUnsigned = false;
  if (auto intTy = dyn_cast<IntegerType>(elementType)) {
    isUnsigned = intTy.getSignedness() == IntegerType::Unsigned;
  } else if (!isa<IndexType>(elementType)) {
    return std::nullopt;
  }

  std::string literal;
  literal.push_back('{');
  bool first = true;
  for (const APInt &value : attr) {
    if (!first)
      literal.append(", ");
    first = false;
    appendEmitCIntegerAttrLiteral(literal, value, isUnsigned);
  }
  literal.push_back('}');
  return literal;
}

static Attribute normalizeEmitCPrintedAttrForCppEmission(MLIRContext *ctx,
                                                         Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return emitc::OpaqueAttr::get(ctx, getEmitCIntegerAttrLiteral(intAttr));

  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (std::optional<std::string> literal =
            getEmitCDenseIntElementsAttrLiteral(denseAttr))
      return emitc::OpaqueAttr::get(ctx, *literal);
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    llvm::SmallVector<Attribute> normalized;
    normalized.reserve(arrayAttr.size());
    bool changed = false;
    for (Attribute element : arrayAttr) {
      Attribute normalizedElement =
          normalizeEmitCPrintedAttrForCppEmission(ctx, element);
      changed |= normalizedElement != element;
      normalized.push_back(normalizedElement);
    }
    if (changed)
      return ArrayAttr::get(ctx, normalized);
  }

  return attr;
}

static IntegerAttr normalizeEmitCIndexPlaceholderAttr(MLIRContext *ctx,
                                                      IntegerAttr attr) {
  const APInt &value = attr.getValue();
  int64_t index = value.getBitWidth() == 0 ? 0 : value.getSExtValue();
  return IntegerAttr::get(IndexType::get(ctx), APInt(64, index));
}

static ArrayAttr normalizeEmitCCallArgsForCppEmission(MLIRContext *ctx,
                                                      ArrayAttr args) {
  llvm::SmallVector<Attribute> normalized;
  normalized.reserve(args.size());
  bool changed = false;

  for (Attribute attr : args) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      if (isa<IndexType>(intAttr.getType())) {
        Attribute normalizedAttr =
            normalizeEmitCIndexPlaceholderAttr(ctx, intAttr);
        changed |= normalizedAttr != attr;
        normalized.push_back(normalizedAttr);
        continue;
      }

      Attribute normalizedAttr =
          normalizeEmitCPrintedAttrForCppEmission(ctx, attr);
      changed |= normalizedAttr != attr;
      normalized.push_back(normalizedAttr);
      continue;
    }

    Attribute normalizedAttr =
        normalizeEmitCPrintedAttrForCppEmission(ctx, attr);
    changed |= normalizedAttr != attr;
    normalized.push_back(normalizedAttr);
  }

  return changed ? ArrayAttr::get(ctx, normalized) : args;
}

static ArrayAttr normalizeEmitCTemplateArgsForCppEmission(MLIRContext *ctx,
                                                          ArrayAttr args) {
  llvm::SmallVector<Attribute> normalized;
  normalized.reserve(args.size());
  bool changed = false;

  for (Attribute attr : args) {
    Attribute normalizedAttr =
        normalizeEmitCPrintedAttrForCppEmission(ctx, attr);
    changed |= normalizedAttr != attr;
    normalized.push_back(normalizedAttr);
  }

  return changed ? ArrayAttr::get(ctx, normalized) : args;
}

static void normalizeEmitCIntegerAttrsForCppEmission(Operation *rootOp) {
  MLIRContext *ctx = rootOp->getContext();
  rootOp->walk([&](Operation *op) {
    if (auto constant = dyn_cast<emitc::ConstantOp>(op)) {
      Attribute value = constant.getValue();
      Attribute normalized =
          normalizeEmitCPrintedAttrForCppEmission(ctx, value);
      if (normalized != value)
        constant.getProperties().setValue(normalized);
      return;
    }

    if (auto variable = dyn_cast<emitc::VariableOp>(op)) {
      Attribute value = variable.getValue();
      Attribute normalized =
          normalizeEmitCPrintedAttrForCppEmission(ctx, value);
      if (normalized != value)
        variable.getProperties().setValue(normalized);
      return;
    }

    if (auto global = dyn_cast<emitc::GlobalOp>(op)) {
      std::optional<Attribute> initialValue = global.getInitialValue();
      if (!initialValue)
        return;
      Attribute normalized =
          normalizeEmitCPrintedAttrForCppEmission(ctx, *initialValue);
      if (normalized != *initialValue)
        global.getProperties().setInitialValue(normalized);
      return;
    }

    if (auto call = dyn_cast<emitc::CallOpaqueOp>(op)) {
      if (std::optional<ArrayAttr> args = call.getArgs()) {
        ArrayAttr normalized = normalizeEmitCCallArgsForCppEmission(ctx, *args);
        if (normalized != *args)
          call.getProperties().setArgs(normalized);
      }
      if (std::optional<ArrayAttr> templateArgs = call.getTemplateArgs()) {
        ArrayAttr normalized =
            normalizeEmitCTemplateArgsForCppEmission(ctx, *templateArgs);
        if (normalized != *templateArgs)
          call.getProperties().setTemplateArgs(normalized);
      }
      return;
    }
  });
}

static Attribute getDefaultEmitCVariableInitAttr(OpBuilder &builder, Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() == 0)
      return emitc::OpaqueAttr::get(builder.getContext(), "0");
    return builder.getIntegerAttr(intTy, 0);
  }
  if (isa<IndexType>(type))
    return builder.getIndexAttr(0);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(floatTy, 0.0);
  if (isa<emitc::OpaqueType, emitc::PointerType>(type))
    return emitc::OpaqueAttr::get(builder.getContext(), "");
  return Attribute{};
}

static Type getEmitCVariableStorageType(Type valueType) {
  if (isa<emitc::ArrayType, emitc::LValueType>(valueType))
    return valueType;
  return emitc::LValueType::get(valueType);
}

static void materializeControlFlowOperands(Operation *rootOp) {
  llvm::SmallVector<Operation *, kBranchInlineCapacity> branches;
  rootOp->walk([&](Operation *op) {
    if (isa<cf::BranchOp, cf::CondBranchOp>(op))
      branches.push_back(op);
  });

  OpBuilder builder(rootOp->getContext());
  for (Operation *op : branches) {
    builder.setInsertionPoint(op);
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      auto expr = dyn_cast_or_null<emitc::ExpressionOp>(value.getDefiningOp());
      if (!expr)
        continue;

      Attribute initAttr =
          getDefaultEmitCVariableInitAttr(builder, value.getType());
      if (!initAttr)
        continue;

      Value tmp = builder
                      .create<emitc::VariableOp>(
                          op->getLoc(), getEmitCVariableStorageType(value.getType()),
                          initAttr)
                      .getResult();
      builder.create<emitc::AssignOp>(op->getLoc(), tmp, value);
      if (auto lvalueTy = dyn_cast<emitc::LValueType>(tmp.getType())) {
        Value loaded = builder
                           .create<emitc::LoadOp>(op->getLoc(),
                                                  lvalueTy.getValueType(), tmp)
                           .getResult();
        operand.set(loaded);
      } else {
        operand.set(tmp);
      }
    }
  }
}

static void dropEmptyEmitCExpressions(Operation *rootOp) {
  llvm::SmallVector<emitc::ExpressionOp, kEmptyExpressionInlineCapacity>
      toErase;
  rootOp->walk([&](emitc::ExpressionOp expr) {
    Block *body = expr.getBody();
    if (!body)
      return;
    auto yield = dyn_cast<emitc::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return;
    Value yielded = yield.getOperand(0);
    Operation *defOp = yielded.getDefiningOp();
    bool yieldedFromOutside = !defOp || defOp->getBlock() != body;
    if (!yieldedFromOutside && expr.getRootOp())
      return;
    expr.getResult().replaceAllUsesWith(yielded);
    toErase.push_back(expr);
  });
  for (emitc::ExpressionOp expr : llvm::reverse(toErase))
    expr.erase();
}

template <typename BuildReplacementFn>
static bool rewriteMarkerCalls(std::string &cpp, llvm::StringRef marker,
                               BuildReplacementFn buildReplacement) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, marker, searchPos); call;
       call = findNextMarkerCall(cpp, marker, searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + marker.size();
      continue;
    }

    std::optional<std::string> replacement = buildReplacement(*call);
    if (!replacement) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    cpp.replace(call->markerPos, (call->rparenPos - call->markerPos) + 1,
                *replacement);
    changed = true;
    searchPos = call->markerPos + replacement->size();
  }
  return changed;
}

static std::optional<ParsedMarkerCall>
findNextMarkerCall(const std::string &cpp, llvm::StringRef marker,
                   size_t searchPos) {
  ParsedMarkerCall call;
  call.markerPos = cpp.find(marker.str(), searchPos);
  if (call.markerPos == std::string::npos)
    return std::nullopt;

  size_t lparenPos = call.markerPos + marker.size();
  if (lparenPos >= cpp.size() || cpp[lparenPos] != '(')
    return ParsedMarkerCall{call.markerPos, std::string::npos, {}};

  size_t argsBegin = lparenPos + 1;
  int parenDepth = 0;
  for (size_t i = argsBegin; i < cpp.size(); ++i) {
    char c = cpp[i];
    if (c == '(') {
      ++parenDepth;
      continue;
    }
    if (c != ')')
      continue;
    if (parenDepth == 0) {
      call.rparenPos = i;
      break;
    }
    --parenDepth;
  }
  if (call.rparenPos == std::string::npos)
    return call;

  llvm::StringRef argsRef(cpp.data() + argsBegin, call.rparenPos - argsBegin);
  size_t partBegin = 0;
  int depth = 0;
  for (size_t i = 0; i < argsRef.size(); ++i) {
    char c = argsRef[i];
    if (c == '(') {
      ++depth;
      continue;
    }
    if (c == ')') {
      if (depth > 0)
        --depth;
      continue;
    }
    if (c == ',' && depth == 0) {
      call.args.push_back(argsRef.slice(partBegin, i).trim());
      partBegin = i + 1;
    }
  }
  if (partBegin > argsRef.size())
    return call;
  call.args.push_back(argsRef.drop_front(partBegin).trim());
  return call;
}

static bool rewriteMarkerCallToMember(std::string &cpp, llvm::StringRef marker,
                                      llvm::StringRef memberName,
                                      unsigned expectedNumArgs) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;

        std::string replacement;
        replacement.reserve(marker.size() + kMarkerCallReserveExtra);
        replacement.append(call.args[0].str());
        replacement.push_back('.');
        replacement.append(memberName.str());
        replacement.push_back('(');
        if (expectedNumArgs >= kMarkerRewriteMinArgCount)
          replacement.append(call.args[1].str());
        if (expectedNumArgs == kMarkerRewriteTernaryArgCount) {
          replacement.append(", ");
          replacement.append(call.args[2].str());
        }
        replacement.push_back(')');
        return replacement;
      });
}

static void rewriteMarkerCallsToMembers(
    std::string &cpp, llvm::ArrayRef<MarkerRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToMember(cpp, rewrite.marker,
                                           rewrite.memberName,
                                           rewrite.expectedNumArgs);
    }
  }
}

static bool rewriteMarkerCallToField(std::string &cpp, llvm::StringRef marker,
                                     llvm::StringRef fieldName,
                                     size_t expectedNumArgs) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;
        if (call.args.empty())
          return std::nullopt;
        std::string replacement;
        replacement.reserve(call.args.front().size() + fieldName.size() + 1);
        replacement.append(call.args.front().str());
        replacement.push_back('.');
        replacement.append(fieldName.str());
        return replacement;
      });
}

static bool rewriteMarkerCallToSubscript(std::string &cpp, llvm::StringRef marker,
                                         unsigned expectedNumArgs,
                                         bool isStore) {
  return rewriteMarkerCalls(
      cpp, marker, [&](const ParsedMarkerCall &call) -> std::optional<std::string> {
        if (call.args.size() != expectedNumArgs)
          return std::nullopt;
        if (isStore) {
          return (call.args[0] + "[" + call.args[1] + "] = " + call.args[2])
              .str();
        }
        return (call.args[0] + "[" + call.args[1] + "]").str();
      });
}

static void rewriteMarkerCallsToSubscripts(
    std::string &cpp, llvm::ArrayRef<MarkerSubscriptRewriteSpec> rewrites) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (const MarkerSubscriptRewriteSpec &rewrite : rewrites) {
      changed |= rewriteMarkerCallToSubscript(cpp, rewrite.marker,
                                              rewrite.expectedNumArgs,
                                              rewrite.isStore);
    }
  }
}

static void rewriteTileGetSetValueMarkers(std::string &cpp) {
  static const MarkerRewriteSpec kTileMarkerRewrites[] = {
      {"PTOAS__TILE_SET_VALUE", "SetValue", 3},
      {"PTOAS__TILE_GET_VALUE", "GetValue", 2},
      {"PTOAS__TILE_DATA", "data", 1},
      {"PTOAS__TILE_SET_VALIDSHAPE", "SetValidShape", 3},
      {"PTOAS__TILE_GET_VALID_ROW", "GetValidRow", 1},
      {"PTOAS__TILE_GET_VALID_COL", "GetValidCol", 1},
  };
  rewriteMarkerCallsToMembers(cpp, kTileMarkerRewrites);
}

static void rewriteAsyncEventMarkers(std::string &cpp) {
  static const MarkerRewriteSpec kAsyncEventMarkerRewrites[] = {
      {"PTOAS__ASYNC_EVENT_WAIT", "Wait", 2},
      {"PTOAS__ASYNC_EVENT_TEST", "Test", 2},
  };
  rewriteMarkerCallsToMembers(cpp, kAsyncEventMarkerRewrites);
  (void)rewriteMarkerCallToField(cpp, "PTOAS__PREFETCH_CTX_SESSION",
                                 "session", 1);
}

static void rewritePtrScalarMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kPtrMarkerRewrites[] = {
      {"PTOAS__PTR_LOAD", 2, false},
      {"PTOAS__PTR_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kPtrMarkerRewrites);
}

static void rewriteEventIdArrayMarkers(std::string &cpp) {
  static const MarkerSubscriptRewriteSpec kEventIdMarkerRewrites[] = {
      {"PTOAS__EVENTID_ARRAY_LOAD", 2, false},
      {"PTOAS__EVENTID_ARRAY_STORE", 3, true},
  };
  rewriteMarkerCallsToSubscripts(cpp, kEventIdMarkerRewrites);
}

static std::string getLineIndent(llvm::StringRef line) {
  size_t firstNonSpace = line.find_first_not_of(" \t");
  if (firstNonSpace == llvm::StringRef::npos)
    return line.str();
  return line.take_front(firstNonSpace).str();
}

static bool isAICOREFunctionStart(llvm::StringRef trimmed) {
  if (trimmed.empty() || trimmed.starts_with("#") || trimmed.starts_with("//"))
    return false;
  if (!trimmed.contains("AICORE"))
    return false;
  return trimmed.contains("(");
}

static int countBraceDelta(llvm::StringRef line) {
  int delta = 0;
  for (char c : line) {
    if (c == '{')
      ++delta;
    else if (c == '}')
      --delta;
  }
  return delta;
}

static void appendScalarGMFlush(std::string &out, llvm::StringRef indent) {
  out.append(indent.str());
  out.append("pipe_barrier(PIPE_ALL);\n");
  out.append(indent.str());
  out.append("dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);\n");
  out.append(indent.str());
  out.append("dsb((mem_dsb_t)0);\n");
}

static bool stripScalarGMFlushMarkersFromLine(std::string &line) {
  static constexpr llvm::StringLiteral kMarker =
      "PTOAS__SCALAR_GM_STORE_FLUSH";

  bool changed = false;
  size_t searchPos = 0;
  while (true) {
    auto call = findNextMarkerCall(line, kMarker, searchPos);
    if (!call)
      break;
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + kMarker.size();
      continue;
    }

    size_t eraseBegin = call->markerPos;
    while (eraseBegin > 0 &&
           (line[eraseBegin - 1] == ' ' || line[eraseBegin - 1] == '\t'))
      --eraseBegin;

    size_t eraseEnd = call->rparenPos + 1;
    while (eraseEnd < line.size() &&
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t'))
      ++eraseEnd;
    if (eraseEnd < line.size() && line[eraseEnd] == ';')
      ++eraseEnd;
    while (eraseEnd < line.size() &&
           (line[eraseEnd] == ' ' || line[eraseEnd] == '\t'))
      ++eraseEnd;

    line.erase(eraseBegin, eraseEnd - eraseBegin);
    changed = true;
    searchPos = eraseBegin;
  }
  return changed;
}

static bool previousSignificantLineIsTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index) {
  for (size_t i = index; i > 0; --i) {
    llvm::StringRef prev = llvm::StringRef(lines[i - 1]).trim();
    if (prev.empty())
      continue;
    return prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static bool previousSignificantLineIsExitOrTailFlushPoint(
    llvm::ArrayRef<std::string> lines, size_t index) {
  for (size_t i = index; i > 0; --i) {
    llvm::StringRef prev = llvm::StringRef(lines[i - 1]).trim();
    if (prev.empty())
      continue;
    return prev.starts_with("return") ||
           prev.starts_with("#endif // __DAV_") ||
           prev.starts_with("ptoas_auto_sync_tail(");
  }
  return false;
}

static std::string rewriteScalarGMStoreFlushMarkersInFunction(
    llvm::ArrayRef<std::string> functionLines, bool hasTrailingNewline) {
  bool needsScalarGMFlush = false;
  llvm::SmallVector<std::string, 32> lines;
  lines.reserve(functionLines.size());

  for (const std::string &rawLine : functionLines) {
    std::string line = rawLine;
    bool hadMarker = stripScalarGMFlushMarkersFromLine(line);
    needsScalarGMFlush |= hadMarker;
    if (hadMarker && llvm::StringRef(line).trim().empty()) {
      continue;
    }
    lines.push_back(std::move(line));
  }

  if (!needsScalarGMFlush) {
    std::string unchanged;
    unchanged.reserve(kRewriteOutputReserveExtra);
    for (size_t i = 0; i < lines.size(); ++i) {
      unchanged.append(lines[i]);
      if (i + 1 < lines.size() || hasTrailingNewline)
        unchanged.push_back('\n');
    }
    return unchanged;
  }

  std::string out;
  out.reserve(kRewriteOutputReserveExtra);
  bool inserted = false;
  size_t fallbackIndex = lines.size();
  for (size_t i = lines.size(); i > 0; --i) {
    llvm::StringRef trimmed = llvm::StringRef(lines[i - 1]).trim();
    if (trimmed.empty())
      continue;
    if (trimmed.starts_with("}"))
      fallbackIndex = i - 1;
    break;
  }

  for (size_t i = 0; i < lines.size(); ++i) {
    llvm::StringRef lineRef(lines[i]);
    llvm::StringRef trimmed = lineRef.trim();
    bool insertHere = false;
    if (trimmed.starts_with("return")) {
      insertHere = !previousSignificantLineIsTailFlushPoint(lines, i);
    } else {
      insertHere = trimmed.starts_with("#endif // __DAV_") ||
                   trimmed.starts_with("ptoas_auto_sync_tail(");
    }
    if (i == fallbackIndex &&
        !previousSignificantLineIsExitOrTailFlushPoint(lines, i))
      insertHere = true;
    if (insertHere) {
      appendScalarGMFlush(out, getLineIndent(lineRef));
      inserted = true;
    }
    out.append(lines[i]);
    if (i + 1 < lines.size() || hasTrailingNewline)
      out.push_back('\n');
  }

  if (!inserted)
    appendScalarGMFlush(out, "  ");
  return out;
}

static void rewriteScalarGMStoreFlushMarkers(std::string &cpp) {
  std::string out;
  out.reserve(cpp.size() + kRewriteOutputReserveExtra);

  llvm::SmallVector<std::string, 32> functionLines;
  bool inFunction = false;
  bool sawFunctionBrace = false;
  int braceDepth = 0;

  auto flushFunction = [&](bool hasTrailingNewline) {
    out.append(rewriteScalarGMStoreFlushMarkersInFunction(functionLines,
                                                         hasTrailingNewline));
    functionLines.clear();
    inFunction = false;
    sawFunctionBrace = false;
    braceDepth = 0;
  };

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    std::string line = split.first.str();
    bool hadNewline = !split.second.empty();
    ref = split.second;

    llvm::StringRef trimmed = llvm::StringRef(line).trim();
    if (!inFunction && isAICOREFunctionStart(trimmed))
      inFunction = true;

    if (!inFunction) {
      out.append(line);
      if (hadNewline)
        out.push_back('\n');
      continue;
    }

    functionLines.push_back(std::move(line));
    int delta = countBraceDelta(functionLines.back());
    if (delta != 0)
      sawFunctionBrace = true;
    braceDepth += delta;
    if (sawFunctionBrace && braceDepth == 0)
      flushFunction(hadNewline);
  }

  if (!functionLines.empty())
    flushFunction(false);
  cpp.swap(out);
}

static bool isPreprocessorDirectiveLine(llvm::StringRef trimmedLine) {
  return trimmedLine.starts_with("#");
}

static void rewriteMalformedVerbatimSemicolons(std::string &cpp) {
  if (cpp.empty())
    return;

  llvm::StringRef input(cpp);
  std::string rewritten;
  rewritten.reserve(cpp.size());

  bool prevWasPreprocessorDirective = false;
  size_t offset = 0;
  while (offset < input.size()) {
    size_t newlinePos = input.find('\n', offset);
    bool hasNewline = newlinePos != llvm::StringRef::npos;
    llvm::StringRef line =
        hasNewline ? input.slice(offset, newlinePos) : input.drop_front(offset);
    std::string current(line.str());
    llvm::StringRef trimmed = llvm::StringRef(current).trim();

    if (trimmed == ";" && prevWasPreprocessorDirective) {
      prevWasPreprocessorDirective = false;
    } else {
      if (isPreprocessorDirectiveLine(trimmed) && trimmed.ends_with(";")) {
        size_t semicolonPos = current.find_last_of(';');
        if (semicolonPos != std::string::npos)
          current.erase(semicolonPos, 1);
      } else if (!trimmed.empty() && !trimmed.starts_with("//") &&
                 !trimmed.starts_with("/*") && trimmed.ends_with(";;")) {
        size_t semicolonPos = current.find_last_of(';');
        if (semicolonPos != std::string::npos)
          current.erase(semicolonPos, 1);
      }

      rewritten.append(current);
      if (hasNewline)
        rewritten.push_back('\n');
      prevWasPreprocessorDirective =
          isPreprocessorDirectiveLine(llvm::StringRef(current).trim());
    }

    if (!hasNewline)
      break;
    offset = newlinePos + 1;
  }

  cpp.swap(rewritten);
}

static void stripHintMarkersWithPrefix(std::string &cpp,
                                       llvm::StringRef markerPrefix) {
  std::string out;
  out.reserve(cpp.size());
  size_t searchPos = 0;
  while (searchPos < cpp.size()) {
    size_t markerPos = cpp.find(markerPrefix.str(), searchPos);
    if (markerPos == std::string::npos) {
      out.append(cpp, searchPos, std::string::npos);
      break;
    }

    out.append(cpp, searchPos, markerPos - searchPos);
    size_t markerEnd = cpp.find("*/", markerPos + markerPrefix.size());
    if (markerEnd == std::string::npos) {
      out.append(cpp, markerPos, std::string::npos);
      break;
    }
    markerEnd += 2;
    while (markerEnd < cpp.size() &&
           (cpp[markerEnd] == '\r' || cpp[markerEnd] == '\n'))
      ++markerEnd;
    searchPos = markerEnd;
  }
  cpp.swap(out);
}

static void stripAllHintMarkers(std::string &cpp) {
  stripHintMarkersWithPrefix(cpp, "/* PTOAS_PROVENANCE:");
}

static std::string sanitizeCommentText(llvm::StringRef text) {
  auto hexDigit = [](unsigned value) -> char {
    return value < 10 ? static_cast<char>('0' + value)
                      : static_cast<char>('A' + (value - 10));
  };

  std::string sanitized;
  sanitized.reserve(text.size());
  for (unsigned char c : text.bytes()) {
    switch (c) {
    case '\n':
      sanitized.append("\\n");
      break;
    case '\r':
      sanitized.append("\\r");
      break;
    case '\t':
      sanitized.append("\\t");
      break;
    default:
      if (std::iscntrl(c)) {
        sanitized.push_back('\\');
        sanitized.push_back('x');
        sanitized.push_back(hexDigit((c >> 4) & 0xF));
        sanitized.push_back(hexDigit(c & 0xF));
      } else {
        sanitized.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  return sanitized;
}

static std::string buildHintMarker(llvm::StringRef prefix,
                                   llvm::ArrayRef<std::string> hints) {
  auto encodeHintMarkerToken = [](llvm::StringRef token) {
    auto hexDigit = [](unsigned value) -> char {
      return value < 10 ? static_cast<char>('0' + value)
                        : static_cast<char>('A' + (value - 10));
    };

    auto isSafeMarkerChar = [](unsigned char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
    };

    std::string encoded;
    encoded.reserve(token.size());
    for (unsigned char c : token.bytes()) {
      if (isSafeMarkerChar(c)) {
        encoded.push_back(static_cast<char>(c));
        continue;
      }
      encoded.push_back('%');
      encoded.push_back(hexDigit((c >> 4) & 0xF));
      encoded.push_back(hexDigit(c & 0xF));
    }
    return encoded;
  };

  std::string marker = ("/* " + prefix + ":").str();
  for (size_t i = 0; i < hints.size(); ++i) {
    if (i != 0)
      marker.push_back(',');
    marker.append(encodeHintMarkerToken(hints[i]));
  }
  marker.append(" */\n");
  return marker;
}

static std::optional<llvm::SmallVector<std::string, 4>>
parseNameHintMarker(llvm::StringRef markerBody) {
  auto decodeHintMarkerToken = [](llvm::StringRef token) {
    auto hexValue = [](char c) -> int {
      if (c >= '0' && c <= '9')
        return c - '0';
      if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
      if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
      return -1;
    };

    std::string decoded;
    decoded.reserve(token.size());
    for (size_t i = 0; i < token.size();) {
      if (token[i] == '%' && i + 2 < token.size()) {
        int hi = hexValue(token[i + 1]);
        int lo = hexValue(token[i + 2]);
        if (hi >= 0 && lo >= 0) {
          decoded.push_back(
              static_cast<char>((static_cast<unsigned>(hi) << 4) | lo));
          i += 3;
          continue;
        }
      }
      decoded.push_back(token[i]);
      ++i;
    }
    return decoded;
  };

  llvm::SmallVector<std::string, 4> hints;
  markerBody = markerBody.trim();
  if (markerBody.empty())
    return std::nullopt;

  size_t start = 0;
  while (start <= markerBody.size()) {
    size_t comma = markerBody.find(',', start);
    llvm::StringRef token = markerBody.slice(
        start, comma == llvm::StringRef::npos ? markerBody.size() : comma);
    token = token.trim();
    if (!token.empty())
      hints.push_back(decodeHintMarkerToken(token));
    if (comma == llvm::StringRef::npos)
      break;
    start = comma + 1;
  }

  if (hints.empty())
    return std::nullopt;
  return hints;
}

static void emitProvenanceComments(std::string &segment) {
  static constexpr llvm::StringLiteral kProvenancePrefix =
      "/* PTOAS_PROVENANCE:";
  std::string out;
  out.reserve(segment.size() + 128);
  size_t i = 0;
  while (i < segment.size()) {
    size_t mp = segment.find(kProvenancePrefix.str(), i);
    if (mp == std::string::npos) {
      out.append(segment, i, std::string::npos);
      break;
    }
    out.append(segment, i, mp - i);
    size_t me = segment.find("*/", mp + kProvenancePrefix.size());
    if (me == std::string::npos) {
      out.append(segment, i, std::string::npos);
      break;
    }
    auto names = parseNameHintMarker(
        llvm::StringRef(segment).slice(mp + kProvenancePrefix.size(), me));
    if (names && !names->empty()) {
      out.append("// pto: ");
      for (size_t idx = 0; idx < names->size(); ++idx) {
        if (idx != 0)
          out.append(", ");
        out.push_back('%');
        out.append(sanitizeCommentText((*names)[idx]));
      }
      out.push_back('\n');
    }
    me += 2;
    while (me < segment.size() &&
           (segment[me] == '\r' || segment[me] == '\n'))
      ++me;
    i = me;
  }
  segment.swap(out);
}

static void rewriteNameHintMarkers(std::string &cpp) {
  emitProvenanceComments(cpp);
  stripAllHintMarkers(cpp);
}

static llvm::SmallVector<std::string, 8>
collectExpressionProvenance(emitc::ExpressionOp expr) {
  llvm::SmallVector<std::string, 8> provenance;
  auto appendUnique = [&](llvm::ArrayRef<std::string> names) {
    for (const std::string &name : names) {
      if (name.empty())
        continue;
      if (std::find(provenance.begin(), provenance.end(), name) !=
          provenance.end())
        continue;
      provenance.push_back(name);
    }
  };

  expr.walk<WalkOrder::PreOrder>([&](Operation *nested) {
    if (nested == expr.getOperation())
      return WalkResult::advance();
    if (nested->getNumResults() == 0 || isa<emitc::VerbatimOp>(nested))
      return WalkResult::advance();
    appendUnique(getRawResultProvenance(nested));
    return WalkResult::advance();
  });
  appendUnique(getRawResultProvenance(expr.getOperation()));
  return provenance;
}

static void annotateEmitCProvenanceHints(ModuleOp module) {
  struct ProvenanceMarker {
    Operation *op = nullptr;
    llvm::SmallVector<std::string, 8> names;
  };

  llvm::SmallVector<ProvenanceMarker, 32> opsToAnnotate;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getNumResults() == 0 || isa<emitc::VerbatimOp>(op))
      return WalkResult::advance();

    if (auto expr = dyn_cast<emitc::ExpressionOp>(op)) {
      llvm::SmallVector<std::string, 8> provenance = collectExpressionProvenance(expr);
      if (provenance.empty())
        return WalkResult::skip();
      opsToAnnotate.push_back(
          ProvenanceMarker{op, llvm::SmallVector<std::string, 8>(provenance)});
      return WalkResult::skip();
    }

    if (op->getParentOfType<emitc::ExpressionOp>())
      return WalkResult::advance();
    llvm::SmallVector<std::string, 4> provenance = getRawResultProvenance(op);
    if (provenance.empty())
      return WalkResult::advance();
    opsToAnnotate.push_back(ProvenanceMarker{
        op, llvm::SmallVector<std::string, 8>(provenance.begin(), provenance.end())});
    return WalkResult::advance();
  });

  OpBuilder builder(module.getContext());
  for (const ProvenanceMarker &marker : opsToAnnotate) {
    if (!marker.names.empty()) {
      builder.setInsertionPoint(marker.op);
      builder.create<emitc::VerbatimOp>(
          marker.op->getLoc(),
          builder.getStringAttr(buildHintMarker("PTOAS_PROVENANCE",
                                               marker.names)));
    }
  }
}

static LogicalResult reorderEmitCFunctions(ModuleOp module) {
  llvm::SmallVector<emitc::FuncOp> declarations;
  llvm::SmallVector<emitc::FuncOp> definitions;
  llvm::DenseMap<StringAttr, emitc::FuncOp> definitionsByName;

  for (auto func : module.getOps<emitc::FuncOp>()) {
    if (func.isDeclaration()) {
      declarations.push_back(func);
      continue;
    }
    definitions.push_back(func);
    definitionsByName[func.getSymNameAttr()] = func;
  }

  llvm::DenseMap<Operation *, unsigned> indegree;
  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>> outgoing;
  for (auto func : definitions)
    indegree[func.getOperation()] = 0;

  for (auto caller : definitions) {
    Operation *callerOp = caller.getOperation();
    llvm::SmallPtrSet<Operation *, kSeenCalleeInlineCapacity> seenCallees;
    bool hasCycle = false;
    caller.walk([&](emitc::CallOp call) {
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        return;
      auto it = definitionsByName.find(calleeAttr.getLeafReference());
      if (it == definitionsByName.end())
        return;
      Operation *calleeOp = it->second.getOperation();
      if (calleeOp == callerOp) {
        hasCycle = true;
        return;
      }
      if (!seenCallees.insert(calleeOp).second)
        return;
      outgoing[calleeOp].push_back(callerOp);
      ++indegree[callerOp];
    });
    if (hasCycle) {
      caller.emitOpError()
          << "recursive function calls are not supported for EmitC C++ "
             "emission";
      return failure();
    }
  }

  llvm::SmallVector<Operation *> ready;
  for (auto func : definitions) {
    if (indegree[func.getOperation()] == 0)
      ready.push_back(func.getOperation());
  }

  llvm::SmallVector<emitc::FuncOp> sortedDefinitions;
  while (!ready.empty()) {
    Operation *next = ready.front();
    ready.erase(ready.begin());
    auto nextFunc = cast<emitc::FuncOp>(next);
    sortedDefinitions.push_back(nextFunc);

    for (Operation *user : outgoing[next]) {
      unsigned &userIndegree = indegree[user];
      if (--userIndegree == 0)
        ready.push_back(user);
    }
  }

  if (sortedDefinitions.size() != definitions.size()) {
    module.emitError()
        << "cyclic function call graph is not supported for EmitC C++ emission";
    return failure();
  }

  if (declarations.empty() && definitions.size() <= 1)
    return success();

  llvm::SmallVector<emitc::FuncOp> desiredOrder;
  desiredOrder.append(declarations.begin(), declarations.end());
  desiredOrder.append(sortedDefinitions.begin(), sortedDefinitions.end());

  Block &body = module.getBodyRegion().front();
  Operation *anchor = nullptr;
  for (Operation &op : body.getOperations()) {
    if (isa<emitc::FuncOp>(op)) {
      anchor = &op;
      break;
    }
  }
  if (!anchor)
    return success();

  auto advanceAnchor = [&]() {
    while (anchor) {
      anchor = anchor->getNextNode();
      if (!anchor || isa<emitc::FuncOp>(anchor))
        return;
    }
  };

  for (auto func : desiredOrder) {
    if (func.getOperation() == anchor) {
      advanceAnchor();
      continue;
    }
    if (anchor)
      func->moveBefore(anchor);
    else
      func->moveBefore(&body, body.end());
  }

  return success();
}

static bool shouldDeclareVariablesAtTop(ModuleOp module) {
  auto hasMultiBlockFunc = [](auto func) { return func.getBlocks().size() > 1; };
  return llvm::any_of(module.getOps<func::FuncOp>(), hasMultiBlockFunc) ||
         llvm::any_of(module.getOps<emitc::FuncOp>(), hasMultiBlockFunc);
}

static LogicalResult emitCppFromEmitCModule(
    ModuleOp module, bool emitAddPtrTrace, std::string &cppOutput) {
  if (failed(reorderEmitCFunctions(module)))
    return failure();
  annotateEmitCProvenanceHints(module);

  llvm::raw_string_ostream cppOS(cppOutput);
  if (failed(emitc::translateToCpp(module, cppOS,
                                   /*declareVariablesAtTop=*/
                                   shouldDeclareVariablesAtTop(module)))) {
    return failure();
  }
  cppOS.flush();

  rewriteTileGetSetValueMarkers(cppOutput);
  rewriteAsyncEventMarkers(cppOutput);
  rewritePtrScalarMarkers(cppOutput);
  rewriteScalarGMStoreFlushMarkers(cppOutput);
  rewriteEventIdArrayMarkers(cppOutput);
  pto::rewriteLastUseMarkersInCpp(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTrace);
  rewriteMalformedVerbatimSemicolons(cppOutput);
  rewriteScalarConstantDecls(cppOutput);
  rewriteHoistedGlobalTensorDecls(cppOutput);
  rewriteNameHintMarkers(cppOutput);
  return success();
}

static bool isGeneratedGlobalTensorDecl(llvm::StringRef trimmed,
                                        llvm::StringRef &decl,
                                        llvm::StringRef &varName) {
  if (!trimmed.starts_with("GlobalTensor<") || !trimmed.ends_with(";") ||
      trimmed.contains('=') || trimmed.contains('(')) {
    return false;
  }

  decl = trimmed.drop_back().rtrim();
  size_t lastWs = decl.find_last_of(" \t");
  if (lastWs == llvm::StringRef::npos)
    return false;
  varName = decl.drop_front(lastWs + 1);
  if (!varName.starts_with("v") || varName.size() <= 1)
    return false;
  return llvm::all_of(varName.drop_front(1),
                      [](char c) { return std::isdigit(c); });
}

static void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
  std::string out;
  out.reserve(cpp.size() + kRewriteOutputReserveExtra);

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;

    llvm::StringRef trimmed = line.trim();
    bool rewritten = false;
    llvm::StringRef decl;
    llvm::StringRef varName;
    if (isGeneratedGlobalTensorDecl(trimmed, decl, varName)) {
      size_t indentLen = line.find_first_not_of(" \t");
      if (indentLen == std::string::npos)
        indentLen = 0;
      llvm::StringRef indent = line.take_front(indentLen);

      out.append(indent.str());
      out.append(decl.str());
      out.append("(nullptr);");
      rewritten = true;
    }

    if (!rewritten)
      out.append(line.str());
    if (!rest.empty())
      out.push_back('\n');
    ref = rest;
  }

  cpp.swap(out);
}

static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
  size_t searchPos = 0;
  bool changed = false;
  for (auto call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos);
       call; call = findNextMarkerCall(cpp, "PTOAS__ADDPTR_TRACE", searchPos)) {
    if (call->rparenPos == std::string::npos) {
      searchPos = call->markerPos + 1;
      continue;
    }
    if (call->args.size() != kMarkerRewriteTernaryArgCount) {
      searchPos = call->rparenPos + 1;
      continue;
    }

    std::string replacement;
    if (showTrace) {
      replacement.reserve(kRewriteOutputReserveExtra);
      replacement.append("/* ADDPTR_TRACE: ");
      replacement.append(call->args[0].str());
      replacement.append(" = ");
      replacement.append(call->args[1].str());
      replacement.append(" + ");
      replacement.append(call->args[2].str());
      replacement.append(" */");
    }

    size_t replaceEnd = call->rparenPos;
    if (!showTrace) {
      size_t i = call->rparenPos + 1;
      while (i < cpp.size() && std::isspace(static_cast<unsigned char>(cpp[i])))
        ++i;
      if (i < cpp.size() && cpp[i] == ';')
        replaceEnd = i;
    }

    cpp.replace(call->markerPos, (replaceEnd - call->markerPos) + 1,
                replacement);
    changed = true;
    searchPos = call->markerPos + replacement.size();
  }
  return changed;
}

static LogicalResult finalizeEmitCModuleForCppEmissionImpl(
    ModuleOp module, const FunctionBlockArgHintMap &blockArgHints,
    bool emitAddPtrTrace, std::string &cppOutput) {
  (void)blockArgHints;
  splitDerivedSingleResultProvenanceLocs(module.getOperation());
  narrowUnusedMultiResultProvenanceLocs(module.getOperation());
  splitDerivedSingleResultProvenanceLocs(module.getOperation());
  dropEmptyEmitCExpressions(module.getOperation());
  materializeControlFlowOperands(module.getOperation());
  normalizeEmitCIntegerAttrsForCppEmission(module.getOperation());
  if (failed(emitCppFromEmitCModule(module, emitAddPtrTrace, cppOutput)))
    return failure();
  return success();
}

} // namespace

LogicalResult mlir::pto::finalizeEmitCModuleForCppEmission(
    ModuleOp module, const FunctionBlockArgHintMap &blockArgHints,
    bool emitAddPtrTrace, std::string &cppOutput) {
  return ::finalizeEmitCModuleForCppEmissionImpl(module, blockArgHints,
                                                 emitAddPtrTrace, cppOutput);
}
