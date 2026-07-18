// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTOASNameHints.h"
#include "ptoas.h"

#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

using namespace mlir;
using namespace pto;

namespace {

static llvm::SmallVector<std::string, 4> getValueNameHints(Value value);

static bool isCppIdentifierStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

static bool isCppIdentifierChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static std::optional<std::string>
getTextualNameFromSMRange(llvm::SMRange range) {
  if (!range.Start.isValid() || !range.End.isValid())
    return std::nullopt;
  const char *begin = range.Start.getPointer();
  const char *end = range.End.getPointer();
  if (!begin || !end || end < begin)
    return std::nullopt;
  llvm::StringRef name(begin, static_cast<size_t>(end - begin));
  if (name.empty())
    return std::nullopt;
  name = name.trim();
  if (name.consume_front("%") && name.empty())
    return std::nullopt;
  return name.str();
}

static llvm::SmallVector<std::string, 4>
expandTextualResultGroupHints(const AsmParserState::OperationDefinition &opDef,
                              unsigned groupIndex) {
  llvm::SmallVector<std::string, 4> hints;
  if (groupIndex >= opDef.resultGroups.size())
    return hints;
  const auto &group = opDef.resultGroups[groupIndex];
  std::optional<std::string> baseName =
      getTextualNameFromSMRange(group.definition.loc);
  if (!baseName)
    return hints;

  unsigned resultStart = group.startIndex;
  unsigned resultEnd = groupIndex + 1 == opDef.resultGroups.size()
                           ? opDef.op->getNumResults()
                           : opDef.resultGroups[groupIndex + 1].startIndex;
  if (resultStart >= resultEnd)
    return hints;
  if (resultEnd - resultStart == 1) {
    hints.push_back(*baseName);
    return hints;
  }
  for (unsigned idx = resultStart; idx < resultEnd; ++idx)
    hints.push_back(*baseName + "#" + std::to_string(idx - resultStart));
  return hints;
}

static std::string sanitizeCppIdentifier(llvm::StringRef name) {
  std::string sanitized;
  sanitized.reserve(name.size() + 4);

  auto appendUnderscore = [&]() {
    if (sanitized.empty() || sanitized.back() != '_')
      sanitized.push_back('_');
  };

  for (char c : name) {
    if (isCppIdentifierChar(c))
      sanitized.push_back(c);
    else
      appendUnderscore();
  }

  while (!sanitized.empty() && sanitized.back() == '_')
    sanitized.pop_back();

  if (sanitized.empty())
    return {};
  if (!isCppIdentifierStart(sanitized.front()))
    sanitized.insert(sanitized.begin(), '_');
  return sanitized;
}

static void appendLocationNameHints(Location loc,
                                    llvm::SmallVectorImpl<std::string> &hints) {
  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    std::string sanitized = sanitizeCppIdentifier(nameLoc.getName().getValue());
    if (!sanitized.empty())
      hints.push_back(std::move(sanitized));
    return;
  }

  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    if (Attribute metadata = fusedLoc.getMetadata()) {
      if (auto strAttr = dyn_cast<StringAttr>(metadata)) {
        std::string sanitized = sanitizeCppIdentifier(strAttr.getValue());
        if (!sanitized.empty())
          hints.push_back(std::move(sanitized));
        return;
      }
      if (auto arrayAttr = dyn_cast<ArrayAttr>(metadata)) {
        for (Attribute attr : arrayAttr) {
          auto strAttr = dyn_cast<StringAttr>(attr);
          if (!strAttr)
            continue;
          std::string sanitized = sanitizeCppIdentifier(strAttr.getValue());
          if (!sanitized.empty())
            hints.push_back(std::move(sanitized));
        }
        if (!hints.empty())
          return;
      }
    }
    return;
  }

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(loc)) {
    appendLocationNameHints(callSiteLoc.getCallee(), hints);
    if (hints.empty())
      appendLocationNameHints(callSiteLoc.getCaller(), hints);
  }
}

static bool hasLocationNameHints(Location loc) {
  llvm::SmallVector<std::string, 4> hints;
  appendLocationNameHints(loc, hints);
  return !hints.empty();
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

static void applyValueNameHints(Value value, llvm::ArrayRef<std::string> hints) {
  if (!value || hints.empty() || hasLocationNameHints(value.getLoc()))
    return;
  value.setLoc(attachLocationNameHints(value.getLoc(), hints, value.getContext()));
}

static void applyOperationResultNameHints(Operation *op,
                                          llvm::ArrayRef<std::string> hints) {
  if (!op || op->getNumResults() == 0 || hints.empty() ||
      hasLocationNameHints(op->getLoc()))
    return;

  llvm::SmallVector<std::string, 4> limitedHints;
  limitedHints.reserve(std::min<size_t>(op->getNumResults(), hints.size()));
  for (size_t i = 0, e = std::min<size_t>(op->getNumResults(), hints.size());
       i < e; ++i)
    limitedHints.push_back(hints[i]);
  if (limitedHints.empty())
    return;

  op->setLoc(attachLocationNameHints(op->getLoc(), limitedHints, op->getContext()));
}

static void collectNonEntryBlocksInSourceOrder(
    Operation *op, llvm::SmallVectorImpl<Block *> &blocks) {
  for (Region &region : op->getRegions()) {
    bool isEntryBlock = true;
    for (Block &block : region) {
      if (!isEntryBlock && block.getNumArguments() != 0)
        blocks.push_back(&block);
      isEntryBlock = false;
      for (Operation &nestedOp : block)
        collectNonEntryBlocksInSourceOrder(&nestedOp, blocks);
    }
  }
}

static llvm::SmallVector<std::string, 4> getValueNameHints(Value value) {
  llvm::SmallVector<std::string, 4> hints;
  if (!value)
    return hints;
  appendLocationNameHints(value.getLoc(), hints);
  if (hints.size() > 1)
    hints.resize(1);
  return hints;
}

} // namespace

mlir::pto::FunctionBlockArgHintMap
mlir::pto::collectFunctionBlockArgNameHints(ModuleOp module) {
  FunctionBlockArgHintMap hintsByFunction;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    llvm::SmallVector<Block *, 8> nonEntryBlocks;
    collectNonEntryBlocksInSourceOrder(func.getOperation(), nonEntryBlocks);
    if (nonEntryBlocks.empty())
      continue;

    llvm::SmallVector<llvm::SmallVector<std::string, 4>, 4> blockHints;
    blockHints.reserve(nonEntryBlocks.size());
    for (Block *block : nonEntryBlocks) {
      llvm::SmallVector<std::string, 4> argHints;
      bool hasAllHints = block->getNumArguments() != 0;
      for (BlockArgument arg : block->getArguments()) {
        llvm::SmallVector<std::string, 4> hints = getValueNameHints(arg);
        if (hints.empty()) {
          hasAllHints = false;
          break;
        }
        argHints.push_back(std::move(hints.front()));
      }
      if (hasAllHints)
        blockHints.push_back(std::move(argHints));
    }

    if (!blockHints.empty())
      hintsByFunction[func.getSymNameAttr()] = std::move(blockHints);
  }
  return hintsByFunction;
}

void mlir::pto::applyTextualNameHintsToModule(ModuleOp module,
                                              const AsmParserState &parserState) {
  if (!module)
    return;

  for (const AsmParserState::BlockDefinition &blockDef : parserState.getBlockDefs()) {
    if (!blockDef.block)
      continue;
    for (auto [argIndex, argDef] : llvm::enumerate(blockDef.arguments)) {
      if (argIndex >= blockDef.block->getNumArguments())
        break;
      std::optional<std::string> hint = getTextualNameFromSMRange(argDef.loc);
      if (!hint)
        continue;
      applyValueNameHints(blockDef.block->getArgument(argIndex),
                          llvm::ArrayRef<std::string>{*hint});
    }
  }

  for (const AsmParserState::OperationDefinition &opDef : parserState.getOpDefs()) {
    if (!opDef.op || opDef.op->getNumResults() == 0)
      continue;

    llvm::SmallVector<std::string, 4> hints;
    hints.reserve(opDef.op->getNumResults());
    for (unsigned groupIndex = 0, e = opDef.resultGroups.size(); groupIndex < e;
         ++groupIndex) {
      llvm::SmallVector<std::string, 4> groupHints =
          expandTextualResultGroupHints(opDef, groupIndex);
      hints.append(groupHints.begin(), groupHints.end());
    }
    if (hints.empty())
      continue;
    applyOperationResultNameHints(opDef.op, hints);
  }
}
