// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- AllocToPointerCast.cpp - convert alloc_tile to pto.pointer_cast. ----===//
//===----------------------------------------------------------------------===//

#include "AllocToPointerCast.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ALLOCTOPOINTERCAST
#include "PTO/Transforms/Passes.h.inc"

} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {} // namespace

namespace {
constexpr uint64_t kDefaultAllocAlignmentBytes = 4096;
constexpr uint64_t kF16ByteSize = 2;
constexpr uint64_t kF32ByteSize = 4;
constexpr unsigned kBitsPerByte = 8;
constexpr size_t kStaticValidShapeRank = 2;

static SmallVector<uint64_t> getAllocatedOffsets(
    pto::AllocTileOp op, pto::TileBufType tileType,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets,
    uint64_t &fallbackNextOffset) {
  auto iter = buffer2Offsets.find(op.getResult());
  SmallVector<uint64_t> offsets;
  if (iter != buffer2Offsets.end())
    offsets = iter->second;

  if (offsets.empty()) {
    // Estimate tile size in bytes using the static tile descriptor.
    uint64_t bytes = kDefaultAllocAlignmentBytes;
    uint64_t elemBytes = 0;
    Type elemTy = tileType.getElementType();
    if (elemTy.isF16() || elemTy.isBF16())
      elemBytes = kF16ByteSize;
    else if (elemTy.isF32())
      elemBytes = kF32ByteSize;
    else if (auto it = dyn_cast<IntegerType>(elemTy))
      elemBytes = it.getWidth() / kBitsPerByte;

    if (elemBytes != 0) {
      uint64_t numel = 1;
      bool allStatic = true;
      for (int64_t d : tileType.getShape()) {
        if (d == ShapedType::kDynamic) {
          allStatic = false;
          break;
        }
        numel *= static_cast<uint64_t>(d);
      }
      if (allStatic && numel != 0)
        bytes = numel * elemBytes;
    }

    uint64_t stride = ((bytes + kDefaultAllocAlignmentBytes - 1) /
                       kDefaultAllocAlignmentBytes) *
                      kDefaultAllocAlignmentBytes;
    uint64_t off = fallbackNextOffset;
    fallbackNextOffset +=
        std::max<uint64_t>(stride, kDefaultAllocAlignmentBytes);
    offsets.push_back(off);
  }
  return offsets;
}

static std::pair<Value, Value>
getValidShapeValues(pto::AllocTileOp op, pto::TileBufType tileType,
                    PatternRewriter &rewriter) {
  Value vRow = op.getValidRow();
  Value vCol = op.getValidCol();
  auto validShape = tileType.getValidShape();
  if (validShape.size() >= kStaticValidShapeRank) {
    auto indexType = rewriter.getIndexType();
    Location loc = op.getLoc();
    if (!vRow && validShape[0] >= 0)
      vRow = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIndexAttr(validShape[0]));
    if (!vCol && validShape[1] >= 0)
      vCol = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIndexAttr(validShape[1]));
  }
  return {vRow, vCol};
}
} // namespace

LogicalResult AllocTileOpToPointerCastOpPattern::matchAndRewrite(
    pto::AllocTileOp op, PatternRewriter &rewriter) const {
  // Manual-address alloc_tile is already fully bound and must not be remapped.
  if (op.getAddr())
    return failure();

  auto tileType = dyn_cast<pto::TileBufType>(op.getResult().getType());
  if (!tileType)
    return failure();

  SmallVector<uint64_t> offsets = getAllocatedOffsets(
      op, tileType, buffer2Offsets, fallbackNextOffset);
  SmallVector<Value> addrs;
  addrs.reserve(offsets.size());
  for (uint64_t offset : offsets) {
    auto constantIntOffsetOp =
        rewriter.create<arith::ConstantIntOp>(op->getLoc(), offset, 64);
    addrs.push_back(constantIntOffsetOp);
  }

  auto [vRow, vCol] = getValidShapeValues(op, tileType, rewriter);
  auto ptoPointerCastOp = rewriter.create<pto::PointerCastOp>(
      op.getLoc(), tileType, ValueRange(addrs), vRow ? vRow : Value(),
      vCol ? vCol : Value(), tileType.getConfigAttr());

  rewriter.replaceOp(op, ptoPointerCastOp->getResults());
  return success();
}
