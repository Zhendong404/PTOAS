// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- FoldTileBufIntrinsics.cpp ------------------------------------------===//
//
// After TileLang DSL template functions are inlined, the IR contains
// structured-view intrinsics that reference template parameters:
//
// tile_buf family:
//   - pto.tile_buf_addr   → extract memref address from tile_buf
//   - pto.tile_valid_rows → extract valid row count
//   - pto.tile_valid_cols → extract valid column count
//
// tensor_view family:
//   - pto.tensor_view_addr       → extract memref/ptr from tensor_view
//   - pto.get_tensor_view_dim    → extract dimension size
//   - pto.get_tensor_view_stride → extract dimension stride
//
// This pass resolves them against the concrete values at the call site. For
// tile_buf intrinsics, the pass reads native !pto.tile_buf producer metadata
// and materializes the requested VPTO-side memref/ptr/index values. For
// tensor_view intrinsics, the pass reads native pto.make_tensor_view /
// pto.partition_view producer metadata and materializes VPTO-side
// memref/ptr/index values without relying on the removed View2Memref chain.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_FOLDTILEBUFINTRINSICS
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

static bool getConstIndexValue(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexValue(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexValue(truncOp.getIn(), out);
  return false;
}

struct TileBufMetadata {
  pto::TileBufType tileType;
  Value address;
  Value validRow;
  Value validCol;
};

struct TensorViewMetadata {
  Type viewType;
  Value basePtr;
  SmallVector<int64_t, 4> shape;
  SmallVector<Value, 4> shapeOperands;
  SmallVector<Value, 4> strides;
  SmallVector<Value, 4> offsets;
};

static bool isZeroIndex(Value value) {
  int64_t constant = 0;
  return getConstIndexValue(value, constant) && constant == 0;
}

static bool isCompatibleTensorViewShape(ArrayRef<int64_t> sourceShape,
                                        ArrayRef<int64_t> targetShape) {
  if (sourceShape.size() != targetShape.size())
    return false;

  for (auto [sourceDim, targetDim] : llvm::zip(sourceShape, targetShape)) {
    if (targetDim != ShapedType::kDynamic && sourceDim != targetDim)
      return false;
  }
  return true;
}

static FailureOr<TensorViewMetadata>
adaptNativeTensorViewCastMetadata(TensorViewMetadata metadata,
                                  Type castResultType, Operation *user) {
  auto sourceTensorViewType = dyn_cast<pto::TensorViewType>(metadata.viewType);
  auto sourcePartitionViewType =
      dyn_cast<pto::PartitionTensorViewType>(metadata.viewType);
  auto targetTensorViewType = dyn_cast<pto::TensorViewType>(castResultType);
  auto targetPartitionViewType =
      dyn_cast<pto::PartitionTensorViewType>(castResultType);

  auto emitUnsupportedCast = [&]() -> FailureOr<TensorViewMetadata> {
    return user->emitError(
               "FoldTileBufIntrinsics: unsupported native tensor_view cast "
               "shape change from ")
           << metadata.viewType << " to " << castResultType;
  };

  ArrayRef<int64_t> sourceShape;
  ArrayRef<int64_t> targetShape;
  Type sourceElementType;
  Type targetElementType;

  if (sourceTensorViewType && targetTensorViewType) {
    sourceShape = sourceTensorViewType.getShape();
    targetShape = targetTensorViewType.getShape();
    sourceElementType = sourceTensorViewType.getElementType();
    targetElementType = targetTensorViewType.getElementType();
  } else if (sourcePartitionViewType && targetPartitionViewType) {
    sourceShape = sourcePartitionViewType.getShape();
    targetShape = targetPartitionViewType.getShape();
    sourceElementType = sourcePartitionViewType.getElementType();
    targetElementType = targetPartitionViewType.getElementType();
  } else {
    return emitUnsupportedCast();
  }

  if (sourceElementType != targetElementType ||
      !isCompatibleTensorViewShape(sourceShape, targetShape))
    return emitUnsupportedCast();

  if (metadata.shapeOperands.size() != targetShape.size() ||
      metadata.strides.size() != targetShape.size())
    return user->emitError(
        "FoldTileBufIntrinsics: native tensor_view cast metadata rank "
        "mismatch");

  metadata.viewType = castResultType;
  metadata.shape.assign(targetShape.begin(), targetShape.end());
  return metadata;
}

static FailureOr<TensorViewMetadata>
resolveNativeTensorViewMetadata(Value tensorView, Operation *user) {
  Type viewType = tensorView.getType();
  auto tensorViewType = dyn_cast<pto::TensorViewType>(viewType);
  auto partitionViewType = dyn_cast<pto::PartitionTensorViewType>(viewType);
  if (!tensorViewType && !partitionViewType) {
    return user->emitError(
               "FoldTileBufIntrinsics: expected native tensor_view or "
               "partition_tensor_view input, got ")
           << viewType;
  }

  Operation *def = tensorView.getDefiningOp();
  if (!def) {
    return user->emitError(
        "FoldTileBufIntrinsics: cannot resolve tensor_view metadata from a "
        "block argument; pass a native pto.make_tensor_view or "
        "pto.partition_view producer");
  }

  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (castOp.getNumOperands() == 1 &&
        isa<MemRefType>(castOp.getOperand(0).getType())) {
      return user->emitError(
          "FoldTileBufIntrinsics: memref-sourced tensor_view bridge is not "
          "supported; expected native pto.make_tensor_view / "
          "pto.partition_view producer");
    }
    if (castOp.getNumOperands() == 1 &&
        isa<pto::TensorViewType, pto::PartitionTensorViewType>(
            castOp.getOperand(0).getType())) {
      FailureOr<TensorViewMetadata> source =
          resolveNativeTensorViewMetadata(castOp.getOperand(0), user);
      if (failed(source))
        return failure();
      return adaptNativeTensorViewCastMetadata(*source, viewType, user);
    }
    return user->emitError(
               "FoldTileBufIntrinsics: unsupported tensor_view cast producer ")
           << castOp->getName()
           << "; expected native pto.make_tensor_view / pto.partition_view";
  }

  if (auto makeOp = dyn_cast<pto::MakeTensorViewOp>(def)) {
    auto resultType = dyn_cast<pto::TensorViewType>(makeOp.getResult().getType());
    if (!resultType) {
      return user->emitError(
          "FoldTileBufIntrinsics: pto.make_tensor_view result must be "
          "!pto.tensor_view");
    }

    int64_t rank = resultType.getRank();
    if (static_cast<int64_t>(makeOp.getShape().size()) != rank ||
        static_cast<int64_t>(makeOp.getStrides().size()) != rank) {
      return user->emitError(
          "FoldTileBufIntrinsics: pto.make_tensor_view shape/stride operand "
          "counts must match tensor_view rank");
    }

    TensorViewMetadata metadata;
    metadata.viewType = viewType;
    metadata.basePtr = makeOp.getPtr();
    metadata.shape.assign(resultType.getShape().begin(),
                          resultType.getShape().end());
    metadata.shapeOperands.append(makeOp.getShape().begin(),
                                  makeOp.getShape().end());
    metadata.strides.append(makeOp.getStrides().begin(),
                            makeOp.getStrides().end());
    return metadata;
  }

  if (auto partOp = dyn_cast<pto::PartitionViewOp>(def)) {
    auto resultType =
        dyn_cast<pto::PartitionTensorViewType>(partOp.getResult().getType());
    if (!resultType) {
      return user->emitError(
          "FoldTileBufIntrinsics: pto.partition_view result must be "
          "!pto.partition_tensor_view");
    }

    FailureOr<TensorViewMetadata> source =
        resolveNativeTensorViewMetadata(partOp.getSource(), user);
    if (failed(source))
      return failure();

    int64_t rank = resultType.getRank();
    if (static_cast<int64_t>(partOp.getOffsets().size()) != rank ||
        static_cast<int64_t>(partOp.getSizes().size()) != rank ||
        static_cast<int64_t>(source->strides.size()) != rank) {
      return user->emitError(
          "FoldTileBufIntrinsics: pto.partition_view offset/size counts and "
          "source strides must match partition_tensor_view rank");
    }

    TensorViewMetadata metadata;
    metadata.viewType = viewType;
    metadata.basePtr = source->basePtr;
    metadata.shape.assign(resultType.getShape().begin(),
                          resultType.getShape().end());
    metadata.shapeOperands.append(partOp.getSizes().begin(),
                                  partOp.getSizes().end());
    metadata.strides = std::move(source->strides);
    metadata.offsets.append(partOp.getOffsets().begin(),
                            partOp.getOffsets().end());
    return metadata;
  }

  return user->emitError(
             "FoldTileBufIntrinsics: unsupported native tensor_view producer ")
         << def->getName()
         << "; expected pto.make_tensor_view or pto.partition_view";
}

static Value materializeTensorViewElementOffset(OpBuilder &builder,
                                                Location loc,
                                                TensorViewMetadata metadata) {
  Value linearOffset;
  for (auto [offset, stride] : llvm::zip(metadata.offsets, metadata.strides)) {
    if (isZeroIndex(offset))
      continue;
    Value term = builder.create<arith::MulIOp>(loc, offset, stride);
    linearOffset =
        linearOffset ? builder.create<arith::AddIOp>(loc, linearOffset, term)
                     : term;
  }
  return linearOffset;
}

static FailureOr<Value>
materializeTensorViewPtr(OpBuilder &builder, Location loc,
                         pto::PtrType resultPtrType,
                         TensorViewMetadata metadata) {
  Value basePtr = metadata.basePtr.getType() == resultPtrType
                      ? metadata.basePtr
                      : builder.create<pto::CastPtrOp>(loc, resultPtrType,
                                                       metadata.basePtr)
                            .getResult();
  Value linearOffset =
      materializeTensorViewElementOffset(builder, loc, metadata);
  if (!linearOffset)
    return basePtr;
  return builder.create<pto::AddPtrOp>(loc, resultPtrType, basePtr,
                                       linearOffset)
      .getResult();
}

static FailureOr<Value>
materializeTensorViewAddress(OpBuilder &builder, pto::TensorViewAddrOp addrOp,
                             TensorViewMetadata metadata) {
  Type resultType = addrOp.getDst().getType();
  Location loc = addrOp.getLoc();

  if (auto resultPtrType = dyn_cast<pto::PtrType>(resultType))
    return materializeTensorViewPtr(builder, loc, resultPtrType, metadata);

  auto resultMemrefType = dyn_cast<MemRefType>(resultType);
  if (!resultMemrefType) {
    return addrOp.emitError(
        "FoldTileBufIntrinsics: tensor_view_addr result must be memref or "
        "!pto.ptr");
  }

  auto gmSpace =
      pto::AddressSpaceAttr::get(builder.getContext(), pto::AddressSpace::GM);
  auto ptrType = pto::PtrType::get(builder.getContext(),
                                   resultMemrefType.getElementType(), gmSpace);
  FailureOr<Value> ptr =
      materializeTensorViewPtr(builder, loc, ptrType, metadata);
  if (failed(ptr))
    return failure();
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{resultMemrefType},
                                          ValueRange{*ptr})
      .getResult(0);
}

static FailureOr<unsigned> getConstantDimIndex(Value dimIndex, Operation *user,
                                               int64_t rank,
                                               StringRef intrinsicName) {
  int64_t dimIdx = 0;
  if (!getConstIndexValue(dimIndex, dimIdx)) {
    return user->emitError()
           << "FoldTileBufIntrinsics: " << intrinsicName
           << " requires a constant dim index";
  }
  if (dimIdx < 0 || dimIdx >= rank) {
    return user->emitError()
           << "FoldTileBufIntrinsics: " << intrinsicName
           << " dim index out of bounds";
  }
  return static_cast<unsigned>(dimIdx);
}

static FailureOr<Value>
resolveTensorViewDim(OpBuilder &builder, Operation *user,
                     TensorViewMetadata metadata, unsigned dim) {
  if (metadata.shape.size() <= dim || metadata.shapeOperands.size() <= dim)
    return user->emitError(
        "FoldTileBufIntrinsics: tensor_view metadata rank mismatch");

  int64_t staticDim = metadata.shape[dim];
  if (staticDim != ShapedType::kDynamic)
    return builder.create<arith::ConstantIndexOp>(user->getLoc(), staticDim)
        .getResult();
  return metadata.shapeOperands[dim];
}

static FailureOr<std::pair<int64_t, int64_t>>
getTileBufPhysicalStrides(pto::TileBufType tileType) {
  auto shape = tileType.getShape();
  if (shape.size() != 2 || shape[0] == ShapedType::kDynamic ||
      shape[1] == ShapedType::kDynamic)
    return failure();

  auto configAttr = tileType.getConfigAttr();
  if (!configAttr)
    return failure();

  int32_t bLayout = 0;
  if (auto attr = dyn_cast<pto::BLayoutAttr>(configAttr.getBLayout()))
    bLayout = static_cast<int32_t>(attr.getValue());
  else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getBLayout()))
    bLayout = static_cast<int32_t>(intAttr.getInt());

  int32_t sLayout = 0;
  if (auto attr = dyn_cast<pto::SLayoutAttr>(configAttr.getSLayout()))
    sLayout = static_cast<int32_t>(attr.getValue());
  else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getSLayout()))
    sLayout = static_cast<int32_t>(intAttr.getInt());

  int64_t rows = shape[0];
  int64_t cols = shape[1];
  bool boxed = sLayout != 0;
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (boxed) {
    int32_t fractal = 512;
    if (auto attr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
      fractal = static_cast<int32_t>(attr.getInt());

    unsigned elemBytes =
        pto::getPTOStorageElemByteSize(tileType.getElementType());
    if (elemBytes == 0)
      return failure();

    switch (fractal) {
    case 1024:
      innerRows = 16;
      innerCols = 16;
      break;
    case 32:
      innerRows = 16;
      innerCols = 2;
      break;
    case 512:
      if (sLayout == 1) {
        innerRows = 16;
        innerCols = 32 / elemBytes;
      } else if (sLayout == 2) {
        innerRows = 32 / elemBytes;
        innerCols = 16;
      } else {
        return failure();
      }
      break;
    default:
      return failure();
    }
    if (innerRows <= 0 || innerCols <= 0)
      return failure();
  }

  if (!boxed) {
    if (bLayout == 1)
      return std::pair<int64_t, int64_t>(/*rowStride=*/1,
                                         /*colStride=*/rows);
    return std::pair<int64_t, int64_t>(/*rowStride=*/cols,
                                       /*colStride=*/1);
  }

  if (bLayout == 1) {
    if (sLayout != 1)
      return failure();
    return std::pair<int64_t, int64_t>(/*rowStride=*/innerCols,
                                       /*colStride=*/rows);
  }

  return std::pair<int64_t, int64_t>(/*rowStride=*/cols,
                                     /*colStride=*/innerRows);
}

static Value asIndexValue(OpBuilder &builder, Location loc, Value value) {
  if (!value || value.getType().isIndex())
    return value;
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value);
}

static FailureOr<Value>
materializeTileSubviewByteOffset(OpBuilder &builder, Operation *user,
                                 pto::TileBufType sourceType,
                                 ValueRange offsets) {
  if (sourceType.getShape().size() != 2 || offsets.size() != 2) {
    return user->emitError(
        "FoldTileBufIntrinsics: pto.subview address resolution requires a "
        "rank-2 tile_buf source and two offsets");
  }

  FailureOr<std::pair<int64_t, int64_t>> strides =
      getTileBufPhysicalStrides(sourceType);
  if (failed(strides)) {
    return user->emitError(
        "FoldTileBufIntrinsics: failed to compute physical tile strides for "
        "pto.subview address resolution");
  }

  int64_t elemBytes =
      pto::getPTOStorageElemByteSize(sourceType.getElementType());
  if (elemBytes <= 0) {
    return user->emitError(
        "FoldTileBufIntrinsics: unsupported tile element type for pto.subview "
        "address byte offset");
  }

  Location loc = user->getLoc();
  Value row = asIndexValue(builder, loc, offsets[0]);
  Value col = asIndexValue(builder, loc, offsets[1]);
  Value rowStride =
      builder.create<arith::ConstantIndexOp>(loc, strides->first);
  Value colStride =
      builder.create<arith::ConstantIndexOp>(loc, strides->second);
  Value elemBytesValue =
      builder.create<arith::ConstantIndexOp>(loc, elemBytes);
  Value rowOffset = builder.create<arith::MulIOp>(loc, row, rowStride);
  Value colOffset = builder.create<arith::MulIOp>(loc, col, colStride);
  Value linearOffset = builder.create<arith::AddIOp>(loc, rowOffset, colOffset);
  Value byteOffset =
      builder.create<arith::MulIOp>(loc, linearOffset, elemBytesValue);
  return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                            byteOffset)
      .getResult();
}

static Value addI64AddressOffset(OpBuilder &builder, Location loc,
                                 Value address, Value byteOffset) {
  if (!byteOffset)
    return address;

  int64_t constantOffset = 0;
  if (getConstIndexValue(byteOffset, constantOffset) && constantOffset == 0)
    return address;

  Value normalizedOffset = byteOffset;
  if (!normalizedOffset.getType().isInteger(64))
    normalizedOffset =
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                           normalizedOffset);

  if (!address.getType().isInteger(64))
    return Value();
  return builder.create<arith::AddIOp>(loc, address, normalizedOffset);
}

static LogicalResult mergeTileMetadata(TileBufMetadata &metadata,
                                       FailureOr<TileBufMetadata> source,
                                       Operation *user) {
  if (failed(source))
    return failure();
  metadata.address = source->address;
  if (!metadata.validRow)
    metadata.validRow = source->validRow;
  if (!metadata.validCol)
    metadata.validCol = source->validCol;
  if (!metadata.address && !source->address)
    return user->emitError(
        "FoldTileBufIntrinsics: failed to resolve native tile address "
        "metadata");
  return success();
}

static FailureOr<TileBufMetadata> resolveTileBufMetadata(Value tileBuf,
                                                         Operation *user,
                                                         OpBuilder &builder) {
  auto tileType = dyn_cast<pto::TileBufType>(tileBuf.getType());
  if (!tileType)
    return user->emitError(
        "FoldTileBufIntrinsics: expected a native !pto.tile_buf value, got ")
           << tileBuf.getType();

  Operation *def = tileBuf.getDefiningOp();
  if (!def) {
    return user->emitError(
        "FoldTileBufIntrinsics: cannot resolve tile_buf metadata from a block "
        "argument; pass a native producer carrying address and dynamic valid "
        "shape metadata");
  }

  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (castOp.getNumOperands() == 1 &&
        isa<MemRefType>(castOp.getOperand(0).getType())) {
      return user->emitError(
          "FoldTileBufIntrinsics: memref-sourced tile_buf bridge is not "
          "supported; expected native pto.pointer_cast, pto.alloc_tile, "
          "pto.bind_tile, pto.subview, pto.bitcast, or pto.treshape");
    }
    return user->emitError(
               "FoldTileBufIntrinsics: unsupported tile_buf cast producer ")
           << castOp->getName()
           << "; expected native pto.pointer_cast, pto.alloc_tile, "
              "pto.bind_tile, pto.subview, pto.bitcast, or pto.treshape";
  }

  TileBufMetadata metadata{tileType, Value(), Value(), Value()};

  if (auto op = dyn_cast<pto::PointerCastOp>(def)) {
    if (op.getAddrs().empty()) {
      return user->emitError(
          "FoldTileBufIntrinsics: pto.pointer_cast tile_buf producer has no "
          "address operand");
    }
    metadata.address = op.getAddrs().front();
    metadata.validRow = op.getValidRow();
    metadata.validCol = op.getValidCol();
    return metadata;
  }

  if (auto op = dyn_cast<pto::AllocTileOp>(def)) {
    if (!op.getAddr()) {
      return user->emitError(
          "FoldTileBufIntrinsics: pto.alloc_tile tile_buf producer has no "
          "native address metadata; run memory planning or provide addr=");
    }
    metadata.address = op.getAddr();
    metadata.validRow = op.getValidRow();
    metadata.validCol = op.getValidCol();
    return metadata;
  }

  if (auto op = dyn_cast<pto::BindTileOp>(def)) {
    metadata.validRow = op.getValidRow();
    metadata.validCol = op.getValidCol();
    if (failed(mergeTileMetadata(metadata,
                                 resolveTileBufMetadata(op.getSource(), user,
                                                        builder),
                                 user)))
      return failure();
    return metadata;
  }

  if (auto op = dyn_cast<pto::SubViewOp>(def)) {
    metadata.validRow = op.getValidRow();
    metadata.validCol = op.getValidCol();

    FailureOr<TileBufMetadata> source =
        resolveTileBufMetadata(op.getSource(), user, builder);
    if (failed(source))
      return failure();

    metadata.address = source->address;
    if (!metadata.validRow)
      metadata.validRow = source->validRow;
    if (!metadata.validCol)
      metadata.validCol = source->validCol;

    builder.setInsertionPoint(user);
    FailureOr<Value> byteOffset =
        materializeTileSubviewByteOffset(builder, user,
                                         cast<pto::TileBufType>(
                                             op.getSource().getType()),
                                         op.getOffsets());
    if (failed(byteOffset))
      return failure();
    Value shiftedAddress =
        addI64AddressOffset(builder, user->getLoc(), metadata.address,
                            *byteOffset);
    if (!shiftedAddress) {
      return user->emitError(
          "FoldTileBufIntrinsics: pto.subview address resolution currently "
          "requires an i64 native address");
    }
    metadata.address = shiftedAddress;
    return metadata;
  }

  if (auto op = dyn_cast<pto::BitcastOp>(def)) {
    FailureOr<TileBufMetadata> source =
        resolveTileBufMetadata(op.getSrc(), user, builder);
    if (failed(source))
      return failure();
    metadata.address = source->address;
    metadata.validRow = source->validRow;
    metadata.validCol = source->validCol;
    return metadata;
  }

  if (auto op = dyn_cast<pto::TReshapeOp>(def)) {
    FailureOr<TileBufMetadata> source =
        resolveTileBufMetadata(op.getSrc(), user, builder);
    if (failed(source))
      return failure();
    metadata.address = source->address;
    metadata.validRow = source->validRow;
    metadata.validCol = source->validCol;
    return metadata;
  }

  return user->emitError(
             "FoldTileBufIntrinsics: unsupported native tile_buf producer ")
         << def->getName()
         << "; expected pto.pointer_cast, pto.alloc_tile, pto.bind_tile, "
            "pto.subview, pto.bitcast, or pto.treshape";
}

static FailureOr<Value> materializeTileAddress(OpBuilder &builder,
                                               pto::TileBufAddrOp addrOp,
                                               TileBufMetadata metadata) {
  Type resultType = addrOp.getDst().getType();
  Location loc = addrOp.getLoc();

  if (auto resultPtrType = dyn_cast<pto::PtrType>(resultType)) {
    if (metadata.address.getType() == resultPtrType)
      return metadata.address;
    return builder.create<pto::CastPtrOp>(loc, resultPtrType,
                                          metadata.address)
        .getResult();
  }

  auto resultMemrefType = dyn_cast<MemRefType>(resultType);
  if (!resultMemrefType) {
    return addrOp.emitError(
        "FoldTileBufIntrinsics: tile_buf_addr result must be memref or "
        "!pto.ptr");
  }

  auto ptrMemorySpace =
      dyn_cast_or_null<pto::AddressSpaceAttr>(resultMemrefType.getMemorySpace());
  if (!ptrMemorySpace) {
    return addrOp.emitError(
        "FoldTileBufIntrinsics: cannot materialize tile_buf_addr memref "
        "result without a PTO address space");
  }

  auto ptrType = pto::PtrType::get(builder.getContext(),
                                   resultMemrefType.getElementType(),
                                   ptrMemorySpace);
  Value ptr = metadata.address.getType() == ptrType
                  ? metadata.address
                  : builder.create<pto::CastPtrOp>(loc, ptrType,
                                                   metadata.address)
                        .getResult();
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{resultMemrefType},
                                          ValueRange{ptr})
      .getResult(0);
}

static FailureOr<Value>
resolveValidDim(OpBuilder &builder, Operation *user, Value tileBuf,
                TileBufMetadata metadata, unsigned dim,
                llvm::StringRef intrinsicName,
                llvm::StringRef dimOperandName) {
  if (metadata.tileType.getValidShape().size() <= dim) {
    return user->emitError() << intrinsicName << ": invalid tile_buf type";
  }

  int64_t staticDim = metadata.tileType.getValidShape()[dim];
  if (staticDim != ShapedType::kDynamic)
    return builder.create<arith::ConstantIndexOp>(user->getLoc(), staticDim)
        .getResult();

  Value dynamicDim = dim == 0 ? metadata.validRow : metadata.validCol;
  if (!dynamicDim) {
    return user->emitError()
           << intrinsicName << ": dynamic valid "
           << (dim == 0 ? "row" : "col")
           << " requires native tile metadata carrying " << dimOperandName
           << "; source is " << tileBuf.getDefiningOp()->getName();
  }
  if (dynamicDim.getType() != user->getResult(0).getType()) {
    return user->emitError()
           << intrinsicName << ": resolved " << dimOperandName
           << " has type " << dynamicDim.getType() << ", expected "
           << user->getResult(0).getType();
  }
  return dynamicDim;
}

struct FoldTileBufIntrinsicsPass
    : public pto::impl::FoldTileBufIntrinsicsBase<FoldTileBufIntrinsicsPass> {
  using FoldTileBufIntrinsicsBase::FoldTileBufIntrinsicsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // Leftover TileLang template instances (private, uncalled after
    // PTOInlineLibCall) still contain pto.tile_buf_addr / tile_valid_*
    // ops on tile_buf function arguments — they have no bind_tile to
    // fold against and will be removed by later DCE.  Skip them.
    if (func->hasAttr("pto.tilelang.instance"))
      return;

    SmallVector<pto::TileBufAddrOp, 8> addrOps;
    SmallVector<pto::TileValidRowsOp, 8> rowsOps;
    SmallVector<pto::TileValidColsOp, 8> colsOps;
    SmallVector<pto::TensorViewAddrOp, 8> tvAddrOps;
    SmallVector<pto::GetTensorViewDimOp, 8> tvDimOps;
    SmallVector<pto::GetTensorViewStrideOp, 8> tvStrideOps;

    func.walk([&](Operation *op) {
      if (auto addr = dyn_cast<pto::TileBufAddrOp>(op))
        addrOps.push_back(addr);
      else if (auto rows = dyn_cast<pto::TileValidRowsOp>(op))
        rowsOps.push_back(rows);
      else if (auto cols = dyn_cast<pto::TileValidColsOp>(op))
        colsOps.push_back(cols);
      else if (auto tvAddr = dyn_cast<pto::TensorViewAddrOp>(op))
        tvAddrOps.push_back(tvAddr);
      else if (auto tvDim = dyn_cast<pto::GetTensorViewDimOp>(op))
        tvDimOps.push_back(tvDim);
      else if (auto tvStride = dyn_cast<pto::GetTensorViewStrideOp>(op))
        tvStrideOps.push_back(tvStride);
    });

    // Fold pto.tile_buf_addr from native tile_buf producer metadata. This is
    // the VPTO-side materialization point for tile addresses; it must not rely
    // on the old memref bridge.
    for (auto addrOp : addrOps) {
      builder.setInsertionPoint(addrOp);
      FailureOr<TileBufMetadata> metadata =
          resolveTileBufMetadata(addrOp.getSrc(), addrOp, builder);
      if (failed(metadata))
        return signalPassFailure();

      FailureOr<Value> replacement =
          materializeTileAddress(builder, addrOp, *metadata);
      if (failed(replacement))
        return signalPassFailure();
      addrOp.getDst().replaceAllUsesWith(*replacement);
      addrOp.erase();
    }

    // Fold pto.tile_valid_rows → arith.constant (static) or native producer
    // valid_row metadata (dynamic).
    for (auto rowsOp : rowsOps) {
      builder.setInsertionPoint(rowsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(rowsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().empty()) {
        rowsOp.emitError("tile_valid_rows: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vRow = tbTy.getValidShape()[0];
      Value replacement;
      if (vRow != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(rowsOp.getLoc(), vRow);
      } else {
        FailureOr<TileBufMetadata> metadata =
            resolveTileBufMetadata(rowsOp.getSrc(), rowsOp, builder);
        if (failed(metadata))
          return signalPassFailure();
        FailureOr<Value> resolved =
            resolveValidDim(builder, rowsOp, rowsOp.getSrc(), *metadata,
                            /*dim=*/0, "tile_valid_rows", "valid_row");
        if (failed(resolved))
          return signalPassFailure();
        replacement = *resolved;
      }
      rowsOp.getResult().replaceAllUsesWith(replacement);
      rowsOp.erase();
    }

    // Fold pto.tile_valid_cols → arith.constant (static) or native producer
    // valid_col metadata (dynamic).
    for (auto colsOp : colsOps) {
      builder.setInsertionPoint(colsOp);
      auto tbTy = dyn_cast<pto::TileBufType>(colsOp.getSrc().getType());
      if (!tbTy || tbTy.getValidShape().size() < 2) {
        colsOp.emitError("tile_valid_cols: invalid tile_buf type");
        return signalPassFailure();
      }

      int64_t vCol = tbTy.getValidShape()[1];
      Value replacement;
      if (vCol != ShapedType::kDynamic) {
        replacement =
            builder.create<arith::ConstantIndexOp>(colsOp.getLoc(), vCol);
      } else {
        FailureOr<TileBufMetadata> metadata =
            resolveTileBufMetadata(colsOp.getSrc(), colsOp, builder);
        if (failed(metadata))
          return signalPassFailure();
        FailureOr<Value> resolved =
            resolveValidDim(builder, colsOp, colsOp.getSrc(), *metadata,
                            /*dim=*/1, "tile_valid_cols", "valid_col");
        if (failed(resolved))
          return signalPassFailure();
        replacement = *resolved;
      }
      colsOp.getResult().replaceAllUsesWith(replacement);
      colsOp.erase();
    }

    for (auto addrOp : tvAddrOps) {
      builder.setInsertionPoint(addrOp);
      FailureOr<TensorViewMetadata> metadata =
          resolveNativeTensorViewMetadata(addrOp.getSrc(), addrOp);
      if (failed(metadata))
        return signalPassFailure();

      FailureOr<Value> replacement =
          materializeTensorViewAddress(builder, addrOp, *metadata);
      if (failed(replacement))
        return signalPassFailure();

      addrOp.getDst().replaceAllUsesWith(*replacement);
      addrOp.erase();
    }

    for (auto dimOp : tvDimOps) {
      FailureOr<TensorViewMetadata> metadata =
          resolveNativeTensorViewMetadata(dimOp.getTensorView(), dimOp);
      if (failed(metadata))
        return signalPassFailure();

      FailureOr<unsigned> dimIdx =
          getConstantDimIndex(dimOp.getDimIndex(), dimOp,
                              static_cast<int64_t>(metadata->shape.size()),
                              "get_tensor_view_dim");
      if (failed(dimIdx))
        return signalPassFailure();

      builder.setInsertionPoint(dimOp);
      FailureOr<Value> replacement =
          resolveTensorViewDim(builder, dimOp, *metadata, *dimIdx);
      if (failed(replacement))
        return signalPassFailure();

      dimOp.getResult().replaceAllUsesWith(*replacement);
      dimOp.erase();
    }

    for (auto strideOp : tvStrideOps) {
      FailureOr<TensorViewMetadata> metadata =
          resolveNativeTensorViewMetadata(strideOp.getTensorView(), strideOp);
      if (failed(metadata))
        return signalPassFailure();

      FailureOr<unsigned> dimIdx =
          getConstantDimIndex(strideOp.getDimIndex(), strideOp,
                              static_cast<int64_t>(metadata->strides.size()),
                              "get_tensor_view_stride");
      if (failed(dimIdx))
        return signalPassFailure();

      builder.setInsertionPoint(strideOp);
      strideOp.getResult().replaceAllUsesWith(metadata->strides[*dimIdx]);
      strideOp.erase();
    }
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass() {
  return std::make_unique<FoldTileBufIntrinsicsPass>();
}

} // namespace pto
} // namespace mlir
