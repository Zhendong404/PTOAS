//===- PTOToA5VM.cpp - PTO to A5VM lowering helpers ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace mlir {
namespace pto {

namespace {

std::optional<int64_t> getConstInt(Value value) {
  if (!value)
    return std::nullopt;

  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>())
    return constIndex.value();
  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>())
    return constInt.value();
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

StringRef stringifyTileDomain(A5VMTileDomain domain) {
  switch (domain) {
  case A5VMTileDomain::Vec:
    return "vec";
  case A5VMTileDomain::Acc:
    return "acc";
  case A5VMTileDomain::Mat:
    return "mat";
  }
  llvm_unreachable("unknown A5VM tile domain");
}

StringRef stringifyTileLayout(TileBufType type) {
  if (auto layoutAttr = dyn_cast_or_null<BLayoutAttr>(type.getBLayoutAttr())) {
    switch (layoutAttr.getValue()) {
    case BLayout::RowMajor:
      return "row_major";
    case BLayout::ColMajor:
      return "col_major";
    }
  }
  return "row_major";
}

StringRef stringifyPadModeAttr(PadModeAttr padMode) {
  if (!padMode)
    return "none";

  switch (padMode.getPadmode()) {
  case PadMode::PadNull:
    return "none";
  case PadMode::PadFirstElem:
    return "first_elem";
  case PadMode::PadValue:
    return "value";
  }
  return "none";
}

StringRef stringifyLayoutAttr(Attribute layoutAttr) {
  if (auto attr = dyn_cast_or_null<LayoutAttr>(layoutAttr))
    return stringifyLayout(attr.getLayout());
  return "nd";
}

A5VMTileDomain deriveTileDomain(Attribute memorySpace) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case AddressSpace::ACC:
      return A5VMTileDomain::Acc;
    case AddressSpace::MAT:
      return A5VMTileDomain::Mat;
    case AddressSpace::VEC:
    default:
      return A5VMTileDomain::Vec;
    }
  }
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(AddressSpace::ACC):
      return A5VMTileDomain::Acc;
    case static_cast<int64_t>(AddressSpace::MAT):
      return A5VMTileDomain::Mat;
    default:
      return A5VMTileDomain::Vec;
    }
  }
  return A5VMTileDomain::Vec;
}

void getValidShape(TileBufType type, int64_t &rows, int64_t &cols) {
  ArrayRef<int64_t> validShape = type.getValidShape();
  rows = validShape.size() > 0 ? validShape[0] : ShapedType::kDynamic;
  cols = validShape.size() > 1 ? validShape[1] : ShapedType::kDynamic;
}

void appendStaticSizes(ValueRange values, SmallVectorImpl<int64_t> &out,
                       bool &hasDynamic) {
  out.clear();
  hasDynamic = false;
  out.reserve(values.size());
  for (Value value : values) {
    if (std::optional<int64_t> constant = getConstInt(value)) {
      out.push_back(*constant);
      continue;
    }
    out.push_back(ShapedType::kDynamic);
    hasDynamic = true;
  }
}

Value resolveTensorViewBase(Value value, Attribute &layoutAttr,
                            SmallVectorImpl<int64_t> &shape,
                            SmallVectorImpl<int64_t> &strides) {
  if (!value)
    return {};

  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    if (auto source = part.getSource().getDefiningOp<MakeTensorViewOp>()) {
      layoutAttr = source.getLayoutAttr();
      auto tensorType = dyn_cast<TensorViewType>(source.getResult().getType());
      shape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
      strides.clear();
      for (Value stride : source.getStrides()) {
        if (std::optional<int64_t> constant = getConstInt(stride))
          strides.push_back(*constant);
        else
          strides.push_back(ShapedType::kDynamic);
      }
      return source.getPtr();
    }
    return resolveTensorViewBase(part.getSource(), layoutAttr, shape, strides);
  }

  if (auto source = value.getDefiningOp<MakeTensorViewOp>()) {
    layoutAttr = source.getLayoutAttr();
    auto tensorType = dyn_cast<TensorViewType>(source.getResult().getType());
    shape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
    strides.clear();
    for (Value stride : source.getStrides()) {
      if (std::optional<int64_t> constant = getConstInt(stride))
        strides.push_back(*constant);
      else
        strides.push_back(ShapedType::kDynamic);
    }
    return source.getPtr();
  }

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
    int64_t offset = 0;
    if (failed(mlir::getStridesAndOffset(memrefType, strides, offset)))
      strides.assign(shape.size(), ShapedType::kDynamic);
    return value;
  }

  return {};
}

a5vm::VecType getA5VMVecType(MLIRContext *context, Type elementType) {
  unsigned bitWidth = 0;
  if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  else if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();

  if (bitWidth == 0 || 2048 % bitWidth != 0)
    return {};
  return a5vm::VecType::get(context, 2048 / bitWidth, elementType);
}

ArrayAttr asI64ArrayAttr(Builder &builder, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

void attachTraceAttrs(Operation *op, const A5VMPartitionTrace &trace,
                      Builder &builder) {
  op->setAttr("trace_offsets", asI64ArrayAttr(builder, trace.offsets));
  op->setAttr("trace_sizes", asI64ArrayAttr(builder, trace.sizes));
}

LogicalResult lowerUnsupportedAccStore(Location loc) {
  emitError(loc) << "TSTORE ACC lowering TODO for a5vm backend";
  return failure();
}

LogicalResult lowerUnsupportedMatStore(Location loc) {
  emitError(loc) << "TSTORE MAT lowering TODO for a5vm backend";
  return failure();
}

} // namespace

A5VMPartitionTrace extractPartitionTrace(Value value) {
  A5VMPartitionTrace trace;
  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    appendStaticSizes(part.getOffsets(), trace.offsets, trace.hasDynamicOffsets);
    appendStaticSizes(part.getSizes(), trace.sizes, trace.hasDynamicSizes);
  }
  return trace;
}

A5VMLoadContract extractTLoadContract(TLoadOp op) {
  A5VMLoadContract contract;
  contract.trace = extractPartitionTrace(op.getSrc());

  Attribute layoutAttr;
  Value base = resolveTensorViewBase(op.getSrc(), layoutAttr, contract.srcShape,
                                     contract.srcStrides);
  (void)base;
  contract.layout = stringifyLayoutAttr(layoutAttr);

  auto dstType = cast<TileBufType>(op.getDst().getType());
  contract.tileLayout = stringifyTileLayout(dstType);
  contract.tileDomain = deriveTileDomain(dstType.getMemorySpace());
  getValidShape(dstType, contract.validRows, contract.validCols);
  contract.padMode = stringifyPadModeAttr(op.getPadModeAttr());
  contract.hasPadValue = static_cast<bool>(op.getPadValue());
  contract.leftPaddingPresent = static_cast<bool>(op.getLeftPaddingNum());
  contract.rightPaddingPresent = static_cast<bool>(op.getRightPaddingNum());
  contract.initOutBuffer = op.getInitOutBuffer();
  contract.hasInitCondition = static_cast<bool>(op.getInitCondition());
  return contract;
}

A5VMUnaryContract extractTAbsContract(TAbsOp op) {
  A5VMUnaryContract contract;
  contract.family = "abs";

  auto srcType = cast<TileBufType>(op.getSrc().getType());
  contract.tileDomain = deriveTileDomain(srcType.getMemorySpace());
  contract.tileLayout = stringifyTileLayout(srcType);
  getValidShape(srcType, contract.validRows, contract.validCols);
  contract.elementType = srcType.getElementType();
  return contract;
}

A5VMStoreContract extractTStoreContract(TStoreOp op) {
  A5VMStoreContract contract;
  contract.trace = extractPartitionTrace(op.getDst());

  auto srcType = cast<TileBufType>(op.getSrc().getType());
  contract.srcDomain = deriveTileDomain(srcType.getMemorySpace());
  getValidShape(srcType, contract.validRows, contract.validCols);

  Attribute layoutAttr;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  Value base = resolveTensorViewBase(op.getDst(), layoutAttr, shape, strides);
  (void)base;
  contract.dstLayout = stringifyLayoutAttr(layoutAttr);
  contract.dstShape = std::move(shape);
  contract.dstStrides = std::move(strides);
  return contract;
}

void attachLoadContractAttrs(Operation *op, const A5VMLoadContract &contract) {
  Builder builder(op->getContext());
  op->setAttr("layout", builder.getStringAttr(contract.layout));
  op->setAttr("src_shape", asI64ArrayAttr(builder, contract.srcShape));
  op->setAttr("src_strides", asI64ArrayAttr(builder, contract.srcStrides));
  op->setAttr("tile_layout", builder.getStringAttr(contract.tileLayout));
  op->setAttr("tile_domain",
              builder.getStringAttr(stringifyTileDomain(contract.tileDomain)));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
  op->setAttr("pad_mode", builder.getStringAttr(contract.padMode));
  op->setAttr("has_pad_value", builder.getBoolAttr(contract.hasPadValue));
  op->setAttr("left_padding_present",
              builder.getBoolAttr(contract.leftPaddingPresent));
  op->setAttr("right_padding_present",
              builder.getBoolAttr(contract.rightPaddingPresent));
  op->setAttr("init_out_buffer", builder.getBoolAttr(contract.initOutBuffer));
  op->setAttr("has_init_condition",
              builder.getBoolAttr(contract.hasInitCondition));
  attachTraceAttrs(op, contract.trace, builder);
}

void attachUnaryContractAttrs(Operation *op, const A5VMUnaryContract &contract) {
  Builder builder(op->getContext());
  op->setAttr("unary_family", builder.getStringAttr(contract.family));
  op->setAttr("tile_domain",
              builder.getStringAttr(stringifyTileDomain(contract.tileDomain)));
  op->setAttr("tile_layout", builder.getStringAttr(contract.tileLayout));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
}

void attachStoreContractAttrs(Operation *op, const A5VMStoreContract &contract) {
  Builder builder(op->getContext());
  op->setAttr("src_domain",
              builder.getStringAttr(stringifyTileDomain(contract.srcDomain)));
  op->setAttr("dst_layout", builder.getStringAttr(contract.dstLayout));
  op->setAttr("dst_shape", asI64ArrayAttr(builder, contract.dstShape));
  op->setAttr("dst_strides", asI64ArrayAttr(builder, contract.dstStrides));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
  attachTraceAttrs(op, contract.trace, builder);
}

LogicalResult lowerUnaryTileOp(StringRef family,
                               const A5VMUnaryContract &contract, Value src,
                               Value dst, PatternRewriter &rewriter,
                               Location loc) {
  auto vecType = getA5VMVecType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported A5VM unary element type";

  Value vectorSrc = src;
  if (src.getType() != vecType)
    vectorSrc = rewriter
                    .create<UnrealizedConversionCastOp>(loc, TypeRange{vecType},
                                                        src)
                    .getResult(0);

  auto absOp = rewriter.create<a5vm::AbsOp>(loc, vecType, vectorSrc);
  A5VMUnaryContract attrs = contract;
  attrs.family = family;
  attachUnaryContractAttrs(absOp, attrs);

  if (dst && dst.getType() != vecType)
    rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{dst.getType()},
                                                absOp.getResult());
  return success();
}

LogicalResult lowerTLOAD(TLoadOp op, PatternRewriter &rewriter) {
  A5VMLoadContract contract = extractTLoadContract(op);

  Attribute layoutAttr;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  Value base = resolveTensorViewBase(op.getSrc(), layoutAttr, shape, strides);
  if (!base || !isa<MemRefType>(base.getType()))
    return op.emitOpError("requires a memref-backed source for A5VM lowering");

  auto dstType = dyn_cast<TileBufType>(op.getDst().getType());
  if (!dstType)
    return op.emitOpError("requires tile_buf destination for A5VM lowering");

  auto vecType = getA5VMVecType(rewriter.getContext(), dstType.getElementType());
  if (!vecType)
    return op.emitOpError("requires A5VM-compatible tile element type");

  Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  auto loadOp = rewriter.create<a5vm::LoadOp>(
      op.getLoc(), vecType, base, zero, rewriter.getStringAttr(contract.layout),
      rewriter.getStringAttr(stringifyTileDomain(contract.tileDomain)),
      rewriter.getI64IntegerAttr(contract.validRows),
      rewriter.getI64IntegerAttr(contract.validCols));
  attachLoadContractAttrs(loadOp, contract);
  return success();
}

LogicalResult lowerTABS(TAbsOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTAbsContract(op);
  if (contract.tileDomain != A5VMTileDomain::Vec)
    return op.emitOpError("TABS lowering requires tile domain vec");
  if (contract.tileLayout != "row_major")
    return op.emitOpError("TABS lowering requires row-major tile layout");
  if (contract.validRows != contract.validCols)
    return op.emitOpError(
        "TABS lowering requires matching valid rows and valid cols");
  if (!contract.elementType.isF16() && !contract.elementType.isF32())
    return op.emitOpError("TABS lowering supports only f16 and f32 element types");

  return lowerUnaryTileOp("abs", contract, op.getSrc(), op.getDst(), rewriter,
                          op.getLoc());
}

LogicalResult lowerTSTORE(TStoreOp op, PatternRewriter &rewriter) {
  A5VMStoreContract contract = extractTStoreContract(op);

  if (contract.srcDomain == A5VMTileDomain::Acc)
    return lowerUnsupportedAccStore(op.getLoc());
  if (contract.srcDomain == A5VMTileDomain::Mat)
    return lowerUnsupportedMatStore(op.getLoc());

  Attribute layoutAttr;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  Value base = resolveTensorViewBase(op.getDst(), layoutAttr, shape, strides);
  if (!base || !isa<MemRefType>(base.getType()))
    return op.emitOpError("requires a memref-backed destination for A5VM lowering");

  auto srcType = dyn_cast<TileBufType>(op.getSrc().getType());
  if (!srcType)
    return op.emitOpError("requires tile_buf source for A5VM lowering");

  auto vecType = getA5VMVecType(rewriter.getContext(), srcType.getElementType());
  if (!vecType)
    return op.emitOpError("requires A5VM-compatible tile element type");

  Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  Value vectorSrc = rewriter
                        .create<UnrealizedConversionCastOp>(op.getLoc(),
                                                            TypeRange{vecType},
                                                            op.getSrc())
                        .getResult(0);
  auto storeOp = rewriter.create<a5vm::StoreOp>(
      op.getLoc(), vectorSrc, base, zero,
      rewriter.getStringAttr(contract.dstLayout),
      rewriter.getStringAttr(stringifyTileDomain(contract.srcDomain)));
  attachStoreContractAttrs(storeOp, contract);
  return success();
}

} // namespace pto
} // namespace mlir
