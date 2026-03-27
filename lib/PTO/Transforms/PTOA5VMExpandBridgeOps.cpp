#include "PTO/IR/A5VM.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOA5VMEXPANDBRIDGEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static unsigned getLLVMAddressSpace(Attribute memorySpace) {
  if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace))
    return static_cast<unsigned>(addrSpace.getAddressSpace());
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
    return static_cast<unsigned>(intAttr.getInt());
  return static_cast<unsigned>(pto::AddressSpace::GM);
}

static Value materializeBufferPointer(Value value, PatternRewriter &rewriter,
                                      Location loc) {
  if (!value)
    return {};

  if (isa<LLVM::LLVMPointerType>(value.getType()))
    return value;

  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return {};

  unsigned llvmAddressSpace = getLLVMAddressSpace(memrefType.getMemorySpace());
  auto canonicalMemRefType = MemRefType::get(
      memrefType.getShape(), memrefType.getElementType(), memrefType.getLayout(),
      rewriter.getI64IntegerAttr(llvmAddressSpace));
  if (value.getType() != canonicalMemRefType)
    value = rewriter.create<memref::MemorySpaceCastOp>(loc, canonicalMemRefType,
                                                       value);

  LLVMTypeConverter typeConverter(rewriter.getContext());
  Type llvmDescriptorType = typeConverter.convertType(canonicalMemRefType);
  if (!llvmDescriptorType)
    return {};

  Value descriptor =
      rewriter
          .create<UnrealizedConversionCastOp>(loc, TypeRange{llvmDescriptorType},
                                              value)
          .getResult(0);
  MemRefDescriptor memrefDescriptor(descriptor);
  return memrefDescriptor.alignedPtr(rewriter, loc);
}

static Value offsetBufferPointer(Value basePtr, Type elementType,
                                 Value elementOffset,
                                 PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  Value offsetI64 = elementOffset.getType().isIndex()
                        ? rewriter.create<arith::IndexCastUIOp>(
                              loc, rewriter.getI64Type(), elementOffset)
                        : elementOffset;
  return rewriter.create<LLVM::GEPOp>(loc, basePtr.getType(), elementType,
                                      basePtr, ValueRange{offsetI64});
}

struct ExpandUvldPattern : public OpRewritePattern<a5vm::UvldOp> {
  using OpRewritePattern<a5vm::UvldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(a5vm::UvldOp op,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<a5vm::VecType>(op.getResult().getType());
    if (!vecType)
      return failure();

    Value basePtr = materializeBufferPointer(op.getSource(), rewriter, op.getLoc());
    if (!basePtr)
      return op.emitOpError(
          "requires a recoverable pointer base for uvld expansion");

    Value loadPtr = offsetBufferPointer(basePtr, vecType.getElementType(),
                                       op.getOffset(), rewriter, op.getLoc());
    auto alignType = a5vm::AlignType::get(rewriter.getContext());
    Value align =
        rewriter.create<a5vm::VldasOp>(op.getLoc(), alignType, loadPtr);
    auto load = rewriter.create<a5vm::VldusOp>(
        op.getLoc(), TypeRange{vecType, alignType, loadPtr.getType()},
        ValueRange{loadPtr, align});
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct PTOA5VMExpandBridgeOpsPass
    : public pto::impl::PTOA5VMExpandBridgeOpsBase<
          PTOA5VMExpandBridgeOpsPass> {
  using pto::impl::PTOA5VMExpandBridgeOpsBase<
      PTOA5VMExpandBridgeOpsPass>::PTOA5VMExpandBridgeOpsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandUvldPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOA5VMExpandBridgeOpsPass() {
  return std::make_unique<PTOA5VMExpandBridgeOpsPass>();
}
