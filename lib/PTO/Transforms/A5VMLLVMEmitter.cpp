//===- A5VMLLVMEmitter.cpp - A5VM to official LLVM IR text emitter -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLLVMEmitter.h"

#include "PTO/IR/A5VM.h"
#include "PTO/Transforms/HIVMIntrinsicNaming.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::pto {
namespace {

constexpr llvm::StringLiteral kDefaultBishengPath =
    "/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/bin/bisheng";

struct QueriedTargetAttrs {
  std::string targetCPU;
  std::string targetFeatures;
};

static Type getElementTypeFromVectorLike(Type type);

static std::optional<uint64_t> parsePipeImmediate(llvm::StringRef pipe) {
  if (pipe == "PIPE_S")
    return 0;
  if (pipe == "PIPE_V")
    return 1;
  if (pipe == "PIPE_M")
    return 2;
  if (pipe == "PIPE_MTE1")
    return 3;
  if (pipe == "PIPE_MTE2")
    return 4;
  if (pipe == "PIPE_MTE3")
    return 5;
  if (pipe == "PIPE_ALL")
    return 6;
  if (pipe == "PIPE_MTE4")
    return 7;
  if (pipe == "PIPE_MTE5")
    return 8;
  if (pipe == "PIPE_V2")
    return 9;
  if (pipe == "PIPE_FIX")
    return 10;
  if (pipe == "VIRTUAL_PIPE_MTE2_L1A")
    return 11;
  if (pipe == "VIRTUAL_PIPE_MTE2_L1B")
    return 12;
  return std::nullopt;
}

static std::optional<uint64_t> parseEventImmediate(llvm::StringRef event) {
  if (!event.consume_front("EVENT_ID"))
    return std::nullopt;
  uint64_t value = 0;
  if (event.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static std::optional<uint64_t> parseLoadDistImmediate(llvm::StringRef dist) {
  if (dist.empty() || dist == "NORM")
    return 0;
  if (dist == "BLK")
    return 15;
  if (dist == "UNPK_B16")
    return 14;
  if (dist == "DINTLV_B32")
    return 19;
  return std::nullopt;
}

static std::optional<uint64_t> parseStoreDistImmediate(Type valueType,
                                                       llvm::StringRef dist) {
  Type elementType = getElementTypeFromVectorLike(valueType);
  if (!elementType)
    return std::nullopt;

  if (dist.empty()) {
    unsigned bitWidth = 0;
    if (auto intType = dyn_cast<IntegerType>(elementType))
      bitWidth = intType.getWidth();
    else if (auto floatType = dyn_cast<FloatType>(elementType))
      bitWidth = floatType.getWidth();
    switch (bitWidth) {
    case 8:
      return 0;
    case 16:
      return 1;
    case 32:
      return 2;
    default:
      return std::nullopt;
    }
  }

  if (dist == "NORM_B8")
    return 0;
  if (dist == "NORM_B16")
    return 1;
  if (dist == "NORM_B32")
    return 2;
  if (dist == "ONEPT_B8")
    return 3;
  if (dist == "ONEPT_B16")
    return 4;
  if (dist == "ONEPT_B32")
    return 5;
  if (dist == "PK_B16")
    return 6;
  if (dist == "PK_B32")
    return 7;
  if (dist == "INTLV_B8")
    return 8;
  if (dist == "INTLV_B16")
    return 9;
  if (dist == "PK_B64")
    return 10;
  if (dist == "INTLV_B32")
    return 11;
  if (dist == "PK4_B32")
    return 12;
  if (dist == "MRG4CHN_B8")
    return 13;
  if (dist == "MRG2CHN_B8")
    return 14;
  if (dist == "MRG2CHN_B16")
    return 15;
  return std::nullopt;
}

static Type convertA5VMType(Type type, Builder &builder) {
  if (auto vecType = dyn_cast<a5vm::VecType>(type))
    return VectorType::get({vecType.getElementCount()}, vecType.getElementType());
  if (isa<a5vm::MaskType>(type))
    return VectorType::get({256}, builder.getI1Type());
  if (isa<a5vm::AlignType>(type))
    return builder.getI64Type();
  return type;
}

static Type getElementTypeFromVectorLike(Type type) {
  if (auto vecType = dyn_cast<a5vm::VecType>(type))
    return vecType.getElementType();
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  return {};
}

static Value castIntegerLikeTo(Operation *anchor, Value value, Type targetType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  if (value.getType() == targetType)
    return value;

  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (value.getType().isIndex() && targetInt)
    return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  if (auto sourceInt = dyn_cast<IntegerType>(value.getType())) {
    if (targetInt) {
      if (sourceInt.getWidth() < targetInt.getWidth())
        return builder.create<arith::ExtUIOp>(anchor->getLoc(), targetType, value);
      if (sourceInt.getWidth() > targetInt.getWidth())
        return builder.create<arith::TruncIOp>(anchor->getLoc(), targetType, value);
      return value;
    }
    if (targetType.isIndex())
      return builder.create<arith::IndexCastOp>(anchor->getLoc(), targetType, value);
  }

  return {};
}

static FailureOr<Value> convertElementOffsetToBytes(Operation *anchor, Value offset,
                                                    Type elementType) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value offsetI32 = castIntegerLikeTo(anchor, offset, builder.getI32Type());
  if (!offsetI32)
    return failure();

  unsigned bitWidth = 0;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return failure();

  Value scale = builder.create<arith::ConstantOp>(
      anchor->getLoc(), builder.getI32IntegerAttr(bitWidth / 8));
  return builder.create<arith::MulIOp>(anchor->getLoc(), offsetI32, scale)
      .getResult();
}

static Value getI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(value))
      .getResult();
}

static Value getI32Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(value))
      .getResult();
}

static FailureOr<Value> packLoopPair(Operation *anchor, Value low, Value high) {
  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);

  Value lowI64 = castIntegerLikeTo(anchor, low, builder.getI64Type());
  Value highI64 = castIntegerLikeTo(anchor, high, builder.getI64Type());
  if (!lowI64 || !highI64)
    return failure();

  Value shift = getI64Constant(builder, anchor->getLoc(), 40);
  Value highShifted =
      builder.create<arith::ShLIOp>(anchor->getLoc(), highI64, shift).getResult();
  return builder.create<arith::OrIOp>(anchor->getLoc(), highShifted, lowI64)
      .getResult();
}

static FailureOr<Value>
packCopyGmToUbConfig0(Operation *anchor, a5vm::CopyGmToUbufOp op,
                      ValueRange operands) {
  if (operands.size() != 12)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(4);
  Value nBurst = getI64Operand(5);
  Value lenBurst = getI64Operand(6);
  Value leftPadding = getI64Operand(7);
  Value rightPadding = getI64Operand(8);
  Value cacheCtl = getI64Operand(9);
  if (!sid || !nBurst || !lenBurst || !leftPadding || !rightPadding || !cacheCtl)
    return failure();

  Value dataSelect =
      getI64Constant(builder, loc,
                     op.getDataSelectBit().has_value() && *op.getDataSelectBit());

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 25));
  config = bitOr(config, shl(leftPadding, 46));
  config = bitOr(config, shl(rightPadding, 52));
  config = bitOr(config, shl(dataSelect, 58));
  config = bitOr(config, shl(cacheCtl, 60));
  return config;
}

static FailureOr<Value>
packCopyGmToUbConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 12)
    return failure();
  return packLoopPair(anchor, operands[10], operands[11]);
}

static FailureOr<Value>
packCopyUbToGmConfig0(Operation *anchor, ValueRange operands) {
  if (operands.size() != 10)
    return failure();

  OpBuilder builder(anchor);
  builder.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  auto getI64Operand = [&](unsigned idx) -> Value {
    return castIntegerLikeTo(anchor, operands[idx], builder.getI64Type());
  };

  Value sid = getI64Operand(4);
  Value nBurst = getI64Operand(5);
  Value lenBurst = getI64Operand(6);
  Value reserved = getI64Operand(7);
  if (!sid || !nBurst || !lenBurst || !reserved)
    return failure();

  auto shl = [&](Value value, uint64_t amount) -> Value {
    return builder.create<arith::ShLIOp>(loc, value,
                                         getI64Constant(builder, loc, amount));
  };
  auto bitOr = [&](Value lhs, Value rhs) -> Value {
    return builder.create<arith::OrIOp>(loc, lhs, rhs);
  };

  Value config = sid;
  config = bitOr(config, shl(nBurst, 4));
  config = bitOr(config, shl(lenBurst, 25));
  config = bitOr(config, shl(reserved, 60));
  return config;
}

static FailureOr<Value>
packCopyUbToGmConfig1(Operation *anchor, ValueRange operands) {
  if (operands.size() != 10)
    return failure();
  return packLoopPair(anchor, operands[8], operands[9]);
}

static func::FuncOp getOrCreateExternalFunc(ModuleOp module, StringRef name,
                                            FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto func = builder.create<func::FuncOp>(module.getLoc(), name, type);
  func.setPrivate();
  return func;
}

static FailureOr<StringRef> getConfirmedAbsPathCallee(Operation *op) {
  if (isa<a5vm::SetLoop2StrideOutToUbOp>(op))
    return StringRef("llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB");
  if (isa<a5vm::SetLoop1StrideOutToUbOp>(op))
    return StringRef("llvm.hivm.SET.LOOP1.STRIDE");
  if (isa<a5vm::SetLoopSizeOutToUbOp>(op))
    return StringRef("llvm.hivm.SET.LOOP.SIZE.OUTTOUB");
  if (isa<a5vm::SetLoop2StrideUbToOutOp>(op))
    return StringRef("llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT");
  if (isa<a5vm::SetLoop1StrideUbToOutOp>(op))
    return StringRef("llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT");
  if (isa<a5vm::SetLoopSizeUbToOutOp>(op))
    return StringRef("llvm.hivm.SET.LOOP.SIZE.UBTOOUT");
  if (isa<a5vm::CopyGmToUbufOp>(op))
    return StringRef("llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV");
  if (isa<a5vm::CopyUbufToGmOp>(op))
    return StringRef("llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.f32.DV");
  if (isa<a5vm::SetFlagOp>(op))
    return StringRef("llvm.hivm.SET.FLAG.IMM");
  if (isa<a5vm::WaitFlagOp>(op))
    return StringRef("llvm.hivm.WAIT.FLAG.IMM");
  if (isa<a5vm::PipeBarrierOp>(op))
    return StringRef("llvm.hivm.BARRIER");
  if (isa<a5vm::PltB32Op>(op))
    return StringRef("llvm.hivm.plt.b32.v300");
  if (isa<a5vm::VldsOp>(op))
    return StringRef("llvm.hivm.vldsx1");
  if (isa<a5vm::VabsOp>(op))
    return StringRef("llvm.hivm.vabs.v64f32.x");
  if (isa<a5vm::VstsOp>(op))
    return StringRef("llvm.hivm.vstsx1");
  return failure();
}

static LogicalResult rewriteA5VMOp(Operation *op, ModuleOp module,
                                   llvm::raw_ostream &diagOS) {
  auto calleeName = getConfirmedAbsPathCallee(op);
  if (failed(calleeName)) {
    diagOS << "A5VM LLVM emission failed: unsupported Abs-path op "
           << op->getName().getStringRef() << "\n";
    return failure();
  }

  IRRewriter builder(op->getContext());
  builder.setInsertionPoint(op);
  Location loc = op->getLoc();

  SmallVector<Type> resultTypes;
  for (Type type : op->getResultTypes())
    resultTypes.push_back(convertA5VMType(type, builder));

  SmallVector<Value> callArgs;

  if (isa<a5vm::SetLoop2StrideOutToUbOp, a5vm::SetLoop1StrideOutToUbOp,
          a5vm::SetLoopSizeOutToUbOp, a5vm::SetLoop2StrideUbToOutOp,
          a5vm::SetLoop1StrideUbToOutOp, a5vm::SetLoopSizeUbToOutOp>(op)) {
    auto packed = packLoopPair(op, op->getOperand(0), op->getOperand(1));
    if (failed(packed))
      return failure();
    callArgs.push_back(*packed);
  } else if (auto copy = dyn_cast<a5vm::CopyGmToUbufOp>(op)) {
    auto config0 = packCopyGmToUbConfig0(op, copy, op->getOperands());
    auto config1 = packCopyGmToUbConfig1(op, op->getOperands());
    if (failed(config0) || failed(config1))
      return failure();
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(*config0);
    callArgs.push_back(*config1);
  } else if (isa<a5vm::CopyUbufToGmOp>(op)) {
    auto config0 = packCopyUbToGmConfig0(op, op->getOperands());
    auto config1 = packCopyUbToGmConfig1(op, op->getOperands());
    if (failed(config0) || failed(config1))
      return failure();
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(*config0);
    callArgs.push_back(*config1);
  } else if (auto setFlag = dyn_cast<a5vm::SetFlagOp>(op)) {
    auto src = parsePipeImmediate(setFlag.getSrcPipe());
    auto dst = parsePipeImmediate(setFlag.getDstPipe());
    auto event = parseEventImmediate(setFlag.getEventId());
    if (!src || !dst || !event)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *src));
    callArgs.push_back(getI64Constant(builder, loc, *dst));
    callArgs.push_back(getI64Constant(builder, loc, *event));
  } else if (auto waitFlag = dyn_cast<a5vm::WaitFlagOp>(op)) {
    auto src = parsePipeImmediate(waitFlag.getSrcPipe());
    auto dst = parsePipeImmediate(waitFlag.getDstPipe());
    auto event = parseEventImmediate(waitFlag.getEventId());
    if (!src || !dst || !event)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *src));
    callArgs.push_back(getI64Constant(builder, loc, *dst));
    callArgs.push_back(getI64Constant(builder, loc, *event));
  } else if (auto barrier = dyn_cast<a5vm::PipeBarrierOp>(op)) {
    auto pipe = parsePipeImmediate(barrier.getPipe());
    if (!pipe)
      return failure();
    callArgs.push_back(getI64Constant(builder, loc, *pipe));
  } else if (isa<a5vm::PltB32Op>(op)) {
    Value laneCount = castIntegerLikeTo(op, op->getOperand(0), builder.getI32Type());
    if (!laneCount)
      return failure();
    callArgs.push_back(laneCount);
  } else if (auto vlds = dyn_cast<a5vm::VldsOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vlds.getResult().getType());
    auto offsetBytes = convertElementOffsetToBytes(
        op, op->getOperand(1), elementType);
    auto dist = parseLoadDistImmediate(vlds.getDist().value_or("NORM"));
    if (!elementType || failed(offsetBytes) || !dist)
      return failure();
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
  } else if (auto vabs = dyn_cast<a5vm::VabsOp>(op)) {
    Value input = op->getOperand(0);
    Value mask = op->getOperand(1);
    Type vecType = resultTypes.front();
    Type maskType = convertA5VMType(mask.getType(), builder);
    if (input.getType() != vecType || mask.getType() != maskType) {
      diagOS << "A5VM LLVM emission failed: unexpected vabs operand types\n";
      return failure();
    }
    callArgs.push_back(input);
    callArgs.push_back(mask);
  } else if (auto vsts = dyn_cast<a5vm::VstsOp>(op)) {
    Type elementType = getElementTypeFromVectorLike(vsts.getValue().getType());
    auto offsetBytes = convertElementOffsetToBytes(
        op, op->getOperand(2), elementType);
    auto dist = parseStoreDistImmediate(vsts.getValue().getType(),
                                        vsts.getDist().value_or(""));
    if (!elementType || failed(offsetBytes) || !dist)
      return failure();
    callArgs.push_back(op->getOperand(0));
    callArgs.push_back(op->getOperand(1));
    callArgs.push_back(*offsetBytes);
    callArgs.push_back(getI32Constant(builder, loc, *dist));
    callArgs.push_back(getI32Constant(builder, loc, 0));
    callArgs.push_back(op->getOperand(3));
  } else {
    diagOS << "A5VM LLVM emission failed: Abs path does not yet support "
           << op->getName().getStringRef() << "\n";
    return failure();
  }

  SmallVector<Type> argTypes;
  for (Value arg : callArgs)
    argTypes.push_back(arg.getType());

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  auto callee = getOrCreateExternalFunc(module, *calleeName, funcType);
  auto call = builder.create<func::CallOp>(loc, callee, callArgs);
  if (op->getNumResults() == 0)
    builder.eraseOp(op);
  else
    builder.replaceOp(op, call.getResults());
  return success();
}

static LogicalResult rewriteA5VMOps(ModuleOp module, llvm::raw_ostream &diagOS) {
  SmallVector<Operation *> opsToRewrite;
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "a5vm")
      opsToRewrite.push_back(op);
  });

  for (Operation *op : opsToRewrite) {
    if (failed(rewriteA5VMOp(op, module, diagOS)))
      return failure();
  }

  bool hasA5VM = false;
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "a5vm")
      hasA5VM = true;
  });
  return success(!hasA5VM);
}

static llvm::StringMap<unsigned>
collectVecScopeLoopCounts(ModuleOp module) {
  llvm::StringMap<unsigned> counts;
  module.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("llvm.loop.aivector_scope"))
      return;
    auto func = forOp->getParentOfType<func::FuncOp>();
    if (!func)
      return;
    counts[func.getName().str()]++;
  });
  return counts;
}

static bool ensureDummyPredForAIVectorScopeLatch(llvm::Loop *loop) {
  llvm::BasicBlock *latch = loop->getLoopLatch();
  if (!latch)
    return false;

  llvm::SmallVector<llvm::BasicBlock *, 4> preds(llvm::predecessors(latch));
  if (preds.size() != 1)
    return false;

  llvm::BasicBlock *pred = preds.front();
  auto *predTerm = pred->getTerminator();
  if (!predTerm || predTerm->getNumSuccessors() <= 1)
    return false;

  llvm::Function *function = latch->getParent();
  if (!function)
    return false;

  llvm::BasicBlock *dummy =
      llvm::BasicBlock::Create(function->getContext(), "aivscope.dummy", function, latch);
  llvm::BranchInst::Create(latch, dummy);
  predTerm->replaceUsesOfWith(latch, dummy);
  return true;
}

static void attachAIVectorScopeMetadata(llvm::Module &llvmModule,
                                        const llvm::StringMap<unsigned> &counts) {
  for (llvm::Function &function : llvmModule) {
    auto it = counts.find(function.getName());
    if (it == counts.end() || it->second == 0)
      continue;
    if (it->second != 1)
      continue;

    llvm::DominatorTree dt(function);
    llvm::LoopInfo loopInfo(dt);
    if (loopInfo.empty())
      continue;

    llvm::Loop *loop = *loopInfo.begin();
    (void)ensureDummyPredForAIVectorScopeLatch(loop);

    dt.recalculate(function);
    loopInfo.releaseMemory();
    loopInfo.analyze(dt);
    if (loopInfo.empty())
      continue;
    loop = *loopInfo.begin();

    llvm::BasicBlock *latch = loop->getLoopLatch();
    if (!latch)
      continue;
    auto *terminator = latch->getTerminator();
    if (!terminator)
      continue;

    llvm::LLVMContext &ctx = llvmModule.getContext();
    llvm::Metadata *ops[] = {
        nullptr, llvm::MDNode::get(ctx, llvm::MDString::get(ctx, "llvm.loop.aivector_scope"))};
    auto *loopID = llvm::MDNode::getDistinct(ctx, ops);
    loopID->replaceOperandWith(0, loopID);
    terminator->setMetadata(llvm::LLVMContext::MD_loop, loopID);
  }
}

static FailureOr<std::string> extractQuotedLLVMFnAttr(llvm::StringRef ir,
                                                      llvm::StringRef key) {
  std::string pattern = "\"";
  pattern += key.str();
  pattern += "\"=\"";
  size_t start = ir.find(pattern);
  if (start == llvm::StringRef::npos)
    return failure();
  start += pattern.size();
  size_t end = ir.find('"', start);
  if (end == llvm::StringRef::npos || end <= start)
    return failure();
  return ir.slice(start, end).str();
}

static FailureOr<QueriedTargetAttrs>
queryDefaultTargetAttrs(const A5VMEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  static llvm::StringMap<QueriedTargetAttrs> cache;

  if (options.targetTriple.empty() || options.march.empty() ||
      options.aicoreArch.empty()) {
    diagOS << "A5VM LLVM emission failed: missing target query options\n";
    return failure();
  }

  std::string cacheKey =
      options.targetTriple + "|" + options.march + "|" + options.aicoreArch;
  if (auto it = cache.find(cacheKey); it != cache.end())
    return it->second;

  llvm::SmallString<64> inputPath;
  llvm::SmallString<64> outputPath;
  int inputFD = -1;
  int outputFD = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-a5vm-target-query",
                                                   "c", inputFD, inputPath)) {
    diagOS << "A5VM LLVM emission failed: cannot create bisheng query input: "
           << ec.message() << "\n";
    return failure();
  }
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-a5vm-target-query",
                                                   "ll", outputFD, outputPath)) {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(inputFD);
    diagOS << "A5VM LLVM emission failed: cannot create bisheng query output: "
           << ec.message() << "\n";
    return failure();
  }

  auto cleanup = llvm::make_scope_exit([&]() {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::fs::remove(outputPath);
  });

  {
    llvm::raw_fd_ostream inputOS(inputFD, /*shouldClose=*/false);
    inputOS << "void f(void) {}\n";
  }
  llvm::sys::Process::SafelyCloseFileDescriptor(inputFD);
  llvm::sys::Process::SafelyCloseFileDescriptor(outputFD);

  llvm::SmallString<128> stderrPath;
  int stderrFD = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-a5vm-target-query",
                                                   "stderr", stderrFD,
                                                   stderrPath)) {
    diagOS << "A5VM LLVM emission failed: cannot create bisheng query stderr: "
           << ec.message() << "\n";
    return failure();
  }
  auto stderrCleanup = llvm::make_scope_exit([&]() {
    llvm::sys::fs::remove(stderrPath);
  });
  llvm::sys::Process::SafelyCloseFileDescriptor(stderrFD);

  llvm::SmallVector<std::string> argStorage = {
      kDefaultBishengPath.str(),
      ("--target=" + options.targetTriple),
      ("-march=" + options.march),
      ("--cce-aicore-arch=" + options.aicoreArch),
      "--cce-aicore-only",
      "-x",
      "c",
      inputPath.str().str(),
      "-S",
      "-emit-llvm",
      "-o",
      outputPath.str().str(),
  };
  llvm::SmallVector<llvm::StringRef> args;
  args.reserve(argStorage.size());
  for (const std::string &arg : argStorage)
    args.push_back(arg);

  std::string execErr;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(
      kDefaultBishengPath, args, std::nullopt,
      {std::nullopt, std::nullopt, llvm::StringRef(stderrPath)}, 0, 0,
      &execErr, &execFailed);

  auto stderrBuffer = llvm::MemoryBuffer::getFile(stderrPath);
  llvm::StringRef stderrText =
      stderrBuffer ? stderrBuffer.get()->getBuffer() : llvm::StringRef();

  if (execFailed || rc != 0) {
    diagOS << "A5VM LLVM emission failed: bisheng target query failed\n";
    diagOS << "Command:";
    for (llvm::StringRef arg : args)
      diagOS << " " << arg;
    diagOS << "\n";
    if (!execErr.empty())
      diagOS << execErr << "\n";
    if (!stderrText.empty())
      diagOS << stderrText << "\n";
    return failure();
  }

  auto outputBuffer = llvm::MemoryBuffer::getFile(outputPath);
  if (!outputBuffer) {
    diagOS << "A5VM LLVM emission failed: cannot read bisheng query output\n";
    return failure();
  }

  FailureOr<std::string> targetCPU =
      extractQuotedLLVMFnAttr(outputBuffer.get()->getBuffer(), "target-cpu");
  FailureOr<std::string> targetFeatures = extractQuotedLLVMFnAttr(
      outputBuffer.get()->getBuffer(), "target-features");
  if (failed(targetCPU) || failed(targetFeatures)) {
    diagOS << "A5VM LLVM emission failed: cannot parse bisheng target attrs\n";
    diagOS << outputBuffer.get()->getBuffer() << "\n";
    return failure();
  }

  QueriedTargetAttrs attrs{*targetCPU, *targetFeatures};
  cache[cacheKey] = attrs;
  return attrs;
}

static LogicalResult
applyQueriedTargetAttrs(ModuleOp module, const A5VMEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  FailureOr<QueriedTargetAttrs> attrs = queryDefaultTargetAttrs(options, diagOS);
  if (failed(attrs))
    return failure();

  MLIRContext *ctx = module.getContext();
  StringAttr cpuAttr = StringAttr::get(ctx, attrs->targetCPU);
  LLVM::TargetFeaturesAttr featureAttr =
      LLVM::TargetFeaturesAttr::get(ctx, attrs->targetFeatures);
  module.walk([&](LLVM::LLVMFuncOp funcOp) {
    funcOp.setTargetCpuAttr(cpuAttr);
    funcOp.setTargetFeaturesAttr(featureAttr);
  });
  return success();
}

} // namespace

LogicalResult
translateA5VMModuleToLLVMText(ModuleOp module, llvm::raw_ostream &os,
                              const A5VMEmissionOptions &options,
                              llvm::raw_ostream &diagOS) {
  OwningOpRef<ModuleOp> cloned(cast<ModuleOp>(module->clone()));
  auto vecScopeCounts = collectVecScopeLoopCounts(*cloned);

  if (failed(rewriteA5VMOps(*cloned, diagOS))) {
    diagOS << "A5VM LLVM emission failed: A5VM-to-call rewriting failed\n";
    return failure();
  }

  PassManager pm(cloned->getContext());
  pm.enableVerifier();
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertIndexToLLVMPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(*cloned))) {
    diagOS << "A5VM LLVM emission failed: official lowering pipeline failed\n";
    return failure();
  }

  if (failed(applyQueriedTargetAttrs(*cloned, options, diagOS)))
    return failure();

  registerBuiltinDialectTranslation(*cloned->getContext());
  registerLLVMDialectTranslation(*cloned->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(cloned.get(), llvmContext);
  if (!llvmModule) {
    diagOS << "A5VM LLVM emission failed: LLVM IR export failed\n";
    return failure();
  }

  attachAIVectorScopeMetadata(*llvmModule, vecScopeCounts);
  llvmModule->setModuleIdentifier("ptoas.hivm.official");
  llvmModule->setSourceFileName("ptoas.hivm.official");
  llvmModule->print(os, nullptr);
  return success();
}

} // namespace mlir::pto
