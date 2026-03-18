//===- A5VMTextEmitter.cpp - A5VM textual LLVM-like emitter --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMTextEmitter.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;

namespace mlir::pto {
namespace {

struct IntrinsicSelection {
  std::string name;
  std::string resultType;
  std::string argumentTypes;
  std::string mappingKey;
  std::string opName;
  std::string vectorType;
};

static std::string stringifyType(Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << type;
  return storage;
}

static std::string moduleID(ModuleOp module) {
  Location loc = module.getLoc();
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc.getFilename().str();
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return nameLoc.getName().str();
  return "<unknown>";
}

static std::string llvmScalarType(Type type) {
  if (type.isF16())
    return "half";
  if (type.isBF16())
    return "bfloat";
  if (type.isF32())
    return "float";
  if (type.isF64())
    return "double";
  if (auto intType = dyn_cast<IntegerType>(type))
    return "i" + std::to_string(intType.getWidth());
  if (isa<IndexType>(type))
    return "i64";
  if (isa<MemRefType>(type))
    return "ptr";
  return "ptr";
}

static std::string llvmValueType(Type type) {
  if (auto vecType = dyn_cast<a5vm::VecType>(type))
    return "<" + std::to_string(vecType.getElementCount()) + " x " +
           llvmScalarType(vecType.getElementType()) + ">";
  return llvmScalarType(type);
}

static std::string llvmFunctionResultType(func::FuncOp func) {
  FunctionType type = func.getFunctionType();
  if (type.getNumResults() == 0)
    return "void";
  return llvmValueType(type.getResult(0));
}

static bool supportsAddLikeType(Type type) {
  return isa<IntegerType, IndexType>(type);
}

static std::string elementCode(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (type.isF64())
    return "f64";
  if (auto intType = dyn_cast<IntegerType>(type))
    return "i" + std::to_string(intType.getWidth());
  if (isa<IndexType>(type))
    return "index";
  return "unknown";
}

static std::string valueName(Value value,
                             llvm::DenseMap<Value, std::string> &names,
                             unsigned &nextID) {
  auto it = names.find(value);
  if (it != names.end())
    return it->second;

  std::string name;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    name = "%arg" + std::to_string(blockArg.getArgNumber());
  } else {
    name = "%" + std::to_string(nextID++);
  }
  names.try_emplace(value, name);
  return name;
}

static std::optional<IntrinsicSelection> selectIntrinsic(Operation *op) {
  auto vectorType = [&]() -> a5vm::VecType {
    if (auto load = dyn_cast<a5vm::VldsOp>(op))
      return cast<a5vm::VecType>(load.getResult().getType());
    if (auto abs = dyn_cast<a5vm::VabsOp>(op))
      return cast<a5vm::VecType>(abs.getResult().getType());
    if (auto store = dyn_cast<a5vm::VstsOp>(op))
      return cast<a5vm::VecType>(store.getValue().getType());
    return {};
  }();
  if (!vectorType)
    return std::nullopt;

  const std::string vecSuffix =
      elementCode(vectorType.getElementType()) + "x" +
      std::to_string(vectorType.getElementCount());
  const std::string vectorTypeStr = stringifyType(vectorType);

  if (auto load = dyn_cast<a5vm::VldsOp>(op)) {
    IntrinsicSelection selection;
    selection.opName = load->getName().getStringRef().str();
    selection.vectorType = vectorTypeStr;
    selection.mappingKey = "vlds." + vecSuffix;
    selection.name = "@llvm.hivm." + selection.mappingKey;
    selection.resultType = llvmValueType(load.getResult().getType());
    selection.argumentTypes = "ptr, i64";
    return selection;
  }

  if (auto abs = dyn_cast<a5vm::VabsOp>(op)) {
    IntrinsicSelection selection;
    selection.opName = abs->getName().getStringRef().str();
    selection.vectorType = vectorTypeStr;
    selection.mappingKey = "vabs." + vecSuffix;
    selection.name = "@llvm.hivm." + selection.mappingKey;
    selection.resultType = llvmValueType(abs.getResult().getType());
    selection.argumentTypes = llvmValueType(abs.getInput().getType());
    return selection;
  }

  if (auto store = dyn_cast<a5vm::VstsOp>(op)) {
    IntrinsicSelection selection;
    selection.opName = store->getName().getStringRef().str();
    selection.vectorType = vectorTypeStr;
    selection.mappingKey = "vsts." + vecSuffix;
    selection.name = "@llvm.hivm." + selection.mappingKey;
    selection.resultType = "void";
    selection.argumentTypes =
        llvmValueType(store.getValue().getType()) + ", ptr, i64";
    return selection;
  }

  return std::nullopt;
}

static std::string unresolvedRecord(Operation *op, StringRef vectorType,
                                    StringRef mappingKey) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "op=" << op->getName().getStringRef() << " type=" << vectorType
     << " mapping=" << mappingKey;
  return storage;
}

static std::string unresolvedMappingKey(Operation *op) {
  if (auto copy = dyn_cast<a5vm::CopyGmToUbufOp>(op))
    return copy->getName().getStringRef().str();
  if (auto load = dyn_cast<a5vm::VldsOp>(op)) {
    auto vecType = cast<a5vm::VecType>(load.getResult().getType());
    return "vlds." + elementCode(vecType.getElementType()) + "x" +
           std::to_string(vecType.getElementCount());
  }
  if (auto abs = dyn_cast<a5vm::VabsOp>(op)) {
    auto vecType = cast<a5vm::VecType>(abs.getResult().getType());
    return "vabs." + elementCode(vecType.getElementType()) + "x" +
           std::to_string(vecType.getElementCount());
  }
  if (auto store = dyn_cast<a5vm::VstsOp>(op)) {
    auto vecType = cast<a5vm::VecType>(store.getValue().getType());
    return "vsts." + elementCode(vecType.getElementType()) + "x" +
           std::to_string(vecType.getElementCount());
  }
  if (auto copy = dyn_cast<a5vm::CopyUbufToGmOp>(op))
    return copy->getName().getStringRef().str();
  return op->getName().getStringRef().str();
}

static LogicalResult writeReportFile(const A5VMEmissionOptions &options,
                                     ArrayRef<std::string> unresolved) {
  if (options.unresolvedReportPath.empty())
    return success();

  std::error_code ec;
  llvm::raw_fd_ostream reportOS(options.unresolvedReportPath, ec,
                                llvm::sys::fs::OF_Text);
  if (ec)
    return failure();
  for (const std::string &record : unresolved)
    reportOS << record << "\n";
  return success();
}

} // namespace

LogicalResult translateA5VMModuleToText(ModuleOp module, llvm::raw_ostream &os,
                                        const A5VMEmissionOptions &options,
                                        llvm::raw_ostream &diagOS) {
  llvm::StringSet<> declaredIntrinsics;
  llvm::SmallVector<IntrinsicSelection, 8> declarations;
  llvm::SmallVector<std::string, 8> functionBodies;
  llvm::SmallVector<std::string, 8> unresolvedRecords;

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    std::string body;
    llvm::raw_string_ostream bodyOS(body);
    llvm::DenseMap<Value, std::string> valueNames;
    unsigned nextValueID = 0;

    bodyOS << "define " << llvmFunctionResultType(func) << " @"
           << func.getSymName() << "(";
    for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
      if (i)
        bodyOS << ", ";
      Value arg = func.getArgument(i);
      bodyOS << llvmValueType(arg.getType()) << " "
             << valueName(arg, valueNames, nextValueID);
    }
    bodyOS << ") {\n";

    for (Block &block : func.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (auto copy = dyn_cast<a5vm::CopyGmToUbufOp>(op)) {
          bodyOS << "  ; A5VM-PRIMITIVE: " << copy->getName().getStringRef()
                 << "\n";
          continue;
        }

        if (auto load = dyn_cast<a5vm::VldsOp>(op)) {
          auto selection = selectIntrinsic(load);
          if (!selection) {
            llvm_unreachable("load must resolve");
          }
          if (declaredIntrinsics.insert(selection->name).second)
            declarations.push_back(*selection);
          if (options.printIntrinsicSelections) {
            diagOS << "A5VM intrinsic: op=" << selection->opName
                   << " type=" << selection->vectorType
                   << " mapping=" << selection->mappingKey
                   << " intrinsic=" << selection->name << "\n";
          }
          bodyOS << "  " << valueName(load.getResult(), valueNames, nextValueID)
                 << " = call " << selection->resultType << " "
                 << selection->name << "(ptr "
                 << valueName(load.getSource(), valueNames, nextValueID) << ", i64 "
                 << valueName(load.getOffset(), valueNames, nextValueID) << ")\n";
          continue;
        }

        if (auto abs = dyn_cast<a5vm::VabsOp>(op)) {
          auto selection = selectIntrinsic(abs);
          if (!selection) {
            llvm_unreachable("abs must resolve");
          }
          if (declaredIntrinsics.insert(selection->name).second)
            declarations.push_back(*selection);
          if (options.printIntrinsicSelections) {
            diagOS << "A5VM intrinsic: op=" << selection->opName
                   << " type=" << selection->vectorType
                   << " mapping=" << selection->mappingKey
                   << " intrinsic=" << selection->name << "\n";
          }
          bodyOS << "  " << valueName(abs.getResult(), valueNames, nextValueID)
                 << " = call " << selection->resultType << " "
                 << selection->name << "(" << selection->resultType << " "
                 << valueName(abs.getInput(), valueNames, nextValueID) << ")\n";
          continue;
        }

        if (auto store = dyn_cast<a5vm::VstsOp>(op)) {
          auto selection = selectIntrinsic(store);
          if (!selection) {
            auto vecTypeStr = stringifyType(store.getValue().getType());
            std::string mappingKey = unresolvedMappingKey(store);
            std::string record =
                unresolvedRecord(&op, vecTypeStr, mappingKey);
            if (!options.allowUnresolved) {
              diagOS << "A5VM emission failed: " << record << "\n";
              return failure();
            }
            unresolvedRecords.push_back(record);
            bodyOS << "  ; A5VM-UNRESOLVED: " << record << "\n";
            continue;
          }
          if (declaredIntrinsics.insert(selection->name).second)
            declarations.push_back(*selection);
          if (options.printIntrinsicSelections) {
            diagOS << "A5VM intrinsic: op=" << selection->opName
                   << " type=" << selection->vectorType
                   << " mapping=" << selection->mappingKey
                   << " intrinsic=" << selection->name << "\n";
          }
          bodyOS << "  call " << selection->resultType << " " << selection->name
                 << "(" << llvmValueType(store.getValue().getType()) << " "
                 << valueName(store.getValue(), valueNames, nextValueID) << ", ptr "
                 << valueName(store.getDestination(), valueNames, nextValueID)
                 << ", i64 "
                 << valueName(store.getOffset(), valueNames, nextValueID) << ")\n";
          continue;
        }

        if (auto copy = dyn_cast<a5vm::CopyUbufToGmOp>(op)) {
          bodyOS << "  ; A5VM-PRIMITIVE: " << copy->getName().getStringRef()
                 << "\n";
          continue;
        }

        if (auto ret = dyn_cast<func::ReturnOp>(op)) {
          if (ret.getNumOperands() == 0) {
            bodyOS << "  ret void\n";
          } else {
            Value result = ret.getOperand(0);
            bodyOS << "  ret " << llvmValueType(result.getType()) << " "
                   << valueName(result, valueNames, nextValueID) << "\n";
          }
          continue;
        }

        if (auto add = dyn_cast<arith::AddIOp>(op)) {
          Type resultType = add.getResult().getType();
          if (!supportsAddLikeType(resultType)) {
            diagOS << "A5VM emission failed: unsupported op="
                   << op.getName().getStringRef() << "\n";
            return failure();
          }
          std::string resultTypeStr = llvmValueType(resultType);
          bodyOS << "  " << valueName(add.getResult(), valueNames, nextValueID)
                 << " = add " << resultTypeStr << " "
                 << valueName(add.getLhs(), valueNames, nextValueID) << ", "
                 << valueName(add.getRhs(), valueNames, nextValueID) << "\n";
          continue;
        }

        if (auto loop = dyn_cast<scf::ForOp>(op)) {
          bodyOS << "  ; A5VM-NONLOWERED: " << op.getName().getStringRef()
                 << "\n";
          for (OpResult result : loop->getResults()) {
            bodyOS << "  " << valueName(result, valueNames, nextValueID)
                   << " = add " << llvmValueType(result.getType()) << " 0, 0\n";
          }
          continue;
        }

        diagOS << "A5VM emission failed: unsupported op="
               << op.getName().getStringRef() << "\n";
        return failure();
      }
    }

    bodyOS << "}\n";
    bodyOS.flush();
    functionBodies.push_back(std::move(body));
  }

  if (failed(writeReportFile(options, unresolvedRecords))) {
    diagOS << "A5VM emission failed: could not write unresolved report to '"
           << options.unresolvedReportPath << "'\n";
    return failure();
  }

  os << "; ModuleID = '" << moduleID(module) << "'\n";
  for (const IntrinsicSelection &selection : declarations)
    os << "declare " << selection.resultType << " " << selection.name << "("
       << selection.argumentTypes << ")\n";
  if (!declarations.empty())
    os << "\n";
  for (const std::string &body : functionBodies)
    os << body;

  return success();
}

} // namespace mlir::pto
