#ifndef MLIR_DIALECT_PTO_TRANSFORMS_A5VMTEXTEMITTER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_A5VMTEXTEMITTER_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::pto {

struct A5VMEmissionOptions {
  bool printIntrinsicSelections = false;
  bool allowUnresolved = false;
  std::string unresolvedReportPath;
};

LogicalResult translateA5VMModuleToText(ModuleOp module, llvm::raw_ostream &os,
                                        const A5VMEmissionOptions &options,
                                        llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_A5VMTEXTEMITTER_H
