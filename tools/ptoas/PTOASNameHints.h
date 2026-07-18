// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_NAME_HINTS_H
#define PTOAS_NAME_HINTS_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir {
class AsmParserState;
}

namespace mlir::pto {

using FunctionBlockArgHintMap =
    llvm::StringMap<llvm::SmallVector<llvm::SmallVector<std::string, 4>, 4>>;

FunctionBlockArgHintMap collectFunctionBlockArgNameHints(ModuleOp module);

} // namespace mlir::pto

#endif
