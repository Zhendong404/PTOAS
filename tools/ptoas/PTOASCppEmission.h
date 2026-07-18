// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_CPP_EMISSION_H
#define PTOAS_CPP_EMISSION_H

#include "PTOASNameHints.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mlir::pto {

LogicalResult finalizeEmitCModuleForCppEmission(
    ModuleOp module, const FunctionBlockArgHintMap &blockArgHints,
    bool emitAddPtrTrace, std::string &cppOutput);

} // namespace mlir::pto

#endif
