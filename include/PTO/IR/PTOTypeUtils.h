// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_PTOTYPEUTILS_H
#define PTO_IR_PTOTYPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::pto {

enum class PTOLowPrecisionKind {
  None,
  Float8E4M3,
  Float8E4M3FN,
  Float8E4M3FNUZ,
  Float8E4M3B11FNUZ,
  Float8E5M2,
  Float8E5M2FNUZ,
  HiFloat8,
  Float4E1M2x2,
  Float4E2M1x2,
};

PTOLowPrecisionKind classifyPTOLowPrecisionType(Type t);
llvm::StringRef stringifyPTOLowPrecisionKind(PTOLowPrecisionKind kind);

bool isPTOFloat8Type(Type t);
bool isPTOHiFloat8Type(Type t);
bool isPTOFloat4PackedType(Type t);
bool isPTOLowPrecisionType(Type t);

unsigned getPTOStorageElemByteSize(Type t);
unsigned getPTOStorageElemBitWidth(Type t);

std::string getPTOLowPrecisionDtypeString(Type t);
std::string getPTOCanonicalDtypeString(Type t);
std::string getPTOLowPrecisionHIVMTypeFragment(Type t);

} // namespace mlir::pto

#endif // PTO_IR_PTOTYPEUTILS_H
