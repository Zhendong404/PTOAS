// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTOTypeUtils.h"

#include "PTO/IR/PTO.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kBitsPerByte = 8;
} // namespace

PTOLowPrecisionKind mlir::pto::classifyPTOLowPrecisionType(Type t) {
  if (t.isFloat8E4M3())
    return PTOLowPrecisionKind::Float8E4M3;
  if (t.isFloat8E4M3FN())
    return PTOLowPrecisionKind::Float8E4M3FN;
  if (t.isFloat8E4M3FNUZ())
    return PTOLowPrecisionKind::Float8E4M3FNUZ;
  if (t.isFloat8E4M3B11FNUZ())
    return PTOLowPrecisionKind::Float8E4M3B11FNUZ;
  if (t.isFloat8E5M2())
    return PTOLowPrecisionKind::Float8E5M2;
  if (t.isFloat8E5M2FNUZ())
    return PTOLowPrecisionKind::Float8E5M2FNUZ;
  if (isa<HiF8Type>(t))
    return PTOLowPrecisionKind::HiFloat8;
  if (isa<F4E1M2x2Type>(t))
    return PTOLowPrecisionKind::Float4E1M2x2;
  if (isa<F4E2M1x2Type>(t))
    return PTOLowPrecisionKind::Float4E2M1x2;
  return PTOLowPrecisionKind::None;
}

llvm::StringRef
mlir::pto::stringifyPTOLowPrecisionKind(PTOLowPrecisionKind kind) {
  switch (kind) {
  case PTOLowPrecisionKind::None:
    return "";
  case PTOLowPrecisionKind::Float8E4M3:
    return "f8e4m3";
  case PTOLowPrecisionKind::Float8E4M3FN:
    return "f8e4m3fn";
  case PTOLowPrecisionKind::Float8E4M3FNUZ:
    return "f8e4m3fnuz";
  case PTOLowPrecisionKind::Float8E4M3B11FNUZ:
    return "f8e4m3b11fnuz";
  case PTOLowPrecisionKind::Float8E5M2:
    return "f8e5m2";
  case PTOLowPrecisionKind::Float8E5M2FNUZ:
    return "f8e5m2fnuz";
  case PTOLowPrecisionKind::HiFloat8:
    return "hif8";
  case PTOLowPrecisionKind::Float4E1M2x2:
    return "f4e1m2x2";
  case PTOLowPrecisionKind::Float4E2M1x2:
    return "f4e2m1x2";
  }
  return "";
}

bool mlir::pto::isPTOFloat8Type(Type t) {
  switch (classifyPTOLowPrecisionType(t)) {
  case PTOLowPrecisionKind::Float8E4M3:
  case PTOLowPrecisionKind::Float8E4M3FN:
  case PTOLowPrecisionKind::Float8E4M3FNUZ:
  case PTOLowPrecisionKind::Float8E4M3B11FNUZ:
  case PTOLowPrecisionKind::Float8E5M2:
  case PTOLowPrecisionKind::Float8E5M2FNUZ:
    return true;
  case PTOLowPrecisionKind::None:
  case PTOLowPrecisionKind::HiFloat8:
  case PTOLowPrecisionKind::Float4E1M2x2:
  case PTOLowPrecisionKind::Float4E2M1x2:
    return false;
  }
  return false;
}

bool mlir::pto::isPTOHiFloat8Type(Type t) {
  return classifyPTOLowPrecisionType(t) == PTOLowPrecisionKind::HiFloat8;
}

bool mlir::pto::isPTOFloat4PackedType(Type t) {
  switch (classifyPTOLowPrecisionType(t)) {
  case PTOLowPrecisionKind::Float4E1M2x2:
  case PTOLowPrecisionKind::Float4E2M1x2:
    return true;
  case PTOLowPrecisionKind::None:
  case PTOLowPrecisionKind::Float8E4M3:
  case PTOLowPrecisionKind::Float8E4M3FN:
  case PTOLowPrecisionKind::Float8E4M3FNUZ:
  case PTOLowPrecisionKind::Float8E4M3B11FNUZ:
  case PTOLowPrecisionKind::Float8E5M2:
  case PTOLowPrecisionKind::Float8E5M2FNUZ:
  case PTOLowPrecisionKind::HiFloat8:
    return false;
  }
  return false;
}

bool mlir::pto::isPTOLowPrecisionType(Type t) {
  return classifyPTOLowPrecisionType(t) != PTOLowPrecisionKind::None;
}

unsigned mlir::pto::getPTOStorageElemByteSize(Type t) {
  if (isPTOLowPrecisionType(t))
    return 1;
  if (auto floatTy = dyn_cast<FloatType>(t))
    return floatTy.getWidth() / kBitsPerByte;
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth() / kBitsPerByte;
  return 0;
}

unsigned mlir::pto::getPTOStorageElemBitWidth(Type t) {
  if (isPTOLowPrecisionType(t))
    return getPTOStorageElemByteSize(t) * kBitsPerByte;
  if (auto floatTy = dyn_cast<FloatType>(t))
    return floatTy.getWidth();
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  return 0;
}

std::string mlir::pto::getPTOLowPrecisionDtypeString(Type t) {
  return stringifyPTOLowPrecisionKind(classifyPTOLowPrecisionType(t)).str();
}

std::string mlir::pto::getPTOCanonicalDtypeString(Type t) {
  std::string lowPrecision = getPTOLowPrecisionDtypeString(t);
  if (!lowPrecision.empty())
    return lowPrecision;

  if (t.isInteger(1))
    return "i1";
  if (t.isF32())
    return "f32";
  if (t.isF16())
    return "f16";
  if (t.isBF16())
    return "bf16";

  if (auto intTy = dyn_cast<IntegerType>(t)) {
    std::string width = std::to_string(intTy.getWidth());
    if (intTy.isUnsigned())
      return "ui" + width;
    if (intTy.isSigned())
      return "si" + width;
    return "i" + width;
  }

  return "";
}

std::string mlir::pto::getPTOLowPrecisionHIVMTypeFragment(Type t) {
  switch (classifyPTOLowPrecisionType(t)) {
  case PTOLowPrecisionKind::Float8E4M3:
  case PTOLowPrecisionKind::Float8E4M3FN:
  case PTOLowPrecisionKind::Float8E4M3FNUZ:
  case PTOLowPrecisionKind::Float8E4M3B11FNUZ:
    return "e4m3";
  case PTOLowPrecisionKind::Float8E5M2:
  case PTOLowPrecisionKind::Float8E5M2FNUZ:
    return "e5m2";
  case PTOLowPrecisionKind::HiFloat8:
    return "hif8";
  case PTOLowPrecisionKind::Float4E1M2x2:
    return "e1m2x2";
  case PTOLowPrecisionKind::Float4E2M1x2:
    return "e2m1x2";
  case PTOLowPrecisionKind::None:
    return "";
  }
  return "";
}
