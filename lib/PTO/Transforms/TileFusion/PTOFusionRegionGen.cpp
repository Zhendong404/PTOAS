// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONREGIONGEN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PTOFusionRegionGenPass
    : public pto::impl::PTOFusionRegionGenBase<PTOFusionRegionGenPass> {
  using pto::impl::PTOFusionRegionGenBase<
      PTOFusionRegionGenPass>::PTOFusionRegionGenBase;

  void runOnOperation() override { markAllAnalysesPreserved(); }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionRegionGenPass() {
  return std::make_unique<PTOFusionRegionGenPass>();
}
