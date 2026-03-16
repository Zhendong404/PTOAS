//===- PTOOpLibMatch.h - OP-Lib match descriptor --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines OP-Lib matching descriptors shared by PTO IR and lowering.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOOPLIBMATCH_H_
#define MLIR_DIALECT_PTO_IR_PTOOPLIBMATCH_H_

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace pto {

enum class OpLibArgRole : int64_t {
  Tile = 0,
  Scalar = 1,
};

// A compact op-level description that can be translated into MatchRequest.
struct OpLibMatchDescriptor {
  std::string kind;
  std::string opName;
  SmallVector<Value, 4> operands;
  SmallVector<int64_t, 4> operandRoles;
  // Family-logic audit hotspots carried through lowering:
  // - compare/select currently exposes cmpMode but not yet the byte-mask
  //   contract itself;
  // - tile_scalar direction-sensitive ops need operandOrder in addition to the
  //   normalized scalarPos;
  // - broadcast_row_binary cannot recover full-tile vs row-broadcast roles
  //   from wildcard template arg matches alone;
  // - reduce_colsum binary semantics must keep both isBinary and an explicit
  //   variant pin to avoid collapsing to linear.
  std::optional<std::string> operandOrder;
  std::optional<int64_t> fullTilePos;
  std::optional<int64_t> rowBroadcastPos;
  std::optional<int64_t> scalarPos;
  std::optional<std::string> cmpMode;
  std::optional<bool> isBinary;
  std::optional<std::string> requiredVariantId;
};

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_IR_PTOOPLIBMATCH_H_
