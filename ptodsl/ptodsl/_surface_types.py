# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public PTODSL surface markers and enums."""

from ._bootstrap import make_context  # noqa: F401
from ._host_tensors import TensorSpec, tensor_spec

from mlir.dialects import pto as _pto


class _ConstexprMarker:
    """Marker annotation for PTODSL compile-time specialization parameters."""

    def __repr__(self):
        return "pto.constexpr"


constexpr = _ConstexprMarker()


class MemorySpace:
    """Public PTODSL memory-space enum aliases."""

    GM = _pto.AddressSpace.GM
    UB = _pto.AddressSpace.VEC
    VEC = _pto.AddressSpace.VEC
    MAT = _pto.AddressSpace.MAT
    LEFT = _pto.AddressSpace.LEFT
    RIGHT = _pto.AddressSpace.RIGHT
    ACC = _pto.AddressSpace.ACC
    BIAS = _pto.AddressSpace.BIAS
    SCALING = _pto.AddressSpace.SCALING


class BarrierType:
    """Public PTODSL memory-barrier kind aliases."""

    VV_ALL = "VV_ALL"
    VST_VLD = "VST_VLD"
    VLD_VST = "VLD_VST"
    VST_VST = "VST_VST"
    VS_ALL = "VS_ALL"
    VST_LD = "VST_LD"
    VLD_ST = "VLD_ST"
    VST_ST = "VST_ST"
    SV_ALL = "SV_ALL"
    ST_VLD = "ST_VLD"
    LD_VST = "LD_VST"
    ST_VST = "ST_VST"
    SS_ALL = "SS_ALL"
    ST_LD = "ST_LD"
    LD_ST = "LD_ST"
    ST_ST = "ST_ST"


class Pipe:
    """Public PTODSL pipeline aliases for pipeline-level sync ops."""

    S = _pto.PIPE.PIPE_S
    V = _pto.PIPE.PIPE_V
    M = _pto.PIPE.PIPE_M
    MTE1 = _pto.PIPE.PIPE_MTE1
    MTE2 = _pto.PIPE.PIPE_MTE2
    MTE3 = _pto.PIPE.PIPE_MTE3
    MTE4 = _pto.PIPE.PIPE_MTE4
    MTE5 = _pto.PIPE.PIPE_MTE5
    V2 = _pto.PIPE.PIPE_V2
    FIX = _pto.PIPE.PIPE_FIX
    ALL = _pto.PIPE.PIPE_ALL


class MaskPattern:
    """Public PTODSL mask-pattern tokens."""

    ALL = "PAT_ALL"
    ALLF = "PAT_ALLF"
    H = "PAT_H"
    Q = "PAT_Q"
    M3 = "PAT_M3"
    M4 = "PAT_M4"


for _vl in range(1, 129):
    setattr(MaskPattern, f"VL{_vl}", f"PAT_VL{_vl}")


class CmpMode:
    """Public PTODSL compare-mode tokens."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class PredicatePart:
    """Public PTODSL predicate pack/unpack part tokens."""

    LOWER = "LOWER"
    HIGHER = "HIGHER"


class PredicateDist:
    """Public PTODSL predicate load/store distribution tokens."""

    NORM = "NORM"
    US = "US"
    DS = "DS"
    PK = "PK"


AlignType = _pto.AlignType


class TensorView:
    """Authoring-time marker for a tensor-view descriptor value."""


class PartitionTensorView:
    """Authoring-time marker for a partitioned tensor-view descriptor value."""


class Tile:
    """Authoring-time marker for an on-chip tile value."""


__all__ = [
    "constexpr",
    "TensorSpec",
    "MemorySpace",
    "BarrierType",
    "Pipe",
    "MaskPattern",
    "CmpMode",
    "PredicatePart",
    "PredicateDist",
    "AlignType",
    "TensorView",
    "PartitionTensorView",
    "Tile",
    "tensor_spec",
]
