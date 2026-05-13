# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Lightweight wrappers around the low-level MLIR Python bindings for the PTO
dialect.  The goal is to eliminate boilerplate so that a vPTO kernel body can
be written in plain-looking Python without manual InsertionPoint management,
verbose type constructors, or raw Operation.create() calls.

Design rules
────────────
• Every helper is a plain function or a contextlib.contextmanager – no classes.
• All helpers work with the *current* MLIR context / location / insertion-point
  (set by `pto_context` and `vpto_kernel`); no context parameter is threaded.
• The module is self-contained: only mlir.* imports are allowed.
"""

from contextlib import contextmanager

from mlir.ir import (
    Attribute,
    Context,
    IntegerType,
    IndexType,
    InsertionPoint,
    Location,
    Module,
    Operation,
    ShapedType,
    StringAttr,
    Type,
    UnitAttr,
)
from mlir.dialects import arith, func, pto, scf

# Mapping from the textual address-space name used in !pto.ptr<elem, NAME>
# to the AddressSpace enum value exposed by the C extension.
_ADDR_SPACE = {
    "ub":  pto.AddressSpace.VEC,   # "ub" (unified buffer) prints as VEC
    "gm":  pto.AddressSpace.GM,
    "vec": pto.AddressSpace.VEC,
    "l1":  pto.AddressSpace.MAT,
}


# ─── Type constructors ────────────────────────────────────────────────────────

def i32_type():
    """Signless 32-bit integer type."""
    return IntegerType.get_signless(32)


def i64_type():
    """Signless 64-bit integer type."""
    return IntegerType.get_signless(64)


def idx_type():
    """MLIR index type."""
    return IndexType.get()


def ptr_type(elem_type, space="ub"):
    """PTO pointer type: !pto.ptr<{elem_type}, {space}>.

    Uses ``pto.PtrType.get`` with an ``AddressSpaceAttr`` when the address-space
    name is known; falls back to ``Type.parse`` for unknown spaces.
    """
    enum_val = _ADDR_SPACE.get(space)
    if enum_val is not None:
        space_attr = pto.AddressSpaceAttr.get(enum_val)
        return pto.PtrType.get(elem_type, memory_space=space_attr)
    return Type.parse(f"!pto.ptr<{elem_type}, {space}>")


def vreg_type(lanes, elem_type):
    """PTO vector-register type: !pto.vreg<{lanes}x{elem_type}>.

    VRegType has no Python-binding constructor; Type.parse is the only path.
    """
    return Type.parse(f"!pto.vreg<{lanes}x{elem_type}>")


def mask_type(bits="b32"):
    """PTO mask/predicate type: !pto.mask<{bits}>  (b8 | b16 | b32).

    MaskType has no Python-binding constructor; Type.parse is the only path.
    """
    return Type.parse(f"!pto.mask<{bits}>")


def tensor_view_type(rank, elem_type):
    """PTO tensor-view type with all-dynamic dimensions: !pto.tensor_view<?x…xelem>.

    Uses ``pto.TensorViewType.get(rank, elem_type)``.
    """
    return pto.TensorViewType.get(rank, elem_type)


def part_tensor_view_type(rank, elem_type):
    """PTO partition-tensor-view type with all-dynamic dims: !pto.partition_tensor_view<?x…xelem>.

    Uses ``pto.PartitionTensorViewType.get([kDynamic]*rank, elem_type)``.
    ``ShapedType.get_dynamic_size()`` (``INT64_MIN``) is the correct MLIR
    sentinel; plain ``-1`` would produce a different printed form.
    """
    kDynamic = ShapedType.get_dynamic_size()
    return pto.PartitionTensorViewType.get([kDynamic] * rank, elem_type)


def tile_buf_type(shape, elem_type, valid_shape, *,
                  blayout="RowMajor", address_space="ub",
                  slayout="NoneBox", fractal_size=512, pad="Null"):
    """PTO tile-buffer type via ``pto.TileBufType.get``.

    ``valid_shape`` entries may be ``-1`` for dynamic (``?``) dimensions.
    ``blayout`` selects the block layout: ``"RowMajor"`` (default, omitted in
    the printed form) or ``"ColMajor"`` (printed as ``blayout=col_major``).

    Common usage::

        # !pto.tile_buf<vec, 8x128xf32, valid=?x?>
        tile_buf_type([8, 128], f32, [-1, -1])

        # !pto.tile_buf<vec, 8x1xf32, valid=?x1, blayout=col_major>
        tile_buf_type([8, 1], f32, [-1, 1], blayout="ColMajor")
    """
    space_enum = _ADDR_SPACE.get(address_space)
    if space_enum is None:
        raise ValueError(f"Unknown address_space '{address_space}'; "
                         f"known: {list(_ADDR_SPACE)}")
    space_attr = pto.AddressSpaceAttr.get(space_enum)
    cfg = pto.TileBufConfigAttr.get(
        pto.BLayoutAttr.get(getattr(pto.BLayout, blayout)),
        pto.SLayoutAttr.get(getattr(pto.SLayout, slayout)),
        fractal_size,
        pto.PadValueAttr.get(getattr(pto.PadValue, pad)),
    )
    return pto.TileBufType.get(shape, elem_type, space_attr, valid_shape, cfg)


# ─── Constant builders ───────────────────────────────────────────────────────

def c_idx(value):
    """Emit an index constant."""
    return arith.ConstantOp(IndexType.get(), value).result


def c_i32(value):
    """Emit a 32-bit integer constant."""
    return arith.ConstantOp(IntegerType.get_signless(32), value).result


def c_i64(value):
    """Emit a 64-bit integer constant."""
    return arith.ConstantOp(IntegerType.get_signless(64), value).result


# ─── Arithmetic shorthands ───────────────────────────────────────────────────

def muli(lhs, rhs):
    """arith.muli"""
    return arith.MulIOp(lhs, rhs).result


def addi(lhs, rhs):
    """arith.addi"""
    return arith.AddIOp(lhs, rhs).result


def subi(lhs, rhs):
    """arith.subi"""
    return arith.SubIOp(lhs, rhs).result


# ─── PTO vector / pointer operations ────────────────────────────────────────

def castptr(int_addr, result_ptr_type):
    """Cast an integer address to a typed PTO pointer (pto.castptr)."""
    return pto.CastPtrOp(result_ptr_type, int_addr).result


def addptr(base_ptr, index_offset):
    """Advance a PTO pointer by an index offset (pto.addptr)."""
    return pto.AddPtrOp(base_ptr, index_offset).result


def vlds(src_ptr, offset, result_vreg_type):
    """Vector load from a PTO pointer at *offset* (pto.vlds)."""
    return pto.VldsOp(result_vreg_type, src_ptr, offset).result


def vadd(lhs, rhs, mask, result_vreg_type):
    """Element-wise vector add under a predicate mask (pto.vadd)."""
    return pto.VaddOp(result_vreg_type, lhs, rhs, mask).result


def vsts(val, dst_ptr, offset, mask):
    """Vector store to a PTO pointer at *offset* under a mask (pto.vsts)."""
    pto.VstsOp(val, dst_ptr, offset, mask)


def plt_b32(scalar):
    """
    Predicate-load from a 32-bit scalar value (pto.plt_b32).

    Returns (mask_value, scalar_out) – the mask is typically the only value
    used downstream; scalar_out can be discarded with ``_``.
    """
    plt_op = pto.PltB32Op(mask_type("b32"), i32_type(), scalar)
    return plt_op.mask, plt_op.scalar_out


# ─── Scope context managers ──────────────────────────────────────────────────

@contextmanager
def vecscope():
    """
    Emit a ``pto.vecscope { ... }`` region.

    Usage::

        with vecscope():
            ptr = castptr(addr, ptr_f32)
            ...
    """
    op = pto.VecScopeOp()
    block = op.body.blocks.append()
    with InsertionPoint(block):
        yield


@contextmanager
def for_range(start, stop, step):
    """
    Emit an ``scf.for`` loop; yield the induction variable.
    The mandatory ``scf.yield`` terminator is inserted automatically on exit.

    Usage::

        with for_range(c0, c16, c1) as i:
            off = muli(i, c64)
            ...
    """
    for_op = scf.ForOp(start, stop, step)
    with InsertionPoint(for_op.body):
        yield for_op.induction_variable
        scf.YieldOp([])


# ─── Top-level module / kernel builder ───────────────────────────────────────

@contextmanager
def pto_context():
    """
    Activate an MLIR context with the PTO dialect registered.
    Must wrap all other utility calls.

    Usage::

        with pto_context():
            f32 = F32Type.get()
            with vpto_kernel("MyKernel", arch="a5") as mod:
                ...
    """
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown():
            yield ctx


@contextmanager
def vpto_kernel(func_name, *, arch="a5"):
    """
    Build the standard two-level nested-module + no-arg ``func.func`` shell
    for a vPTO vector kernel, then yield the outer ``Module`` as the context
    variable.  ``func.ReturnOp`` and ``module.verify()`` are inserted/called
    automatically on context exit.

    The emitted skeleton is::

        module attributes {pto.target_arch = arch} {
          module attributes {pto.kernel_kind = #pto.kernel_kind<vector>,
                             pto.target_arch = arch} {
            func.func @func_name() {
              <your code here>
              return
            }
          }
        }

    Usage::

        with vpto_kernel("TADD", arch="a5") as mod:
            c0 = c_idx(0)
            ...
        return mod
    """
    arch_attr = StringAttr.get(arch)
    kind_attr = Attribute.parse("#pto.kernel_kind<vector>")

    outer_mod = Module.create()
    outer_mod.operation.attributes["pto.target_arch"] = arch_attr

    with InsertionPoint(outer_mod.body):
        # Module.create() ignores the active InsertionPoint, so use
        # Operation.create("builtin.module") to insert the inner module.
        inner_op = Operation.create("builtin.module", regions=1)
        inner_op.attributes["pto.target_arch"] = arch_attr
        inner_op.attributes["pto.kernel_kind"] = kind_attr
        inner_body = inner_op.regions[0].blocks.append()

        with InsertionPoint(inner_body):
            fn = func.FuncOp(func_name, func.FunctionType.get([], []))
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            yield outer_mod
            func.ReturnOp([])

    outer_mod.operation.verify()


# ─── Flat single-module builders (for direct func inside module) ─────────────

@contextmanager
def flat_pto_module(arch="a5"):
    """
    Flat single-level module with ``pto.target_arch`` and
    ``pto.kernel_kind = #pto.kernel_kind<vector>``.

    Usage::

        with flat_pto_module("a5") as mod:
            with pto_aicore_func("MyKernel", [ptr_gm, i32]) as args:
                ...
        return mod
    """
    m = Module.create()
    m.operation.attributes["pto.target_arch"] = StringAttr.get(arch)
    m.operation.attributes["pto.kernel_kind"] = Attribute.parse(
        "#pto.kernel_kind<vector>"
    )
    with InsertionPoint(m.body):
        yield m
    m.operation.verify()


@contextmanager
def pto_aicore_func(func_name, arg_types, *, ret_types=None):
    """
    Create a ``func.func`` with the ``pto.aicore`` attribute.
    Yields the function's block arguments tuple.
    ``func.return`` is inserted automatically on exit.

    Usage::

        with pto_aicore_func("f", [ptr_gm, ptr_gm, i32]) as (p0, p1, n):
            ...
    """
    fn_ty = func.FunctionType.get(arg_types, ret_types or [])
    fn = func.FuncOp(func_name, fn_ty)
    fn.attributes["pto.aicore"] = UnitAttr.get()
    entry = fn.add_entry_block()
    with InsertionPoint(entry):
        yield tuple(entry.arguments)
        func.ReturnOp([])


# ─── Additional control-flow helpers ─────────────────────────────────────────

@contextmanager
def if_ctx(cond):
    """
    Emit ``scf.if cond { ... }`` with no results and no else branch.
    The mandatory ``scf.yield`` terminator is inserted automatically.

    Usage::

        with if_ctx(has_rows):
            tload(part, tile)
            ...
    """
    op = scf.IfOp(cond)
    with InsertionPoint(op.then_block):
        yield
        scf.YieldOp([])


def if_op_returning(cond, result_types):
    """
    Create a ``scf.if`` with results *and* an else branch.
    Returns the raw ``IfOp`` so the caller can manage the two blocks
    manually with ``InsertionPoint`` and close each with ``yield_vals()``.

    Usage::

        br = if_op_returning(has_chunk, [vreg_f32, vreg_f32])
        with InsertionPoint(br.then_block):
            ...
            yield_vals(merged_max, merged_sum)
        with InsertionPoint(br.else_block):
            yield_vals(running_max, running_sum)
        next_max, next_sum = br.results
    """
    return scf.IfOp(cond, result_types, hasElse=True)


@contextmanager
def for_range_iter(start, stop, step, init_vals):
    """
    Emit ``scf.for`` with iter_args.  Yields the raw ``ForOp`` so the
    caller can access ``induction_variable``, ``inner_iter_args``, and
    ``results`` (after the ``with`` block).

    The caller **must** call ``yield_vals(...)`` at the end of the body.

    Usage::

        with for_range_iter(c0, c128, c64, [a, b]) as cf:
            i   = cf.induction_variable
            x, y = cf.inner_iter_args
            ...
            yield_vals(new_x, new_y)
        final_x, final_y = cf.results
    """
    for_op = scf.ForOp(start, stop, step, init_vals)
    with InsertionPoint(for_op.body):
        yield for_op


def yield_vals(*vals):
    """Emit ``scf.yield`` with the given values (shorthand for scf.YieldOp)."""
    scf.YieldOp(list(vals))


# ─── Arithmetic helpers ───────────────────────────────────────────────────────

def index_cast(result_type, val):
    """arith.index_cast from/to index."""
    return arith.IndexCastOp(result_type, val).result


def cmpi_sgt(lhs, rhs):
    """arith.cmpi sgt (signed greater-than)."""
    return arith.CmpIOp(arith.CmpIPredicate.sgt, lhs, rhs).result


def select_val(cond, true_val, false_val):
    """arith.select."""
    return arith.SelectOp(cond, true_val, false_val).result


# ─── PTO hardware helpers ─────────────────────────────────────────────────────

def get_block_idx():
    """pto.get_block_idx → i64 block index."""
    return pto.GetBlockIdxOp().result


def barrier_all():
    """pto.barrier #pto.pipe<PIPE_ALL>."""
    pto.BarrierOp(pto.PipeAttr.get(pto.PIPE.PIPE_ALL))


# ─── Tile-domain helpers ──────────────────────────────────────────────────────

def tile_view(tv_type, ptr, shape, strides):
    """pto.make_tensor_view → tensor_view SSA value."""
    return pto.MakeTensorViewOp(tv_type, ptr, shape, strides).result


def part_view(ptv_type, tv, offsets, sizes):
    """pto.partition_view → partition_tensor_view SSA value."""
    return pto.PartitionViewOp(ptv_type, tv, offsets, sizes).result


def alloc_tile(tile_type, *, addr, valid_row, valid_col=None):
    """pto.alloc_tile with optional valid_col."""
    return pto.AllocTileOp(tile_type, addr=addr, valid_row=valid_row,
                           valid_col=valid_col).result


def tload(part, tile):
    """pto.tload ins(part) outs(tile)."""
    pto.TLoadOp(None, part, tile)


def tstore(tile, part):
    """pto.tstore ins(tile) outs(part)."""
    pto.TStoreOp(None, tile, part)


def tile_ptr(tile, result_ptr_type):
    """pto.tile_buf_addr – materialise a UB pointer from a tile handle."""
    return pto.TileBufAddrOp(result_ptr_type, tile).result


# ─── Mask helpers ─────────────────────────────────────────────────────────────

def pset_b32(pattern):
    """pto.pset_b32 "PATTERN" → !pto.mask<b32> (all-true when "PAT_ALL")."""
    return pto.PsetB32Op(mask_type("b32"), pattern).result


# ─── Vector load / store with dist attribute ──────────────────────────────────

def vbrc_load(src_ptr, offset, result_vreg_type):
    """pto.vlds with dist="BRC_B32" – broadcast a scalar into all lanes."""
    return pto.VldsOp(result_vreg_type, src_ptr, offset,
                      dist="BRC_B32").result


def vsts_1pt(val, dst_ptr, offset, mask):
    """pto.vsts with dist="1PT_B32" – store only the lowest lane."""
    pto.VstsOp(val, dst_ptr, offset, mask, dist="1PT_B32")


# ─── Vector math (result type inferred from first operand) ────────────────────
#
# These wrappers follow the convention: if result_type is None the type is
# taken from the first operand (all PTO binary vector ops return the same
# type as their inputs).
#

def vcmax(v, mask):
    """pto.vcmax – cross-lane maximum reduction."""
    return pto.VcmaxOp(v.type, v, mask).result


def vdup_lowest(v, mask):
    """pto.vdup {position="LOWEST"} – broadcast lane-0 to all lanes."""
    return pto.VdupOp(v.type, v, mask, position="LOWEST").result


def vmax(lhs, rhs, mask):
    """pto.vmax – element-wise maximum."""
    return pto.VmaxOp(lhs.type, lhs, rhs, mask).result


def vexpdif(inp, ref, mask, part="ODD"):
    """pto.vexpdif – exp(inp − ref), selecting ODD or EVEN lanes."""
    return pto.VexpdifOp(inp.type, inp, ref, mask, part).result


def vmul(lhs, rhs, mask):
    """pto.vmul – element-wise multiply."""
    return pto.VmulOp(lhs.type, lhs, rhs, mask).result


def vcadd(v, mask):
    """pto.vcadd – cross-lane add (sum reduction)."""
    return pto.VcaddOp(v.type, v, mask).result


def vdiv(lhs, rhs, mask):
    """pto.vdiv – element-wise divide."""
    return pto.VdivOp(lhs.type, lhs, rhs, mask).result


# Override vadd to make result_type optional (inferred from lhs when omitted)
_vadd_impl = vadd


def vadd(lhs, rhs, mask, result_type=None):  # type: ignore[misc]
    """pto.vadd – element-wise add (result_type inferred from lhs if None)."""
    rt = result_type if result_type is not None else lhs.type
    return pto.VaddOp(rt, lhs, rhs, mask).result

