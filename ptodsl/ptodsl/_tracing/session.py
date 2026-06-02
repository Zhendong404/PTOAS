# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Trace-session objects shared by PTODSL tracing runtimes."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

from .._control_flow import _ExplicitReturnSignal
from .control_flow import (
    build_carry_loop_frame,
    finish_carry_loop_frame,
    yield_carry_loop_state,
)
from .module_builder import create_container_child_module
from .._ops import const
from .._kernel_signature import RuntimeScalarParameterSpec
from .._surface_values import unwrap_surface_value, wrap_like_surface_value

from mlir.dialects import arith, func
from mlir.dialects import pto as _pto
from mlir.ir import Attribute, InsertionPoint, IntegerType, StringAttr, UnitAttr


@dataclass(frozen=True)
class HelperFunctionSpec:
    """Declarative description of a helper function emitted during tracing."""

    symbol_name: str
    arg_types: tuple
    result_types: tuple = ()
    attributes: tuple[tuple[str, object], ...] = ()

    def cache_key(self) -> tuple:
        """Return one stable ABI-sensitive cache key for this helper signature."""
        return (
            self.symbol_name,
            tuple(str(arg_type) for arg_type in self.arg_types),
            tuple(str(result_type) for result_type in self.result_types),
            tuple((attr_name, str(attr_value)) for attr_name, attr_value in self.attributes),
        )

    def specialized_symbol_name(self) -> str:
        """Return one stable symbol name that is unique for this helper ABI."""
        digest = hashlib.sha1(repr(self.cache_key()).encode("utf-8")).hexdigest()[:10]
        return f"{self.symbol_name}__ptodsl_{digest}"


@dataclass(frozen=True)
class KernelModuleImportRecord:
    """One private import declaration emitted for a kernel-module callsite."""

    caller_symbol_name: str
    import_symbol_name: str
    target_symbol_name: str


@dataclass(frozen=True)
class KernelModuleGraphSnapshot:
    """Immutable snapshot of traced kernel-module imports and dependencies."""

    imports: tuple[KernelModuleImportRecord, ...] = ()
    dependencies: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class TracedChildModuleRecord:
    """Metadata for one child module assembled during tracing."""

    symbol_name: str
    primary_symbol_name: str
    role: str
    module_spec: object


@dataclass(frozen=True)
class SubkernelTraceFrame:
    """Active inline-lowering frame for one PTODSL subkernel call."""

    role: str
    symbol_name: str
    target: str


class TraceSession:
    """Shared per-build state for a traced PTODSL module."""

    def __init__(self, module_spec, module, entry_function):
        self.module_spec = module_spec
        self.module = module
        self.entry_function = entry_function
        self.entry_block = None
        self._function_stack = [entry_function]
        self._entry_child_op = entry_function.operation.parent.parent
        self._entry_child_symbol_table = entry_function.operation.parent.regions[0].blocks[0]
        self._function_symbol_table = self._entry_child_symbol_table
        self._helpers: dict[str, object] = {}
        self._kernel_module_primary_functions: dict[tuple, object] = {}
        self._kernel_module_private_imports: dict[tuple[str, str], object] = {}
        self._kernel_module_dependencies: dict[str, set[str]] = {}
        self._kernel_module_child_symbol_tables: dict[str, object] = {
            self.current_function_symbol_name: self._entry_child_symbol_table,
        }
        self._kernel_module_child_records: dict[str, TracedChildModuleRecord] = {
            self.current_function_symbol_name: TracedChildModuleRecord(
                symbol_name=f"{self.current_function_symbol_name}$child",
                primary_symbol_name=self.current_function_symbol_name,
                role="entry" if module_spec.entry else "kernel_module",
                module_spec=module_spec,
            )
        }
        self._subkernel_stack: list[SubkernelTraceFrame] = []
        self._carry_loop_stack = []

    @property
    def current_function(self):
        return self._function_stack[-1]

    @property
    def current_function_symbol_name(self):
        return self.current_function.name.value

    @property
    def current_function_module_spec(self):
        """Return the module spec that owns the actively lowered function."""
        current_record = self._kernel_module_child_records.get(self.current_function_symbol_name)
        if current_record is not None:
            return current_record.module_spec
        return self.module_spec

    @property
    def current_subkernel(self):
        if not self._subkernel_stack:
            return None
        return self._subkernel_stack[-1]

    @property
    def subkernel_stack_depth(self):
        return len(self._subkernel_stack)

    @property
    def current_carry_loop(self):
        if not self._carry_loop_stack:
            return None
        return self._carry_loop_stack[-1]

    def bind_entry_block(self, entry_block) -> None:
        """Record the root entry block for the active trace."""
        self.entry_block = entry_block

    @contextmanager
    def enter_function(self, ir_fn):
        """Push *ir_fn* as the current active function in this session."""
        self._function_stack.append(ir_fn)
        try:
            yield ir_fn
        finally:
            popped = self._function_stack.pop()
            if popped is not ir_fn:
                raise RuntimeError("PTODSL trace-session function stack corruption detected")

    @contextmanager
    def enter_inline_subkernel(self, role: str, symbol_name: str, target: str):
        """Push one inline subkernel frame onto the active tracing stack."""
        frame = SubkernelTraceFrame(
            role=role,
            symbol_name=symbol_name,
            target=target,
        )
        self._subkernel_stack.append(frame)
        try:
            section_op = None
            if role == "simd":
                section_op = _pto.SectionVectorOp()
            elif role == "cube":
                section_op = _pto.SectionCubeOp()

            if section_op is None:
                yield frame
                return

            block = section_op.body.blocks.append()
            with InsertionPoint(block):
                yield frame
        finally:
            popped = self._subkernel_stack.pop()
            if popped is not frame:
                raise RuntimeError("PTODSL trace-session subkernel stack corruption detected")

    @contextmanager
    def enter_subkernel(self, subkernel):
        """Push *subkernel* as the current active inline-lowering frame."""
        with self.enter_inline_subkernel(
            subkernel.spec.role.value,
            subkernel.spec.symbol_name,
            subkernel.spec.target,
        ) as frame:
            yield frame

    @contextmanager
    def suspend_subkernel_scope(self):
        """Temporarily clear caller-owned subkernel scope while lowering a new function body."""
        saved_stack = self._subkernel_stack
        self._subkernel_stack = []
        try:
            yield
        finally:
            self._subkernel_stack = saved_stack

    def lower_inline_subkernel(self, subkernel, *args, **kwargs):
        """Lower one inline PTODSL subkernel call through the shared session."""
        with self.enter_subkernel(subkernel):
            return subkernel.emit_body(*args, **kwargs)

    def begin_carry_loop(self, start, stop, step, state_items):
        """Materialize one authored ``pto.for_(...).carry(...)`` loop body."""
        frame = build_carry_loop_frame(start, stop, step, state_items)
        self._carry_loop_stack.append(frame)
        return frame

    def update_carry_loop(self, frame, **kwargs):
        """Emit the one legal ``loop.update(...)`` for the active carry loop."""
        active = self.current_carry_loop
        if active is None or active is not frame:
            raise RuntimeError("loop.update(...) may only be called inside the active carry loop body")
        yield_carry_loop_state(frame, **kwargs)

    def finish_carry_loop(self, frame, exc_type, exc, tb):
        """Finalize one active authored carry loop and close its body insertion point."""
        if not self._carry_loop_stack:
            raise RuntimeError("carry-loop exit without a matching active PTODSL trace-session frame")
        popped = self._carry_loop_stack.pop()
        if popped is not frame:
            raise RuntimeError("PTODSL trace-session carry-loop stack corruption detected")
        finish_carry_loop_frame(frame, exc_type, exc, tb)

    def lower_simt_helper_subkernel(self, subkernel, *args, **kwargs):
        """Lower one ``@pto.simt`` call through a dedicated helper function."""
        outer_frame = self.current_subkernel
        if outer_frame is not None and outer_frame.role == "simt":
            raise RuntimeError("@pto.simt helper lowering does not support nested SIMT helper calls")

        arg_templates = tuple(args)
        arg_types = tuple(unwrap_surface_value(arg).type for arg in arg_templates)
        helper_spec = HelperFunctionSpec(
            symbol_name=subkernel.spec.symbol_name,
            arg_types=arg_types,
            attributes=(("pto.simt_entry", UnitAttr.get()),),
        )
        helper_fn, created = self.get_or_create_helper_function(helper_spec)

        if created:
            entry_block = helper_fn.add_entry_block()
            wrapped_args = tuple(
                wrap_like_surface_value(template, value)
                for template, value in zip(arg_templates, entry_block.arguments)
            )
            with self.enter_function(helper_fn), self.enter_subkernel(subkernel), InsertionPoint(entry_block):
                returned_early = False
                try:
                    subkernel.emit_body(*wrapped_args, **kwargs)
                except _ExplicitReturnSignal:
                    returned_early = True
                if not returned_early:
                    func.ReturnOp([])

        i32 = IntegerType.get_signless(32)
        dim_z = arith.ConstantOp(i32, 1).result
        dim_y = arith.ConstantOp(i32, 1).result
        dim_x = arith.ConstantOp(i32, 1).result
        _pto.StoreVfSimtInfoOp(dim_z, dim_y, dim_x)
        func.CallOp(helper_fn, [unwrap_surface_value(arg) for arg in arg_templates])

    def lower_kernel_module_call(self, kernel_handle, *args, **kwargs):
        """Lower one ``@pto.jit(entry=False)`` kernel-module call in the active trace."""
        if kwargs:
            raise TypeError("@pto.jit(entry=False) kernel module calls do not support keyword arguments yet")

        compiler = kernel_handle._compiler
        kernel_signature = compiler._kernel_signature
        if kernel_signature.constexpr_parameters:
            raise RuntimeError(
                "@pto.jit(entry=False) kernel modules do not support constexpr specialization parameters"
            )
        positional_params = kernel_signature.positional_parameters
        if len(args) != len(positional_params):
            raise TypeError(
                f"@pto.jit(entry=False) kernel module {kernel_handle._py_name!r} expects "
                f"{len(positional_params)} argument(s), got {len(args)}"
            )
        arg_templates = tuple(
            const(arg, dtype=param.annotation)
            if isinstance(param, RuntimeScalarParameterSpec) and not hasattr(unwrap_surface_value(arg), "type")
            else arg
            for param, arg in zip(positional_params, args)
        )

        arg_types = tuple(unwrap_surface_value(arg).type for arg in arg_templates)
        helper_spec = HelperFunctionSpec(
            symbol_name=compiler._module_spec.function_name,
            arg_types=arg_types,
        )
        helper_fn, created = self.get_or_create_kernel_module_primary_function(
            helper_spec,
            compiler._module_spec,
        )

        if created:
            entry_block = helper_fn.add_entry_block()
            wrapped_args = tuple(
                wrap_like_surface_value(template, value)
                for template, value in zip(arg_templates, entry_block.arguments)
            )
            with self.enter_function(helper_fn), self.suspend_subkernel_scope(), InsertionPoint(entry_block):
                returned_early = False
                try:
                    compiler._callback(*wrapped_args)
                except _ExplicitReturnSignal:
                    returned_early = True
                if not returned_early:
                    func.ReturnOp([])

        caller_symbol_name = self.current_function_symbol_name
        import_fn, _ = self.get_or_create_kernel_module_import_declaration(
            caller_symbol_name,
            helper_spec,
        )
        self.record_kernel_module_dependency(caller_symbol_name, helper_spec.symbol_name)
        call_args = [unwrap_surface_value(arg) for arg in arg_templates]
        func.CallOp(import_fn, call_args)

    def lookup_helper(self, symbol_name: str):
        """Return a previously declared helper function, or ``None``."""
        return self._helpers.get(symbol_name)

    def get_or_create_helper_function(self, spec: HelperFunctionSpec):
        """
        Look up or create a helper ``func.func`` in the current symbol table.

        Returns ``(helper_fn, created)`` where *created* reports whether a new
        symbol was emitted in this trace session.
        """
        helper = self._helpers.get(spec.symbol_name)
        if helper is not None:
            return helper, False

        fn_ty = func.FunctionType.get(list(spec.arg_types), list(spec.result_types))
        with InsertionPoint(self._function_symbol_table):
            helper = func.FuncOp(spec.symbol_name, fn_ty)
            for attr_name, attr_value in spec.attributes:
                helper.attributes[attr_name] = attr_value
        self._helpers[spec.symbol_name] = helper
        return helper, True

    def get_or_create_kernel_module_primary_function(self, spec: HelperFunctionSpec, module_spec):
        """Look up or create the primary definition for one kernel-module callee."""
        cache_key = spec.cache_key()
        helper = self._kernel_module_primary_functions.get(cache_key)
        if helper is not None:
            return helper, False

        fn_ty = func.FunctionType.get(list(spec.arg_types), list(spec.result_types))
        specialized_symbol_name = spec.specialized_symbol_name()
        symbol_table = self.get_or_create_kernel_module_child_symbol_table(specialized_symbol_name, module_spec)
        with InsertionPoint(symbol_table):
            helper = func.FuncOp(specialized_symbol_name, fn_ty)
            helper.attributes["sym_visibility"] = StringAttr.get("public")
            if (
                module_spec.backend == "emitc"
                and not module_spec.entry
                and module_spec.kernel_kind in {"cube", "vector"}
            ):
                helper.attributes["pto.kernel_kind"] = Attribute.parse(
                    f"#pto.kernel_kind<{module_spec.kernel_kind}>"
                )
            for attr_name, attr_value in spec.attributes:
                helper.attributes[attr_name] = attr_value
        self._kernel_module_primary_functions[cache_key] = helper
        return helper, True

    def kernel_module_import_symbol_name(self, caller_symbol_name: str, callee_symbol_name: str) -> str:
        """Return the import declaration symbol for one caller/callee pair."""
        _ = caller_symbol_name
        return callee_symbol_name

    def get_or_create_kernel_module_import_declaration(
        self,
        caller_symbol_name: str,
        spec: HelperFunctionSpec,
    ):
        """Look up or create the private import declaration for one kernel-module callee."""
        target_symbol_name = spec.specialized_symbol_name()
        key = (caller_symbol_name, target_symbol_name)
        helper = self._kernel_module_private_imports.get(key)
        if helper is not None:
            return helper, False

        fn_ty = func.FunctionType.get(list(spec.arg_types), list(spec.result_types))
        import_symbol_name = self.kernel_module_import_symbol_name(caller_symbol_name, target_symbol_name)
        caller_symbol_table = self.get_or_create_kernel_module_child_symbol_table(
            caller_symbol_name,
            self._kernel_module_child_records[caller_symbol_name].module_spec,
        )
        with InsertionPoint(caller_symbol_table):
            helper = func.FuncOp(import_symbol_name, fn_ty)
            helper.attributes["sym_visibility"] = StringAttr.get("private")
        self._kernel_module_private_imports[key] = helper
        return helper, True

    def record_kernel_module_dependency(self, caller_symbol_name: str, callee_symbol_name: str) -> None:
        """Record one caller->callee dependency edge for kernel-module assembly."""
        deps = self._kernel_module_dependencies.setdefault(caller_symbol_name, set())
        deps.add(callee_symbol_name)

    def snapshot_kernel_module_graph(self) -> KernelModuleGraphSnapshot:
        """Return an immutable snapshot of traced kernel-module imports/dependencies."""
        imports = tuple(
            sorted(
                (
                    KernelModuleImportRecord(
                        caller_symbol_name=caller_symbol_name,
                        import_symbol_name=helper.name.value,
                        target_symbol_name=callee_symbol_name,
                    )
                    for (caller_symbol_name, callee_symbol_name), helper in self._kernel_module_private_imports.items()
                ),
                key=lambda record: (
                    record.caller_symbol_name,
                    record.target_symbol_name,
                    record.import_symbol_name,
                ),
            )
        )
        dependencies = tuple(
            (caller_symbol_name, tuple(sorted(callee_symbol_names)))
            for caller_symbol_name, callee_symbol_names in sorted(self._kernel_module_dependencies.items())
        )
        return KernelModuleGraphSnapshot(imports=imports, dependencies=dependencies)

    def get_or_create_kernel_module_child_symbol_table(self, primary_symbol_name: str, module_spec):
        """Return the child-module symbol table that owns *primary_symbol_name*."""
        symbol_table = self._kernel_module_child_symbol_tables.get(primary_symbol_name)
        if symbol_table is not None:
            return symbol_table

        child_op, symbol_table = create_container_child_module(self.module, module_spec)
        self._kernel_module_child_symbol_tables[primary_symbol_name] = symbol_table
        self._kernel_module_child_records[primary_symbol_name] = TracedChildModuleRecord(
            symbol_name=f"{primary_symbol_name}$child",
            primary_symbol_name=primary_symbol_name,
            role="kernel_module",
            module_spec=module_spec,
        )
        return symbol_table

    def validate_final_state(self) -> None:
        """Check that tracing-time session stacks were fully unwound."""
        if self._subkernel_stack:
            raise RuntimeError("PTODSL trace-session exited with an open subkernel lowering frame")
        if self._carry_loop_stack:
            raise RuntimeError("PTODSL trace-session exited with an open loop-carry lowering frame")


__all__ = [
    "HelperFunctionSpec",
    "KernelModuleGraphSnapshot",
    "KernelModuleImportRecord",
    "SubkernelTraceFrame",
    "TraceSession",
]
