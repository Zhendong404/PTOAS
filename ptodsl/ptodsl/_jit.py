# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""``@pto.jit`` decorator and compiled-kernel handles."""

from __future__ import annotations

import inspect

from ._diagnostics import (
    invalid_jit_backend_error,
    invalid_jit_mode_error,
    kernel_module_launch_error,
)
from ._kernel_compilation import CompiledKernelHandle, KernelCompiler
from ._kernel_signature import parse_jit_kernel_signature
from ._tracing import (
    KernelModuleSpec,
    ModuleArtifact,
    ModuleStyle,
    current_runtime,
)

from mlir.ir import InsertionPoint


_MODULE_ATTRS = ("pto.target_arch",)


def _normalize_mode(mode: str, *, fn=None) -> str:
    if mode not in {"auto", "explicit"}:
        source_file = None
        source_line = None
        function_name = None
        if fn is not None:
            function_name = fn.__name__
            try:
                source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
            except (OSError, TypeError):
                source_file = None
            source_line = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
        raise invalid_jit_mode_error(
            mode,
            function_name=function_name,
            source_file=source_file,
            source_line=source_line,
        )
    return mode


def _normalize_backend(backend: str, *, fn=None) -> str:
    if backend not in {"vpto", "emitc"}:
        source_file = None
        source_line = None
        function_name = None
        if fn is not None:
            function_name = fn.__name__
            try:
                source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
            except (OSError, TypeError):
                source_file = None
            source_line = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
        raise invalid_jit_backend_error(
            backend,
            function_name=function_name,
            source_file=source_file,
            source_line=source_line,
        )
    return backend


def _module_attr_map(module):
    attrs = module.operation.attributes
    return {name: str(attrs[name]) for name in _MODULE_ATTRS if name in attrs}


def merge_jit_modules(*kernels: KernelHandle):
    """
    Merge multiple ``@pto.jit`` kernels into one MLIR module.

    Each handle must have been compiled with compatible outer-container
    attributes. Child module order follows *kernels*.
    """
    if not kernels:
        raise ValueError("merge_jit_modules() requires at least one kernel handle")

    merged = kernels[0].build()
    expected_attrs = _module_attr_map(merged)

    for kernel in kernels[1:]:
        module = kernel.build()
        actual_attrs = _module_attr_map(module)
        if actual_attrs != expected_attrs:
            raise ValueError(
                "merge_jit_modules() requires compatible module attributes; "
                f"expected {expected_attrs}, got {actual_attrs}"
            )
        with InsertionPoint(merged.body):
            for op in module.body.operations:
                if op.operation.name != "builtin.module":
                    raise ValueError(
                        "merge_jit_modules() expects backend-partitioned PTODSL containers "
                        "whose payload is expressed entirely through child modules"
                    )
                op.operation.clone()

    merged.operation.verify()
    return merged


def jit(
    name=None,
    *,
    target: str = "a5",
    kernel_kind: str = "vector",
    backend: str = "vpto",
    entry: bool = True,
    mode: str = "auto",
    insert_sync: bool | None = None,
):
    """
    Decorator that wraps a Python function as a PTODSL JIT kernel template.

    Parameters
    ----------
    name:        IR function name (defaults to the Python function name).
    target:      Target architecture string, e.g. ``"a5"``.
    kernel_kind: ``"vector"`` or ``"cube"`` – sets ``pto.kernel_kind``.
    backend:     ``"vpto"`` or ``"emitc"`` – records the intended backend.
    entry:       ``True`` for launchable kernel entries, ``False`` for helpers.
    mode:        ``"auto"`` or ``"explicit"`` – feeds child compile policy.
    insert_sync: ``True``/``False`` to explicitly control PTOAS sync insertion
                 for launch builds. ``None`` keeps the mode-based default
                 behavior.

    The decorated function is replaced by a :class:`KernelHandle` that:

    - supports ``my_kernel.compile(**constexprs)`` specialization,
    - prints as the default-specialization MLIR text,
    - exposes ``my_kernel.mlir_module()`` / ``verify()`` / ``emit()`` on the
      default specialization for convenience.
    - emits a backend-partitioned outer container by default.
    """

    def decorator(fn):
        fn_name = name or fn.__name__
        kernel_signature = parse_jit_kernel_signature(fn, entry=entry)
        normalized_mode = _normalize_mode(mode, fn=fn)
        normalized_backend = _normalize_backend(backend, fn=fn)
        source_file = None
        try:
            source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
        except (OSError, TypeError):
            source_file = None
        compiler = KernelCompiler(
            fn.__name__,
            KernelModuleSpec(
                function_name=fn_name,
                target_arch=target,
                kernel_kind=kernel_kind,
                backend=normalized_backend,
                entry=entry,
                mode=normalized_mode,
                insert_sync=insert_sync,
                module_style=ModuleStyle.BACKEND_PARTITIONED,
                source_file=source_file,
                source_line=getattr(fn.__code__, "co_firstlineno", None),
            ),
            kernel_signature,
            fn,
        )
        return KernelHandle(fn.__name__, compiler)

    return decorator


class KernelHandle(ModuleArtifact):
    """
    Represents a JIT kernel template plus its compiled specializations.

    ``handle.compile(**constexprs)`` returns one compiled specialization.
    ``print(handle)`` emits the default-specialization MLIR module text.
    """

    def __init__(self, py_name: str, compiler: KernelCompiler):
        self._compiler = compiler
        super().__init__(py_name, module_factory=self._build_default_module)

    def compile(self, **constexpr_bindings) -> CompiledKernelHandle:
        return self._compiler.compile(**constexpr_bindings)

    def __call__(self, *args, **kwargs):
        if self._compiler._module_spec.entry is not False:
            raise TypeError(
                f"@pto.jit entry kernel {self._py_name!r} is not callable from traced PTODSL bodies; "
                "only @pto.jit(entry=False) kernel modules may be called there."
            )
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                f"@pto.jit(entry=False) kernel module {self._py_name!r} may only be called while tracing "
                "a compatible PTODSL kernel"
            )
        return runtime.dispatch_kernel_module_call(self, *args, **kwargs)

    def __getitem__(self, launch_spec):
        if self._compiler._module_spec.entry is False:
            raise kernel_module_launch_error(self._py_name)
        return self.compile().__getitem__(launch_spec)

    def cached_specializations(self):
        return self._compiler.cached_specializations()

    def _build_default_module(self):
        return self.compile().build()


__all__ = ["CompiledKernelHandle", "jit", "KernelHandle", "merge_jit_modules"]
