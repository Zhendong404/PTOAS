#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import difflib
import json
import sys
from pathlib import Path

from family_dsl import ensure_catalog_sync, load_family_dsl


SCRIPT_DIR = Path(__file__).resolve().parent
SKELETON_DIR = SCRIPT_DIR / "skeletons"
CATALOG_PATH = SKELETON_DIR / "catalog.json"
MODULE_TEMPLATE_PATH = SKELETON_DIR / "module.tmpl.mlir"

TILE_TYPE_FMT = (
    "!pto.tile_buf<loc=vec, dtype={dtype}, rows=32, cols=32, v_row=?, "
    "v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>"
)
MEMREF_TYPE_FMT = (
    "memref<?x?x{dtype}, strided<[32, 1], offset: 0>, #pto.address_space<vec>>"
)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_catalog() -> dict:
    catalog = json.loads(load_text(CATALOG_PATH))
    ensure_catalog_sync(catalog, load_family_dsl())
    return catalog


def replace_all(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace(f"@@{key}@@", value)
    return out


def sanitize_symbol(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch == "_":
            chars.append(ch.lower())
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def lower_name(value: str) -> str:
    return value.lower()


def tile_type(dtype: str) -> str:
    return TILE_TYPE_FMT.format(dtype=dtype)


def scalar_type(dtype: str) -> str:
    return dtype


def vector_type(dtype: str) -> str:
    return f"vector<64x{dtype}>"


def mask_vector_type() -> str:
    return "vector<64xi1>"


def memref_type(dtype: str) -> str:
    return MEMREF_TYPE_FMT.format(dtype=dtype)


FLOAT_BINARY_CORE_OPS = {
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.maximumf",
    "arith.minimumf",
}


def dense_literal(dtype: str, value: str) -> str:
    if value == "zero":
        return "dense<0.0>" if dtype.startswith("f") else "dense<0>"
    if value == "one":
        return "dense<1.0>" if dtype.startswith("f") else "dense<1>"
    if value == "neg_one":
        return "dense<-1.0>" if dtype.startswith("f") else "dense<-1>"
    raise ValueError(f"unsupported dense literal '{value}' for dtype {dtype}")


def format_vector_compute(op_name: str, lhs: str, rhs: str, vector_ty: str,
                          dtype: str) -> str:
    attrs = ""
    if dtype.startswith("f") and op_name in FLOAT_BINARY_CORE_OPS:
        attrs = ' {pto.simd.exec_mode = "MODE_ZEROING"}'
    return f"          %result = {op_name} {lhs}, {rhs}{attrs} : {vector_ty}\n"


def format_vector_assign(name: str, op_name: str, lhs: str, rhs: str,
                         vector_ty: str, dtype: str) -> str:
    attrs = ""
    if dtype.startswith("f") and op_name in FLOAT_BINARY_CORE_OPS:
        attrs = ' {pto.simd.exec_mode = "MODE_ZEROING"}'
    return f"          %{name} = {op_name} {lhs}, {rhs}{attrs} : {vector_ty}\n"


def build_arg_decls(arg_roles: list[str], input_dtype: str,
                    result_dtype: str | None) -> str:
    decls: list[str] = []
    tile_index = 0
    scalar_index = 0
    for idx, role in enumerate(arg_roles):
        is_last = idx == len(arg_roles) - 1
        if role == "tile":
            if is_last:
                name = "dst"
                dtype = result_dtype or input_dtype
            else:
                name = f"src{tile_index}"
                dtype = input_dtype
                tile_index += 1
            decls.append(f"      %{name}: {tile_type(dtype)}")
            continue
        name = "scalar" if scalar_index == 0 else f"scalar{scalar_index}"
        scalar_index += 1
        decls.append(f"      %{name}: {scalar_type(input_dtype)}")
    return ",\n".join(decls)


def build_match_attrs(arg_roles: list[str]) -> str:
    lines: list[str] = []
    for idx, role in enumerate(arg_roles):
        if role != "tile":
            continue
        lines.extend(
            [
                f"        pto.oplib.match.arg{idx}.rows = -1 : i64,",
                f"        pto.oplib.match.arg{idx}.cols = -1 : i64,",
                f'        pto.oplib.match.arg{idx}.blayout = "row_major",',
                f'        pto.oplib.match.arg{idx}.slayout = "any",',
                f"        pto.oplib.match.arg{idx}.fractal = -1 : i64,",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def build_func_name(pattern: dict, family: dict, op_info: dict, op_name: str,
                    variant_id: str, dtype: str) -> str:
    func_name_format = op_info.get("func_name_format", family.get("func_name_format"))
    if func_name_format:
        return func_name_format.format(
            pattern_id=pattern["id"],
            family_id=family["family_id"],
            kind=family["kind"],
            op=op_name,
            variant_id=variant_id,
            dtype=dtype,
        )
    return sanitize_symbol(
        f"__pto_oplib_{pattern['id']}_{family['kind']}_{op_name}_{variant_id}_{dtype}"
    )


def build_compute_block(pattern_id: str, op_info: dict, dtype: str,
                        variant_id: str, vector_ty: str) -> tuple[str, str]:
    extra_setup = ""
    if pattern_id == "binary":
        compute = format_vector_compute(
            op_info["core_op"], "%lhs", "%rhs", vector_ty, dtype
        )
        return extra_setup, compute

    if pattern_id == "tile_scalar":
        body_kind = op_info.get("body_kind")
        if body_kind == "lrelu":
            extra_setup = (
                f"      %zeroVec = arith.constant {dense_literal(dtype, 'zero')} : "
                f"{vector_ty}\n"
            )
            compute = (
                f"          %positive = arith.cmpf ogt, %lhs, %zeroVec : {vector_ty}\n"
                + format_vector_assign(
                    "scaled", "arith.mulf", "%lhs", "%scalarVec", vector_ty, dtype
                )
                + f"          %result = arith.select %positive, %lhs, %scaled : "
                f"vector<64xi1>, {vector_ty}\n"
            )
            return extra_setup, compute
        operand_order = op_info.get("operand_order", variant_id)
        lhs = "%lhs"
        rhs = "%scalarVec"
        if operand_order == "scalar_tile":
            lhs, rhs = rhs, lhs
        compute = format_vector_compute(op_info["core_op"], lhs, rhs, vector_ty, dtype)
        return extra_setup, compute

    if pattern_id == "unary":
        body_kind = op_info.get("body_kind", op_info["op"])
        if body_kind == "not":
            extra_setup = (
                f"      %allOnes = arith.constant {dense_literal(dtype, 'neg_one')} : "
                f"{vector_ty}\n"
            )
            compute = f"          %result = arith.xori %lhs, %allOnes : {vector_ty}\n"
            return extra_setup, compute
        if body_kind == "neg":
            compute = f"          %result = arith.negf %lhs : {vector_ty}\n"
            return extra_setup, compute
        if body_kind == "abs":
            compute = (
                f"          %neg = arith.negf %lhs : {vector_ty}\n"
                + format_vector_compute(
                    "arith.maximumf", "%neg", "%lhs", vector_ty, dtype
                )
            )
            return extra_setup, compute
        if body_kind == "recip":
            extra_setup = (
                f"      %ones = arith.constant {dense_literal(dtype, 'one')} : "
                f"{vector_ty}\n"
            )
            compute = format_vector_compute(
                "arith.divf", "%ones", "%lhs", vector_ty, dtype
            )
            return extra_setup, compute
        if body_kind == "relu":
            extra_setup = (
                f"      %zeroVec = arith.constant {dense_literal(dtype, 'zero')} : "
                f"{vector_ty}\n"
            )
            compute = format_vector_compute(
                "arith.maximumf", "%lhs", "%zeroVec", vector_ty, dtype
            )
            return extra_setup, compute
        if body_kind == "exp":
            compute = f"          %result = math.exp %lhs : {vector_ty}\n"
            return extra_setup, compute
        if body_kind == "log":
            compute = f"          %result = math.log %lhs : {vector_ty}\n"
            return extra_setup, compute
        if body_kind == "sqrt":
            compute = f"          %result = math.sqrt %lhs : {vector_ty}\n"
            return extra_setup, compute
        if body_kind == "rsqrt":
            extra_setup = (
                f"      %ones = arith.constant {dense_literal(dtype, 'one')} : "
                f"{vector_ty}\n"
            )
            compute = (
                f"          %sqrt = math.sqrt %lhs : {vector_ty}\n"
                + format_vector_compute(
                    "arith.divf", "%ones", "%sqrt", vector_ty, dtype
                )
            )
            return extra_setup, compute

    raise ValueError(
        f"unsupported compute body for pattern={pattern_id} op={op_info['op']} dtype={dtype}"
    )


def expand_family_instances(pattern: dict, family: dict) -> list[dict]:
    instances: list[dict] = []
    dtypes = family["dtypes"]
    ops = family["ops"]
    conditions = family.get("conditions", [None])
    result_dtype = family.get("result_dtype")
    scalar_pos = family.get("scalar_pos")
    passive_vectors = family.get("passive_vectors", {})
    has_scalar_operand = "scalar" in family["arg_roles"]
    for op_info in ops:
        op_dtypes = op_info.get("dtypes", dtypes)
        for dtype in op_dtypes:
            for condition_info in conditions:
                condition = condition_info
                cmp_predicate = ""
                variant_id = family.get("variant_id_by_dtype", {}).get(
                    dtype, op_info.get("variant_id")
                )
                if isinstance(condition_info, dict):
                    condition = condition_info["mode"]
                    cmp_predicate = condition_info.get("predicate", "")
                    variant_id = condition_info.get("variant_id", variant_id)
                if variant_id is None:
                    variant_id = lower_name(condition) if condition else lower_name(
                        op_info["op"]
                    )
                cost = family.get("cost", 10)
                priority = family.get("priority", 0)
                if isinstance(condition_info, dict):
                    cost = condition_info.get("cost", cost)
                    priority = condition_info.get("priority", priority)
                cost = op_info.get("cost", cost)
                priority = op_info.get("priority", priority)
                func_name = build_func_name(
                    pattern, family, op_info, op_info["op"], variant_id, dtype
                )
                axis_values = [
                    f"dtype={dtype}",
                    f"core_op={op_info['core_op']}",
                    f"variant_id={variant_id}",
                ]
                if condition:
                    axis_values.insert(1, f"condition={condition}")
                rhs_memref_cast = ""
                rhs_setup = ""
                rhs_load = ""
                rhs_value = "%src1v"
                if has_scalar_operand:
                    rhs_setup = (
                        f"      %scalarVec = vector.splat %scalar : {vector_type(dtype)}\n"
                    )
                    rhs_value = "%scalarVec"
                else:
                    rhs_memref_cast = (
                        f"    %m1 = pto.simd.tile_to_memref %src1 : {tile_type(dtype)} "
                        f"to {memref_type(dtype)}\n"
                    )
                    rhs_load = (
                        f"          %src1v = vector.maskedload %m1[%r, %cidx], %mask, "
                        f"%passive {{pto.simd.vld_dist = \"NORM\"}} : {memref_type(dtype)}, "
                        f"{mask_vector_type()}, {vector_type(dtype)} into {vector_type(dtype)}\n"
                    )
                extra_setup = ""
                compute = ""
                if pattern["id"] in {"binary", "tile_scalar", "unary"}:
                    extra_setup, compute = build_compute_block(
                        pattern["id"],
                        op_info,
                        dtype,
                        variant_id,
                        vector_type(result_dtype or dtype),
                    )
                instances.append(
                    {
                        "family_id": family["family_id"],
                        "func_name": func_name,
                        "kind": family["kind"],
                        "op_name": op_info["op"],
                        "variant_id": variant_id,
                        "match_dtype": dtype,
                        "arg_decls": build_arg_decls(
                            family["arg_roles"], dtype, result_dtype
                        ),
                        "match_attrs": build_match_attrs(family["arg_roles"]),
                        "core_op": op_info["core_op"],
                        "cmp_mode": condition or "",
                        "cmp_predicate": cmp_predicate,
                        "scalar_pos_attr": (
                            f"        pto.oplib.match.scalar_pos = {scalar_pos} : i64,\n"
                            if scalar_pos is not None
                            else ""
                        ),
                        "scalar_pos": "" if scalar_pos is None else str(scalar_pos),
                        "axis_values": ", ".join(axis_values),
                        "input_tile_type": tile_type(dtype),
                        "result_tile_type": tile_type(result_dtype or dtype),
                        "input_memref_type": memref_type(dtype),
                        "result_memref_type": memref_type(result_dtype or dtype),
                        "input_vector_type": vector_type(dtype),
                        "result_vector_type": vector_type(result_dtype or dtype),
                        "mask_vector_type": mask_vector_type(),
                        "passive_vector": passive_vectors.get(
                            dtype, dense_literal(dtype, "zero")
                        ),
                        "rhs_memref_cast": rhs_memref_cast,
                        "rhs_setup": rhs_setup,
                        "rhs_load": rhs_load,
                        "rhs_value": rhs_value,
                        "extra_setup": extra_setup,
                        "compute": compute,
                        "cost": str(cost),
                        "priority": str(priority),
                    }
                )
    return instances


def render_pattern(pattern: dict, families: list[dict] | None = None,
                   output_role: str | None = None) -> str:
    instance_template = load_text(SKELETON_DIR / pattern["template"])
    module_template = load_text(MODULE_TEMPLATE_PATH)
    instance_blocks: list[str] = []
    for family in families or pattern["families"]:
        for instance in expand_family_instances(pattern, family):
            instance_blocks.append(
                replace_all(
                    instance_template,
                    {
                        "FAMILY_ID": instance["family_id"],
                        "AXIS_VALUES": instance["axis_values"],
                        "FUNC_NAME": instance["func_name"],
                        "ARG_DECLS": instance["arg_decls"],
                        "KIND": instance["kind"],
                        "OP_NAME": instance["op_name"],
                        "VARIANT_ID": instance["variant_id"],
                        "MATCH_DTYPE": instance["match_dtype"],
                        "MATCH_ATTRS": instance["match_attrs"],
                        "CORE_OP": instance["core_op"],
                        "CMP_MODE": instance["cmp_mode"],
                        "CMP_PREDICATE": instance["cmp_predicate"],
                        "SCALAR_POS_ATTR": instance["scalar_pos_attr"],
                        "SCALAR_POS": instance["scalar_pos"],
                        "INPUT_TILE_TYPE": instance["input_tile_type"],
                        "RESULT_TILE_TYPE": instance["result_tile_type"],
                        "INPUT_MEMREF_TYPE": instance["input_memref_type"],
                        "RESULT_MEMREF_TYPE": instance["result_memref_type"],
                        "INPUT_VECTOR_TYPE": instance["input_vector_type"],
                        "RESULT_VECTOR_TYPE": instance["result_vector_type"],
                        "VECTOR_TYPE": instance["input_vector_type"],
                        "MASK_VECTOR_TYPE": instance["mask_vector_type"],
                        "PASSIVE_VECTOR": instance["passive_vector"],
                        "RHS_MEMREF_CAST": instance["rhs_memref_cast"],
                        "RHS_SETUP": instance["rhs_setup"],
                        "RHS_LOAD": instance["rhs_load"],
                        "RHS_VALUE": instance["rhs_value"],
                        "EXTRA_SETUP": instance["extra_setup"],
                        "COMPUTE": instance["compute"],
                        "COST": instance["cost"],
                        "PRIORITY": instance["priority"],
                    },
                ).rstrip()
            )
    rendered = replace_all(
        module_template,
        {
            "PATTERN_ID": pattern["id"],
            "TEMPLATE_PATH": f"skeletons/{pattern['template']}",
            "AXES": ", ".join(pattern["axes"]),
            "OUTPUT_ROLE": output_role
            or pattern.get(
                "output_role",
                "generated concrete templates synchronized from skeleton source.",
            ),
            "INSTANCE_BLOCKS": "\n\n".join(instance_blocks),
        },
    )
    return rendered.rstrip() + "\n"


def render_outputs() -> dict[Path, str]:
    outputs: dict[Path, str] = {}
    for pattern in load_catalog()["patterns"]:
        for family in pattern["families"]:
            for target in family.get("targets", []):
                target_family = copy.deepcopy(family)
                target_family["dtypes"] = target["dtypes"]
                if "func_name_format" in target:
                    target_family["func_name_format"] = target["func_name_format"]
                outputs[SCRIPT_DIR / target["output"]] = render_pattern(
                    pattern,
                    families=[target_family],
                    output_role=target.get(
                        "output_role",
                        "generated concrete templates synchronized from skeleton source.",
                    ),
                )
            family_output = family.get("output")
            if not family_output:
                continue
            outputs[SCRIPT_DIR / family_output] = render_pattern(
                pattern,
                families=[family],
                output_role=family.get(
                    "output_role",
                    "generated concrete templates synchronized from skeleton source.",
                ),
            )
    return outputs


def write_outputs(outputs: dict[Path, str]) -> int:
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(SCRIPT_DIR)}")
    return 0


def check_outputs(outputs: dict[Path, str]) -> int:
    failures = 0
    for path, expected in outputs.items():
        if not path.exists():
            print(f"missing generated file: {path.relative_to(SCRIPT_DIR)}", file=sys.stderr)
            failures += 1
            continue
        actual = load_text(path)
        if actual == expected:
            print(f"ok {path.relative_to(SCRIPT_DIR)}")
            continue
        failures += 1
        print(f"drift detected: {path.relative_to(SCRIPT_DIR)}", file=sys.stderr)
        diff = difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(path),
            tofile=f"{path} (expected)",
            lineterm="",
        )
        for line in diff:
            print(line, file=sys.stderr)
    return 1 if failures else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Level-3 concrete templates from skeleton sources."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write generated concrete templates")
    mode.add_argument("--check", action="store_true", help="check generated concrete templates")
    mode.add_argument("--list", action="store_true", help="list generated concrete output paths")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    outputs = render_outputs()
    if args.list:
        for path in sorted(outputs):
            print(path.relative_to(SCRIPT_DIR))
        return 0
    if args.write:
        return write_outputs(outputs)
    return check_outputs(outputs)


if __name__ == "__main__":
    sys.exit(main())
