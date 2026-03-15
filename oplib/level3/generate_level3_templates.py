#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

from family_dsl import ensure_catalog_sync, load_family_dsl, load_snippet_contracts


SCRIPT_DIR = Path(__file__).resolve().parent
SKELETON_DIR = SCRIPT_DIR / "skeletons"
SNIPPET_DIR = SCRIPT_DIR / "families" / "snippets"
CATALOG_PATH = SKELETON_DIR / "catalog.json"
MODULE_TEMPLATE_PATH = SKELETON_DIR / "module.tmpl.mlir"
GENERATED_FILE_BANNER = "// AUTO-GENERATED: do not edit directly."

TILE_TYPE_FMT = (
    "!pto.tile_buf<loc=vec, dtype={dtype}, rows=32, cols=32, v_row=?, "
    "v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>"
)
MEMREF_TYPE_FMT = (
    "memref<?x?x{dtype}, strided<[32, 1], offset: 0>, #pto.address_space<vec>>"
)

SETUP_BEGIN = "// SNIPPET_SETUP_BEGIN"
SETUP_END = "// SNIPPET_SETUP_END"
COMPUTE_BEGIN = "// SNIPPET_COMPUTE_BEGIN"
COMPUTE_END = "// SNIPPET_COMPUTE_END"

FLOAT_BINARY_CORE_OPS = {
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.maximumf",
    "arith.minimumf",
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_family_doc() -> dict[str, Any]:
    family_doc = load_family_dsl()
    catalog = json.loads(load_text(CATALOG_PATH))
    ensure_catalog_sync(catalog, family_doc)
    return family_doc


def is_generated_concrete_file(path: Path) -> bool:
    if path.suffix != ".mlir" or not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    return GENERATED_FILE_BANNER in lines[:4]


def list_existing_generated_outputs() -> list[Path]:
    return sorted(path for path in SCRIPT_DIR.glob("*.mlir") if is_generated_concrete_file(path))


def find_stale_generated_outputs(outputs: dict[Path, str]) -> list[Path]:
    expected = set(outputs)
    return [path for path in list_existing_generated_outputs() if path not in expected]


def validate_expected_output(path: Path, content: str) -> None:
    unresolved = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", content)))
    if unresolved:
        unresolved_list = ", ".join(unresolved)
        raise ValueError(
            f"generated output {path.relative_to(SCRIPT_DIR)} still contains unresolved placeholders: {unresolved_list}"
        )


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


def dense_literal(dtype: str, value: str) -> str:
    if value == "zero":
        return "dense<0.0>" if dtype.startswith("f") else "dense<0>"
    if value == "one":
        return "dense<1.0>" if dtype.startswith("f") else "dense<1>"
    if value == "neg_one":
        return "dense<-1.0>" if dtype.startswith("f") else "dense<-1>"
    raise ValueError(f"unsupported dense literal '{value}' for dtype {dtype}")


def exec_mode_attr(core_op: str, dtype: str) -> str:
    if dtype.startswith("f") and core_op in FLOAT_BINARY_CORE_OPS:
        return ' {pto.simd.exec_mode = "MODE_ZEROING"}'
    return ""


def build_arg_decls(arg_roles: list[str], input_dtype: str, result_dtype: str | None) -> str:
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


def build_func_name(
    pattern: dict[str, Any],
    family: dict[str, Any],
    op_info: dict[str, Any],
    variant_id: str,
    dtype: str,
) -> str:
    func_name_format = op_info.get("func_name_format", family["metadata"].get("func_name_format"))
    if func_name_format:
        return func_name_format.format(
            pattern_id=pattern["id"],
            family_id=family["family"],
            kind=family["kind"],
            op=op_info["name"],
            variant_id=variant_id,
            dtype=dtype,
        )
    return sanitize_symbol(
        f"__pto_oplib_{pattern['id']}_{family['kind']}_{op_info['name']}_{variant_id}_{dtype}"
    )


def family_scalar_index(family: dict[str, Any]) -> int | None:
    for index, role in enumerate(family["parameter_roles"]):
        if role["kind"] == "scalar":
            return index
    return None


def expand_ops(family: dict[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for op in family["ops"]:
        variants = op.get("variants", [])
        if not variants:
            expanded.append(op)
            continue
        for variant in variants:
            entry = dict(op)
            entry.pop("variants", None)
            entry["variant_id"] = variant["id"]
            for key, value in variant.items():
                if key in {"id", "body", "metadata", "request_keys"}:
                    continue
                entry[key] = value
            body = variant.get("body", {})
            entry.update(body)
            metadata = variant.get("metadata", {})
            entry.update(metadata)
            request_keys = variant.get("request_keys")
            if request_keys:
                entry["request_keys"] = request_keys
            expanded.append(entry)
    return expanded


def snippet_relpath(op_info: dict[str, Any]) -> str:
    snippet = str(op_info.get("snippet", "")).strip()
    if not snippet:
        raise ValueError(f"active generated op {op_info['name']} is missing snippet")
    return snippet


def load_snippet_sections(relative_path: str) -> tuple[str, str]:
    path = SNIPPET_DIR / relative_path
    text = load_text(path)
    if SETUP_BEGIN in text or COMPUTE_BEGIN in text:
        setup = extract_marked_section(text, SETUP_BEGIN, SETUP_END)
        compute = extract_marked_section(text, COMPUTE_BEGIN, COMPUTE_END)
    else:
        setup = ""
        compute = text.strip()
    return setup, compute


def extract_marked_section(text: str, begin: str, end: str) -> str:
    if begin not in text:
        return ""
    pattern = re.compile(rf"{re.escape(begin)}\n(.*?){re.escape(end)}", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"snippet section {begin} is missing closing marker {end}")
    return match.group(1).strip()


def indent_block(text: str, prefix: str) -> str:
    body = text.strip()
    if not body:
        return ""
    return "".join(f"{prefix}{line}\n" if line else "\n" for line in body.splitlines())


def validate_rendered_snippet(
    relative_path: str,
    contract: dict[str, Any],
    setup_text: str,
    compute_text: str,
) -> None:
    combined = "\n".join(part for part in (setup_text, compute_text) if part)
    for op_name in contract["forbidden_ops"]:
        if op_name in combined:
            raise ValueError(f"snippet {relative_path} uses forbidden op for contract {contract['id']}: {op_name}")

    result_ssa = contract["result_ssa"]
    if not re.search(rf"^\s*{re.escape(result_ssa)}\s*=", compute_text, re.MULTILINE):
        raise ValueError(f"snippet {relative_path} must define {result_ssa} in compute section")

    for ssa_name in contract["snippet_must_not_define"]:
        if ssa_name == result_ssa:
            continue
        if re.search(rf"^\s*{re.escape(ssa_name)}\s*=", combined, re.MULTILINE):
            raise ValueError(
                f"snippet {relative_path} redefines reserved SSA {ssa_name} for contract {contract['id']}"
            )


def build_snippet_mapping(
    op_info: dict[str, Any],
    dtype: str,
    result_dtype: str | None,
    cmp_predicate: str,
    rhs_value: str,
) -> dict[str, str]:
    operand_order = op_info.get("operand_order", "tile_scalar")
    snippet_lhs = "%lhs"
    snippet_rhs = "%scalarVec"
    if operand_order == "scalar_tile":
        snippet_lhs, snippet_rhs = snippet_rhs, snippet_lhs
    input_vector_ty = vector_type(dtype)
    result_vector_ty = vector_type(result_dtype or dtype)
    return {
        "CORE_OP": op_info["core_op"],
        "VECTOR_TYPE": result_vector_ty,
        "INPUT_VECTOR_TYPE": input_vector_ty,
        "RESULT_VECTOR_TYPE": result_vector_ty,
        "CMP_PREDICATE": cmp_predicate,
        "RHS_VALUE": rhs_value,
        "SNIPPET_LHS": snippet_lhs,
        "SNIPPET_RHS": snippet_rhs,
        "EXEC_MODE_ATTR": exec_mode_attr(op_info["core_op"], dtype),
        "ZERO_LITERAL": dense_literal(result_dtype or dtype, "zero"),
        "ONE_LITERAL": dense_literal(result_dtype or dtype, "one"),
        "NEG_ONE_LITERAL": dense_literal(result_dtype or dtype, "neg_one"),
    }


def build_snippet_blocks(
    op_info: dict[str, Any],
    contract: dict[str, Any],
    dtype: str,
    result_dtype: str | None,
    cmp_predicate: str,
    rhs_value: str,
) -> tuple[str, str]:
    relative_path = snippet_relpath(op_info)
    setup_text, compute_text = load_snippet_sections(relative_path)
    mapping = build_snippet_mapping(op_info, dtype, result_dtype, cmp_predicate, rhs_value)
    rendered_setup = replace_all(setup_text, mapping)
    rendered_compute = replace_all(compute_text, mapping)
    validate_rendered_snippet(relative_path, contract, rendered_setup, rendered_compute)
    return indent_block(rendered_setup, "      "), indent_block(rendered_compute, "          ")


def expand_family_instances(pattern: dict[str, Any], family: dict[str, Any], dtypes: list[str]) -> list[dict[str, str]]:
    contract_map = {
        contract["id"]: contract for contract in load_snippet_contracts()["contracts"]
    }
    contract = contract_map[family["snippet_contract"]]

    instances: list[dict[str, str]] = []
    conditions = family["variant_axis"].get("values", [None]) if family["variant_axis"]["basis"] == "condition_list" else [None]
    result_dtype = family.get("dtype_axis", {}).get("result_dtype")
    scalar_pos = family_scalar_index(family)
    passive_vectors = family.get("dtype_axis", {}).get("passive_vectors", {})
    has_scalar_operand = any(role["kind"] == "scalar" for role in family["parameter_roles"])
    arg_roles = [role["kind"] for role in family["parameter_roles"]]

    for op_info in expand_ops(family):
        requested_dtypes = op_info.get("dtypes")
        op_dtypes = [dtype for dtype in dtypes if requested_dtypes is None or dtype in requested_dtypes]
        for dtype in op_dtypes:
            effective_op_info = dict(op_info)
            effective_op_info["core_op"] = op_info.get("core_op_by_dtype", {}).get(dtype, op_info["core_op"])
            for condition_info in conditions:
                condition = ""
                cmp_predicate = ""
                variant_id = effective_op_info.get(
                    "variant_id",
                    family.get("dtype_axis", {}).get("variant_id_by_dtype", {}).get(dtype),
                )
                if isinstance(condition_info, dict):
                    condition = condition_info["matcher"]["cmpMode"]
                    cmp_predicate = condition_info.get("metadata", {}).get("predicate", "")
                    variant_id = condition_info["id"]
                if variant_id is None:
                    variant_id = lower_name(condition) if condition else lower_name(op_info["name"])

                cost = int(op_info.get("cost", family["metadata"].get("cost", 10)))
                priority = int(op_info.get("priority", family["metadata"].get("priority", 0)))
                func_name = build_func_name(pattern, family, effective_op_info, variant_id, dtype)

                axis_values = [
                    f"dtype={dtype}",
                    f"core_op={effective_op_info['core_op']}",
                    f"variant_id={variant_id}",
                ]
                if condition:
                    axis_values.insert(1, f"condition={condition}")

                rhs_memref_cast = ""
                rhs_setup = ""
                rhs_load = ""
                rhs_value = "%src1v"
                if has_scalar_operand:
                    rhs_setup = f"      %scalarVec = vector.splat %scalar : {vector_type(dtype)}\n"
                    rhs_value = "%scalarVec"
                elif pattern["id"] == "compare":
                    rhs_memref_cast = (
                        f"    %m1 = pto.simd.tile_to_memref %src1 : {tile_type(dtype)} to {memref_type(dtype)}\n"
                    )
                    rhs_load = (
                        f"          %src1v = vector.maskedload %m1[%r, %cidx], %mask, %passive "
                        f'{{pto.simd.vld_dist = "NORM"}} : {memref_type(dtype)}, {mask_vector_type()}, '
                        f"{vector_type(dtype)} into {vector_type(dtype)}\n"
                    )
                elif pattern["id"] in {"binary", "partial_binary"}:
                    rhs_value = "%rhs"

                extra_setup, compute = build_snippet_blocks(
                    effective_op_info,
                    contract,
                    dtype,
                    result_dtype,
                    cmp_predicate,
                    rhs_value,
                )

                instances.append(
                    {
                        "family_id": family["family"],
                        "func_name": func_name,
                        "kind": family["kind"],
                        "op_name": effective_op_info["name"],
                        "variant_id": variant_id,
                        "match_dtype": dtype,
                        "arg_decls": build_arg_decls(arg_roles, dtype, result_dtype),
                        "match_attrs": build_match_attrs(arg_roles),
                        "core_op": effective_op_info["core_op"],
                        "cmp_mode": condition,
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
                        "passive_vector": passive_vectors.get(dtype, dense_literal(dtype, "zero")),
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


def render_family_output(
    pattern: dict[str, Any],
    family: dict[str, Any],
    dtypes: list[str],
    output_role: str,
) -> str:
    instance_template = load_text(SKELETON_DIR / pattern["template"])
    module_template = load_text(MODULE_TEMPLATE_PATH)
    instance_blocks: list[str] = []
    for instance in expand_family_instances(pattern, family, dtypes):
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
            "OUTPUT_ROLE": output_role,
            "INSTANCE_BLOCKS": "\n\n".join(instance_blocks),
        },
    )
    return rendered.rstrip() + "\n"


def render_outputs() -> dict[Path, str]:
    family_doc = load_family_doc()
    pattern_map = {pattern["id"]: pattern for pattern in family_doc["patterns"]}
    outputs: dict[Path, str] = {}

    for family in family_doc["families"]:
        if family["status"] != "active":
            continue
        pattern = pattern_map[family["pattern"]]
        metadata = family["metadata"]
        family_dtypes = list(family["dtype_axis"]["values"])

        if "targets" in metadata:
            for target in metadata["targets"]:
                outputs[SCRIPT_DIR / target["output"]] = render_family_output(
                    pattern,
                    family,
                    list(target["dtypes"]),
                    target.get(
                        "output_role",
                        "generated concrete templates synchronized from skeleton source.",
                    ),
                )

        if "output" in metadata:
            outputs[SCRIPT_DIR / metadata["output"]] = render_family_output(
                pattern,
                family,
                family_dtypes,
                metadata.get(
                    "output_role",
                    "generated concrete templates synchronized from skeleton source.",
                ),
            )
    for path, content in outputs.items():
        validate_expected_output(path, content)
    return outputs


def write_outputs(outputs: dict[Path, str]) -> int:
    stale_outputs = find_stale_generated_outputs(outputs)
    for path in stale_outputs:
        path.unlink()
        print(f"removed stale {path.relative_to(SCRIPT_DIR)}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(SCRIPT_DIR)}")
    return 0


def check_outputs(outputs: dict[Path, str]) -> int:
    failures = 0
    stale_outputs = find_stale_generated_outputs(outputs)
    for path in stale_outputs:
        print(
            "unexpected generated file: "
            f"{path.relative_to(SCRIPT_DIR)} (remove it or regenerate with --write)",
            file=sys.stderr,
        )
        failures += 1
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
        description="Generate Level-3 concrete templates from Family DSL and snippet sources."
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
