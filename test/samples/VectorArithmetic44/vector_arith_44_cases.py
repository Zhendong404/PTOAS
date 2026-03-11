#!/usr/bin/env python3

import argparse


STATIC_TILE = (
    "!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, "
    "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
)

DYNAMIC_TILE = (
    "!pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, "
    "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
)


def indent(lines, depth=4):
    prefix = " " * depth
    return "\n".join(prefix + line if line else "" for line in lines)


def render_alloc(name, tile_ty, valid_row=None, valid_col=None):
    if valid_row is not None and valid_col is not None:
        return (
            f"%{name} = pto.alloc_tile valid_row = %{valid_row} valid_col = %{valid_col} "
            f": {tile_ty}"
        )
    return f"%{name} = pto.alloc_tile : {tile_ty}"


def render_ops(tile_ty):
    return [
        f"pto.tadd ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tadd_out : {tile_ty})",
        f"pto.tsub ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tsub_out : {tile_ty})",
        f"pto.tmul ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tmul_out : {tile_ty})",
        f"pto.tdiv ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tdiv_out : {tile_ty})",
        f"pto.tmax ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tmax_out : {tile_ty})",
        f"pto.tmin ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tmin_out : {tile_ty})",
        f"pto.trem ins(%a, %b : {tile_ty}, {tile_ty}) outs(%trem_out : {tile_ty})",
        f"pto.tprelu ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tprelu_out : {tile_ty})",
        f"pto.tadds ins(%a, %scale : {tile_ty}, f32) outs(%tadds_out : {tile_ty})",
        f"pto.tsubs ins(%a, %scale : {tile_ty}, f32) outs(%tsubs_out : {tile_ty})",
        f"pto.tmuls ins(%a, %scale : {tile_ty}, f32) outs(%tmuls_out : {tile_ty})",
        f"pto.tdivs ins(%a, %scale : {tile_ty}, f32) outs(%tdivs_ts_out : {tile_ty})",
        f"pto.tdivs ins(%scale, %a : f32, {tile_ty}) outs(%tdivs_st_out : {tile_ty})",
        f"pto.tmaxs ins(%a, %scale : {tile_ty}, f32) outs(%tmaxs_out : {tile_ty})",
        f"pto.tmins ins(%a, %scale : {tile_ty}, f32) outs(%tmins_out : {tile_ty})",
        f"pto.trems ins(%a, %scale : {tile_ty}, f32) outs(%trems_out : {tile_ty})",
        f"pto.taddc ins(%a, %b, %c : {tile_ty}, {tile_ty}, {tile_ty}) outs(%taddc_out : {tile_ty})",
        f"pto.tsubc ins(%a, %b, %c : {tile_ty}, {tile_ty}, {tile_ty}) outs(%tsubc_out : {tile_ty})",
        f"pto.taddsc ins(%a, %bias, %b : {tile_ty}, f32, {tile_ty}) outs(%taddsc_out : {tile_ty})",
        f"pto.tsubsc ins(%a, %bias, %b : {tile_ty}, f32, {tile_ty}) outs(%tsubsc_out : {tile_ty})",
        f"pto.tabs ins(%a : {tile_ty}) outs(%tabs_out : {tile_ty})",
        f"pto.tneg ins(%a : {tile_ty}) outs(%tneg_out : {tile_ty})",
        f"pto.texp ins(%a : {tile_ty}) outs(%texp_out : {tile_ty})",
        f"pto.tlog ins(%a : {tile_ty}) outs(%tlog_out : {tile_ty})",
        f"pto.tsqrt ins(%a : {tile_ty}) outs(%tsqrt_out : {tile_ty})",
        f"pto.trsqrt ins(%a : {tile_ty}) outs(%trsqrt_out : {tile_ty})",
        f"pto.trecip ins(%a : {tile_ty}) outs(%trecip_out : {tile_ty})",
        f"pto.trelu ins(%a : {tile_ty}) outs(%trelu_out : {tile_ty})",
        f"pto.tlrelu ins(%a, %slope : {tile_ty}, f32) outs(%tlrelu_out : {tile_ty})",
    ]


def render_partial_ops(tile_ty):
    return [
        f"pto.tpartadd ins(%part0, %part1 : {tile_ty}, {tile_ty}) outs(%tpartadd_out : {tile_ty})",
        f"pto.tpartmax ins(%part0, %part1 : {tile_ty}, {tile_ty}) outs(%tpartmax_out : {tile_ty})",
        f"pto.tpartmin ins(%part0, %part1 : {tile_ty}, {tile_ty}) outs(%tpartmin_out : {tile_ty})",
    ]


def render_static_case():
    allocs = [
        render_alloc("a", STATIC_TILE),
        render_alloc("b", STATIC_TILE),
        render_alloc("c", STATIC_TILE),
    ]
    allocs.extend(
        render_alloc(name, STATIC_TILE)
        for name in [
            "tadd_out",
            "tsub_out",
            "tmul_out",
            "tdiv_out",
            "tmax_out",
            "tmin_out",
            "trem_out",
            "tprelu_out",
            "tadds_out",
            "tsubs_out",
            "tmuls_out",
            "tdivs_ts_out",
            "tdivs_st_out",
            "tmaxs_out",
            "tmins_out",
            "trems_out",
            "taddc_out",
            "tsubc_out",
            "taddsc_out",
            "tsubsc_out",
            "tabs_out",
            "tneg_out",
            "texp_out",
            "tlog_out",
            "tsqrt_out",
            "trsqrt_out",
            "trecip_out",
            "trelu_out",
            "tlrelu_out",
            "part0",
            "part1",
            "tpartadd_out",
            "tpartmax_out",
            "tpartmin_out",
        ]
    )

    lines = [
        "module {",
        "  func.func @vector_arith_44_static() {",
        "    %scale = arith.constant 2.000000e+00 : f32",
        "    %bias = arith.constant -1.250000e-01 : f32",
        "    %slope = arith.constant 3.125000e-01 : f32",
        indent(allocs),
        indent(render_ops(STATIC_TILE)),
        indent(render_partial_ops(STATIC_TILE)),
        "    return",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_dynamic_case():
    allocs = [
        render_alloc("a", DYNAMIC_TILE, "vrow", "vcol"),
        render_alloc("b", DYNAMIC_TILE, "vrow", "vcol"),
        render_alloc("c", DYNAMIC_TILE, "vrow", "vcol"),
    ]
    allocs.extend(
        render_alloc(name, DYNAMIC_TILE, "vrow", "vcol")
        for name in [
            "tadd_out",
            "tsub_out",
            "tmul_out",
            "tdiv_out",
            "tmax_out",
            "tmin_out",
            "trem_out",
            "tprelu_out",
            "tadds_out",
            "tsubs_out",
            "tmuls_out",
            "tdivs_ts_out",
            "tdivs_st_out",
            "tmaxs_out",
            "tmins_out",
            "trems_out",
            "taddc_out",
            "tsubc_out",
            "taddsc_out",
            "tsubsc_out",
            "tabs_out",
            "tneg_out",
            "texp_out",
            "tlog_out",
            "tsqrt_out",
            "trsqrt_out",
            "trecip_out",
            "trelu_out",
            "tlrelu_out",
        ]
    )
    allocs.extend(
        [
            render_alloc("part0", DYNAMIC_TILE, "part0_row", "part0_col"),
            render_alloc("part1", DYNAMIC_TILE, "part1_row", "part1_col"),
            render_alloc("tpartadd_out", DYNAMIC_TILE, "partd_row", "partd_col"),
            render_alloc("tpartmax_out", DYNAMIC_TILE, "partd_row", "partd_col"),
            render_alloc("tpartmin_out", DYNAMIC_TILE, "partd_row", "partd_col"),
        ]
    )

    lines = [
        "module {",
        "  func.func @vector_arith_44_dynamic(",
        "      %valid_row: i32, %valid_col: i32,",
        "      %part0_valid_row: i32, %part0_valid_col: i32,",
        "      %part1_valid_row: i32, %part1_valid_col: i32,",
        "      %partd_valid_row: i32, %partd_valid_col: i32) {",
        "    %scale = arith.constant 2.000000e+00 : f32",
        "    %bias = arith.constant -1.250000e-01 : f32",
        "    %slope = arith.constant 3.125000e-01 : f32",
        "    %vrow = arith.index_cast %valid_row : i32 to index",
        "    %vcol = arith.index_cast %valid_col : i32 to index",
        "    %part0_row = arith.index_cast %part0_valid_row : i32 to index",
        "    %part0_col = arith.index_cast %part0_valid_col : i32 to index",
        "    %part1_row = arith.index_cast %part1_valid_row : i32 to index",
        "    %part1_col = arith.index_cast %part1_valid_col : i32 to index",
        "    %partd_row = arith.index_cast %partd_valid_row : i32 to index",
        "    %partd_col = arith.index_cast %partd_valid_col : i32 to index",
        indent(allocs),
        indent(render_ops(DYNAMIC_TILE)),
        indent(render_partial_ops(DYNAMIC_TILE)),
        "    return",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def render_emitc_case():
    tile_ty = DYNAMIC_TILE
    lines = [
        "module {",
        "  func.func @vector_arith_44_emitc(",
        "      %valid_row: i32, %valid_col: i32,",
        "      %part0_valid_row: i32, %part0_valid_col: i32,",
        "      %part1_valid_row: i32, %part1_valid_col: i32,",
        "      %partd_valid_row: i32, %partd_valid_col: i32) {",
        "    %scale = arith.constant 2.000000e+00 : f32",
        "    %slope = arith.constant 3.125000e-01 : f32",
        "    %vrow = arith.index_cast %valid_row : i32 to index",
        "    %vcol = arith.index_cast %valid_col : i32 to index",
        "    %part0_row = arith.index_cast %part0_valid_row : i32 to index",
        "    %part0_col = arith.index_cast %part0_valid_col : i32 to index",
        "    %part1_row = arith.index_cast %part1_valid_row : i32 to index",
        "    %part1_col = arith.index_cast %part1_valid_col : i32 to index",
        "    %partd_row = arith.index_cast %partd_valid_row : i32 to index",
        "    %partd_col = arith.index_cast %partd_valid_col : i32 to index",
        indent(
            [
                render_alloc("a", tile_ty, "vrow", "vcol"),
                render_alloc("b", tile_ty, "vrow", "vcol"),
                render_alloc("trem_out", tile_ty, "vrow", "vcol"),
                render_alloc("tprelu_out", tile_ty, "vrow", "vcol"),
                render_alloc("tlrelu_out", tile_ty, "vrow", "vcol"),
                render_alloc("texp_out", tile_ty, "vrow", "vcol"),
                render_alloc("tsqrt_out", tile_ty, "vrow", "vcol"),
                render_alloc("part0", tile_ty, "part0_row", "part0_col"),
                render_alloc("part1", tile_ty, "part1_row", "part1_col"),
                render_alloc("tpartadd_out", tile_ty, "partd_row", "partd_col"),
            ]
        ),
        indent(
            [
                f"pto.trem ins(%a, %b : {tile_ty}, {tile_ty}) outs(%trem_out : {tile_ty})",
                f"pto.tprelu ins(%a, %b : {tile_ty}, {tile_ty}) outs(%tprelu_out : {tile_ty})",
                f"pto.tlrelu ins(%a, %slope : {tile_ty}, f32) outs(%tlrelu_out : {tile_ty})",
                f"pto.texp ins(%a : {tile_ty}) outs(%texp_out : {tile_ty})",
                f"pto.tsqrt ins(%a : {tile_ty}) outs(%tsqrt_out : {tile_ty})",
                f"pto.tpartadd ins(%part0, %part1 : {tile_ty}, {tile_ty}) outs(%tpartadd_out : {tile_ty})",
            ]
        ),
        "    return",
        "  }",
        "}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic", "emitc"],
        required=True,
        help="Select which 4.4 vector arithmetic PTO case to emit.",
    )
    args = parser.parse_args()

    if args.mode == "static":
        print(render_static_case(), end="")
        return
    if args.mode == "dynamic":
        print(render_dynamic_case(), end="")
        return
    print(render_emitc_case(), end="")


if __name__ == "__main__":
    main()
