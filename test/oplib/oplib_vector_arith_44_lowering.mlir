// RUN: python3 %S/../samples/VectorArithmetic44/vector_arith_44_cases.py --mode static > %t.static.pto
// RUN: python3 %S/../samples/VectorArithmetic44/vector_arith_44_cases.py --mode dynamic > %t.dynamic.pto
// RUN: cat %t.dynamic.pto | FileCheck %s --check-prefix=DYN-SRC
// RUN: { ptoas %t.static.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --op-fusion-debug --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=STATIC
// RUN: { ptoas %t.dynamic.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --op-fusion-debug --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=DYNAMIC

// DYN-SRC: func.func @vector_arith_44_dynamic(
// DYN-SRC: %part0 = pto.alloc_tile valid_row = %part0_row valid_col = %part0_col : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
// DYN-SRC: %part1 = pto.alloc_tile valid_row = %part1_row valid_col = %part1_col : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
// DYN-SRC: %tpartadd_out = pto.alloc_tile valid_row = %partd_row valid_col = %partd_col : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
// DYN-SRC: pto.tdivs ins(%a, %scale : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32) outs(%tdivs_ts_out : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
// DYN-SRC: pto.tdivs ins(%scale, %a : f32, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tdivs_st_out : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tadd dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tadd__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tsub dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tsub__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tmul dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tmul__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tdiv dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tdiv__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tmax dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tmax__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tmin dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tmin__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=trem dtype=f32 variant_id=tile_tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_partial_binary_template op=tpartadd dtype=f32 variant_id=__seed__seed_l3_float_partial_binary_core__tpartadd__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_partial_binary_template op=tpartmax dtype=f32 variant_id=__seed__seed_l3_float_partial_binary_core__tpartmax__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_partial_binary_template op=tpartmin dtype=f32 variant_id=__seed__seed_l3_float_partial_binary_core__tpartmin__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_special_template op=tprelu dtype=f32 variant_id=tile_tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tadds dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tadds__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tsubs dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tsubs__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tmuls dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tmuls__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tdivs dtype=f32 variant_id=tile_scalar cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tdivs dtype=f32 variant_id=scalar_tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tmaxs dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tmaxs__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tmins dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tmins__f32
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=trems dtype=f32 variant_id=tile_scalar cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_template op=taddc dtype=f32 variant_id=tile_tile_tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_template op=tsubc dtype=f32 variant_id=tile_tile_tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_scalar_template op=taddsc dtype=f32 variant_id=tile_scalar_tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_scalar_template op=tsubsc dtype=f32 variant_id=tile_scalar_tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=tabs dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=tneg dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=texp dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=tlog dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=tsqrt dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=trsqrt dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=trecip dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=trelu dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// STATIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_scalar_template op=tlrelu dtype=f32 variant_id=tile_scalar cost=10 priority=0 source=variant
// STATIC-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// STATIC: func.func private @__pto_oplib_inst_l3_float_binary_special_template_tprelu_tile_tile(
// STATIC: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_trem_tile_tile(
// STATIC: func.func private @__pto_oplib_inst_l3_float_unary_scalar_template_tlrelu_tile_scalar(
// STATIC: func.func private @__pto_oplib_inst_l3_float_unary_math_template_texp_tile(
// STATIC: func.func private @__pto_oplib_inst_l3_float_unary_math_template_tsqrt_tile(

// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tadd dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tadd__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tsub dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tsub__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tmul dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tmul__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tdiv dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tdiv__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tmax dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tmax__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=tmin dtype=f32 variant_id=__seed__seed_l3_float_binary_elementwise_core__tmin__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_elementwise_template op=trem dtype=f32 variant_id=tile_tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_partial_binary_template op=tpartadd dtype=f32 variant_id=__seed__seed_l3_float_partial_binary_core__tpartadd__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_partial_binary_template op=tpartmax dtype=f32 variant_id=__seed__seed_l3_float_partial_binary_core__tpartmax__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_partial_binary_template op=tpartmin dtype=f32 variant_id=__seed__seed_l3_float_partial_binary_core__tpartmin__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_binary_special_template op=tprelu dtype=f32 variant_id=tile_tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tadds dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tadds__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tsubs dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tsubs__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tmuls dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tmuls__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tdivs dtype=f32 variant_id=tile_scalar cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tdivs dtype=f32 variant_id=scalar_tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tmaxs dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tmaxs__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=tmins dtype=f32 variant_id=__seed__seed_l3_float_tile_scalar_core__tmins__f32
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_tile_scalar_template op=trems dtype=f32 variant_id=tile_scalar cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_template op=taddc dtype=f32 variant_id=tile_tile_tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_template op=tsubc dtype=f32 variant_id=tile_tile_tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_scalar_template op=taddsc dtype=f32 variant_id=tile_scalar_tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_ternary_tile_scalar_template op=tsubsc dtype=f32 variant_id=tile_scalar_tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=tabs dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=tneg dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=texp dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=tlog dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=tsqrt dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_math_template op=trsqrt dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=trecip dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_template op=trelu dtype=f32 variant_id=tile cost=10 priority=0 source=variant
// DYNAMIC-DAG: [op-fusion] selected variant: kind=l3_float_unary_scalar_template op=tlrelu dtype=f32 variant_id=tile_scalar cost=10 priority=0 source=variant
// DYNAMIC-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// DYNAMIC: func.func private @__pto_oplib_inst_l3_float_binary_special_template_tprelu_tile_tile(
// DYNAMIC: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_trem_tile_tile(
// DYNAMIC: func.func private @__pto_oplib_inst_l3_float_unary_scalar_template_tlrelu_tile_scalar(
// DYNAMIC: func.func private @__pto_oplib_inst_l3_float_unary_math_template_texp_tile(
// DYNAMIC: func.func private @__pto_oplib_inst_l3_float_unary_math_template_tsqrt_tile(
// DYNAMIC: func.func private @__pto_oplib_inst_l3_float_tile_scalar_template_tdivs_scalar_tile(
// DYNAMIC: func.func private @__pto_oplib_inst_l3_float_partial_binary_template_tpartadd___seed__seed_l3_float_partial_binary_core__tpartadd__f32(
