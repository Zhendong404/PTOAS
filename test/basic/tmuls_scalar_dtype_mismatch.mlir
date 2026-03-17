// RUN: bash -c 'set +e; out=$(ptoas %s -o /dev/null 2>&1); status=$?; test $status -ne 0; echo "$out" | grep -F "expects scalar type to match tile element type"'

module {
  func.func @tmuls_bad_scalar_type(
      %src: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f32,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {
    pto.tmuls ins(%src, %scalar : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, f32)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
