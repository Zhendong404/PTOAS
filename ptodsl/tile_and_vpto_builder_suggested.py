# minimum Pythonic mapping of test/lit/vpto/expand_tileop_to_vpto_result.pto

import pto
s = pto.scalar

@pto.to_ir(name="TADD", kernel_kind="vector", arch="a5")
def vpto_demo():
    c0_i64    = pto.const(0, dtype=pto.int64)
    c16       = pto.const(16, dtype=pto.index)  # if no dtype passed, default to pto.index
    c4096_i64 = pto.const(4096, dtype=pto.int64)
    c0        = pto.const(0)
    c1        = pto.const(1)
    c64_i32   = pto.const(64, dtype=pto.int32)
    c64       = pto.const(64)
    with pto.vecscope():
        ptr_type  = pto.ptr(pto.float32, "UB")
        ptr_src = pto.castptr(c4096_i64, ptr_type)
        ptr_dst = pto.castptr(c0_i64, ptr_type)
        vreg_type  = vreg_type(64, pto.float32)
        with for_(c0, c16, step=c1) as tile_idx:
            mask, _ = pto.plt_b32(c64_i32)
            tile_off = s.muli(tile_idx, c64)  # can optionally overload __mul__
            va = pto.vlds(pto.addptr(ptr_src, tile_off), c0, vreg_type)
            ptr_dst_tile = pto.addptr(ptr_dst, tile_off)
            vb = pto.vlds(ptr_dst_tile, c0, vreg_type)
            vc = pto.vadd(va, vb, mask, vreg_type)
            pto.vsts(vc, ptr_dst_tile, c0, mask)
    # by default return None, matches IR `return`


if __name__ == "__main__":
    print(vpto_demo)
