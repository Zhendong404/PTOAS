# A5VM Builtin Coverage Checklist

Updated: 2026-03-20

Purpose:

- Compare current `a5vm` op coverage against
  `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`.
- Separate:
  - already-covered CCE builtin families
  - families intentionally abstracted by existing `a5vm` ops
  - real gaps worth defining as new `a5vm` ops
- Avoid duplicate op design while broadening backend support.

Source of comparison:

- CCE intrinsic wrapper header:
  `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`
- Current dialect surface:
  [`include/PTO/IR/A5VMOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/A5VMOps.td)

Current explicit `a5vm` op count:

- `64`

## A. Already Covered By Existing A5VM Ops

These CCE builtin families already have a direct or near-direct `a5vm` carrier.

### A1. Sync / Pipe / Buffer

- `set_flag` / `wait_flag`
  Covered by:
  `a5vm.set_flag`, `a5vm.wait_flag`
- `pipe_barrier`
  Covered by:
  `a5vm.pipe_barrier`
- `get_buf` / `rls_buf`
  Covered by:
  `a5vm.get_buf`, `a5vm.rls_buf`

### A2. Copy-Family DMA

- `copy_gm_to_ub*`
  Covered by:
  `a5vm.copy_gm_to_ubuf` plus `a5vm.set_loop*_outtoub`
- `copy_ub*to_gm`
  Covered by:
  `a5vm.copy_ubuf_to_gm` plus `a5vm.set_loop*_ubtoout`
- UB-to-UB copy helper path
  Covered by:
  `a5vm.copy_ubuf_to_ubuf`

### A3. Load / Store Families Already Abstracted

- `vlds`
  Covered by:
  `a5vm.vlds`
- `vldas`
  Covered by:
  `a5vm.vldas`
- `vldus`
  Covered by:
  `a5vm.vldus`
- `plds`
  Covered by:
  `a5vm.plds`
- `vsts`
  Covered by:
  `a5vm.vsts`
- `psts`
  Covered by:
  `a5vm.psts`
- `vscatter`
  Covered by:
  `a5vm.vscatter`
- predicated tail store
  Covered by:
  `a5vm.vsts_pred`

### A4. Predicate / Mask Materialization

- `pset_b8`
  Covered by:
  `a5vm.pset_b8`
- `pset_b16`
  Covered by:
  `a5vm.pset_b16`
- `ppack`
  Covered by:
  `a5vm.ppack`
- `punpack`
  Covered by:
  `a5vm.punpack`
- `pdintlv_b8`
  Covered by:
  `a5vm.pdintlv_b8`
- `pintlv_b16`
  Covered by:
  `a5vm.pintlv_b16`

### A5. Basic Vector Math / Compare / Select

- `vbr`, `vdup`
  Covered by:
  `a5vm.vbr`, `a5vm.vdup`
- unary vec:
  `vabs`, `vexp`, `vln`, `vsqrt`, `vrec`, `vrelu`, `vnot`,
  `vcadd`, `vcmax`, `vcmin`
  Covered by same-name `a5vm` ops
- binary vec:
  `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`
  Covered by same-name `a5vm` ops
- vec-scalar:
  `vmuls`, `vadds`, `vmaxs`, `vmins`, `vlrelu`
  Covered by same-name `a5vm` ops
- compare / select:
  `vsel`, `vcmp`, `vcmps`
  Covered by same-name `a5vm` ops

### A6. Conversion / Index / Sort / Indexed Access

- `vtrc`
  Covered by:
  `a5vm.vtrc`
- `vcvt`
  Covered by:
  `a5vm.vcvt`
- `vci`
  Covered by:
  `a5vm.vci`
- `vbitsort`
  Covered by:
  `a5vm.vbitsort`
- `vmrgsort4`
  Covered by:
  `a5vm.vmrgsort4`
- `vgather2`
  Covered by:
  `a5vm.vgather2`
- `vgatherb`
  Covered by:
  `a5vm.vgatherb`

## B. Families That Are Currently Covered By A5VM Abstraction Rather Than 1:1 Builtins

These are not gaps by themselves.

### B1. `vld` / `vst` Wrapper Families

- CCE has many wrapper spellings:
  `vld`, `vldx2`, `vldoncex1`, `vldoncex2`, `vlds`, `vldsx2`, `vldi`, `vsldb`,
  `vst`, `vstx2`, `vsts`, `vsti`
- Current `a5vm` stance:
  keep a smaller normalized carrier set when the missing variant can still be
  reconstructed from existing MLIR information.

Today this means:

- scalar/vector UB load path is normalized to:
  `a5vm.vlds`
- align-sensitive UB load path is normalized to:
  `a5vm.vldas` + `a5vm.vldus`
- normal vector store path is normalized to:
  `a5vm.vsts`

Status:

- `vld` / `vst` x1 paths used by current PTO seam are already adequately covered.
- x2, post-update, stride-specific, and align-store-specific variants are not.

### B2. `vmov`

- CCE often materializes masked writeback through `vmov_*_m`.
- Current `a5vm` stance:
  do not define `a5vm.vmov` unless needed.
  When seam IR already carries old-dst and mask, merging semantics can be
  represented as `a5vm.vsel(new_value, old_value, mask)`.

Status:

- No direct `a5vm.vmov` needed yet for current supported PTO seam.

### B3. Part Of `vcvt` Space

- CCE exposes many builtin spellings:
  `vfcvt`, `vsfcvt`, `vcvtfi`, `vcvtif`, `vcvtii`, `vscvt`, etc.
- Current `a5vm` stance:
  keep one generic `a5vm.vcvt` carrier with attrs such as
  `round_mode`, `sat`, `part`.

Status:

- For currently used PTO seam branches, this abstraction is sufficient.
- Wider integer/float/mask conversion space is still only partially modeled.

### B4. Alias / Wrapper Surface That Does Not Need Independent A5VM Ops Yet

- `pge`
  Current stance:
  prefer width-specific forms (`pge_b8/b16/b32`) if needed.
- `pset`
  Current stance:
  prefer width-specific forms (`pset_b8/b16/b32`) if needed.
- `pset_2xvl_b64`
  Current stance:
  treat as a specialized mask-materialization wrapper, not a first-wave
  primitive op until PTO seam requires its exact semantics.
- `pdintlv`, `pdintlvv2`
  Current stance:
  current mask coverage is represented through `a5vm.pdintlv_b8`.
- `pintlv`, `pintlvv2`
  Current stance:
  current mask coverage is represented through `a5vm.pintlv_b16`.
- `vload`, `vstore`
  Current stance:
  these are library convenience wrappers over lower-level load/store families;
  they should not get separate `a5vm` ops unless PTO seam depends on them directly.
- `vdiv_high_precision`
  Current stance:
  this is a library algorithm built from lower-level ops, not a primitive op.
- `vlda`, `vldu`
  Current stance:
  the current dialect already carries the split form
  `a5vm.vldas + a5vm.vldus`, which is a better SSA representation.
- `vdupi`, `vdups`
  Current stance:
  do not add until a PTO seam requires the post/update or repeated-lane semantics
  directly.
- `vneg`
  Current stance:
  current PTO lowering intentionally models this through existing multiply-scalar
  machinery rather than a dedicated builtin carrier.
- `plt_2xvl_b64`, `pltm_2xvl_b64`
  Current stance:
  treat as specialized helper wrappers rather than first-wave primitive ops.

### B5. Families That Look Like Library Algorithms, Not First-Wave Primitive Ops

- `vavg`
- `vaxpy`
- `vag_b8`
- `vag_b16`
- `vag_b32`
- `vcgadd`
- `vcgaddv2`
- `vcp`
- `vcpadd`
- `vcpaddv2`
- `vintegral`
- `vintegrals1`
- `vintegrals2`
- `vintegrals3`
- `vintegralv2`
- `vmod`
- `vmulscvt`
- `vexpdif`
- `vext`
- `vextfa`
- `vpack`
- `vpackv2`
- `vunpack`
- `vslide`
- `pslide`
- `vsqz`
- `vusqz`
- `chist0`
- `chist1`
- `chist2`
- `chist3`
- `chistv2`
- `dhist0`
- `dhist1`
- `dhist2`
- `dhist3`
- `dhistv2`

Current stance:

- keep these out of the first-wave `a5vm` primitive set unless PTO seam
  requires them directly.
- many of them look like composite library utilities, domain-specific helpers,
  or hardware-specialized kernels rather than the minimal primitive layer we
  want first.

## C. Confirmed Missing Or Under-Modeled Families

These are the main candidates for new `a5vm` ops because current `a5vm`
surface cannot faithfully express them without losing semantics.

### C1. Predicate / Mask Gaps

- `pge_b8`
- `pge_b16`
- `pge_b32`
- `pset_b32`
- `pld`
- `pldi`
- `pst`
- `psti`
- `pstu`
- `pnot`
- `psel`

Why these are real gaps:

- `pge_*` is not equivalent to `pset_*`.
- `pnot` / `psel` cannot be reconstructed from current mask ops.
- `pld/pst` family has distinct load/store semantics beyond current `plds/psts`.

### C2. Extended Load / Store Gaps

- `vldx2` / `vldsx2`
- `vldoncex2`
- `vsldb`
- `vsld`
- `vsst`
- `vsstb`
- `vstu`
- `vstus`
- `vsta`
- `vstas`
- `vstur`
- `vstar`

Why these are real gaps:

- x2 forms need multi-result or paired-vector semantics.
- stride forms (`vsld/vsst`) are not the same as current `vlds/vsts`.
- align-store/update families (`vstu/vstas/...`) need `!a5vm.align`-driven
  semantics that current `a5vm` store ops do not represent.

### C3. Carry / Widen / Shift / Count Arithmetic Gaps

- `vaddc`
- `vaddcs`
- `vsubc`
- `vsubcs`
- `vmull`
- `vmula`
- `vshl`
- `vshr`
- `vshls`
- `vshrs`
- `vbcnt`
- `vcls`

Why these are real gaps:

- carry ops produce more than a plain vector result
- widening multiply / multiply-accumulate is not representable as plain `vmul`
- shift/count/classification families are not expressible with current core vec ops

### C4. SPR / Misc Hardware State Gaps

- `sprclr`
- `sprsti`
- `sprsts`
- `mem_bar`
- `set_ub_addr_upper_bound`
- `prpnset`
- `pmov`
- `tensor_range_bind`

Why these are real gaps:

- these target special registers / runtime binding state, not ordinary vector SSA
- they are closer to hardware/runtime state mutation than plain vector SSA ops

### C5. Extended Indexed / Select / Pairing Gaps

- `vgather2_bc`
- `vselr`
- `vselrv2`
- `vintlv`
- `vintlvv2`
- `vdintlv`
- `vdintlvv2`

Why these are real gaps:

- broadcast gather and lane-reordering select forms are not representable with
  current `vgather2/vsel`.
- pair interleave/deinterleave vector families are richer than current
  predicate-only interleave coverage.

### C6. Wider Conversion Family Gaps

- `vfcvt`
- `vsfcvt`
- `vscvt`
- `vcvtfi`
- `vcvtif`
- `vcvtii`
- `vcvt_bf162e6m2`
- `vcvt_bf162e8m0`
- `vcvt_e6m22bf16`
- `vcvt_e8m02bf16`
- `vcvt_f162s4`
- `vcvt_rcpe6m22bf16`
- `vcvt_s162s4`
- `vcvt_s42s16`
- `wcvt48`

Why these are real gaps:

- some can eventually map onto generic `a5vm.vcvt`, but today the dialect does
  not explicitly preserve enough information for the full conversion family.
- low-bit and exotic-format conversion branches likely need additional attrs or
  dedicated carriers.

### C7. Word / Wide Register Families

- `wadd`
- `wmov`
- `wmul`
- `wmula`
- `wmulas`
- `wmuls`
- `wpack`
- `wpacks`
- `wdups`
- `wfifr2`
- `wfifr2a`
- `wfifr2s`

Why these are real gaps:

- these are not naturally covered by current `!a5vm.vec<...>` surface and may
  imply a distinct word/wide-register model.

### C8. Low-Priority Deprecated / Variant Wrapper Gaps

- `vldi`
- `vldui`
- `vstai`
- `vstui`

Why these are real gaps:

- they do exist in the header, but they are deprecated/variant wrapper spellings.
- they should not be first-wave additions unless a real PTO seam still depends
  on them.

## D. Suggested Next Definition Batch

These are the highest-value additions if we want to broaden `a5vm` without
exploding the dialect surface too early.

### D1. First Batch

- `a5vm.pge_b8`
- `a5vm.pge_b16`
- `a5vm.pge_b32`
- `a5vm.pset_b32`
- `a5vm.pnot`
- `a5vm.psel`
- `a5vm.vaddc`
- `a5vm.vaddcs`
- `a5vm.vsubc`
- `a5vm.vsubcs`
- `a5vm.vshl`
- `a5vm.vshr`
- `a5vm.vshls`
- `a5vm.vshrs`
- `a5vm.vbcnt`
- `a5vm.vcls`
- `a5vm.vsld`
- `a5vm.vsst`

Reason:

- all of these are semantically meaningful, appear in CCE wrappers, and are
  not cleanly replaceable by today’s `a5vm` ops.

### D2. Second Batch

- `a5vm.vmull`
- `a5vm.vmula`
- `a5vm.vldx2`
- `a5vm.vstx2`
- `a5vm.vstu`
- `a5vm.vstus`
- `a5vm.vsta`
- `a5vm.vstas`
- `a5vm.vstur`
- `a5vm.vstar`

Reason:

- useful, but they need more careful result typing / align-state modeling.

### D3. Deferred Unless PTO Seam Demands Them

- `sprclr`
- `sprsti`
- `sprsts`
- `mem_bar`
- `set_ub_addr_upper_bound`
- `prpnset`
- `pmov`
- `tensor_range_bind`
- deprecated wrapper spellings such as `vldi`, `vsti`
- wide-register families such as `wadd/wmul/wpack/...`
- composite helper families such as `vavg/vaxpy/vintegral/vmod/...`

Reason:

- they are not needed by the current PTO seam and would widen the backend
  surface without immediate payoff.

## E. Header Family Sweep Status

This section exists only to ensure the checklist really covers the family names
present in the CCE wrapper header.

### E1. Covered Or Intentionally Abstracted

- `pge`
- `pset`
- `ppack`
- `punpack`
- `plds`
- `psts`
- `vabs`
- `vbr`
- `vcadd`
- `vci`
- `vcls`
- `vcmp`
- `vcmps`
- `vcvt`
- `vdiv`
- `vdup`
- `vgather2`
- `vgatherb`
- `vld`
- `vlda`
- `vldas`
- `vlds`
- `vldu`
- `vldus`
- `vmov`
- `vmul`
- `vmuls`
- `vneg`
- `vnot`
- `vscatter`
- `vsel`
- `vshl`
- `vshls`
- `vshr`
- `vshrs`
- `vsld`
- `vsldb`
- `vsst`
- `vsstb`
- `vst`
- `vsta`
- `vstar`
- `vstas`
- `vsts`
- `vstu`
- `vstur`
- `vstus`
- `vtrc`

Status note:

- every family above is now at least mentioned in this checklist as covered,
  abstracted, or gap/defer.

### E2. Explicit Gap / Defer Buckets Added By This Sweep

- `mem_bar`
- `movvp`
- `pmov`
- `prpnset`
- `pslide`
- `pset_2xvl_b64`
- `plt_2xvl_b64`
- `pltm_2xvl_b64`
- `set_ub_addr_upper_bound`
- `vag_b8`
- `vag_b16`
- `vag_b32`
- `vavg`
- `vaxpy`
- `vcgadd`
- `vcgaddv2`
- `vcp`
- `vcpadd`
- `vcpaddv2`
- `vcvt_bf162e6m2`
- `vcvt_bf162e8m0`
- `vcvt_e6m22bf16`
- `vcvt_e8m02bf16`
- `vcvt_f162s4`
- `vcvt_rcpe6m22bf16`
- `vcvt_s162s4`
- `vcvt_s42s16`
- `vdiv_high_precision`
- `vdupi`
- `vdups`
- `vexpdif`
- `vext`
- `vextfa`
- `vgather2_bc`
- `vintegral`
- `vintegrals1`
- `vintegrals2`
- `vintegrals3`
- `vintegralv2`
- `vintlv`
- `vintlvv2`
- `vdintlv`
- `vdintlvv2`
- `vload`
- `vmod`
- `vmula`
- `vmull`
- `vmulscvt`
- `vnop`
- `vpack`
- `vpackv2`
- `vselr`
- `vselrv2`
- `vsfcvt`
- `vsqz`
- `vstore`
- `vunpack`
- `vusqz`
- `wadd`
- `wcvt48`
- `wdups`
- `wfifr2`
- `wfifr2a`
- `wfifr2s`
- `wmov`
- `wmul`
- `wmula`
- `wmulas`
- `wmuls`
- `wpack`
- `wpacks`

Status note:

- after this sweep, the checklist is no longer just a narrow PTO-seam list.
- it is now a real family-coverage map for the current CCE wrapper header,
  though many items remain intentionally deferred.

## E. Rules For Future Additions

- Do not add a new `a5vm` op if existing `a5vm` ops can already preserve the
  same semantics and all lowering-relevant information.
- Do add a new `a5vm` op when collapsing to existing ops would lose:
  - carry/mask side-result information
  - stride or align-specific memory semantics
  - x2 paired-vector structure
  - special-register / hardware-state side effects
- Prefer builtin-family names that stay close to CCE spelling.
- Keep namespace in `mlir::a5vm`, not `mlir::pto::a5vm`.
