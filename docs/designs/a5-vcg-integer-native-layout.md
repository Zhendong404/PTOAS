# Native A5 16-bit integer group-sum layout

## Producer contract

A5 integer `vcgadd` consumes 16-bit elements and produces eight 32-bit sums.
The logical VMI result still has 16-bit elements. Viewed as halfwords, the
register contains:

| Physical halfword lane | 0 | 1 | 2 | 3 | … | 14 | 15 |
|---|---|---|---|---|---|---|---|
| Contents | low(S0) | high(S0) | low(S1) | high(S1) | … | low(S7) | high(S7) |
| Logical group | 0 | padding | 1 | padding | … | 7 | padding |

This is `gs(8, 2)`: logical group `g` is in physical packet `g / 8`, at
halfword lane `2 * (g % 8)`. The layout follows the producer instruction,
independently of stores, casts, broadcasts, or the number of users. No
producer-side `vpack` is necessary to represent the logical result.

`VMIGroupReduceKind` distinguishes integer addition in the shared layout
queries. The reduction table specifies the 16-bit integer result separately
from the default result. Assignment, propagation, validation and lowering use
the same facts. Floating-point addition and integer max/min retain their
existing layouts. Full-row and compact scalar `vcadd` paths retain their
existing row-local result handling.

An explicitly assigned native integer sum must use `gs(8, 2)`. A caller that
needs consecutive slots requests `ensure_layout` to `gs(8)` on the result.

## Partial sums and consumers

Two- and four-block paths combine the native partial sums in a 32-bit view
with a `b32` mask. Each result packet activates at most eight lanes; a final
partial packet activates only its remaining groups. The combined register is
then viewed as 16-bit elements with `gs(8, 2)`. This preserves the sum modulo
2^16 without packing every intermediate result.

The high halfwords are padding from the logical value's perspective. They
are not additional groups and must not be assumed to be zero. In particular,
extending the result to 32 bits must extend each **low 16-bit value**, rather
than forward the full hardware sum. Examples:

| Logical reduction and extension | Result |
|---|---|
| `ui16(65535 + 1)` → `ui32` | `0` |
| `si16(32767 + 1)` → `si32` | `-32768` |
| `si16(-32768 - 1)` → `si32` | `32767` |

The existing group-slot cast lowering implements extension with `vcvt EVEN`.
Narrowing to `ui8` records `gs(8, 4)` and covers the 16 halfword positions
with its conversion mask, for both saturating and nonsaturating casts. Stores
use their normal layout conversions where needed.
Integer grouped broadcasts use the existing selector's source stride for
both the index ramp and its base slot. The added broadcast table rows are
limited to 16-bit integers, preserving floating-point layout choices.
An `ls(2)` view of a single `gs(8, 2)` packet forwards the same register; this
also lets a subsequent half-block reduction read the low halfwords directly.

The current `si16/ui16` VL64/VL128 regression cases illustrate the distinction:

| Consumer chain | Explicit `vpack` count | Handling |
|---|---|---|
| 16-to-32 extension and store | 0 | `vcvt EVEN` reads the low halfwords |
| Direct 16-bit store or unit-stride group store | 1 | Convert to consecutive slots, then `NORM_B16` store |
| Broadcast and store | 0 | Select the strided source slots; VL64 uses a `PK_B32` output store |
| Shared store, extension and broadcast | 1 | Pack only for the direct store; extension and broadcast use the native value |

These counts describe the complete cases in
[`vmi_vcg_integer_native_layout.pto`](../../test/lit/vmi_new/vmi_vcg_integer_native_layout.pto),
not all possible consumers. `PK_B32` also performs packing as part of a store,
even though it emits no separate `vpack` instruction. A packing-store fold
requires its own address, mask and memory-layout proof.

The former `vpto-optimize-vcvt` cast-to-store peephole is removed. There is no
single-user or particular-store requirement on the native reduction layout.
Consumer-side packing remains legal and can be necessary. The physical
reduction-tree combiner preserves the native 16-bit integer `vcgadd`
boundary: distributing a 16-bit arithmetic operation before an instruction
that produces 32-bit sums is not a valid physical rewrite. In a half-block
path it would also put a `b32` reduction predicate on a `b16` arithmetic op.

## Regression coverage

- `vmi_native_group_sum_producer.pto` checks explicitly assigned native
  results, multiple packets, and floating-point/max/min/scalar controls.
- `vmi_vcg_integer_native_layout.pto` checks automatic layout assignment for
  stores, broadcasts, casts and shared users, plus the 32-bit partial-sum
  predicates and pack-free widening chains through the public CLI.
- `vmi_native_group_sum_trunc.pto` checks strided 16-to-8-bit narrowing
  and the predicate that includes all eight logical values, plus the matching
  mask-granularity cast (both layouts address the same physical `b32` lanes).
- `vmi_to_vpto_ensure_group_slot_layout.pto` checks the matching-stride
  packet/dense identity in both directions.
- `vmi_native_i16_group_sum.py` compares signed, unsigned and signless
  results against independent integer arithmetic, including empty, tail,
  full, holey and middle-empty masks, overflow and underflow. It checks
  stores, broadcasts, extension, truncation, pointwise addition and a
  subsequent reduction, with multiple users of each native sum. Entire
  output buffers are compared, including untouched canaries.

The runtime test has a wrapper at
`test/vpto/cases/vmi_new/native-i16-group-sum.py`. With a configured build,
CANN environment, `PTOAS_BIN`, `PYTHONPATH`, and `WORK_SPACE`, the queued NPU
invocation is:

```sh
task-submit --device auto --run 'ASCEND_RT_VISIBLE_DEVICES="$TASK_DEVICE" DEVICE=NPU CASE_NAME=vmi_new/native-i16-group-sum.py bash test/vpto/scripts/run_host_vpto_validation.sh'
```

This change establishes the native layout contract. Fewer producer packs do
not establish a speedup for every complete consumer chain: consumers may
need packing or additional index arithmetic. Measurements of the removed
cast-store peephole do not describe this implementation.
