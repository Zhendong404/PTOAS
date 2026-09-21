# 5. Reduce

> **Category:** B (VLane-aligned), C (unaligned sub-VLane).
> **Mask:** `Pg req` (governing mask is a required operand).
>
> Reduction ops produce one logical scalar per group, governed by a mask.
> The assigned layout determines each scalar's physical slot position.
> `{group=C}` controls the number of sub-groups. Inactive lane behavior:
> `vcadd` treats inactive as 0; `vcmax`/`vcmin` treat inactive as `-∞`/`+∞`
> (fp) or type min/max (int).


The A5 VPTO backend supports `group = 1, 2, 4, 8` when the group count
divides `L`, for the following one-carrier shapes:

| Element type | Supported logical `L` |
|---|---|
| 8-bit integers (signless, signed, unsigned) | `1, 2, 4, 8, 64, 128, 256` |
| 16-bit integers and `f16` | `1, 2, 4, 8, 64, 128` |
| 32-bit integers and `f32` | `1, 2, 4, 8, 64` |

The compiler prefers executable native layouts, retaining the established
broadcast and memory paths. Where these are unavailable, a dense fallback
intersects each group's logical lane range with the governing mask and packs
the scalar results for grouped stores or broadcasts. Dynamic masks may contain
holes, empty groups, or partial final groups; inactive lanes retain the
identities described above.

A5 has no executable 8-bit row or VCG reduction form. Eight-bit integer inputs are
extended to 16 bits before reducing, with `L = 256` unpacked into two halves
inside instruction lowering. An empty min/max group retains the original
8-bit type's identity across extension. For integer singleton groups
(`group = L`), selection between the input and its identity replaces reduction.
Floating singleton groups retain reduction semantics for NaNs and signed zero.
Integer add results are narrowed modulo the result element width; native
16-bit VCG sums expose the low halfwords of 32-bit results as `gs(8, 2)`.
Floating-point addition still requires `reassoc`.

The dense fallback is bounded by one 256-byte input carrier and at most eight
result slots. Existing executable 16/32-bit multi-carrier layouts remain
available; this does not add arbitrary `L` values or a general multi-carrier
8-bit fallback. Unsupported reduction requests are rejected with
`VMI-UNSUPPORTED` and the element type, `VL`, group count, and bytes per group,
before layout assignment and again during final conversion.

### A5 integer result widths and group slots

The logical result keeps the input element width, but the hardware's sum may
be wider. In particular, **both** 16-bit integer `vcgadd` and `vcadd` produce
32-bit sums. Their layouts reflect how many independent results each instruction produces:

| Physical path | Hardware result for 16-bit integer input | VMI result handling |
|---|---|---|
| Native `vcgadd` | Eight 32-bit sums in one register | Keep the low 16 bits at halfword lanes `0, 2, ..., 14`, yielding `gs(8, 2)` without a producer-side pack |
| Native `vcgmax` / `vcgmin` | Eight consecutive 16-bit extrema | Keep the values at their original width; no sum-narrowing pack |
| Compact `vcadd` | One 32-bit sum in lane zero per invocation | Use the widened result type, combine partial sums if needed, then select each group's low bits into the result packet |

For example, a 32-bit sum of `131091` has low/high 16-bit halves `19` and `2`.
Two such VCG sums appear as `[19, 2, 19, 2]` in a 16-bit view. The logical
16-bit results must be `[19, 19]`, not `[19, 2]`. The high halves are part of
the hardware sums; they are not additional groups or necessarily zero.

In the compact path, `getRowResultType()` selects the widened integer sum
type. After any partial sums are combined, a register bitcast exposes the
original-width low lane. `buildCompactPacket()` uses only that lane from each
group and assembles the logical slots. The bitcast itself does not pack or
clear the other physical lanes. Compact max/min likewise consume only the
extremum value, not the index returned by the row max/min instruction. For
8-bit inputs, the logical result is narrowed back to 8 bits after extension
and reduction. All integer sum narrowing follows modulo arithmetic at the
logical result width; it is not saturation. Floating-point reductions keep
their existing result types and semantics. See the physical contracts in
[Reduction Ops](../micro-isa/10-reduction-ops.md).

### Native integer sum layout

The layout capability table selects `gs(8, 2)` for native 16-bit integer
addition, independently of consumers. Operation kind and element width are
part of the query: native floating-point addition and integer max/min retain
`gs(8)`. Scalar `vcadd` paths retain their existing row-local layout.

Two- and four-block native reductions combine their 32-bit partial sums with
32-bit predicates, then expose the low halfwords as `gs(8, 2)`. The arithmetic
still returns a logical 16-bit result modulo 2^16. The unused high halfwords
are unspecified padding; consumers must not read them as logical values or
assume they are zero.

Consumers use the regular layout support and `ensure_layout` machinery.
A 16-to-32 integer extension reads even halfword lanes with `vcvt EVEN`:
`ui16(65535 + 1)` extends to `0`, and `si16(32767 + 1)` extends to `-32768`.
It must not return the full hardware sum. A store or broadcast can consume
the strided slots directly when its layout supports them, or request a
conversion to consecutive slots. Multiple consumers share the same producer
layout and request their own necessary conversions.

This removes the former single-use cast-to-store peephole from
`vpto-optimize-vcvt`. It does not promise that every complete consumer chain
contains fewer instructions or runs faster. See the
[native sum layout design and validation](../../designs/a5-vcg-integer-native-layout.md).


---

## `pto.vmi.vcadd`

- **semantics:** Masked add-reduction. When `{group=C}` is absent, reduces all
  `L` active lanes to a single scalar (`V<1×T>`).

  ```c
  // Without group: full reduction to scalar
  T sum = 0;
  for (int i = 0; i < L; i++)
      if (mask[i]) sum += src[i];
  dst[0] = sum;

  // With {group=C}: per-group reduction
  int gs = L / C;  // lanes per group
  for (int g = 0; g < C; g++) {
      T sum = 0;
      for (int i = 0; i < gs; i++)
          if (mask[g*gs + i]) sum += src[g*gs + i];
      dst[g] = sum;
  }
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcadd %src, %mask {group = C, reassoc} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Source vector |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<C×T>` | One logical scalar per group in the assigned layout (`C = 1` if no group) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | Positive `C` dividing `L`, subject to the shape support above | `1` (full reduce) | Number of sub-groups |
  | `reassoc` | *(unit attr)* | *(absent)* | Permit reassociation (**required** for fp sources) |
  | `pmode` | `"zero"` | `"zero"` | Inactive-result behavior |

- **datatypes:** `i8`/`i16`/`i32` (signless, signed, unsigned), `f16`/`f32`,
  subject to the shape limits and internal 8-bit extension described above.
- **lowering to `pto.mi`:**

  | Group / W | Category | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|---|
  | No group (`C=1`), `K=1` | B | `1 × pto.vcadd` | `1` | `1` |
  | No group, `K>1` (fold) | B | `(K-1) × vadd` + `1 × vcadd` | `K` | `K` |
  | No group, `K>1` (partial) | B | `K × vcadd` + `(K-1) × vadd` | `2K-1` | `1+⌈log₂K⌉` |
  | W=32B, VLane-aligned | B | `K × pto.vcgadd` (eight groups per full packet) | `K` | `1` |
  | W=64B/128B, `k=2/4` fragments per group | B | Per result packet: `k × vcgadd` + `(k-1) × vadd` | `2k-1` | `1+⌈log₂k⌉` |

  Layout-conversion and consumer instructions are not included in these
  native-path counts. Equivalent-mask source folding is optional and applies
  only when legal. Native 16-bit integer `vcgadd` retains the partial-result
  form with 32-bit adds and `b32` masks; it is excluded from source folding.

- **example:**
  ```mlir
  // Full sum reduction (to scalar)
  %sum = pto.vmi.vcadd %x, %mask {reassoc}
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 256 f16 lanes → 8 groups of 32, two VLanes per group (W=64B)
  %sums = pto.vmi.vcadd %x, %mask {group = 8, reassoc}
      : !pto.vmi.vreg<256×f16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×f16>
  ```


---

## `pto.vmi.vcmax` / `pto.vmi.vcmin`

- **semantics:** Masked max/min reduction.

  ```c
  // vcmax: inactive lanes treated as -∞
  T best = -INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = max(best, src[i]);
  dst[0] = best;

  // vcmin: inactive lanes treated as +∞
  T best = +INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = min(best, src[i]);
  dst[0] = best;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmax %src, %mask {group = C} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:** Same as `vcadd` (without `reassoc`).
- **results:** Same as `vcadd`.
- **attributes:** `group`, `pmode` (same as `vcadd`, no `reassoc`).
- **datatypes:** `i8`/`i16`/`i32`, `f16`/`f32`.
- **lowering to `pto.mi`:**

  | Group / W | Physical lowering |
  |---|---|
  | No group, fold | `(K-1) × vmax` + `1 × vcmax` |
  | VLane-aligned | `K × pto.vcgmax` / `K × pto.vcgmin` |

- **example:**
  ```mlir
  // Full max reduction
  %mx = pto.vmi.vcmax %x, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 8-sub-group max (MX block-scale exponent pattern)
  %maxe = pto.vmi.vcmax %exp, %mask {group = 8}
      : !pto.vmi.vreg<256×ui16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×ui16>
  ```
