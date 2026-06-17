# fp32 NEON SDPA microkernel shapes

Comparison of the QKT (`scores = scale · Q·Kᵀ`) and PV (`O = P̂·V`) microkernel
tile shapes used by the `fp32_packqkv` SDPA path, with measured numbers on the
target box.

- Hardware: HiSilicon ARM (implementer 0x48), 2.9 GHz, 4×128-bit NEON FMA pipes.
- Peak fp32 ≈ 4 × 4 lanes × 2 (FMA) × 2.9 GHz ≈ **92 GFLOP/s** per core.
- All numbers below: single core (`numactl --physcpubind=40 --membind=0`),
  bge-small-zh-v1.5 Q8, L=S=512, E=Ev=64, 8 heads.
- Packed operands are contiguous (HW prefetcher covers them); no compute-time
  prefetch. Reduction is unrolled by 4; the held operand is broadcast via
  `vfmaq_laneq_f32`.

## Register-budget formulations

A `M×N` fp32 tile on 32 NEON registers (`q` = 128-bit = 4 floats):

- **Hold-both** (the 8×8 path): keep both operands resident.
  `C(M·N/4) + A(M) + B(N) ≤ 32`. At 8×8 that is `16 + 8 + 8 = 32` (exact),
  giving **16 accumulators** — the max for this formulation.
- **Stream-B** (the 6×16 path, KleidiAI `f32p16x1b_6x16` style): hold the LHS
  rows, *stream* the packed RHS through 2 registers.
  `C(M·N/4) + A(M) + 2 ≤ 32`. With NR=16 this allows `MR ≤ 6`
  (`24 + 6 + 2 = 32`), giving **24 accumulators** — 50% more independent FMA
  chains, which hides the ~4–5-cycle FMA latency better.
- **Stream-B, NR=8** (the 10×8 path): same budget with NR=8 allows `MR ≤ 10`
  (`20 + 10 + 2 = 32`), giving **20 accumulators** *and* a higher MR (fewer
  K-passes = L/MR) than 8×8 — and it reuses the existing width-8 packers.

Arithmetic intensity (MAC per packed-operand vector load) is ~4.0–4.4 for both;
the win of stream-B is purely the larger accumulator count.

## Shapes

| shape | formulation | C accs | live regs | K-passes (L/MR) | packed K/V layout |
|------|-------------|:---:|:---:|:---:|---|
| 4×4  | hold-both (early) | 4  | 4+4+4   | 128 | sblock4 |
| 4×8  | hold-both         | 8  | 8+4+8   | 128 | sblock8 |
| **8×8**  | **hold-both** (default) | **16** | 16+8+8=32 | 64 | K `[S/8][E][8]`, V `[Ev/8][S][8]` |
| **10×8** | **stream-B, NR=8** | **20** | 20+10+2=32 | 52 | K `[S/8][E][8]`, V `[Ev/8][S][8]` (same as 8×8) |
| **6×16** | **stream-B, NR=16** | **24** | 24+6+2=32 | 86 | K `[S/16][E][16]`, V `[Ev/16][S][16]` |

## Isolated kernel throughput (warm, GFLOP/s)

| kernel | QKT (M512 S512 E64) | PV (M510 Ev64 S512) |
|---|:---:|:---:|
| 8×8 (this repo) | 56.5 | 76.3 |
| **10×8 + fused row-max** | **71.2** | **71.9** |
| 6×16 plain | 63.3 | 74.9 |
| **6×16 + fused row-max** | **76.5** | — |
| KleidiAI 6×16 asm (ref) | 76.9 | 81.4 |

The intrinsic 6×16 reaches KleidiAI hand-asm parity (76.5 GFLOP/s ≈ 84% of peak)
once the 24 accumulators are kept in named registers (an `acc[R][4]` array
spills → only ~40 GFLOP/s; explicit named accumulators are required). 10×8 (20
named accumulators) lands at 71.2 — between 8×8 and 6×16, with no spill.

## In-context per-SDPA-op composition

One op = one layer's attention (8 heads, L=S=512). 268 MFLOP each for QKT/PV.

| stage | 8×8 | 10×8 | 6×16 |
|---|:---:|:---:|:---:|
| **QKT** | 4.72 ms (56.8 GF) | 4.73 ms (56.6 GF) | 6.17 ms (43.4 GF) |
| softmax | 1.08 ms | 0.91 ms | 0.92 ms |
| PV | 3.49 ms (76.8 GF) | 3.62 ms (74 GF) | 3.40 ms (78.8 GF) |
| **main_compute** | **9.46 ms** | **9.63 ms** | **10.81 ms** |

e2e (20×512 inputs, single core, wall-clock): **8×8 ≈ 2.32 s, 10×8 ≈ 2.32 s
(tie), 6×16 ≈ 2.40 s** (6×16 ~3.5% slower).

## Key finding: in-context QKT is capped ~56 GF/s regardless of shape

- **softmax**: 6×16/10×8 are faster than 8×8 (fused row-max in the QKT epilogue
  + SVE exp).
- **PV**: ~parity across shapes (74–79 GF/s, ≈85% of peak — the practical
  ceiling); 10×8's 20 accumulators make it marginally slower than 6×16/8×8.
- **QKT**: the whole story. Isolated, more accumulators = faster (8×8 56.5 <
  10×8 71.2 < 6×16 76.5). **In-context this does NOT carry over**: QKT lands at
  ~56.6 GF/s for *both* 8×8 (64 K-passes) and 10×8 (52 K-passes), and *collapses*
  to 43 for 6×16 (86 passes).

The 10×8 experiment was designed to test the hypothesis that QKT is
**K-traffic-bound** and that throughput ∝ MR (fewer passes = faster). It is
**not confirmed**: cutting passes from 64 (8×8) to 52 (10×8) left in-context QKT
unchanged at ~56.6 GF/s — a tie, not the predicted ~+18%. So there is a common
~56 GF/s in-context ceiling that both 8×8 and 10×8 hit (most likely the short
E=64 reduction plus Q/K load traffic, not the K re-read count). 6×16 sits *below*
that ceiling because its even-higher pass count, combined with the per-pass
overhead of a low-MR kernel, makes it the only shape that is actually
memory-bound in context. 10×8 therefore fully **closes the 6×16 regression**
(back to the 8×8 e2e) but does not beat 8×8.

## Recommendation

- **8×8 and 10×8 are equivalent** for the single-thread S≤512 embedding path
  (e2e tie at ~2.32 s). 8×8 remains the safe default; 10×8 is a validated,
  correct alternative that reuses the same width-8 packers.
- Keep the chosen kernel selectable via `FUSED_CPP_SDPA_KERNEL`
  (`10x8` / `8x8`). All paths match `reference_sdpa_fp32` for
  unmasked / causal / masked / non-multiple-of-MR tails.
- **6×16 is not worth shipping** for this workload (~3.5% slower e2e) despite its
  isolated-kernel and KleidiAI-parity win — the in-context QKT ceiling negates
  it.
- The remaining place higher arithmetic intensity (20/24 vs 16 accumulators)
  might pay off is the **multi-thread server path**, where per-core memory
  bandwidth is lower and QKT may become genuinely compute-bound — untested here.
