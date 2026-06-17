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

Arithmetic intensity (MAC per packed-operand vector load) is ~4.0–4.4 for both;
the win of stream-B is purely the larger accumulator count.

## Shapes

| shape | formulation | C accs | live regs | packed K/V layout |
|------|-------------|:---:|:---:|---|
| 4×4  | hold-both (early) | 4  | 4+4+4   | sblock4 |
| 4×8  | hold-both         | 8  | 8+4+8   | sblock8 |
| **8×8**  | **hold-both** (default) | **16** | 16+8+8=32 | K `[S/8][E][8]`, V `[Ev/8][S][8]` |
| **6×16** | **stream-B** (this work) | **24** | 24+6+2=32 | K `[S/16][E][16]`, V `[Ev/16][S][16]` |

## Isolated kernel throughput (warm, GFLOP/s)

| kernel | QKT (M512 S512 E64) | PV (M510 Ev64 S512) |
|---|:---:|:---:|
| 8×8 (this repo) | 56.5 | 76.3 |
| 6×16 plain | 63.3 | 74.9 |
| **6×16 + fused row-max** | **76.5** | — |
| KleidiAI 6×16 asm (ref) | 76.9 | 81.4 |

The intrinsic 6×16 reaches KleidiAI hand-asm parity (76.5 GFLOP/s ≈ 84% of peak)
once the 24 accumulators are kept in named registers (an `acc[R][4]` array
spills → only ~40 GFLOP/s; explicit named accumulators are required).

## In-context per-SDPA-op composition

One op = one layer's attention (8 heads, L=S=512). 268 MFLOP each for QKT/PV.

| stage | 6×16 | 8×8 | 6×16 GFLOP/s | 8×8 GFLOP/s |
|---|:---:|:---:|:---:|:---:|
| **QKT** | 6.17 ms (57%) | 4.72 ms (48%) | **43.4** | **56.8** |
| softmax | 0.92 ms (8.5%) | 1.08 ms (11%) | — | — |
| PV | 3.40 ms (31%) | 3.49 ms (36%) | 78.8 | 76.8 |
| pack K+V | ~0.25 ms | ~0.25 ms | — | — |
| **main_compute** | **10.81 ms** | **9.46 ms** | | |

e2e (20×512 inputs, single core, wall-clock): **6×16 ≈ 2.40 s vs 8×8 ≈ 2.32 s**
(6×16 ~3.5% slower).

## Key finding: isolated win does not carry to e2e

- **softmax**: 6×16 is faster (fused row-max in the QKT epilogue + SVE exp).
- **PV**: parity (~78 GFLOP/s, ≈85% of peak — the practical ceiling).
- **QKT**: the entire regression. Isolated, 6×16 QKT (76.5) beats 8×8 (56.5);
  **in-context it inverts** — 6×16 falls to 43.4 (57% of isolated) while 8×8
  holds 56.8 (100% of isolated).

Why: with MR=6, L=512 needs **86 K-passes/head** (each query-tile re-reads the
full ~128 KB packed K); 8×8 (MR=8) needs **64**. In the real forward pass K is
evicted between passes (by the interleaved PV V-reads and general model memory
pressure), so QKT becomes **K-traffic-bound** and throughput scales with the
inverse pass count: `56.8 × 64/86 ≈ 43`. The 24-accumulator compute advantage is
wasted because QKT is not compute-bound in this context. MR cannot exceed 6
(register-capped), so the only structural fix is **flash2 S-tiling** to keep K
resident — but for S=512 its online-rescale overhead measured *larger* than the
K-residency saving (SC=128 → 2.50 s, slower than untiled 2.40 s).

## Recommendation

- Keep **8×8 as the default** for the single-thread S≤512 embedding path; it sits
  at a local optimum (QKT bottleneck-free) that the 6×16's isolated kernel
  advantage cannot overcome here.
- Keep **6×16 selectable** via `FUSED_CPP_SDPA_KERNEL=6x16` (default is 6×16 on
  this branch). It is fully correct (matches `reference_sdpa_fp32` for
  unmasked / causal / masked / non-multiple-of-6 tails) and at KleidiAI kernel
  parity.
- The place 6×16's higher arithmetic intensity (24 vs 16 accumulators) may still
  pay off is the **multi-thread server path**, where per-core memory bandwidth is
  lower and QKT is more likely compute-bound — untested here.
