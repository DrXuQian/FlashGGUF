# Executable plan

Workspace: `/home/junfu/humming_quact_810e`.
Implementation: `src/QUactlize-kda-810e`.

## K-pack baseline

1. Bind direct INT4 to `DenseQ4KPack4KernelTypes` and
   `PreparedMultiformatLauncher`, requiring K-pack at compile time.
2. Pack codes with `q4_kpack4::prepare`; independently reconstruct all logical
   codes in the Python harness before any device call.
3. Build and validate:

```sh
CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  venv/bin/python -m pytest -q -p no:cacheprovider \
  src/QUactlize-kda-810e/tests/test_affine_contract.py
venv/bin/python src/QUactlize-kda-810e/tools/build_affine.py \
  --sdk /usr/local/PPU_SDK --output tuning/affine-kpack-sym-r2 \
  --configs 0,2,4,5,10,11,12 --jobs 4
```

4. Check `nvidia-smi -i 1`, then run on idle physical card 1:

```sh
CUDA_VISIBLE_DEVICES=1 PPU_SDK=/usr/local/PPU_SDK \
PYTHONPATH="$PWD/src/QUactlize-kda-810e" \
QUACTLIZE_PPU_PYTHON="$PWD/venv/bin/python" \
HUMMING_CACHE_DIR="$PWD/cache/humming-affine-kda" \
TRITON_CACHE_DIR="$PWD/cache/triton-affine-kda" \
QUACTLIZE_PPU_CACHE_DIR="$PWD/cache/quact-affine-kda" \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
venv/bin/python -u tuning/bench_affine.py \
  --library tuning/affine-kpack-sym-r2/libquactlize_ppu_w4a16.so \
  --output results/affine-kpack-sym-r2-baseline \
  --shapes 1024x3072 3072x1024 2048x3072 \
  --ms 1 12 64 512 --splits 1 2 4 8
```

5. Profile validated small/large-M cases, both backends in one process, with
   ACU `--profile-from-start no --replay-mode kernel --cache-control none
   --clock-control none --kill no --check-exit-code yes`. Record that this
   ACU version documents `none` as clearing L1/L2. Use launch/occupancy,
   compute, memory and instruction sections; retain all required reducers.
6. Record the next single candidate and its expected/disproving observations.
   No kernel promotion until the same contract is correct and faster.

## Remaining integration

The r2 baseline and matched small/large ACU captures are complete; see
`results/affine-kpack-sym-r2-baseline/REPORT.md` in the workspace.
The next build is `tuning/affine-kpack-sym-r3`, configurations 13 through 29.
Configurations 13..23 expand warp-M and pipeline geometries. Configurations
24..26 use the existing Stream-K driver at 1/2/4 CTAs per tensor CU, including
its mandatory lock reset on every run. Configurations 27..29 use the independent
Marlin scheduler with FP32 workspace, at the same grid multipliers.
All use the same K-pack artifact and native symmetric metadata as r2.

R3b completed: 588 valid points, 228 unsupported. Existing CTA schedulers
preserve correctness but do not close the deficit. The best M512/N2048/K3072
ordinary row is 78.705 us against 69.257 us; small-M remains behind r2.

R4 tests a separate QUactlize CUTLASS collective with a Humming-like N/K warp
grid and in-CTA FP32 reduction. It keeps the canonical K-pack4 bytes. The
global/shared writer copies each A/B/scale stage exactly once; K cohorts
consume disjoint group32 intervals. For a lane `l`, each native B n16 atom
uses columns `l/4` and `l/4+8`, K residues `2*(l%4)`, `+1`, `+8`, `+9`.
Two neighboring K-pack residue rows supply these from one g32 word set.
The tested Marlin PPU A-load/MMA and C-coordinate primitives supply the native
register map; no NVIDIA fragment map is inferred. Shared B has an eight-half
row pad; checkpoint bytes do not change. Each B fragment is scaled once and
reused over every M atom in the warp. All K cohorts reduce FP32 in shared
memory before the sole FP16 publication, avoiding a second kernel or locks.

Admission first requires full CPU-FP32 validation and changed-input graph
replay on card 1. Profile native instructions, registers and stage cadence
after correctness. This is a candidate, not a claimed speedup.

R4 passed 36 points. R5 production-AIU cohorts passed 72 points, improving
M12/N1024/K3072 to 7.531 us versus 5.729 us. R6 changes only the cohort reduction
storage from scalar thread-major to aligned four-value chunks across threads
for the existing config35 control. Additional candidate geometries use WN32
instead of WN16 within K cohorts. Large-M ordinary candidates 38..43 test
WN16 with WM64/128 to balance conversion amortization and register billing;
config22 remains the ordinary control. No default selector is promoted.

R6 completed (380 pass, 148 unsupported). The reduction reordering has little
effect; the best small-M K-pack candidate reaches 7.299 us versus 5.729 us.
Ordinary WN16 large-M variants do not beat the existing config22 control.

R7 separates the A reader from the B representation. Config30 remains the R4
control. Config47 uses the SDK's native swzl A load and a matching software
cp.async writer, with unchanged K-pack B. Config48/49/50 use Humming-like
warp-M/N/K geometry for the two larger M buckets. A second build instantiates
the same collective with the repository's existing classic Marlin code
packing, identified by a distinct mapping ID, while retaining identical
logical INT4/g32 codes and unpermuted FP16 scales. It is a representation
control authorized by the user's allowance to adapt the Humming solution,
not a claim about the canonical K-pack ABI. Both layouts must independently
decode every code before device validation; packing stays outside timing.
Final integration must make representation selection explicit.

The first R7 native-A row failed full-output validation (relative L2 0.845);
config30 remained correct. Changing the reader opcode prefix in R7b left the
residual identical: the initial prefix diagnosis was incorrect. The isolated
register-map probe (`tuning/probe_affine_a-r1.cu` in the workspace) demonstrates
that the software XOR writer is incompatible with the swzl reader; no register
permutation repairs it. Humming's actual PPU branch uses `load_aiu`, bypassing
the initially inspected `load_legacy` writer. R7c pairs an opaque AIU writer
with its swzl reader, first through the isolated map probe and then full GEMM
validation. The failed artifacts remain retained and untimed.

The R7c isolated reader probe passes: AIU-written + swzl-read A is exactly
equal to the known-good classic A fragment for every lane/register across
all four K16 atoms in a 16x64 cube (1024 half values, identity register map).
Evidence is `results/affine-native-a-map-r2.txt` and
`tuning/probe_affine_a-r2` in the workspace. No register permutation is needed.
The next gate is the full GEMM sweep of both weight layouts with this writer.

R7c is correct: 60/60 points for each weight representation. M12/N1024/K3072:
K-pack config47 is 8.332 us, classic code layout config47 is 7.199 us, Humming
about 5.74 us. The best existing K-pack cohort row (r6 config46, 7.299 us)
remains competitive with the alternative packing. Large-M R7c still loses:
classic config50 is 104.503 us versus Humming 69.474 us at M512/N2048/K3072.
These results rule out treating offline B reordering alone as the solution.

Next hypotheses, in priority order:

1. Match launch scheduling before another reader rewrite. Humming launches a
   flat 40-CTA grid; the new driver launches a 2D `(Mtiles,Ntiles)` grid.
   Compare a flat grid padded to 40 with early-exit inactive CTAs, preserving
   the same logical M-fast tile order, versus the existing exact 2D launch.
   This is a PPU scheduling hypothesis, not an assumption that extra work wins.
2. The new native mainloop consumes just-loaded A/B fragments immediately.
   Humming holds two register sets. After a matched ACU capture, try pipelining
   the next K16 A/B/scale fragment before the current MMA; watch the 128/256
   register billing thresholds and reject spills. Config49 already spills
   12 bytes in r7b and is not a promotion candidate.
3. For small M, test N128/K512/Stages2/WarpN32/WarpK64 (1024 threads).
   It preserves the same total useful warps as two N64 CTAs while sharing A.
   For larger M, test N-fast raster and a 40-CTA persistent scheduler only as
   separate changes after the compute pipeline is understood.

Primary sweep artifacts are `results/affine-{kpack,classic}-sym-r7c-sweep/`.
Library hashes: K-pack `2dc24a96404e5c23e0817b9eec723521ad169f6315f279c49e7698536401ac9a`,
classic `f2d192a61960002f502da12488826f89e417163508188ef16d0a73c7c0f5a693`.

The R7c matched ACU small-M capture has 534544 versus 406392 instructions,
87665 versus 24485 shared-load conflicts, 19.68% versus 31.41% active warps,
and identical 104448-byte shared allocations (QUactlize versus Humming).
Inspection finds the native driver's FP32 reduction still performs scalar
loads from thread-major partial storage. Prior r6 vectorization changed only
the separate production-cohort driver. Therefore r8 prioritizes this measured
remaining reduction path: config51 changes only to four-value vector chunks
across threads; config52 changes only the flat/padded grid; config53 combines
both. Config47 is rebuilt as the control. Config54/55/56 apply the combined
driver to the validated larger-M geometries, and config57 tests the separate
small-M N128/K512/stage2 geometry. All preserve per-output FP32 summation order.

R8 passed 96/96 rows for each representation. The padded grid does not help.
Vector reduction reduces classic M12/N1024/K3072 from 7.231 to 6.931 us, and
M64/N1024/K3072 reaches 11.737 us. The N128 small-M geometry is slower.
ACU confirms the M64/N2048 gap includes a register cliff: Humming uses 128
registers and 40.39% active warps, candidate54 uses 142 and 24.87%. Shared
allocations are identical, 125952 bytes. Native instructions are 1.210M
versus 1.489M; shared-load conflicts are 54864 versus 162008.

R9 addresses the actual B load branch: Humming uses AIU bulk B delivery,
while r8 still uses thread-partitioned cp.async. Config58/59/60 are exact
51/54/55 controls with native B delivery. Classic weights keep their existing
shared layout and use linear bulk copies. Canonical K-pack uses opaque AIU
swzl copies and the native transposed reader, caching each K64 delivery across
its four K16 MMA atoms; a WK32 warp selects its half of that delivery. The
transport map must pass an isolated bit-exact probe before the full oracle.
No input representation, scales, accumulation or timing boundary changes.

R9 passes both transport probes (8192 opaque b16 and 4096 opaque b32 values,
including nonzero row/column offsets), then 72/72 full-GEMM rows per layout.
Native K-pack M12/N1024/K3072 is 6.869 us, close to classic's 6.714 us; the
representation gap is largely removed. M64/N2048/K3072 remains 24.407 us,
with 134 registers in K-pack and 140 in classic, still over the 128 threshold.
R10 caches the two B fragments and loads A one M16 atom at a time, instead
of retaining every M atom's A fragment. It preserves each output's K order
and aims to drop at least six live registers at M64. Config61/62/63 are the
chunked-A counterparts of rebuilt control58/59/60. Resource usage and full
output/graph checks precede any performance admission.

R10 passes 72/72 rows per layout. Canonical K-pack bests include 6.720 us for
M12/N1024/K3072 (Humming 5.744), 42.057 us for M512/N1024/K3072 (42.023),
and 80.008 us for M512/N2048/K3072 (69.409). It is not admitted.
The predicted M64 register improvement did not occur: candidate62 still has
134 registers. ISA inspection locates registers 128..132 exclusively in the
FP32 reduction additions, rather than the MMA loop (`isa-62.txt` in r10).
R11 therefore bounds the live partial data by looping over peers for one M16
fragment at a time. This keeps the FP32 order and shared layout unchanged.
Config64/65/66 compare against 61/62/63; config67/68 revisit the validated
M128/N64/K128/WK32 geometry with and without the bounded reduction.

After that gate, consider a small-M N32 tile using an overfetched 64-column
native B cube (without changing canonical bytes), and separately specialize
N/K at compile time to match Humming's JIT contract. Humming's small-M cached
source is `cache/humming-affine-kda/603ddb7cf9a066e06ec79d0468ad0ebb/kernel.cu`.
Its disassembly is retained in r9 as `humming-small-isa.txt`; both kernels use
the same native FP32 MMA, while QUactlize currently keeps runtime N/K strides.

R11 passes 96/96 rows per layout. The bounded peer loop reaches 128 registers
for K-pack65 (134 for its control62), but adds latency: M64/N2048 is 18.724 us
versus Humming 15.424. Classic65 remains at 136 registers. The M128/N64
geometry is slower than N128. R12 replaces the serial peer loop with M16
scratch slices, unrolled direct reduction for at most four K cohorts and a
binary tree otherwise. Each tree level owns disjoint scratch slots, requiring
one publication barrier per level and one barrier before reusing the slice.
The tree changes FP32 associativity; the independent oracle and tolerances
remain unchanged. This is informed by Humming's chunked reduction organization,
with a separate slot allocation and barrier protocol.

R12 config69/70/71/72 compare against 61/59/62/63. The geometry ladder then
changes stages 3->2 (73), warp-K 32->64 (74), N64->N128 (75), and A loading
order (76). Focus this round on canonical K-pack, retaining classic's evidence
as the layout control. Full dense coverage to admit eventually is the seven
N/K pairs `(512,3072),(1024,3072),(2048,3072),(3072,1024),(3072,8192),
(8192,8192),(16384,3072)` and M in `1,2,4,6,8,12,16,24,32,48,64,128,
256,512,1024,2048,4096,8192`. The twelve anchors do not establish that coverage.

R12 passes all 144 rows. Chunked tree reduction alone does not close the gap.
The geometry ladder helps M64/N2048: candidate76 reaches 15.069 us versus
Humming 15.450 (still short of 10%). M64/N1024 candidate73 is 11.619 versus
11.447. M512 still prefers candidate63, 41.960/80.041 us at N1024/2048.
R13 specializes N/K without changing the kernel geometry or arithmetic.
Candidates77/78/79 isolate N-only/K-only/both for small M at N1024/K3072;
80 covers N2048, 81/82 specialize the best respective M64 geometries, and
83/84 specialize candidate63. Query rejects mismatched dimensions before launch.
The build now archives its task source and verifies it stayed unchanged during
compilation; this augments the existing source hashes and retained binaries.

R13: 84 supported rows pass, 60 are explicitly unsupported, and all 14 host
contract checks pass including actual DSO negative queries. Specialization
gives small-M 6.557/8.976 us versus Humming 5.737/8.109 at N1024/2048.
M64 is 11.387/14.783 us; M512 still prefers dynamic candidate63. No admission.

R14 corrects another concrete compiler difference: Humming uses
`__launch_bounds__(threads,2)`, whereas the CUTLASS driver requested one block.
Its matched M64/N2048 binary in cache `40bc121ef973b1bda382067fa87c248a`
reports 128 vector registers **and a 28-byte stack**. The earlier register
comparison must not imply that Humming obtained that limit without spills.
R14 first changes only the minimum-block compiler hint (85..91 and 96).
A separate M128 stage2 control92 has hints2/4 in 93/94; candidate95 applies
hint4 to candidate74; 97/98 compare hints1/2 for M64/N128/K128/stage3.
These hints are not occupancy measurements. Record local-memory traffic and
measured latency if a forced register bound spills; preserve FP32 arithmetic.

R14: 200 supported rows pass, 64 reject unsupported dimensions. Forced bounds
produce 124 registers/no stack for 86/87/95/98 and 118/no stack for 88, but
their instruction scheduling is slower. For example, M64/N2048 still prefers
82 (14.780 versus Humming 15.444); 88 is 18.146. Candidate94 has a 568-byte
stack and is decisively rejected (568.067 us at M512/N2048). Candidate92
(M128, stage2) modestly improves M512/N1024 to 41.244 versus Humming 42.017.
Small M remains 6.558/8.976 us at N1024/2048; no 10% admission.

Next controlled candidates:
- Keep candidate82 as the medium anchor. Try M64/N128/K256/WK32/stage2
  (1024 threads, eight K cohorts) against the WK64 control. Chunk reduction
  bounds scratch. This doubles useful warps without duplicating the A tile.
- The small-M N32 reader requires a physical 64-column native B staging cube.
  Overfetch only in shared staging, leaving canonical global bytes unchanged;
  validate nonzero N offsets before measurements.
- Humming's `kernel/humming.cuh:145-229` explicitly pipelines the next
  register load before current MMA, and unrolls stages in complete cycles.
  QUactlize still consumes each stage before advancing and computes dynamic
  stage offsets. First isolate cycle unrolling, then register prefetch.
  Match the original arithmetic and paired cache protocol.

The CUTLASS `device_kernel` already marks its Params with CUTLASS_GRID_CONSTANT.
Humming's non-Stream-K scheduler has slice_count=1 and uses the full static K
range; do not infer hidden CTA-level K splitting merely from its padded grid.

R15 isolates complete-stage-cycle unrolling (99..106) from register prefetch.
Controls are rebuilt with the same extracted, force-inlined consume function.
The geometry controls then test M64/N128/WK32 (107..109), small-M WN16
(110/111), stage2 versions (112/113), and WK32 with 16 K cohorts (114/115),
followed by chunked tree reduction (116/117). All keep a native N64-or-larger
B cube; the N32 staging change has not been implemented.

R15: 188 supported rows pass, 136 unsupported. Cyclic stage unrolling improves
M12/N1024/2048 to 6.296/8.565 us (Humming 5.730/8.117); M64 to
10.983/13.961 us (11.437/15.436); M512/N2048 to 78.611 us (69.433).
The new 1024-thread geometry variants are slower and are not promoted.

R16 tests fused cross-CTA K partitioning, retaining the production fixed-Split-K
actual-last publication protocol, explicit PPU acquire, and fixed-order
volatile FP32 peer reduction. One M tile is admitted; workspace is compact
`[split][M][N]` FP32 plus counters and capped at 64 MiB. Query/prepare reject
workspace aliasing. Preparation initializes counters and completes that work;
run has one kernel, and the actual last CTA resets each counter for replay.
Every writer fences, non-last CTAs return, and none spins waiting for a peer.
Workspace is now poisoned with NaN bytes and guarded at 256-byte alignment.

Candidates118..125 add two global K partitions to 99/100/87/88/101/102/106/105.
The separate small-M N128/K512/WK128/stage2 controls are 126/128, with two
partitions in 127/129. These tests do not change the primary Humming baseline.

R16 fails its first fused output gate (M1/N1024/K3072,118). Both compact FP32
planes contain 1024 finite values and all counters reset to zero, but every
FP16 output is nonfinite. The failed sweep/probe are retained; no fused timing
is admitted. R16b changes only peer loads to explicit scalar `ld.global.cg`
while retaining fixed FP32 addition order, to investigate visibility versus
the inherited volatile-load primitive. Candidate99 remains the arithmetic
control. The current evidence does not yet establish the cause.

R16b has the same failure. Its object contains the publication/acquire/reset
but no final FP32 peer loads or FP16 output stores at all, disproving a cache-only
explanation of that artifact. R16c isolates scalar half constructors from
NumericArrayConverter<half_t,float,8>; the generated output code is a required
gate before this diagnostic proceeds to device validation.

R16c emits FP32 peer loads, native packed FP16 conversion and vector C stores;
both M1 and M12 pass full output and changed-input graph checks. Latencies are
7.926/8.528 us, versus the 99 control's roughly6.2 us: no promotion. The
dependency NumericArrayConverter remains untouched; this extension uses the
same scalar half constructor already used by its validated single-CTA epilogue.

R15 paired large-M ACU evidence: Humming has 3.843M instructions, 114688 shared
conflicts, 68.39% elapsed tensor activity;103 has 4.474M,301056,59.22%.
Both execute256-thread blocks, but Humming uses40 persistent CTAs versus
103's80 padded ordinary CTAs for64 actual output tiles. These are diagnostic
counters, not latency acceptance measurements.

R17 tests two independent axes. 130..135 restrict the FP32 partial writers to
one warp (118/119/120/121/127/129 controls); every actual writer still fences.
136 is103 with a40-CTA persistent MN loop;137/139 specialize103 to N2048/1024
and K3072;138/140 add persistence to those controls;141 adds persistence to104.
The end-of-tile barrier protects shared output against the next tile's prime.

R18 is a two-row numeric gate for decomposing an m16 atom into two real PPU m8
atoms, retaining all four native A registers and the established m16 accumulator
storage/output map. 142 tests A(0,1)/(2,3),143 tests A(0,2)/(1,3);99 is the
known-good m16 control. Neither mapping is assumed correct from NVIDIA x2
behavior, and M12 must validate the upper eight-row half as well as row zero.

R17:96 PASS,408 unsupported. Fused K partitions remain slower, even with one
writer warp; they are not promoted. Persistent141 improves M512/N3072/K1024
to43.275 us (Humming106.637). Large N2048 remains77.305 us with137 against
69.394; persistence alone does not close that gap.

R19 follows the Humming mainloop's advance-A-load ordering, using two M16 A
fragments rather than retaining two full warp-M A tiles. It preloads the next
M fragment before the current MMA and carries the last preload across a K16
boundary. Candidate144 is101 with chunked A to isolate MMA-loop ordering;
145 adds prefetch. 146/147 add prefetch to99/100;148 to102;149/150 to137/139;
151 to141. Source ordering is a hypothesis until emitted code and paired
device measurements establish its effect.

R18:143 fails the first M1 gate (relative L2 0.72852);142 passes full output and
graph replay for M1/M12, proving A(0,1)/(2,3) for this swzl reader. Two m8
instructions are slower:6.425/6.449 us versus the m16 control's roughly6.24 us.
Keep143 as rejected evidence, not an admitted mapping.

R19:76 PASS,104 unsupported, no spills. Prefetch is generally neutral or slower.
M12 remains6.237/8.567 us for N1024/2048; M64 remains11.057/13.971 us.
Large N2048 reaches77.211 us with149 versus Humming69.400, still off target.
The checked native DSO passes14 host tests with one skipped fused-only test.
Library SHA256:2e89aaf48c394ba080e2263fdf6a330a038ea00e25249bd1cc0a9779516b0447.

Next controlled hypothesis: pack only FP16 scale metadata into N16 pairs
(n,n+8), so each lane can load one32-bit pair instead of two16-bit values.
K-pack code bytes, logical group32 scales and dequantization rounding must stay
identical. This needs an explicit scale-layout query and offline host packer,
an independent inverse-layout witness, a row-layout control build, and complete
output/graph checks. The library's code-layout ID must not silently stand in
for a new scale-layout contract. No repacking may occur in timed run.
Reprofile before attributing a gain to bank conflicts or instruction count.

R20 refines that plan to preserve the existing public scale-buffer contract:
prepare performs a bit-preserving copy into caller-provided workspace, while
run reads that prepared metadata. No new external scale layout is required.
The additional workspace is N*K/16 bytes, appended after existing partials and
counters when present, and the total remains capped at64 MiB. It persists for
the plan lifetime. The benchmark retains the original scale allocation and
reports it separately from prepared metadata; it independently inverses the
actual GPU-prepared buffer and checks every FP16 bit. Candidates152..158 pair
scales for99/100/101/102/137/139/141, respectively; row-layout controls remain.

R21 tests the stage-boundary register pipeline omitted by R19: retain two full
A-fragment buffers, prefetch the next stage's first A/raw B before the current
stage's last MMA, and transform the next B after the current MMA. Refill a
future stage at the penultimate K16 step; two-stage pipelines refill the retired
stage after consuming it. Unlike R19, A and transformed B remain live across
stage boundaries. Candidates159..166 derive from99/100/101/102/137/139/141/103.
The generic166/103 pair must cover stage-cycle remainders with K256/512/768/1024
before treating the schedule as general. This is separate from paired scales.

R20:72 PASS with bitwise inverse checks of every prepared scale buffer.
The small N1024 point improves to6.070 us; M512/N1024 to39.636 us and
M512/N2048 to75.416 us. Still no acceptance. Native ISA changes24 b16 shared
loads to12 b32 loads in the small control's unrolled stage cycle.

R21:96 PASS plus32 PASS over all three stage-cycle remainder classes.
Full A buffering is slower for small/mid M. M512/N1024 reaches39.130 us;
M512/N2048 is76.076 us, with256 registers and16 bytes of stack. The generic
166 has52 bytes of stack and regresses. These remain experimental.

R22 separates the logical output tile from the native AIU storage rectangle.
TM8 uses the proven m8 operand partition and a real four-float accumulator;
its A reader initially retains the validated16-row swzl cube. TN32 pads only
the shared B rectangle to64 columns; AIU padz bounds the global read, while
the kernel writes only its logical32 columns. K-pack code bytes remain intact.
167/170 compare paired/row metadata for TM8/TN64,168 changes167 toTN32,
169 changes168's WN to16. 171/172 test TN32 with TM16 and WN32/16.
173/174 combine paired scales with163/164. 175 narrows154 toTN32;
176/177 narrow155 toTN64/32. Numeric gates must include tails in M and N.

R22:84 PASS. TM8/TN32/WN16 reaches5.938 us for M1/N1024, but splitting
M12 across two TM8 tiles regresses. M12/TN32/WN16 reaches6.028 us.
The narrow medium tiles do not improve the prior best. Paired scales with the
full register pipeline reaches39.045/74.985 us at M512/N1024/2048.

R20 paired small ACU: Humming/152 issue406392/402768 instructions and have
12800/12448 shared-load conflicts, respectively. Shared-store conflicts remain
256/1536; registers86/88 are in the same billing bucket. Humming's grid is40,
152's grid is16. Profiler durations are diagnostic, not paired acceptance times.
R23 isolates the launch geometry:178..180 are152 with a flattened exact grid,
a grid rounded to40, and the latter with min-blocks2;181..183 repeat for153.
184..190 vectorize the output of152/153/154/155/156/157/169. Adjacent C values
belong to adjacent lanes: pair conversion is followed by one XOR shuffle and
byte permutation, checked against the authoritative output-coordinate map.
191..197 write those packed values directly to C and remove the terminal
scratch-reuse barrier. 198/199 combine vector output with173/174.

R23:124 PASS and36 additional M-tail/graph cases. Flattening/padding alone
does not improve small M. Packed/shared output generally regresses; direct
output improves only M1/TM8 to5.860 us, still behind Humming5.690 us.
The prior large winners remain39.089/74.992 us, so no output variant is promoted.

R24 tests small-stage depth, WK32, complete cycle unrolling and tree reduction
independently (200..204 for152,205..209 for153). 210/211 unroll154/155.
212..214 and215..217 retain TM128 while splitting it across two WM64 warps,
derived from156/157: whole reduction, chunked reduction, then full A buffering.
Warp order is N, M, K. Both A-fragment rows and C output rows include the
physical M-warp offset; FP32 scratch cohorts include all M/N warps.
The intended large-M tradeoff is lower per-warp C residency and more active
warps against duplicated B conversion. Actual register billing decides whether
that hypothesis holds; compilation alone is not a performance result.

R24:104 PASS plus12 M-warp tail cases. 211 reaches13.6469 us against
Humming15.4500 us at M64/N2048/K3072:11.67% median reduction and the complete
ten-sample ranges meet10%. This is warm single-layer evidence, not full
shape/cache-rotation acceptance. Two-stage200/205 reach5.943/8.321 us at M12;
WK32 and four stages regress. WM64 variants use176..180 registers and regress.
The 213 ISA shows A/raw-B coordinate values retained in VGPRs and several A
loads before the first MMA despite source-level chunking.

R25 isolates compiler scheduling from numerical layout. Parents210/211/213/216
get an empty memory clobber after each K16 step, opaque read-coordinate operands,
their combination, and min-blocks2. Chunk-A parents additionally get a compiler
memory clobber after each M16 fragment. These are compiler constraints, not
thread synchronization. Report register/stack counts and native timing before
claiming that a residency reduction improves occupancy or performance.

R25:50 PASS. The compiler clobbers lower WM64 from176 to168 registers but
remain in the256 billing bucket and regress. Opaque coordinates do not lower
register use. Min-blocks2 yields124 registers without stack for medium tiles,
but M64 becomes slower; for WM64 it yields128 registers with40 bytes of stack
and remains slower. Keep211 for the warm M64/N2048 result; other bests remain
200/205 (small),154/210 (M64/N1024),174/173 (large). No new promotion.

Next hypothesis: reuse a dequantized B tile across M warps instead of having
each M warp repeat conversion. Keep global weights in canonical K-pack and
expand only one CTA-local FP16 B tile, inside the timed CUTLASS collective.
Use WM64, WN32, TK=WK=64, TM128/256 and TN128 initially. WarpK1 removes the
FP32 K reducer. Only the M0 warps read/dequantize each B tile; a CTA barrier
publishes FP16 B to all M warps. The existing next-stage wait/barrier protects
scratch reuse, so one expanded tile suffices, rather than one per pipeline
stage. A proposed row-major half2 scratch map is
`n*(TK/2) + ((k/2) ^ ((n%8)*4))`; validate its bijection and actual lane
ownership against the existing B fragment reader before relying on it.
R26 uses the existing B reader's opaque BPair directly instead of the proposed
logical-coordinate map. Each `(K16 step, N atom, N/K thread)` tuple owns one
Vector128 slot; all M warps use that same tuple. The compiled sizes establish
that the scratch occupies exactly TN*TK FP16 values. This avoids rebuilding a
fragment map when the actual reader is already callable. Full CPU-FP32 GEMM
oracles remain the independent numerical gate.
237..244 target N2048,245..252 target N1024. Each pair differs only in shared
B reuse; TM128/256 and stages2/3 form the geometry grid, always WM64/WN32/TK64/
WK64. TM256 A uses two verified 128-row opaque cubes rather than assuming
that a 256-row AIU cube is supported. Older A geometries retain their layout.
WarpK1 allocates only a one-word unused reduction member, rather than a full
FP32 C tile. All active reduction allocations retain their existing extent.
The tradeoff is one extra publish barrier and more shared B traffic versus
shared conversion and lower register occupancy pressure. No gain is assumed.

R26:36 PASS plus8 M129/257 tail cases. Shared B does not win: registers
increase by18 in the paired rows, and its extra shared traffic/barrier is not
repaid. Baseline174/173 remain38.62/74.82 us at M512/N1024/2048. R26 is
rejected for selection; retain its native correctness and resource evidence.

R27 follows an ISA observation: the compiler emits signed-division correction
around nonnegative thread/warp indices (e.g. shift31, shift27, add, shift5).
253..266 clone200/205/210/211/173/174/237/238/241/242/245/246/249/250 with
unsigned internal thread arithmetic. A single K cohort is explicitly zero.
The public problem ABI stays signed and continues validating its extents.
Compare both resource use and emitted code; a source-level simplification
does not itself establish a speedup.

R27:112 PASS. Unsigned thread arithmetic improves the small and medium rows:
M12/N1024:5.415 vs5.726 us; M12/N2048:7.858 vs8.128 us.
M64/N1024:10.078 vs11.426 us (11.79%, including ranges);
M64/N2048:13.136 vs15.454 us (15.00%, including ranges).
Large rows still prefer signed173/174; unsigned full-pipeline rows regress.
Shared B still does not win. Keep unsigned253..256 as warm candidates only.

R28 targets the remaining small-M gap. 267..276 derive253,277..286 derive254.
First compare a single-buffer TK512 control, then TK1536/WK192, TK3072/WK384,
TK3072/WK192 flat/tree reduction, and TM8/TK3072/WK384. All use stage1 and
bounded 512-K/64-N opaque B cubes; larger K tiles do not enlarge an AIU cube.
The one-stage loop waits for the tile before MMA and fences before scratch
reuse. With TK3072 the complete K interval fits in about204 KiB of shared
memory for TM16/TN64, trading prefetch overlap/CTA occupancy for fewer stage
transitions. Other rows test TN32/WN16 with TM16/8 at stage2 and two kinds of
register prefetch on the existing TK512 geometry. No assumptions about wins.

R28:104 PASS plus104 shape-specialization UNSUPPORTED rows. All new rows
lose to the incumbent for their intended small-M regime. Single-stage
TK3072/WK384 gives6.26/11.26 us at M1/N1024/2048; it does not repay lost
overlap/residency. TN32 and both prefetch variants also lose. No promotion.
Current warm incumbents are253/254 for M1/12,255/256 for M64,174/173 for
M512 at N1024/2048 respectively. The M64 pair again passes the stable10%
criterion; small and large M remain below the acceptance requirement.

Native build identities:
- R26:57dcfbcf8bf59ae4c0b9a077ac4a4fb8a4c84901a1bea28854560bc3f55452f1.
- R27:a9554e22cdd29231a1a55cc85898e445c3b63fa24a05c1af1f73aa4b6def5e09.
- R28:cf93d61eca17c8468751effbd6149c0d3ffa238fb9970f195e9c9b1a69140cea.
All three builds pass15 host contract tests with1 skipped. The independent
full-output/poison/changed-input graph gate precedes every timed PASS row.

Next: refresh matched ACU captures for253 at M12/N1024 and173 at M512/N2048
using the R28 DSO, both against the shipping Humming in the same process on
card1. Inspect actual launches, total instructions, shared conflicts,
register/shared limits, active warps and tensor activity before choosing the
next candidate. ACU2.0.2 uses kernel replay, cache-control none and
clock-control none; its durations do not replace the clean paired timings.

R28 matched profiles confirm two kernels per comparison. Large Q173 has64
useful equal-K CTAs in a padded80 grid, versus Humming's40-worker DP/Stream-K
schedule. It also has16 stack bytes in the native object. The evidence and
limits are in `results/affine-kpack-r28-analysis.md` from the workspace
root (the report is a task artifact, not part of the KDA checkout).
R29 first changes only A-register lifetime: overwrite each current A fragment
with its next-step value after both N MMAs consume it. Compare signed173/174
and unsigned257/258 controls. The intended benefit is one A set instead of
two; resource savings and complete-call improvement must be measured.

R29:8 PASS,8 specialization rejections. Interleaved refill removes the16-byte
stack allocation but loses latency: signed287/288 reach78.02/43.35 us at
N2048/1024; unsigned289/290 reach76.17/39.61 us. Do not promote either change
alone. Library SHA-256:
e68c339289af6b61f2ebfe10e7e520a4b028b737235f05518e37a4bf7980effc.

R30 implements balanced contiguous Stream-K intervals inside the same driver.
Workers own `[floor(total*w/W),floor(total*(w+1)/W))` K-tile units. Whole
output tiles retain the normal epilogue. Partial tiles reuse the actual-last
FP32 publication/acquire protocol with at most two contributors initially.
One scratch slot per worker boundary bounds storage to
`2*workers*TM*TN*sizeof(float)` plus counters, independently of matrix M/N.
Query must reject empty scheduling domains, integer overflow and more than
two contributors; these are candidate admission limits, not format limits.
Host tests will enumerate the actual scheduling helper to prove unique
coverage, contributor numbering and scratch-slot ownership. Device tests
must cover short K slices, output tails and poisoned changed-input replay.

**RETRACTED R28 attribution:** the sentence above calling the selected Humming
schedule "DP/Stream-K" was not justified. Saved `humming_config` records have
`use_stream_k=false` for all primary M1/M12/M512 cases. The40-worker grid
persists over whole MN tiles. Both sides have two full-K waves at M512/N2048.
The source's FP16 inter-CTA addition is not executed for these primary cases;
do not lower their arithmetic precision. The initial exact rounding probe
failed because it incorrectly expected narrowing at M12; preserve its log.

R30:26 PASS points including changed-input graphs and M513/M641 tails.
Balanced Stream-K is slower at the target points: small291/292 need7.79–12.48
us; large293/295/297 need86.32/83.05/82.84 us at N2048. The corresponding N1024
variants need53.10/52.47/51.78 us. The non-register-pipeline293/294 pair has242
registers and zero stack; full pipelines295/296 and297/298 use52/28 stack
bytes. Keep this mechanism as measured rejection evidence, not a default.
M641 does beat its different Humming heuristic, but this does not close M512.
Library:5347fae5b8cf57e37ffa7f4d92660c625b3f013bf8cebad81bfd77b41dd9d879.

R31 tests M112/M96/M80 tiles. Existing source supports16-row multiples; its
AIU writer and native A reader both take the same actual `ACubeM`. For M112,
two112x64 A cubes exactly occupy `112*(128/8)` vectors and seven M16 fragments
cover rows0..111. No software XOR layout is substituted. At M512, M112 gives
40/80 useful CTAs for N1024/2048, so all work in the last wave is useful.
This changes tile geometry, not precision. Compare parent156/157,173/174 and
289/290 independently; add stage2 only as a separately identified M80 rung.

R31:18 PASS points. M112 candidate304 reaches33.786 us versus42.031 us at
M512/N1024 (19.6% reduction; all ten samples meet the threshold);303 reaches
65.128 versus69.422 us at N2048 (6.2%, not admitted). Both have252 registers
and zero stack. M80/M96 regress. Library:
f07adbfb744aa29483f22563f9954d5d8bfa010843f0eaad9ece23358d308921.

The corrected exact-rounding probe (`results/affine-reduction-semantics-r1`)
passes24 alternating changed-input graph replays. Humming and both native
FP32 variants preserve2^-10 at M12/M512/M513. At M641, Humming produces zero
for165888 of1312768 outputs while both native variants preserve2^-10
everywhere. This confirms the conditional FP16 inter-CTA path without using
it to explain or optimize the primary cases. Installed Humming epilogue and
scheduler headers are byte-identical to the pinned source.

R32 separates persistent whole-MN scheduling from raster order. The actual
Humming source maps `m=task/Ntiles,n=task%Ntiles`; native flat grids currently
map `m=task%Mtiles,n=task/Mtiles`. Earlier persistent candidates136/138/140
lacked paired scales;165 used stage2 without paired scales. None combines
the current stage3 full pipeline with paired scales. Compare each flag alone
and together against173/174 and303/304. No inter-CTA reduction is introduced.

R33 specializes M1/M12 from253/254, then separately prunes inactive output
rows in FP32 warp-K reduction and staging, then tests the existing direct
epilogue. Query rejects every other M. The production coordinate map proves
each float4 belongs to a single row. Both producers and consumers apply the
same predicate; all CTA barriers remain unconditional. At M1 only one of
the sixteen logical output rows is read/stored, while every MMA and all
group32 arithmetic retain their existing precision and ordering.

R32:19 PASS points (16 primary,3 M513 tails). N-fast persistence321 reaches
64.572 us versus69.430 us at M512/N2048; it still misses10%. It has256
registers and8 stack bytes. Plain persistent317 has252 registers/zero stack
but takes65.337 us. At N1024,304 remains best at33.682 versus41.959 us.
Neither raster order nor persistence alone closes the remaining large case.
Library:0dd4a581d9c180160f201d28e582fd48b10bea18a0b8d66af937107e8276e38c.

R33 library compiled with zero compiler errors:
4f6c9bc7056d3b52060b59682a16c3edfe3418a1f34ec1b52dc6131718da40b8.
IDs323/324/325 are N1024,M1: specialization only / row pruning / pruning
plus direct output.326/327/328 are M12.329–334 repeat these for N2048.
The paired device sweep is `results/affine-kpack-sym-r33-sweep`.

Possible next large-M experiment, not implemented or measured yet: M104
using six native M16 atoms plus one native M8 tail, rather than replacing
every M16 with two M8 instructions. Five M104 tiles cover M512, preserving
the40/80 useful-CTA grids while cutting the seven-atom M112 critical path.
Existing native M8 uses A registers0/1 and four FP32 C values; the first
half of the final A fragment at local row96 therefore has an existing
device-validated mapping. Physical A storage should remain112 rows. Limit
the first admission to one M warp cohort: WM104 with multiple M cohorts
can make an M16 reader cross a128-row physical cube boundary.

Such a mixed tail needs ceil(WM/16) accumulator fragments and explicit
masking of padded logical rows in reduction and output staging. Otherwise
the final four values per lane would address beyond a TM104 output array.
Use the existing authoritative output-row map and preserve unconditional
barriers. Compare the tail change alone against303/304 and321/322 before
claiming a register or timing benefit. Another bounded candidate is packed
or direct output on the current M112 parents; the older M128 vector-output
controls198/199 were slightly slower, so that is not an established win.

R33:16 PASS,40 explicit specialization rejections. Static-M-only323/326
reach5.313/5.302 us for M1/M12,N1024 (H5.706/5.744);329/332 reach7.680/7.710
us for N2048 (H8.118/8.127). Row pruning and direct output regress despite
correctness: M12 pruning327/333 takes6.284/8.442 us. Retain static-M-only as
the small-M parent; reject pruning/direct variants. No small-M point reaches
10% yet. The compiled native contract tests pass16 and skip1 unbuilt legacy
fused-partition case; the row-owner constexpr gate instantiates successfully.

R34 implements the proposed single-M8-tail M104 ladder as335–338, against
303/304 and321/322. Padding masks also cover the partial reducer; scalar
staging never addresses rows104..111. The physical A cube remains112 rows.
Performance and numerical acceptance remain pending the native build and
full-output device gate. M513 additionally exercises one live row in the
last M8 atom of the final M tile.

R35 varies small-M warp-K from the static-only R33 parents.339–346 use
WK128/WK256 on the existing N64/K512 tile, reducing intra-CTA contributors
from8 to4/2 while keeping FP32 partials.347–350 change only TN to128 from
the WK128 children;351–354 change only TN from the WK64 parents. The native
K reader starts every such cohort at a multiple of64, so `kg/2` remains the
existing raw-fragment selection. No shared-row pruning is inherited.

R34:10 PASS (8 primary plus2 M513 tails), with the incumbent gate fatal.
M104337 reaches64.390 versus69.355 us at M512/N2048, still below10%.
335/336 have242 registers;337/338 have246; all have zero stack. M104 is
correct but its arithmetic reduction did not produce the predicted latency
change. Do not infer proportional native-M8 instruction throughput.
Library:8f8975f0477659bf71cda5ad9bd9a4e466d2d65c0da02a72a219053dad7fd952.

R34 ACU matches two validated kernels at M512/N2048. Both launch40 CTAs of256
threads and admit2 CTAs/CU. Q337 uses113664 shared bytes,246 registers and
zero stack. Compared with Humming it executes4,942,920 versus3,843,048
instructions,137216 versus98304 shared-load conflicts,91520 versus16384
shared-store conflicts, and a0.251312 versus0.109120 memory-dependency
stall/issue ratio. Tensor active/elapsed is78.38/71.87% versus74.48/63.96%.
These are diagnostic counters, not a measured latency decomposition.
Report hash:396d6ee7207b0a0c2d37d3e23ba786382549a11f951157ba4e9142b507c10cd0.
Replay/cache/clock settings match R28; validation and raw CSV are retained.

R35:20 PASS. Larger WK loses at N1024 (WK1285.91–5.94 us, WK2569.63–9.69).
At N2048, N128/WK128349/350 reach7.554/7.560 us against8.138/8.130 us.
These improve the former7.68–7.71 us points but still miss10%. N128/WK64
does not win. Library:
f5dfc6860f35c71a09f88f4b4f22d9b01e2d29913b1c709d4a6780410a723330.

R36 separates packed shared output from direct output on303/321/337.
355/357/359 use packed shared stores;356/358/360 use direct stores.
The direct writer must clamp its row bound to the logical TM as well as
remaining M: M104 has a padded M16 accumulator whose upper8 rows must never
be written into a neighboring tile. M513 remains a required device gate.

R37 tests the unmeasured K1024 rung, retaining WK64:361–364 double TK from
the static-M-only parents;365–368 change only stages2->1;369–372 then change
TM/WM16->8 using the existing real M8 atom;373–376 change stages2->3 instead.
The K reader uses two existing512-wide physical B cubes. Unlike increasing
WK, this increases K cohorts8->16 and uses1024 threads. Read the native
resource report before attributing any change to occupancy. Prior one-stage
K1536/K3072 experiments268–282 already exist and are not repeated here.

The R34 native ISA independently retains redundant group32 scale loads:
at PCs0x1738/0x1748 and0x1b50/0x1b60 it loads the same
`vreg119+0x100/+0x120` addresses twice, for the two K16 halves of a group.
The complete ISA is `tuning/affine-kpack-sym-r34/w4a16_337.isa`. An explicit
per-N-atom scale register cache across the two halves is a candidate after
the output ladder. Refresh on part0, reuse on part1, and carry the cache
through the register pipeline; staged, prefetched and shared-B paths need
their own correctly scoped cache arrays. This is an observed duplicate load,
not yet a demonstrated bottleneck or improvement.

R36:11 PASS (9 primary plus2 M513 tails). Packed output359 reaches63.207 us
versus69.375 at M512/N2048:8.9%, still short of10%. It uses244 registers and
zero stack. Direct output360 is slower64.428 us. M112 packed355/357 do not
improve their parents. Library:
6b71a211a1cedf605d62d0f700d2624676b10f2b6dd34a4e9a1c7141538dc5f8.

R38 adds explicit group32 scale-register reuse.377/378 derive from323/326;
379/380 from349/350;381 from359;382 from303. Part0 initializes each N-atom
entry before part1 can read it. The full register pipeline owns its cache
across step refills; compute-tile and shared-B scopes each have their own
array. False-flag parents are rebuilt to detect code-generation drift.
Acceptance requires native correctness plus an ISA/resource check showing
whether repeated loads disappear without excessive register billing.

R37:22 PASS,66 explicit unsupported-shape rows. K1024 does not improve
small M: M16/stage2 takes5.72–5.78 us at N1024 and10.36–10.48 us at N2048;
one-stage and M8 descendants are slower. Keep323/326 and349/350 as the
small-M parents pending R38. Library:
3350e90d7fc7182fad6c544e9cf88f5bf6a50331aa47b9addf2d480a481917ad.
R38 build is the active next step; it has not yet been device-ranked.

R38 completed:16 PASS.381 removes the repeated large-kernel scale loads
(22->14 scalar TSM loads in the object) but slows M512/N2048 to65.252 us
from63.395 for359. It uses246 registers/zero stack;382 spills8 stack bytes.
Reject scale caching for this large pipeline. In small323/377 the complete
static instruction counts are identical1003;349/379 retain16 scalar scale
loads and differ only1316->1314 total static instructions. Small timings do
not establish a gain: prior CSE already removed the proposed duplicate reads.
Library:b5eaf78a080541105f55b08f08d9e9b428d147d01a161a71fe5aec81de5d7978.

R39 small ladder:383/384 add row pruning to349/350.385–388 compact the
existing pruned FP32 partial layout of324/327/383/384;389–392 then use direct
packed stores with the row predicate before conversion/shuffle. The helper
calls the authoritative Marlin row/column map and is host-checked for full
coverage, bounded slots and no aliasing; native CPU-oracle checks remain
the gate. Shuffle pairs share a row, and early-masked stores use the active
mask. Arithmetic and peer summation order remain FP32.

The independent large ladder derives393 (static M512),394 (unrolled cycles)
and395 (TK64/WK64, one K cohort) from359.395 retains the register pipeline,
unlike the earlier multi-M-warp TK64 experiments237–266. Its measured
register/shared residency and native reader correctness must precede any
occupancy explanation.

R40 completes the TK64 scheduling ladder.395 has228 registers/zero stack
but inherits the fixed40-worker grid, so it cannot expose more than two CTAs
per TFU CU.396 uses80 persistent workers;397 uses the ordinary flattened
N-fast grid with80-task granularity. The existing GridMultiple build control
now determines the persistent worker count too. All older persistent entries
were checked and use40, so their launch grids are unchanged.

R39 completed:23 PASS,49 unsupported combinations. Unrolling394 reaches
60.927 us versus69.324 at M512/N2048 (12.1% lower latency); the ten sample
ranges pass10%. It has254 registers and zero stack. Static-M393 reaches
62.280 but its ranges do not pass. M513394 is61.219 versus67.367 and is
correct, but does not pass the performance threshold.
The compact partial and early-masked output ladder loses at every small
shape despite lower register use; reject385–392. Small incumbents remain
323/326 and349/350. Library:
3fa5ee881282178270070005861539bb5f44a3a0fbb016ce65c18db12b18463e.

R41 applies cycle unrolling to the current small-M parents. The only older
small unroll entries203/208 had stage3, signed indices and dynamic M.
398–403 derive from323/326/329/332/349/350, changing only UnrollCycles.
404/405 specialize the current N1024/N2048 parents to M16;406/407 unroll
those. M16 stays in the acceptance matrix rather than inheriting M12 data.

R40:8 PASS. Raising the TK64 persistent grid to80 workers396 changes
M512/N2048 from83.158 to60.701 us versus H69.188; sample ranges pass10%.
Ordinary397 is61.483 versus69.352 and also passes. M513396 is60.723 versus
67.362, just short of10%. Prefer394/396 for subsequent accepted-path work.
Library:708bdb05737779355281a919d022ec26624caa27c0dd651fba245f372a8ad87a.

R41:16 PASS,80 unsupported rows. N1024 unrolled398/399/406 reach
5.160/5.210/5.223 us at M1/12/16 versus5.717/5.743/5.744; all remain short
of10%. Registers fall80->74. N2048 unrolled N128402/403/407 regress, so keep
349/350/405 (7.537/7.608/7.607 us). N64 unrolled400/401 are7.550/7.554.
Library:7e296a5a6d48a199c2c4cc5cc7aedfad186140ae3f9c94dd4566ec3bdb5df206.

R42 tests the full register pipeline408–413 against those six small-M
parents.414–419 remove the redundant step-(Steps-2) barrier for stage2.
For a live future copy, the next-stage wait/barrier at step-(Steps-1) already
ensures every warp has finished reading the current stage. At the range tail
that next-stage barrier is absent; the new path therefore skips future
copies outside num_tiles. The final run-range wait/barrier still protects
shared-storage reuse. This experiment admits whole K only and requires
fresh full-output and changed-input graph validation.

The independent `tuning/probe_affine_zero.py` calibrates shipping Humming
with all16 codes/zero points and one-hot rows at every group boundary.
It distinguishes centered integer-zero arithmetic from rounded FP16
offsets. Its scope is Humming-only; native asymmetric support is still pending.

Zero calibration r0 passed:384 one-hot rows at N1024/K3072, all16 code and
zero values,393216 outputs per replay, and three changed-input replays.
Every output equals `round_fp16((q-z)*s)`. Rounded-FP16-offset alternatives
differ at135264/135232/135512 elements, so this fixture distinguishes the
required arithmetic. Humming selected Stream-K, but each one-hot output has
only one nonzero contribution; this probe does not certify its multi-partial
reduction semantics.

The first R42 host gate caught a generator insertion before its parent IDs
existed (`IndexError` during module import). No native compilation or device
timing ran. Move the new ladder after R41 and repeat the fatal host gate.

R42 rebuilt and completed:18 PASS,90 unsupported combinations. Full register
pipelines are slower than the staged parents: N1024 around5.69–5.75 us versus
5.20–5.27; removing the stage2 barrier improves them to5.58–5.59 but remains
slower. N2048 remains8.80–8.88 versus7.55–7.61. Reject408–419.
Library:0bca99afaaa1de160bd0c41036a155830772cdfb1b49787ee7e988e9ecb2cae4.

R43 addresses output-transition synchronization without changing the MMA
schedule or reduction arithmetic.420–425 place FP32 cohort scratch after
the input/output union and retain every barrier.426–431 then omit the final
mainloop barrier and the post-reduction alias barrier. The FP32 publication
barrier still waits for every warp after its async drain and final MMA, so
input storage can then become FP16 output storage. The final output barrier
still publishes staged values to the vector global stores. This ladder
requires multiple K cohorts and whole-K flat reduction. Verify actual shared
bytes/resources and oracle/graph results before accepting a timing gain.

R43:18 small PASS plus6 gate/regression PASS (including fused and Stream-K).
Disjoint scratch is98304/131072 bytes, with unchanged74/108 registers and
zero stack. Removing the two alias barriers has little complete-call impact:
M12/N1024 is5.193 versus5.746; M12/N2048 is7.551 versus8.135. No small case
passes10%; reject the memory increase as a default. Large394/396 still pass.
Library:9839ba9c47825206de0c5d5a132327a937bd264997b5b9ef8816bf0e4d6f2900.

R44 tests direct global A delivery for small M, retaining native K-pack B
and the same FP32 MMA/reduction. The register coordinates come from the
actual actlize PPU0010 MMA ALayout; pair contiguity is checked on that object.
432–437 change only the A register reader, retaining the old A copy/storage.
438–443 omit the now-unused A copies, retaining the storage footprint.
444–449 then shrink unused A shared storage. Each rung needs independent
CPU-oracle and changed-input graph validation. This does not infer physical
behavior from CuTe alone; native machine operations and measured performance
must establish whether it helps.

R44:24 PASS,120 unsupported. The authoritative ALayout produced correct
results in every case, but direct global loads are much slower:12.1–18.2 us
at N1024 and19.2–29.3 at N2048. Reject direct A. Native objects confirm the
ladder:398->432 replaces24 static A swzl reads with48 global32-bit reads;
432->438 reduces static AIU instructions12->6. Register reductions alone
do not predict the latency. Library:
b20bd1d9b26b30a5389915b273b3eabb78fd3b88a2703065cc21833760745d9d.

R45 retains shared A, switching450–455 to the existing ordinary cp.async A
copy/reader at current static-M/unsigned-index settings.456/457 are bounded
read-only-load controls for444/445: inspect native code before device timing
and skip them if __ldg emits the same body. Native K-pack B stays fixed.

R44 N2048 profile is saved as
`results/affine-kpack-r44-n2048-profile.acurep`, with a raw CSV, validation
snapshot and `affine-kpack-r44-n2048-analysis.md` in the workspace.350 has
fewer executed instructions than Humming615824 vs791624 and similar
shared-load conflicts24896 vs25600, but active warps12.713 vs26.134 and
memory-dependency ratio0.300691 vs0.213643. Both report two resident CTAs/CU.
Its grid is(1,16,1) versus shipping Humming(40,1,1); this is evidence for
work-allocation/latency-overlap investigation, not a Humming mask bug.
R45 __ldg controls emit kp1 global loads, so they proceed to device checks.

R45:16 PASS,80 unsupported. Ordinary shared-A450–455 loses to AIU:
N1024 M1/12/16 is6.513/6.547/5.922 us; N2048 is9.288/9.378/8.755.
Read-only direct-A456/457 is12.067/17.450, effectively no useful improvement
over444/445. Reject both directions. Native AIU A remains the incumbent.
Library:835642040a2bdea034bcbee21bc707f8628dbc1ae633a03ba9827b5c5832ddb9.

Next investigation follows the R44 profile: grid/work distribution for the
small native reader, including flattened N-fast launch geometry and balanced
CTA work. Keep the current staged AIU parents398/399/406 (N1024) and
349/350/405 (N2048), with existing N64 variants as controls. M512/N2048 has
already met the hot target with394/396; default selector/wheel, native
integer-zero support and the final256MiB weight-rotation acceptance remain
unfinished. No device job is running after the R45 sweep.

R46 narrows warp N32->N16 at fixed CTA M/N/K, WK and pipeline.458–465
descend from398/399/406/349/350/405/400/401. This doubles threads per CTA
512->1024 while halving each warp's output width. Check actual register
billing, correctness and full-call timing; extra warps do not themselves
prove useful overlap. Older110–113 used N16 before paired scales,
unsigned indices and current static-M specialization, so they are historical
counterexamples rather than an exact test of this rung.

R47 is an independent M-tile rung while R46 device measurements run:
466–469 use TM=WM=8 from398/399/349/350, invoking the existing proved native
M8 MMA/output mapping at current paired-scale, static-M and unsigned-index
settings. M1 reduces padded computation and FP32 partial traffic; M12 needs
two M tiles and may lose from duplicated B traffic. Older M8 runs used
stage3, narrow N or one-stage/TK1024/3072; they do not establish this rung's
performance. Compile on the CPU concurrently; serialize device work.

R46:16 PASS,80 unsupported. N1024 candidates458/459/460 reach the hot target
at M1/12/16:5.028/5.088/5.086 us versus5.714/5.745/5.748, with all ten-sample
range criteria passing. They use58 registers and1024 threads, zero stack.
Retain these three as candidates for final product/cache-rotation acceptance.
N2048 candidates461–465 regress and are not promoted.
Library:108a0684ca1ed28c21203c86a13d3037d2bc12c2d68f1fae9804524441657bbb.

R48 focuses on remaining small N2048.470–472 increase TK512->1024 from
349/350/405 at WK128/WN32. This doubles K cohorts, retaining1024 threads
and16 useful CTAs while reducing the mainloop from six tiles to three.
473–475 extend461/462/463 with TK1024/WK256, retaining their four K cohorts
and WN16; this doubles each warp's stage work and halves stage count. Both
lineages require actual resource/spill inspection and independent numerical
validation. No prior candidate used small-M/TN128/TK1024.

R47:8 PASS,24 unsupported. M8 improves M1/N2048 to7.363 versus8.114,
but misses10%; M12/N2048 regresses to11.950. M1/N1024's5.080 versus5.691
does not pass the range criterion and is behind R46's admitted458.
Keep468 as an unadmitted M1 control; reject467/469.
Library:53ff3367be9e81b61c4254399ec502c2e795b7c450ace36ae7b72c3df91b71fc.

R49 tests independent warp-K pipeline progress, rather than more warps
sharing one CTA barrier.476–478 split A/B/scale copy ownership by K cohort
and use WK-wide opaque B cubes, retaining CTA barriers.479–481 then replace
only staged-consumption barriers with actlize NamedBarrier::sync for each
output cohort. User barrier IDs come from the SDK's reserved-count constant,
leaving barrier0 for the final CTA drain/reduction/output protocol. The A,
B and scale ranges of different K cohorts are disjoint; the final async drain
and CTA barrier remain. First validate the copy-only control, then the named
barrier path, with bounded jobs. No timing is admitted before full oracle
and changed-input graph checks. Source/native compilation is independent of
the ongoing R48 device run.

R48:12 PASS,24 unsupported. TK1024 is slower:470–475 are8.10–8.18 us.
All use1024 threads;96 registers for470–472 and108 for473–475, with zero
stack. Reject the larger K tile.
Library:b455b8136c7fbcebfd78b1e3baf42292f780785f7e5ffdc4fb10b88106940fe0.

R49:3 gate PASS plus9 sweep PASS,18 unsupported. Both copy partitioning
and named-barrier semantics pass full oracle and graph checks. Splitting
copies adds cost; named barriers recover some of it but miss the incumbent:
M12 is7.837 (477),7.685 (480), versus7.563 (350), Humming8.15.
Reject as a default. Library:
c1a7d756701c92411794b3c3108df65f313fd508be73ba6257683b4a6d3bbd93.

R50 isolates stage depth at the current native metadata/index specializations:
482–488 set stage3 on349/350/405/400/401/464/465;489–491 set stage4 on
349/350/405. Earlier stage3/4 small cases predate unsigned indices and static M;
none used TN128. For the TN128 grid16, extra shared memory reduces theoretical
residency without reducing the actual single useful CTA/CU. For TN64/grid32,
stage3 still permits two CTAs/CU. Check native resources and matched timing;
this is a latency-overlap hypothesis, not a resource-only prediction.

R50:17 PASS,34 unsupported. All deeper-stage variants regress. TN128 stage3
is7.81–7.84 us, stage4 is8.01–8.04; neither reaches10%. The N64 stage3
variants also lose to stage2. No stack spills; stage3 TN128 uses126 registers,
stage4 uses120. Library:
c25d9a9aa20b7d0d6b3b3fd87dbd56e7aac0977edc4b4ac1dfee9dc020526928.

R51 returns to the valid stage2 parents468 (M1/M8),350 (M12),405 (M16).
492–494 enable packed shared output;495–497 enable direct packed global
output.498–500 use exact-sized flattened N-fast grids;501–503 then pad those
grids to40. Each is a one-axis rung using existing mechanisms. Check complete
output, graph replay and paired timings; do not infer a gain from grid shape
or removed staging alone.

R51:15 PASS,30 unsupported. Packed staging, direct stores and flattened
grids do not pass10%. M12 parent350 is7.554; packed493 is7.677, direct496
7.574, flat4997.847, padded5027.907, against Humming around8.13 us.
M16 direct497 reaches7.546 but remains below the target.
Library:62752e6cbff62a42d327891f29b5630976e32f080354e61cae7e4a4235350ead.

R52 coarsens R49's copy/barrier groups to two K cohorts per pipeline.
504–506 use paired-cohort copy ownership with CTA barriers, from468/350/405.
507–509 then use named barriers. Native B cubes are WK*2=256, so this uses
four B-copy issuers rather than R49's eight, retaining disjoint A/B/scale
ranges between the two pipeline groups. Barrier IDs still start at the
SDK's first user barrier, and the final CTA drain/reduction remains intact.
Validate copy ownership and named synchronization separately before sweeping.

R52:3 gate PASS plus9 sweep PASS,18 unsupported. Paired K-cohort pipelines
remain slower: M12 505/508 are7.822/7.824 us, versus350 at7.563 and
Humming8.13. Neither copy subdivision nor named barriers is promoted.
Library:f2476463712e41fc8662f244540274cf0c6f8588a0568a8672323ad7f219d32a.

R53 keeps the admitted thread count, tile, stage2 reader and FP32 reduction
and changes physical warp assignment only.510–512 map physical warps K-fast;
513–515 XOR the N warp index with K;516–518 XOR K with N. Parents are
468/350/405. Lane IDs are preserved and the same logical thread permutation
feeds copy ownership, MMA and reduction. Limit this experiment to WarpM=1:
the collective's warp_m() still reads the physical thread index. Measure the
actual impact rather than assuming a fixed hardware PU assignment.

R54 is a separate copy-ownership axis on350/M12:519/520/521 move the B-copy
issuers to warps8/10/14 respectively, leaving computation and reduction
ownership unchanged. For this tile, A and scale copies use warps0–7 and
the incumbent also assigns B copies to0–1. The new assignments place B in
otherwise unused copy warps; the source coordinates and copied byte set
are unchanged. This tests whether producer serialization matters, without
claiming that an async instruction blocks until its transfer completes.

R53:4 gate PASS plus12 sweep PASS,24 unsupported. The K-fast assignment
regresses M12/M16 to8.00 us; XOR assignments are7.61–7.63, versus7.58
controls. All M12 mappings use108 registers with zero stack.
Reject physical warp remapping as a default.
Library:deeb0764f1f2ac2eaea6e7a91269f9c5a1009de25a84c13c5e5e91b7a29eb4f7.

R55 fills an actual hole in the geometry table: TN64/WN16/WK128. Prior
TN64/WN16 candidates used WK64, doubling K cohorts and FP32 reduction work.
522 changes only WN32->16 from345 (TN64/WK128);523 changes TN64->32;
524 changes TK512->256;525 then changes WK128->64.526 enables cycle
unrolling on522, and527 changes only WK128->256 on522. Include350,345 and
401 controls so this ladder separates N width, K work per warp and loop
unrolling. All new rows first target M12/N2048/K3072.

R54:4 PASS. Moving B issuers does not close the gap:519/520/521 are
7.582/7.636/7.610 us versus Humming8.13–8.14.
Library:d1e3bf68f8026dcb3d902ac34ec664a1d811144103b387fc91dd810a8c913213.

R55:9 PASS. The missing TN64/WN16/WK128 shape improves to7.483 us (522),
and full cycle unrolling reaches7.381 (526) versus Humming8.136.
This is9.3% lower latency, still short of10% and not promoted.526 has72
registers and no stack. TN32 and WK256 variants regress.
Library:c12fb55ea802705713760d7c6eda617256b58d9ec0da764aa746e98184e9e9dd.

R56 tests local scheduling/output axes on526:528 loads A before B conversion
by disabling ChunkA;529 prefetches the next A fragment;530 caches paired scales;
531 writes packed output directly;532 moves B copies to warp8.533 extends
this geometry to M1/TM8, and534 to M16. These rows retain the native reader,
FP32 reduction and query/prepare/run boundary.

R56:11 PASS,22 unsupported. M1/533 and M12/direct531 pass the median10%
criterion (7.277/8.118 and7.309/8.131 us) but miss the conservative sample
range criterion by about7 ns and27 ns respectively. M16/534 is7.379/8.111,
still short. This is not whole-package acceptance. Library:
bdd16e155417af09e7808e43baa3c1802eaaefb0aaee7f2ad94835b707fb1db1.

R57 extends direct output to535(M1) and536(M16);537–539 add A-first,
scale caching and A prefetch separately on531.540/541 prune padded rows
on531/535.542–544 use the existing disjoint FP32 reduction storage mode2
on531/535/536, removing the final mainloop alias-protection barrier. The
separate partial region remains bounded and must not change data or
reduction precision. Test each row's full output and changed-input graph.

R57:13 PASS,26 unsupported. M1/535 and M16/536 now pass both median and
ten-sample range criteria:7.222/8.126 and7.267/8.123 us. Retain these as hot
module candidates, not package/cache-rotation acceptance. M12 remains thin:
5317.318/8.144,5387.313/8.139; range criteria still fail. Pruning and disjoint
storage do not close this remaining gap.
Library:3086d5c0f75a36605ecd9cd21b44426560473096a6b7e811733fec79b81292b5.

R58 targets only remaining M12.545 preloads all FP32 peer vectors before the
unchanged ordered additions, with a compiler memory fence to keep those reads
before use; inspect native registers/stack and LDS scheduling.546 changes only
WM16->8 on531, using two M warp cohorts and existing M8 MMA mapping.547 changes
TK512->256 on546;548 then changes WK128->64. This ladder keeps potential
register-bucket and duplicated-B work costs visible. All retain FP32 sums.

R58:5 PASS. The two-M8-warp-cohort ladder546–548 is12.52–12.96 us despite
64/64/52 registers without stack; reject it.545 is7.307/8.137 us and still
misses the sample-range gate. Library:
5dfa49d427c060c0db13e52a2e2450e46e43bc9963eccb291c34c5427ee01dde.

R59 tests the output tail on the existing output-map witness.549/550 extend
531/545: convert and exchange both packed words belonging to one output row
before one common row predicate, rather than a separate predicate per word.
All shuffles still execute with their original participation mask; only stores
are grouped. The existing output-row map must prove the four-value group shares
one row. Compare native instruction scheduling as well as paired event latency.

R59:4 PASS.550 reaches7.276875 us against8.142187 (10.63% lower);
max Q7.291875 <=0.9*min H8.125625. Full output and changed-input graph pass.
The R59 grouped output removes one tail predicate/restore pair in native ISA.
Keep550 for M12, with R57's535/536 for M1/M16. This completes the missing
representative hot module rows, not selector/wheel/cache-rotation acceptance.
Library:a768be06e1dc4f5580df3455c3edc805a6159b55cb4ae7efa4663aca32f42d07.
R58 ISA also shows the empty memory-clobber asm emits a device vmem.fence.blk:
the prefetch experiment is not a compiler-only fence, and the incumbent already
issues all six peer LDS vectors before arithmetic. Do not attribute its timing
to an unobserved improvement in LDS overlap.

Integer-zero A0 adds551/552/553 from550/535/536 and generic-M/N control554
at K3072. Public zero_mode1 consumes uint8[K/32,N] values0–15 in prepare,
packs two N-separated zero nibbles per byte alongside paired FP16 scales,
and reads those prepared bytes in the native collective. The converter
subtracts z exactly before FP16 scale multiplication, matching the earlier
Humming calibration; no rounded floating offset is substituted. The source
zeros are checked during prepare. Query/prepare/run ABI structure sizes stay
unchanged; an additive configuration-mode query distinguishes the candidates.
Existing symmetric parameter/storage layouts use empty/conditional types.
Run integer-zero host/ABI guards, exact one-hot probes over every q/z and group
boundary, full outputs, metadata byte decode, invalid-zero preparation, changed
graph inputs, then separately paired integer-zero timings.

A0 passes18 host/ABI tests(1 skipped), the exact384x1024 one-hot comparison
at all group boundaries and q/z values, and three changed-input graph replays.
16/255 zero inputs are rejected; poisoning original scales/zeros after prepare
does not change output.14 generic full-output/tail rows plus3 specialized
small-M rows pass. Asymmetric performance remains unadmitted: M1/M12/M16
are8.637/8.675/8.662 us versus9.040/8.929/8.936, only2.8–4.5% lower.
Library:a6816384615d76f64ca071e76514e18358d7d2bf490681fa3f3591a18a11e461.

A1 changes only the zero reader:555/556/557/558 extend551/552/553/554 with
aligned32-bit shared loads plus byte extraction instead of byte shared loads.
A0 native ISA contains tsm.ld.b8. This tests the bank/broadcast behavior without
expanding metadata or changing arithmetic. Confirm the emitted load width and
repeat the independent decoder, exact probe and paired small-M comparison.

A2 isolates integer-zero converter setup. A0's native code constructs both
1024+z and -(64+z) biases, including two integer multiplies per column. Humming's
uint_to_f16 uses a shared1024+z subtraction after low-nibble extraction.
559/560/561/562/563 extend551/555/552/553/554: shift the high nibble down,
reuse the same exact bias for both half2 subtractions, then multiply by the same
FP16 scale. Preserve A0 arithmetic as the control; metadata stays4.625 bits
per weight. Inspect native operations and repeat the exact q/z probe before
paired timing. This is a source-level hypothesis until those gates pass.

A1 passes18 host/ABI tests(1 skipped), the exact q/z and changed-input graph
probe, and6 paired small-M rows. Native555 eliminates all24 byte LDS operations.
M1/556 improves to8.377 us versus Humming8.921; M12/555 is8.574/8.917.
It remains below the required10% gain. Library:
680bf62a32980f1d82e297e9bfb43458a1e34fc566f387ad6730c8b32d7ec256.

A2 passes18 host/ABI tests(1 skipped), the exact probe and8 paired rows.
The converter removes96 static half2 FMA instructions and uses80 registers
instead of84 at M12 (no stack). M1/561 is8.338/8.954, M12/5598.531/8.941,
M12/5608.555/8.913, M16/5628.577/8.918. No asymmetric row is performance
admitted. Library:da61c4c075642173cd89a5cbb21e4fed6c12b8d2c934592cdce62b33072d3e59.
Next, profile the validated integer-zero path against shipping Humming before
another broad geometry sweep. A separate possible metadata experiment would
prepare exact FP16(1024+z) biases and load paired words, eliminating runtime
integer replication; this would increase native runtime storage to5 bits/weight
and must be reported and tested as such, never presented as4.625. No such
metadata expansion has been implemented.

Production integration remains pending. The existing public library name is
libncp_quactlize.so, built by tools/build_ppu_library.py and packaged through
build.sh --wheel. Link the admitted affine objects into that library rather
than shipping the screening DSO as the acceptance route. Add the affine C
header/exports to its sealed manifest and provide default selection during
query/prepare, keeping run allocation/JIT/conversion-free. The full bundle
builder seals all package sources, so finish source edits before starting it.
Full dense coverage remains the seven N/K pairs and18 M values recorded
above; the hot anchor wins do not certify that matrix or the256MiB rotation.

A2 paired ACU is saved with an analysis report in the workspace results.
The current asymmetric path has comparable active warps25.05/25.87 and fewer
executed instructions903776/933576, but shared-load bank conflicts61440/25600,
group conflicts97504/45458 and TSM-dependency stalls/issue0.338619/0.139785.
This motivates changing the compact zero metadata layout before expansion.

A3 interleaves four group32 zero bytes at each N pair into a word, retaining
the same4.625-bit footprint and source API.564/565 extend559/560 with this
layout (byte/word reads);566/570 and567/571 are M1/M16 byte/word variants;
568/569 are generic-M/N controls. The global copy, preparation and shared reader
all use the same bounded group-quartet layout. Independent host inversion and
one-hot arithmetic checks must precede timing. In particular, a word reader
may reuse a loaded word across four groups; verify this in the actual ISA.

The first A3 build failed before producing a library: its class-scope assertion
referenced CopyK, which is local to copy_stage. Keep the copy-cohort assertion
next to the actual CopyK declaration; rebuild in fresh A3b output. No device
test or performance result came from the failed build.

A3b passes18 host/ABI tests(1 skipped), both generic byte/word exact probes,
and8 paired rows. The compiler widens the interleaved byte accesses itself:
564 has30 scalar word LDS instructions (24 scales +6 zero words), compared
with559's24 scale words +24 zero bytes. Explicit word565 still has48 word
LDS operations, so source load width alone does not predict reuse.
M12/564 improves to8.246 us versus8.907, M16/5678.265/8.929 and M1/566
8.233/9.023. Still no asymmetric range-gate pass. Library:
bf11f8cc7b8d498715b0adc61b21906dab505bd85610579c064391e573540fee.

A4 tests paired integer-bias construction without changing storage:
572/573/574/575 extend564/566/567/568. Multiplying a packed zero byte by
0x1001 places its two nibbles at bit0/16 after masking; append the FP16
1024 exponent then broadcast each halfword. Compare emitted integer
replication/permutation work, exact q/z arithmetic and paired timings.
The old manifest quantization prose said symmetric even in A0–A3b; their
integer_zero lists, ABI mode queries and benchmark/probe contracts identify
mode1 correctly. Fix the prose for new builds; retain old artifacts verbatim.

A4 passes18 host/ABI tests(1 skipped), the exact probe and6 paired rows,
but is rejected: M12/5728.309 us versus control5648.237, M16/5748.329
versus5678.235. Integer replication multiplies fall48→24 while registers stay
76/68; fewer operations do not establish a latency win. Library:
6d24080d5ce5d73334c5f345693b76b46bfff88206ae87240ab7ca1d1f916a22.

A3b follow-up ACU confirms bank conflicts24576 versus Humming25600 and
group conflicts40449 versus46563. TSM-dependency stalls/issue fall to0.154753
(Humming0.138848). The compact layout has resolved the observed shared-memory
gap. A5 changes stages2→3 at576/577/578 from564/566/567, with all reader and
arithmetic choices retained. Extra pipeline storage/occupancy is a tradeoff;
inspect resources before timing.579/580/581 add integer-zero N1024 small-M
anchors;582/583 add M64 N1024/N2048;584/585 add large-M N1024/N2048;
586/587/588 cover N3072/K1024 with M64/M128/M16 tiles.585 retains the ordinary
zero layout because TK64 contains only two group32 entries. These are new
functional/performance candidates, not automatically inherited symmetric wins.

A5 passes18 host/ABI tests(1 skipped), the exact N3072/K1024 probe with all256
code/zero pairs and44 full-output/graph rows covering15 anchor shapes. Eight
anchors pass the paired range gate: all five N3072 rows, both M64 rows, and
M512/N1024(36.936 vs43.976 us). Small N1024 is near the boundary at5.57–5.64
versus6.19–6.24; N2048 small remains8.11–8.27 versus8.91–8.94. M512/N2048
is65.176/70.565 (7.64% lower), so it remains unadmitted. Stage3 does not replace
the stage2 incumbents. Library:
b8bce4fba5415ed45e2a5b3e5e984d049938fe270db2de26007412bb0b911467.

A6 targets those remaining anchors.589 enables two-group zero interleaving
for TK64, using the same compact footprint and a generalized1/2/4-group
encoder/reader.590 uses native M8 for M1/N1024;591/592 group output stores
for M12/M16/N1024.593–595 prefetch A registers at N2048;596–598 change the
warp N/K rectangle32/64 while preserving512 CTA threads;599–601 use the
existing full register pipeline. Each branch has its own parent control.
The A3b profile shows TFU write-after-read stalls/issue0.026752 versus Humming
0.000603; operand lifetime/overlap is a hypothesis, not an accumulation-RAW
finding. Exact probes/whole outputs and native resources precede admission.

The first A6 build stopped at the output-store assertion: N1024 parents580/581
do not enable DirectOutput, unlike the N2048 parents. No library or device
result was produced. Keep the assertion and split this experiment into a numeric
ladder:591/592 enable DirectOutput only, then602/603 add grouped stores. Rebuild
in fresh A6b output; attribute grouping effects against591/592, not580/581.

Product host CI initially reports1367 passes/3 skips and therefore FAIL.
Its strict no-skip rule is retained. The three native-library checks had been
incorrectly assigned to the headers-only host inventory. Move specialization
checks and the combined available-workspace guards into the existing public
accuracy suite; it resolves the verified package by default and records which
compiled configurations exercise fused partitions. Explicit diagnostic DSOs
remain available for CPU-only ABI checks. The16 pure-host affine tests stay in
the host inventory. Rerun the host gate; neither skipped checks nor the initial
FAIL count as release acceptance.

A6b passes18 CPU/ABI checks without skips, the exact G2 N2048 probe, and28
full-output/graph rows. N1024 small-M now passes the range gate:
M1/5905.365 vs6.184 us, M12/6025.452 vs6.184, M16/6035.483 vs6.268.
Eleven of15 asymmetric anchors now have hot range-gate evidence. Full register
pipelining regresses N2048 to9.43–9.45 us; two-group zero packing reduces
registers244→238 but regresses M512/N2048 to66.973 (585 control65.162).
Preserve these failures. Library:
aae624acb23b1935cd09cba50ede8784c6fcfe4fe0cb83488454c67448202df0.

A7 targets the four remaining N2048 anchors.604–606 use direct quartet-word
addressing on593–595, avoiding the signed byte-address division that defeated
reuse in565; the existing569 is rebuilt as the generic exact-probe control.607 increases stages on the
M1 warp32 candidate597.608/609 bound reduction on596/598;613/614 then change
only stage depth on those parents.610–612 increase the large-M tile104→112,
120,128 on585 to test whether additional row reuse amortizes integer-zero
work; inspect spill/occupancy before interpreting results. These candidates
retain integer arithmetic, FP32 accumulation/reduction and the K-pack code ABI.

The initial A7 host registry gate caught a duplicate of569 before native
compilation. Reuse569 instead of creating a second identity; the last two
stage-depth variants consequently occupy613/614. No native binary or timing
was produced by that failed host gate.

A7b then failed native trait checks: the M12 parent uses prefetched reduction,
which is mutually exclusive with bounded reduction, and the aligned M112/M128
tiles cannot retain M104's M8-tail trait. Select the bounded reducer instead of
the prefetched reducer, and derive the tail trait from the new warp-M dimension.
No assertions are weakened. Preserve the compiler logs and rebuild as A7c.

A7c passes18 CPU/ABI checks, the exact quartet-word probe and30 paired rows.
The direct word index restores30 scalar word LDS operations, but does not
replace the byte-reader incumbents. Bounded reduction reduces the M12 warp32
register count134→84 without a complete-call win. M120 spills20 bytes; M112
and M128 have252 registers/no stack but also lose. N2048 still has no new
range-gate pass. Library:
b39b825757dacab285f3200d8eb6425c610e57e12acaafdaf77450db8a1b380c.

Public integration now links the finite affine object set into libncp_quactlize,
ships its C header and sealed affine.json receipt, and adds config=-1 plus
select_v1 without changing existing ABI structures.615–621 are bounded generic
fallbacks, separately unadmitted. The normal builder emits an empty affine
registry on ppu_15, preserving exports while reporting native support absent.
Library() resolves the installed public bundle; explicit tuning paths remain
diagnostic. --auto in the benchmark selects through native query/prepare.
Shipping Humming Stream-K rows are excluded from the FP32-reduction comparison.

A8 also tests622 (cycle-unrolled585),623 (integer-zero version of symmetric394),
and624–626 (paired bias on593–595). The paired-bias implementation now replicates
the whole zero byte once and uses two masks plus a shift, eliminating A4's
byte permutations without expanding metadata. Rebuild575 for the exact oracle.
The previous paired-bias artifacts remain preserved; this is a new source hash.
Default-selector functionality, normal wheel build and device acceptance are
still pending; none of this wiring constitutes production performance admission.

Product host R1 passes all1367 tests with0 skips. The40-configuration A8 module
compiles, and the native specialization/default/workspace ABI tests pass.
The ppu_15 empty registry also compiles and rejects unsupported queries.
The module SHA is d34283dce8e627876401836ec2664e9385b656487c0b39efcba96d7c76ddd6c9.
A8 exact paired-bias verification and18 performance rows pass numerically.
622 reaches63.450 vs70.652 us at M512/N2048; its63.534 us maximum exceeds
0.9*H_min=63.514 us, so it remains unadmitted. The default now selects622 over
585 without claiming admission.623 loses at69.490 us. Small-M paired bias
alone is insufficient.

Default query/prepare passes10 comparable full-output/graph rows per mode;
9 symmetric and6 asymmetric rows pass the range gate in this run. Symmetric
M1/N1024 misses the range gate despite an11.2% median reduction. The five
N3072/K1024 rows select Humming Stream-K and were conservatively excluded;
the historical11/15 asymmetric count is NOT yet a verified same-reduction
production claim. Run the existing exact cancellation witness on all15
anchors before deciding which Stream-K rows actually narrow partial sums.

A9:627 changes symmetric458 to an M8 tile.628–630 form a M1/N2048 ladder:
597 plus register prefetch, then paired bias, then direct word loads.
631/632 apply the existing M8 MMA decomposition to624/626.633/634 change
622's persistent scheduling to ordinary2D/flattened grids; the2D variant
disables the inapplicable flat-grid raster flag.635 specializes622 to M512.
These are independent measurements, not a combined promotion.

The15-anchor exact cancellation probe confirms active Humming narrowing at all
five N3072/K1024 points: one sixth of outputs produce0 where FP32 gives2^-10.
Every QUactlize default output retains2^-10; the other ten Humming anchors also
retain it. Strict comparison admission is therefore6/10 asymmetric anchors,
not11/15. Preserve the historical timings without relabelling their precision.
The benchmark now still validates QUactlize's full output/changed-input graph
for excluded Stream-K rows, but produces no comparison timing for them.
Changed-input graph checks now explicitly request every output cell as well.

627 passes symmetric M1/N1024 at4.926 vs5.699 us (13.6% lower latency), with a
passing range and50 registers/no stack. Select it instead of458 to add margin.
A9 prunes633 for156 bytes of stack spilling. M1's628–630 ladder loses to625.
631 fails full-output accuracy (relative L2=.906); preserve the failure and
exclude its M8-decomposition sibling632 pending a separate numeric investigation.
The remaining M16/large-M cases were not run by that aborted process and need
a fresh continuation. A9 library:
6b41ef53d7f8842c5dc55c2763a369ecb621f61518e8095c9c40c81f06c03c27.

A9 continuation passes7 full-output/full-graph rows.634 reaches62.123 vs70.618 us
at M512/N2048 (12.0% lower latency), passing the range gate;635 also passes
at63.272 us. Select634. This leaves three small-M asymmetric anchors.
All12 generic minimum-K smoke rows pass full output and changed-input graph;
their Humming configurations use Stream-K, so they carry correctness-only
records. This diagnostic continuation accidentally supplied the Python path
as QUACTLIZE_PPU_CACHE_DIR; it used prebuilt affine C ABI calls with no QUactlize
JIT/cache access. Preserve that environment record and use the proper task
cache directory for subsequent acceptance.

A10 caches each integer-zero bias across its two K16 consumers, preserving
the existing zero address/layout rules and integer-before-scale arithmetic.
636–638 isolate zero-bias reuse on625/624/626;639 applies it to634.640 is the
generic exact probe.641–643 then add scale reuse independently;644 is their
generic exact probe. The cache-disabled type remains two half2 values, avoiding
new register storage in retained controls. Inspect native instructions/registers
and test the part0→part1 invariant rather than assuming source-level reuse wins.

A10 compiles, but its small-M optimizations are machine-code no-ops:
636/641 are byte-identical to625,637/642 to624, and638/643 to626. All retain
30 scalar word LDS operations and their original register counts. Exact
instruction-byte hashes are in
`tuning/affine-kpack-asym-r10/cache-body-comparison.json` in the workspace.
Do not run another timing campaign for those identical bodies.639 reduces
large-M registers242→240 but is not needed to replace the admitted634.

A11 tests a concrete remaining register boundary:625 has66 registers, just
above64.645 requests four blocks through the existing launch-bound mechanism.
646/647 reuse its proven M8 instruction mapping for M12/M16 with two output
row tiles;648/649 then add the bound.650/651 apply the bound to624/626.
Inspect spilling before promotion; this does not lower arithmetic precision.

The affine benchmark now supports --auto --full-matrix and --caches hot cold,
using the original seven N/K shapes and eighteen M values. Cold mode requires
at least256 MiB per backend, traverses complete rings with the existing paired
event/ABBA measurement utility, and validates every output of every layer in
eager and changed-input graph execution. Distinct column permutations of a
seeded base matrix provide distinct device allocations and independently
checked code/metadata layouts; complete CPU FP32 references are permuted by
the same verified column identity. No helpers enter measured calls. The first
protocol smoke is running at `results/affine-rotation-protocol-r0` on card1
with the previously validated diagnostic module; this is not wheel acceptance.

The rotation smoke passes full eager/changed-graph output checks for all148
layers. Each arm has269156352 bytes of active weights/metadata (256.69 MiB),
with non-overlapping allocations. At asymmetric M1/N1024 the native default590
takes6.262 us versus6.605 us Humming:5.2% lower latency, below the10% target.
Do not transfer hot admission to this rotation condition.

A11 compiles and its eight eligible rows pass full output and graph checks.
645 reaches62 registers with no stack and8.028 vs8.926 us at M1/N2048:
the median passes, the range does not. Two M8 row tiles (646–649) take15–16 us
and are rejected.650/651 spill20/24 bytes and are pruned before timing.
Use the best validated small-M candidates645/624/626 in the default selector,
without declaring them performance-admitted. A frozen source worktree will
exercise the ordinary full wheel builder while further profiling continues.

```sh
venv/bin/python src/QUactlize-kda-810e/tools/build_affine.py \
  --sdk /usr/local/PPU_SDK --output tuning/affine-kpack-sym-r3 \
  --configs 13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --jobs 4
```

Expose native integer-zero arithmetic separately, extend the tested candidate
inventory and selector, wire the module into the public library/wheel build,
and repeat the paired comparison through that installed package. Retain the
original installed Humming and QUactlize comparison packages. All graph runs
must use preallocated buffers and include algorithm-required work.

## Wheel R0 and A12 continuation

Frozen source: `src/QUactlize-affine-wheel-r0` in the comparison workspace.
The ordinary build.sh --wheel workflow completed:
`wheels/affine-public-r0/quactlize-0.2.1-py3-none-linux_x86_64.whl`.
It was installed with --no-deps into `install-affine-public-r0`, retaining the
original installed comparison packages. Library() from that installation
resolves `quactlize/_native/libncp_quactlize.so`, exposes34 native configs,
and selects627/550/396 for the sampled symmetric anchors and590/624/634
for asymmetric ones. The install receipt and hash are in
`results/affine-public-wheel-r0-install.json`.

Host R2 passes1367/1367 with no skips. The initial full accuracy run on card1
at `results/affine-product-accuracy-r0` was deliberately interrupted after
successful individual checks to release the paired-measurement device.
Its operator-note.json records this; it is not a suite pass. Card0 was
verified idle and the complete suite is restarted there under
`results/affine-product-accuracy-r1-card0`, reusing validated JIT cache entries.
Card1 retains installed affine tests and all paired performance measurements.
Do not claim full accuracy before the replacement summary. The wheel source
snapshot remains frozen while the primary tuning worktree continues.

Fresh ACU reports:
`results/affine-asym-r11-m12-profile.acurep` (cache-control none) and
`results/affine-asym-n1024-forcedcold-profile.acurep` (cache-control all).
Both have raw CSVs and complete-output validation receipts. M12/624 executes
847136 instructions versus Humming933576, with76/96 registers and similar
shared-load conflicts. Its IALU-busy stalls/issue remain.263 versus.168;
TFU WAR is.0184 versus.000345. These are issue ratios, not wall-time fractions.
At forced-cold N1024/M1,590 uses1024 threads versus512, executes522512 versus
477592 instructions and has TSM-dependency stalls/issue.447 versus.157.
Its forced-cold profiler duration is NOT the ordinary rotation result.

A12 compiles successfully, with no spilling in652–664.652 ports the bounded
WK128 body645 to N1024.653 changes only590's warp-N width16→32;654 then enables
direct output.655 adds paired zero bias to590.656/657 prune dead M1 rows in
590/645;658 prunes M12 rows in624.659–661 use a flattened N-fast grid padded
to40 for645/624/626;662–664 change its padding multiple to32. Existing grid
mapping handles these cases; no new layout formula or arithmetic is introduced.
Controls590/624/626/645 are compiled alongside them. A12 SHA:
d52c40c7831c06f534db1959b5ecda99f358aedb9de32f64dbc5f71b26f67e66.
Native specialization/workspace tests pass. Card1 device measurements follow
installed public affine validation; numeric/timing acceptance remains pending.

For the full matrix, the affine harness now computes the complete CPU FP32
reference and absolute-dot denominator once for each fixed input/weight pair.
Negated-input replay uses exact sign symmetry. Every output is still checked;
no timed call, tolerance, sample count, or operator boundary changes. This
avoids repeatedly recomputing the same large CPU GEMM. Validate this harness
revision on the next device cohort before launching full-matrix acceptance.

A12 passes17 complete-output/changed-graph rows with the cached oracle.
The explicit installed public ABI suite passes15 tests on card1;686 unrelated
tests are deselected, not counted as passes. The 6 workspace harness tests and
known-answer cached-oracle/negative-control checks also pass.
Pruning and flattened grids do not close the three N2048 range gates.
At N1024/M1,655's paired bias gives5.267 versus6.189 us and passes the hot
range gate; rotation acceptance is pending. The narrower thread inventory
653 uses512 rather than1024 threads, but takes5.602 versus6.187 us and fails
the range gate. No thread-count benefit is inferred. ISA records are in
`tuning/affine-kpack-asym-r12/instruction-bodies.json`:655 has1301 static
instructions versus590's1329.659/662,660/663,661/664 share their instruction
bytes and differ only in launch padding.

A12 mapping: the source computes `group=warp_k*(WK/32)+step/2`,
`column=warp_n*WN+na*16+lane/4`, with paired columns `column` and `column+8`.
Native B's opaque shared cube uses coordinates `(kk%BCubeK)/4,(column%64)*2`,
where `kk=((warp_k*WK)/64+chunk)*64`. Each swizzled B read returns four32-bit
words/lane; its K64/N16 source footprint is512 bytes. Doubling WN adds one N16
atom and halves warp-N workers, retaining the CTA's unique packed-B footprint.
Scale pair words and quartet zero words retain their existing indices.
All stage copies are16-byte vectors or existing aligned opaque cubes.
For M1, pruning retains only one of eight output rows in the shared FP32
partials; reduction ownership/order and global outputs are unchanged.
These are source request models; the ACU traffic remains the measured evidence.

A13 targets integer issue work.665/666/667 enable direct zero-word indexing
on624/645/626;668/669/670 then change only bias construction.671 applies the
same change to the already validated generic569 for the exhaustive probe.
For quartet word `w`, mask `w & 0x0f0f0f0f` and `(w>>4) & 0x0f0f0f0f`,
then PRMT selects byte `g%4` into both low halfword bytes and0x64 into both
high bytes. This gives exactly the FP16 encoding of1024+z, retaining the
existing subtraction and scale rounding. No metadata bytes, B/A ownership,
grid or reducer change. Compare native instruction counts and register/spill
results before timing; require the generic all-code/all-zero probe first.

A13 passes the exhaustive all-code/all-zero probe and nine paired rows.
669 at M1/N2048 reaches7.912 vs8.940 us,11.5% lower with a passing range gate.
668/670 reach7.997/8.001 us at M12/M16; both median gates pass, but the range
gates miss by approximately.002/.019 us. Preserve that distinction.
The integer denominator now has8/10 hot anchors with range-gate evidence.
PRMT reduces static instruction counts1924→1874 and1915→1865; register
counts76→78 stay in the same billing bucket.669 stays at64 registers/no stack.

Rotation655 passes complete checks for all148 layers but reaches only
6.079 vs6.594 us,7.8% lower.653 reaches6.420/6.613,2.9% lower and is rejected.
Do not promote either as satisfying rotation performance.

A14 extends PRMT to WK64 using a warp-relative word shift: shift the quartet
word by8*((group-kg)%4), then use kg%4 as the byte selector. This makes
the selector constant within the unrolled K loop. All indices are unsigned
and derived from nonnegative warp/group coordinates. For WK128 the shift is
zero; compare native instruction bodies with A13 before attributing changes.
672 extends655,673/674 add paired bias to602/603, and675/676 then add PRMT.
677/678 change only grouped output stores on668/670;679/680 change only
prefetched reduction;681/682 use existing disjoint reduction storage.
683 removes672's static-M restriction for the exhaustive WK64 zero probe.
No new reduction arithmetic or storage format is introduced.

A14 passes the WK64 all-code/all-zero probe and16 paired rows. N2048/M16
670 passes the range gate at7.996 vs8.933 us. M12 remains marginal:
679's serialized partial loads give8.003 vs8.928 us but still miss the range.
N1024 PRMT is not a clear hot win; keep655's earlier body as the incumbent.
The unsigned WK128 address rewrite added2–5 native instructions, so restore
the original A13 expression for WK>=128 and retain the relative selector only
for shorter warp-K spans. Verify binary identity before any renewed claim.

The installed wheel's first full-M extension (`affine-public-r0-integer-n512-hot`)
passes complete outputs/graphs for all18 M values. Twelve Humming Stream-K
cases remain outside the admitted FP32-reduction timing contract. The other
six expose new fallback performance gaps: M256/512 take29.643/29.526 us
against17.019/22.912 us; M1024 is5.5% faster and M2048/4096/8192 are slower.
These are additional gaps outside the ten representative anchors. The generic
fallback's correctness does not imply performance coverage.

A15 performs a bounded N512/3072 large-M extension of existing bodies.
684–687 specialize582/583/584/634;688 specializes621 and removes its
persistent loop.689 narrows TN128→64 and WK64→32;690 changes stages2→3.
691 uses M128/N256/K64/WK64/stages3,692 adds the existing full register
pipeline.693/694 reduce689/690's M/warp-M128→64.695 is the specialized
unchanged621 control,696 adds cycle unrolling,697 adds full register pipelining.
At M256 the original M128/N128 tile has only8 productive CTAs, compared with
Humming's M128/N64 tile's16. Preserve per-M winners: the more parallel small
M configurations need not win at M8192. Check resources, code size and complete
outputs before interpreting timings. The offline code/metadata bytes are fixed;
changed TK uses the existing group-interleave descriptor and preparation path.

## Full-M continuation, 2026-11-16

A15 passes all78 measured full-output/changed-graph cells on physical card1.
The N512/3072 winners are685 at M256 (13.930 vs16.875 us),686 at M1024
(37.769 vs43.845), and687 at M2048/4096 (62.221/121.925 vs70.672/138.412).
All four pass the full-range10% gate. M512 remains behind (25.173 vs22.829);
M8192 improves only4.9% (241.662 vs254.053). Do not extend these admissions.
692/697 were pruned before device timing for32/24 bytes of stack spilling.
The A15 DSO SHA256 is
`3de2783a0a00c8e70f27d36a65e704eafb52e3703c8b47793157df45357ab97f`.
The restored WK128 expression reproduces A13's668/669/670 instruction bytes
exactly;679's serialized reduction is a distinct body. Raw evidence lives in
`results/affine-kpack-asym-r15-n512-hot` and the A15 build directory.

Installed R0 default N1024/3072 passes correctness for all18 M values, but
only the original M1/12/16/64/512 selections pass the performance range gate.
Explicitly reusing582/584 does not repair the other medium/large-M cells.
The generic-M WK64 candidate683 passes M2/4/6/8 at5.36–5.50 us against
6.14–6.34 us, including the range gate. Those four cells still need public
default selection and final-wheel verification.

A16 continues with a finite numeric ladder:

- 698 removes655's exact-M constraint;699/700 specialize583/634 to N1024.
- 701/703 reduce582's M/warp-M64 to32/48;702/704 change only stages2 to3.
  705/706 halve warp-N32 to16, then707/708 increase warp-K32 to64.
- 709/711/712 change686's M/warp-M104 to128/112/96, dropping the M8-tail
  trait for aligned rows.710 adds scale caching to709. The host registry
  rejected an initial interleaved-A variant because686 already enables it;
  no device build or timing was produced for that duplicate.
- 713 doubles688's tile-K128 to256,714 then changes stages2 to3.
  715 changes only warp-N32 to16;716 changes only warp-M128 to64.
- 717 changes687's M/warp-M104 to128;718 adds scale caching.
  719/720 double tile-N128 to256 on687/717;721 adds persistence to719.
- 722/723/724 halve warp-K64 to32 on686/688/709. At M512/N512 this doubles
  CTA-local K workers for the20/16/16 productive CTAs, retaining FP32 partials.
  The maximum reduction storage grows to196608 bytes; query remains the
  authority for actual shared storage and supported launch dimensions.

For each CTA, warp indices still own independent `(m,n,k)` intervals with
`32*(TM/WM)*(TN/WN)*(TK/WK)` threads. Each N16 atom owns columns
`warp_n*WN + na*16 + lane/4` and that column plus8, and group
`warp_k*(WK/32)+step/2`; only warp-M contributes the existing row offset.
Paired scale words and quartet/pair zero words keep the established addresses.
Tile-K256 increases the native A/B and metadata stage sizes through the
existing descriptor; the writer still issues16-byte metadata vectors and
opaque AIU cubes with matching SWZL reads. Tile-N256 shares each A stage over
twice as many N warps. The FP32 reduction remains within the CTA, with the
existing shared allocation/launch bounds checked by native query and compile.
Inspect registers/spills and query sizes before timing. Geometry-derived
coverage and independent full outputs must both hold; resource counts alone
do not establish a performance improvement.

The paired ACU capture at M512/N512 reports256 versus254 registers and
113664 versus114432 shared bytes for Humming/686, with the same two-CTA
resource limits. Achieved active warps are20.11% versus12.94%, while TSM
load/store dependency stalls per issue are.135 versus.256. The productive
grids differ:32 Humming tiles versus20 candidate tiles, both padded to40.
This motivates the additional warp-K ladder instead of assuming a register
cap would improve residency. ACU's24.12/28.92 us are diagnostic replay times,
not replacements for the clean paired25.17/22.83 us result above.

The complete installed R0 accuracy suite on card0 exceeded its7200s deadline;
retain `affine-product-accuracy-r1-card0` as FAIL due to timeout. The cached
repeat uses a new `affine-product-accuracy-r2-card0-cached` directory and a
14400s deadline. Host1367/1367 and the15 public affine device tests remain
separate completed evidence, not substitutes for that unfinished suite.

A16 passes72 full-output/changed-graph cells.708 closes N1024/M48
(8.444 vs9.860 us);699 closes M128 (13.968 vs16.871);700 closes M1024
(63.331 vs70.701). M24/32 remain close at7.415/7.450 vs8.164/8.195.
The N512 gaps remain:715 reaches24.319 vs22.888 at M512;687 stays best at
M8192 (240.612 vs254.069). More K workers (722/724) are slower despite
passing numerics.713/714/717/718/720/723 were pruned for stack spills.
A16 SHA256:
`d99ea3ad3c9909da114e8a4d0086fce4bac81192ed34d3aa693e00e463d102d9`.
The PRMT cold control672 passes all148 layers but only improves7.4%
(6.087 vs6.573 us); retain655's earlier7.8% incumbent without cold admission.

The installed R0 integer full matrix is complete:126 correct QUactlize
eager/changed-graph cells,86 strict comparable cells and40 Humming Stream-K
exclusions. Seven default rows pass the10% range gate; that count concerns
the frozen R0 selector, not the later candidate inventory. In particular,
the original ten anchors do not cover the wider M/N/K acceptance matrix.

A17 tests finite extensions before choosing new defaults.725/726 specialize
707 for M24/32;727/728 add a stage to707/708.729 moves669 to N1024 for the
remaining rotation deficit,730 removes its M constraint at N2048, and731/732
move the medium-M A16 bodies to N2048.733/734 remove cycle unrolling from the
spilling M128 parents717/720;735/736 then reduce stages3 to2.737/738 add the
full register pipeline and then interleaved A to715.739 first halves its
tile-K128 to64,740/741 add those two pipeline changes,742 then uses stage3.
743 changes only709's direct output.744/745 narrow733's warp-N to16 and then
widen tile-N to256.746–751 make selected large-M bodies generic in N/K.
752–764 specialize existing small/medium bodies for the previously unmeasured
N16384/K3072 and N8192/K8192 domains. Exact-M restrictions remain explicit.

No mapping primitive changes in A17. A generic N/K control loses compile-time
address folding but retains the same physical group32 and K-pack contracts.
For larger K, compare looped versus unrolled instruction size and spills before
timing; a repeated compute-heavy body can trade loop overhead for instruction
delivery. Direct output uses the existing compile-time adjacency witness and
FP32 reducer. Preserve the native query, resource, complete-output and graph
gates for every expanded domain.

A18 is a separate metadata-delivery hypothesis for the remaining small-M and
rotation margins. PreparedZeroBias stores one64-bit record for each N-column
pair/group32: the original two FP16 scales and the exact FP16 encodings of
1024+z. The canonical4-bit code bytes and arithmetic remain unchanged:
the reader constructs1024+q, subtracts the prepared1024+z and multiplies by
the original FP16 scale. No scale-times-zero offset is rounded in prepare.
The new copy runs only in prepare and retains invalid-zero rejection. Run
remains a single native CUTLASS launch with preallocated bounded workspace.

This increases active storage from4.625 to5 bits/weight including metadata.
It does not expand weights to FP16. The shared reader requests an aligned
64-bit record at `group*(TN/2)+(warp_n*WN+na*16)/2+lane/4`; the four lanes
sharing a column pair share that request. Consecutive pairs are8 bytes apart,
so the eight distinct requests span64 bytes and16 ordinary shared-memory
banks. Check emitted vector loads rather than assuming that width survives.
The metadata writer uses16-byte copies (two records), preserving full
group/column coverage. Integer-zero scratch disappears from shared storage.

765–771 apply the format to655/668/670/729/707/708/700.772/773 remove exact-M
constraints for the independent WK64/WK128 probes.774–776 additionally cache
scale/bias across the two K16 consumers. Cold reporting now derives QUactlize's
active bytes from the selected encoding, keeps the common logical weight-ring
length, and requires both backends to exceed256 MiB. The extra preparation
metadata and its cache footprint must be reported, not hidden by retaining the
old equal-byte field. No output tolerance or timed-call boundary changes.
Before timing: independently decode the prepared records, reject16/255,
poison original scale/zero inputs, cover all256 code/zero pairs, and validate
changed-input replay. Keep the compact-metadata controls in the same build.

A17 passes141 comparable cells across the gap, medium, generic-large and wide-N
cohorts (28+14+6+30+63); unsupported exact-shape requests are retained separately.
732 closes M48/N2048 (12.238 vs13.678 us) and756 closes M1/N16384 (41.367
vs48.542 us). The M128 looped bodies do not improve the large-M incumbents.
The N512 gaps remain, and729's cold M1/N1024 is6.426/6.597 us (2.6% lower).
730 and743 were pruned for8 and68 stack bytes. A17 SHA256:
`51caf8e328388833c45cd8b9aa58f4166f77a25bad27d3e68b9fe745622eca21`.

A18's WK64/772, WK128/773 and cached/776 probes pass all256 code/zero
pairs, metadata inverse checks, invalid16/255 rejection, poisoned preparation
inputs and three changed one-hot graph inputs. All19 compiled bodies have
zero stack. Native765 uses52 registers/1192 instructions versus655's58/1301;
771 uses234/8675 versus700's242/9836. The merged reads are emitted as
`tsm.ld.b32x2`, not two scalar source-level assumptions.

The first hot cohort has36 PASS cells (5+19+12). Its launch accidentally
assigned the Python executable to QUACTLIZE_PPU_CACHE_DIR and omitted
QUACTLIZE_PPU_PYTHON. The private prebuilt C ABI never accesses that cache,
and the independent probes/outputs passed, but retain these as diagnostic
candidate evidence rather than a clean selector acceptance run. The shell
was stopped after the12 large-M measurement rows, before cold execution.
Use fresh result directories and the correct environment for confirmation.
In that cohort769 closes M24/32/N1024 (6.907/7.006 vs8.162/8.200 us),
770 improves M48 (8.107/9.932), and776 closes M2048 (118.537/135.215).
M12/N2048 remains below the range gate (7757.977/8.915). The large-M N1024
gaps remain M4096/8192 (776231.518/248.321 and455.173/449.007).
A18 SHA256:
`2ad9b96c570bc90058a58873bfd21fffbc659b862463a4baed446a265036682b`.

The separate correctly configured cold765 run passes all148 layers and
changed-input replay:6.029 vs6.577 us,8.33% lower. Its worst QUactlize sample
is6.043 and0.9 times Humming's best sample is5.904 us: the gap is0.139 us.
This is not a cold admission. Actual active bytes are reported separately:
QUactlize uses5 bits/weight including prepared metadata, Humming4.625.

A19 tests whether the reduced metadata live ranges make the previously
spilling M128 unrolled bodies viable.777/778 prepare717/720's metadata;
779/780 specialize them for N1024.781/782 specialize only K3072, retaining
cycle-unrolling opportunity across N.783/784 move the B producer to warp8
for765/775, beyond the eight warps doing A/metadata copies.785–790 transfer
the measured WK128 small-M bodies to wide N, with merged metadata controls.
791/792 narrow689/690's M128 to112;793–796 add merged metadata to that ladder.
797/798 enable existing direct output on771/777, retaining FP32 accumulation.

689/690 already match Humming's M128/N64/K128/WK32 geometry and are spill-free,
but measured29.578/29.511 vs22.909/22.876 us. Do not build those geometries
again as new identities.811–815 instead add unsigned indices, vector output,
full register pipeline, interleaved A and cycle unrolling one at a time;
816 adds merged metadata,817 stage3,818 changes M/warp-M to112 and819 stage3.
799–810 specialize the WK128 M8 small-M bodies for M2/4/6/8: the previous
dynamic-M730 crossed its64-register cap and spilled. Use the native resource
gate before deciding which of these hypotheses reaches device timing.

A20 separates the two costs in A18: bias reconstruction instructions versus
extra metadata bandwidth. ByteZeroWords keeps the unchanged paired FP16 scale
array and stores four integer zeros per32-bit word: `(g,n),(g,n+8),(g+1,n),
(g+1,n+8)`, for even g. Each byte is still an exact integer in[0,15].
Two PRMTs combine the selected bytes with the fixed0x64 exponent to form
1024+z, without nibble masks. A single prepare thread writes each complete
word and validates all four bytes; there are no overlapping narrow stores.

The zero word address is `(group/2)*(TN/2)+(warp_n*WN+na*16)/2+lane/4`.
Each warp has eight distinct contiguous32-bit requests, each broadcast to
four lanes. The scale address remains the independently validated paired
layout. At WK64, adjacent group32 consumers can reuse one zero word; WK32
selects the appropriate group byte by the global group parity. Native code
must decide whether that load reuse survives. The16-byte zero copies cover
eight columns and two groups, with whole group-pair copy cohorts required.
All planes retain128-byte shared alignment and the runtime's aligned packed
workspace. Actual zero storage is NK/32 bytes versus the compact NK/64.

Active storage becomes4.75 bits/weight including metadata, between compact
4.625 and merged5. It remains ordinary INT4/g32 with identical FP16 weights
after centered reconstruction.820–826 copy the A18 control geometries;
827/828 provide unrestricted-M WK64/WK128 exact probes.829 rotates the B
producer,830/831 cache metadata,832/833 use A19's N512 bodies, and834–843
cover the outstanding small-M/wide-N extensions. Timings follow the same
independent metadata decode, invalid-value, poisoned-source, known-answer
and complete-output gates. No runtime conversion or API structure changes.

A19 passes63 hot cells (32 N512,12 direct-output,19 small/wide-N), all with
the correct task environment.775 now records a passing M12/N2048 range
(7.9747/8.9322 us);799/800/801 pass M2/4/6 at N2048.803/804/805 and785/786
pass M2/4/6/12/16 at N16384/K3072. N8192/K8192 small-M remains below target.
The large-M incumbents are unchanged;797's direct output does not replace776.
The M128 unrolled S1 bodies777–782 still spill20–40 bytes;798,802,806,810,
815/816 also spill and were not timed. No identical native bodies were found
among the remaining A19 configurations. Library SHA256:
`f55b9226ec2a8602f8d04163cc3c397d0ef98f97dceea1baa84d482ef15bbc4c`.

Cold783 is6.0312/6.6578 us with a failed range gate. Its QUactlize time is
unchanged versus765's6.0291; the apparent9.4% reduction mainly reflects a
higher paired Humming denominator. Do not attribute that percentage change
to the B-producer rotation or promote it over765.

The clean paired ACU690 capture matches Humming's actual M128/N64/K128,
WM128/WN32/WK32, stage3 geometry at M512/N512. Both have20.1% achieved
active-warps and two-CTA register/shared limits. Nevertheless QUactlize
executes1885640 instructions versus1526888, has52736 versus25088 shared-load
bank conflicts and24576 versus4096 store conflicts. Tensor-active elapsed is
41.37% versus52.71%; DRAM reads are4083072 versus4073728 bytes. This points
to on-chip issue/operand/reduction delivery, not an increased DRAM denominator
or an occupancy difference for this particular matched configuration.
TSM dependency stalls/issue are.183922 versus.135476, not wall-time shares.
Replay durations32.106/24.174 us remain diagnostic only.
Artifacts: `results/affine-asym-r15-690-profile{.acurep,-raw.csv,.sha256}`.

The source inspection confirms intra-CTA Humming reduction is FP32 here.
Its installed `epilogue/smem_reducer.cuh` and `memory/s2r_loader/loader_b.cuh`
are byte-identical to the pinned source archive. Humming's regular packed
WN32 reader uses a64-bit load per K iteration (`kLoadHalfGroup`), whereas
the current native K-pack WK32 reader fetches a K64 fragment and consumes
half. This source fact does not establish that B alone causes the measured
gap: total shared-load transactions differ by only6.1%, and A/output/reducer
traffic must be separated before changing the canonical code layout.

A20 native build passes the16 host gates.823/832/837/839–842 spill and are
pruned;830 is byte-identical to821 and is not retimed.820 uses58 registers,
1262 instructions and18 scalar shared loads versus655's58/1301 and24
scalar/byte loads. The merged765 still has fewer instructions (1192).
Library SHA256:
`840b0c4c09139d824bce732f56afcd09ca2870e48b849f4e2b43bda7cbeafc5c`.
The four exact probes (827/828/831/833), small/large paired cohorts and cold820
were launched sequentially on card1. Their completion must be read from the
new result directories before any admission. The product accuracy R2 remains
active on card0; an INCOMPLETE summary during execution is not a final result.

The cached R0 product accuracy rerun is now complete:1230 passed, zero
failures/errors/skips, exit0 in5418.845 s. The package is the isolated
`install-affine-public-r0/quactlize/_native` tree, not the newer candidates.
Retain R1's7200-second timeout. The final candidate wheel still needs its
own public-path acceptance.

All four A20 exact probes pass. Its30 hot cells (15+6+9) pass complete output
and graph replay, but the new metadata encoding does not solve the gaps.
Cold820 regresses to6.2420/6.6176 us (5.68% lower); retain765 as incumbent.
Merged775 passes M12/N2048 again (7.9719/8.9384 us). Do not treat the smaller
4.75-bit metadata footprint as a measured cold improvement.

After the user's priority change, stage1 is **stable improvement on all86
strict-comparable shapes**, using `max(Q)<min(H)` before pursuing10% across
already-winning shapes. The final10% target remains stage2. Rank remaining
losses by matched latency ratio, emphasizing large M or N. The new snapshot
`results/affine-progress-r20-counts.json` uses the latest10-sample graph row
per shape/config, excludes R18's diagnostic hot cohort, and retains exact
per-shape evidence.46/86 are faster and range-separated;32/86 pass10%.
Historical10% witnesses exist for34/86, with two not sustained in the latest
measurements. The old public selector remains13/86 faster and7/86 at10%.

Largest losses now: M64/N16384/K3072 +37.1%; M128/N2048 and M256/N1024
+29.6%; M2048/N3072/K1024 +27.0%; M48/N16384 +26.7%; M4096/N3072/K1024
+25.4%; M8/N8192/K8192 +22.5%. Freeze already-winning small-M/rotation
fine tuning while addressing these losses.

The new776 large profile at M8192/N1024 is different from the690 diagnosis:
Q/H achieve24.72/23.96% active-warps and87.24/84.57% tensor-active elapsed,
but execute6795264/6291456 FP16 tensor instructions (8.0% extra). M104
needs79 tiles with seven M atoms, versus Humming's64 tiles with eight atoms.
The M8 tail instruction's physical cost is not assumed to be half of M16.
DRAM reads52.52/52.29 MB are again similar. This makes aligned M96/M112
tiles a controlled test of wasted tensor issue without the M128 spills.
Report: `results/affine-asym-r20-776-large-profile{.acurep,-raw.csv,.sha256}`.

A21 first transfers measured bodies to the worst workloads, preserving the
code/metadata contracts.844/845 change776 to aligned M112/M96;846–848 widen
TN to256.849–859 use the already validated M64/M48/M32 medium-M bodies at
N16384 and N8192, including a TK128/WK32/stage3 ladder.860–882 transfer the
aligned large-M bodies across N2048, N16384, N3072 and N8192 with exact K.
883–886 transfer the WN16 and WM64 low-grid alternatives to N1024/N2048.
887–905 test the existing M128/WM64 body at TK64, then prepare metadata,
unsigned indices and register-pipeline/unroll changes one axis at a time.
906–908 remove the64-register launch-bound cap on the compact M8/WK128
bodies which previously spilled8 bytes. The changed occupancy must be read
from native resources, not assumed beneficial. No altered source body is
promoted from compilation or a static instruction count.

A21 is native-built (74 configurations, all zero stack and supported queries),
SHA256 `956f8b9bc1831a8f244ebed0313af38d2f092061932bd660f445ca06a25d491c`.
The complete sequential cohort is in `affine-kpack-asym-r21-*` result folders.
Initial closed gaps include M8/N2048 and N16384 (908/907), M1024/N3072/K1024
(866), and M4096/8192/N16384 (864). Some wins have small margins and require
final selector confirmation. The largest M64/N16384 loss narrows from37.1%
to12.0% (85192.696/82.752 us), while M128/N2048 and M256/N1024 narrow from
29.6% to about5.7% (884/883). Full final cohort counts must come from raw JSONL.

**Resource-gate correction:** the large Humming control has
`launch__stack_size_per_thread=60`, while776 has0. The earlier blanket
zero-stack candidate pruning was too conservative. Stack size alone does not
give spill frequency or measured cost. The matched690 control still has0
on both sides, so that separate observation remains valid. Retain all earlier
PRUNED records, but revisit779–782 in `affine-kpack-asym-r19-spill-trial`
with full correctness, replay and paired timing. Do not claim a win from
occupancy changes or waive any numerical/graph gate.

A22 matches Humming's measured wide-N medium-M geometry:
M64/N16384 uses TM64/TN256/TK128/WM64/WN32/WK64/stage3; N8192/K8192 uses
TN128/WK32 instead.909/910 reach the first geometry from851 one axis at a
time. The parents already use chunked FP32 reduction and no A-register
prefetch, so those are not new hypotheses.911–923 compare cycle looping,
the existing compiler fences, merged metadata, full/interleaved register
pipeline and a two-CTA launch bound.924–929 transfer to M48/M32.
930–945 apply the corresponding N8192 ladder.946–957 test cached and
uncached merged M128/TN256 bodies across the remaining large-M losses,
explicitly allowing measured stack use.958–967 compare opaque copy
coordinates and looped M48 bodies;968–973 enable vector output.
Every changed shape/layout remains independently validated before timing.

A21 final cohort has252 PASS measurement cells and6 explicit out-of-scope
requests. Additional new stable wins: M1024/N3072/K8192 (869441.750/456.504),
M4096/8192/N8192/K8192 (8734509.48/4542.73 and8881.66/9031.56), and
M2048/8192/N2048/K3072 (860231.420/248.652 and861868.433/870.730).
Margins near zero remain provisional until final product confirmation.
The largest remaining losses are still wide-N medium M; those drive A22.

The R19 nonzero-stack revisit completes21 PASS cells and9 out-of-scope
requests. It does not replace the current incumbents. At M8192/N1024,
776 remains best454.689/449.149 us; at M8192/N16384 the best trial781 is
6949.58/6843.40, slower than A21's864. This is measured rejection of these
particular bodies, not a renewed universal ban on local stack usage.

A22 compilation is in progress with four host workers, source configs909–973
plus controls; the16 host contract gates passed. Native resource identities
and device measurements are still pending. Physical card1 is idle after the
completed R19 trial; card0's R0 accuracy1230/1230 is complete. Keep the original
R0 public binaries unchanged while selecting and validating the next product
configuration set.

Latest snapshot `results/affine-progress-r21-counts.json` (including the R19
stack revisit):58/86 faster medians,56/86 with `max(Q)<min(H)`,33/86 with
the10% range gate. The two median-only wins are M24/M32 at N8192/K8192.

A22 builds all72 requested configurations (the logged `compiled` field is
the work item index, not a completion count; no compiler errors).
SHA256 `3152c65131ac21c09fb09317f54b94325d64963eff950cd4c765190d6341aff3`.
Its68 medium-M measurement cells pass full output and changed-input replay,
but none replaces a loss with a range-separated win. The two-CTA bound
helps M64:922 is89.253/82.695 us at N16384,940 is143.521/131.234 at N8192,
still slower. Opaque-coordinate pairs910/958,914/959,919/960,930/961,
934/962,938/963 have identical instruction bytes; do not repeatedly time
them as separate hypotheses. Full resources and stack sizes are saved in
`tuning/affine-kpack-asym-r22/resource-summary.json`.

The measured M32/WK32 cliff is particularly clear:944 uses130 registers
and about150 us at N8192;859 uses108 and about80 us. Both use512 threads,
so the power-of-two register bill changes from256 to128. These bodies also
differ in tile/stages/K work, so the full latency ratio is not attributed
solely to the register threshold. A23 tests that axis directly:
974–986 add the two-CTA bound to M48/M32 parents, including looped controls;
987–994 use stage2 and independently add the bound;995–1006 specialize
actual M=24/32/48 before independently bounding registers. Allow measured
local stack and retain all correctness, replay and timing gates.

A22 large-M trials add46 PASS cells.956 closes M128/N8192/K8192 at
169.784/182.585 us. Other M128 bodies do not replace the existing large-M
incumbents; nonzero stack remains a measured tradeoff, not a promotion signal.

A23 builds49 configs, SHA256
`23f42ac72550d023f90c2cc4384857d70ad8f4efd645b1aa109be3c9e5b844ce`.
Its44 applicable measurement cells pass full output/replay;4 explicit
M-specialization requests are out of scope. The two-CTA bound can change
code scheduling without spilling:925's142 registers become974's124 with
zero stack. At M48/N16384,1005 reaches70.975/67.822 us, reducing the loss
from26.7% to4.65%. M48/N8192 remains15.8% slower with764 as incumbent.
M32/WK32 bodies improve under the128-register limit but remain slower than
the K64-reader859, so the occupancy cliff was not the only cost.

A24 preserves the public ABI and metadata representations. It first tests
TN64 for the remaining equal-area rectangles (M512/N512, M256/N1024,
M128/N2048), changing the16-CTA grid to32 CTAs.1007–1009 are geometry-only;
1010/1011 add unsigned indices and prepared metadata separately.
1015/1016 bound the earlier M48/TK256/WK64 bodies to two CTAs, keeping the
native reader's K64 granularity;1017–1020 add metadata preparation/exact M.

The new optional `InlineReduction` stays in the CUTLASS kernel extension.
Each volatile asm block loads one aligned FP32 shared vector and performs
four `add.rn.f32` operations immediately. This tests temporary liveness
without the per-M-fragment barriers of chunk reduction. It is not a fused
hardware instruction and does not change accumulation precision. Whole-tile
partial writes and the existing read/write barriers remain.1012–1014 and
1021–1034 compare it against existing reducers; validate a changed body
beside1005 before building the complete cohort. The manifest and harness
record the new flag. Workspace may change from chunk to whole-tile partials
and is read from the compiled query.

1035–1070 separately test existing persistent-MN scheduling with40/80 workers
on the M104/M112/M128 large-M bodies across the outstanding N/K families.
The default public selector is not changed by these experiments. Resource
counts, exact probes, full-output/replay and paired device timing are still
required; no candidate is admitted from a source-level liveness argument.

A24 host contracts pass16/16 after adding the new inline flag to the existing
registry identity tuple (the first host attempt correctly caught the stale
identity schema). The first exact probe used fixed-M48 config1024 against
the probe's M384 matrix and was rejected at query; retain that unsuccessful
invocation. Generic1012 then passes all256 code/zero pairs, group boundaries,
poisoned prepare sources and changed-input replay at M384/N512/K3072.
Smoke full-output/replay passes five cells.1011 and1012 are byte-identical
device bodies and have the same launch contract; do not attribute their
22.72/22.85 us rectangle result to inline asm. Geometry/metadata changed.
1005/1024 remain about71.1/67.9 us on M48/N16384.

The full A24 library has78 configurations, SHA256
`f454ee2fd3c27372b75221647912649073a5180144f134ed03730257ac85b6c8`.
`resource-summary.json` records each new config's native registers, stack,
instruction digest and actual query footprint. The persistent40/80 pairs
share device instruction bytes, **but their host launch grids differ**;
they are distinct experiments.1017/1018 reach118 registers with no stack
using M48/WN32/WK64.1022 instead has128 registers/8-byte stack. The complete
sequential A24 device run is active on card1 in `affine-kpack-asym-r24-*`
folders, from the wide-M48 gaps through rectangles and persistent large-M
anchors. No final A24 aggregate exists yet.

The paired764 wide profile at M48/N8192/K8192 confirms a different limiting
factor from the earlier690 example: H/Q have identical786432 FP16 tensor
instructions and39.596/39.640 MB DRAM reads, but4,278,272/7,426,048 shared
load transactions. Load conflict counts are262144/267520, nearly equal.
Q's active-warps fraction is49.03% versus41.92%; tensor-active elapsed is
45.25% versus53.43%. H uses128 registers and8-byte stack; Q96 and0, both
register-limited to two CTAs. Thus further occupancy alone cannot explain
or close this gap. WN16 doubles A-fragment reading relative to WN32 for
equivalent math; test the native K64/WN32 bodies rather than assuming the
entire74% excess is A traffic. Report:
`results/affine-asym-r23-764-wide-profile{.acurep,-raw.csv,.sha256}`.

A diagnostic native reader probe lives at `tuning/probe_native_b32.cu`;
artifacts are in `tuning/native-b32-probe-r0/`. The known m16n16.x1 swzl
transposed control compiles to `full.so`. The attempted
`ppu.ldmatrix.sync.aligned.m8n8.x2.swzl.trans.shared.b16` is explicitly
rejected as unsupported on ppu0010 by the native compiler; there is no
`half.so` and no device result for that form. This only rejects that tested
mnemonic, not every possible partial-reader design. It is a coordinate
diagnostic, never an alternative standalone operator/performance result.

A24 interim new range-separated wins: M48/N8192/K8192 config1020 is
90.627/99.256 us (the earlier15.8% loss becomes8.7% lower latency);
M256/N1024 config1013 is22.658/22.922; M8192/N1024 config1039 is
434.336/449.660; M1024/N16384 config1045 is848.167/870.733.
The persistent M128/40-worker bodies warrant transfer across the remaining
large-M gaps after this cohort completes. M48/N16384 still loses at
71.008/68.010 with1005. The N512/M512 and N2048/M128 rectangle medians
are now slightly faster but latest sample ranges overlap; do not mark
them stage1-complete. The N8192/M24 previous tiny range win also did not
persist in the latest control run. Full results and aggregate are pending.

A24 is complete. The new `tuning/count_affine_progress.py` independently
reproduces the R21 counts58/56/33 before admitting the new runs. It filters
integer/g32/hot/graph, ten samples, complete finite validated outputs and
shipping Humming FP32 accumulation with no Stream-K. It retains the latest
record per shape/config/split and excludes the R18 diagnostic cohort.
`results/affine-progress-r24-counts.json` gives77/86 faster medians,
74/86 with `max(Q)<min(H)`,33/86 at the10% range gate. The range condition
is within each paired run; final independent product remeasurement remains.
Historical range wins are75/86, distinct from the current74/86.

The persistent M128/40-worker transfer passed full output/replay throughout
and closed nine more large-M cells:1045 wins all remaining M128/256/512/2048
at N16384;1051 wins M256/2048 at N8192/K8192;1063 wins M4096/8192 at
N3072/K8192;1069 wins M1024/N2048. Other config1007 also supplies the
N512/M512 range win, so that rectangle is now frozen with the other winners.

A25 targets only the remaining twelve losses.1071–1080 test persistent
40/80-worker scheduling on wide-N M24/32/48/64 controls.1081–1090 extend
the measured K64/WN32 M48 body to M64, independently testing looped code,
the register bound and persistence.1091–1093 test exact32-block or persistent
launches for M128/N2048 (its existing padded grid is40, not80).1094–1105
test group32 scale/zero reuse on those medium-M bodies, with exact-M
specializations for24/32.1106–1115 test TK128/WK128/stage2 on the K1024
large-M loss, independently changing metadata caching, register pipeline,
M tile and unrolling.1116–1124 target the sole remaining M8/N8192 loss
with persistence, metadata reuse/preparation and looped controls.
All candidates retain the K-pack codes, FP16 I/O, FP32 accumulators/reduction
and existing public ABI. Resource inspection precedes device timing; there
is no selector promotion or change to Humming.

A25's first build correctly rejects1112/1113: the M128-to-M104 geometry
mutation omitted the existing `TailM8` feature. Set that feature for the
104-row bodies instead of weakening the native assertion, and extend the
existing host registry check so this omission fails before device compilation.
Keep the failed `affine-kpack-asym-r25` directory/log; the corrected build
uses a new `affine-kpack-asym-r25a` directory.

The paired922 profile is an actual same-tile control at M64/N16384:
both512 threads,128 registers,two resident CTAs and786432 tensor instructions.
Q has76-byte stack against H's8, despite fewer total instructions
(7,545,040 versus7,771,048). Shared reads are almost equal
(4,030,464 versus4,024,320), as are load-conflict counts. Q DRAM reads are
31.919 MB versus29.506 MB, consistent with its larger prepared metadata.
Q tensor-active elapsed is55.89% versus60.23%; active-warps fraction is
49.76% versus42.19%. This case needs separate treatment from764's excess
A-reading diagnosis: examine spill traffic, metadata reuse and the persistent
grid. Report is `results/affine-asym-r25-922-wide-profile{.acurep,-raw.csv}`.

A25a builds all64 requested configurations, SHA256
`5bad76926235da03d05839c7e0eb28e72d329eb538d213f29c4cf7c9a462d9a5`.
Resources are in its `resource-summary.json`. Persistent medium-M unrolled
bodies are rejected before timing when they explode register liveness:
1071 has588-byte stack,1075 has844,1085 has1124. By contrast, looped1086
has20.1077 uses236 registers versus928's114; the register bucket doubles.
This is not the behavior of the admitted S1 persistent large-M kernels.
906/1117,1116/1118 and1119/1120 have identical instruction bytes; do not
claim a metadata-cache benefit from those pairs.

The38 selected A25 cells and four large-M transfers pass complete outputs
and changed-input replay.1098 closes M64/N8192/K8192 at112.072/132.352 us.
1091 closes M128/N2048 at22.640/22.833.1107 reaches121.685/121.641 at
M2048/N3072/K1024 (still a loss), but closes M4096 and8192 at239.914/241.022
and467.194/472.814. The updated snapshot gives80/86 median wins,78/86
range wins and34/86 at the10% range gate. Historical range wins79 are
reported separately. Eight remaining gaps are M64/M48/M24/M32 at N16384,
M8/M24/M32 at N8192/K8192, and M2048/N3072/K1024.

A26 freezes the new winners.1125–1134 test looped persistent medium-M
bodies instead of the spilling unrolled versions.1135–1144 target M8's
64-register boundary: prepared1121 uses66 registers with no stack; disable
register prefetch or bound registers before widening its N tile.1145–1147
test exact-M and raster order on the remaining K1024 tie; the already
winning M4096/8192 versions are not retuned.

1148–1178 test whole-K warps on the wide-N medium-M gaps. Keep the
native K64 reader and FP32 accumulation, but use TK64/WK64 so each CTA
has no K-cohort reduction. There is ample N parallelism to test this
independently of the small-N need for K cohorts. Sweep TN128/256,
stage2/3 and register bounds, with explicit loop scheduling controls.
Admit only compiled-query-valid resources and paired correct device data;
neither reduced barrier count nor a nominal register bound is a result.

A26 builds62 configurations, SHA256
`7d8c508bebf135ec911e8572d0ef7559ccb745782e01925669984d275ada7d6b`.
The66 selected cells all pass full output and changed-input replay.
The looped persistent1129 closes M24/M32 at N8192/K8192 at78.537/80.307
and78.653/80.386 us. The S1 bodies close all four N16384 gaps:
1167 gives51.796/57.688 (M24) and52.003/57.813 (M32);
1159 gives60.832/67.750 (M48);1154 gives74.380/82.658 (M64).
1141 closes M8/N8192/K8192 at55.129/63.618.1147's exact-M/raster variant
closes M2048/N3072/K1024 at120.359/121.746.

`results/affine-progress-r26-counts.json` now records86/86 median wins,
86/86 range wins and36/86 at the10% range gate. These are latest records
per config across cohorts, not yet a simultaneous product-build acceptance.
Freeze its64 distinct selected configurations (65 N/K/config batches), build
them together in `tuning/affine-kpack-asym-r26-selected`, and run
`tuning/revalidate_affine.py` against that exact DSO and frozen JSON.
The runner requires all86 complete PASS records, verifies the requested
selection and an unchanged library hash, and exits nonzero if any paired
sample ranges overlap. Preserve all results even if a tiny margin fails.
Only after this clean full-map check should the final selector be updated.

The paired1107 K1024 profile supports a near-parity diagnosis, not a new
performance admission: both have zero stack, two CTAs,40 grid blocks,
256 threads and1,572,864 tensor instructions. H/Q registers are244/248,
active warps24.28/24.34%,tensor-active elapsed78.24/77.85%. Q executes
8,460,392 instructions versus8,937,576, with7,274,496 versus7,176,192 shared
read transactions. The observed profile durations125.061/126.591 us must
not replace ordinary graph timing. Current future options for the10% phase
include the existing `SkipStage2Barrier` on FPR/stage2 whole-K bodies;
it has not been enabled or admitted for1107.

The selected64-config library builds successfully with SHA256
`233db7a1767046f235d0052e859cb465805aaf693bfcce7f1336b30b51f35f9b`.
The frozen-map repeat is running sequentially on card1 with prefix
`affine-kpack-asym-r26-revalidate`; its driver log is
`results/affine-kpack-asym-r26-revalidate.log`, and completion writes
`results/affine-kpack-asym-r26-revalidate-summary.json`.
The source selector remains V1 pending this clean repeat. Do not combine
an older winning sample with a new losing repeat to claim acceptance.
If the old tiny-margin873/864 large-M winners fail, test the already compiled
1051/1045 persistent M128 bodies on those specific shapes rather than
extrapolating from their lower-M wins.

The frozen-map run finishes with86 correct outputs,86 faster medians,
83 range wins and30 at the10% range gate; its driver exits1 as intended.
Do not promote the selector yet. Failures are1013 at M256/N1024
(median ratio0.9909,range1.1276),625 at M1/N2048
(0.9276,range2.3982), and864 at M4096/N16384
(0.9929,range1.00246). The first two have timing spikes; retain their
records and seek independent paired confirmation rather than discarding
samples. Repeat those controls, and compare864/1045 at M4096/8192,N16384
using the same selected DSO. Prefixes are
`affine-kpack-asym-r26-confirm-{n1024-m256,n2048-m1,n16384-large}`.

The confirmations retain all failures.625 now has range ratio0.90790.
1045 improves M4096/8192,N16384 to3355.50/3449.18 and6671.42/6851.58 us,
with range ratios0.97390/0.97480; use it instead of the tiny-margin864.
1013 still fails the range gate at M256/N1024: median22.810/22.977 us,
range ratio1.03757. Do not resolve this by choosing an earlier winning run.

A27 targets that last rectangle.1179–1181 test the exact32-block grid
and exact-M specialization.1182–1189 split the M128 block into two WM64
warp groups, then test exact grid/M, the register bound, prefetch and
chunk reduction independently. This trades some B reuse for more active
warps without changing the logical INT4 data or FP32 reduction.

A27 rejects1189 at compile time because it inherited `InlineReduction`
while enabling the incompatible chunk reducer. Clear the former flag,
retain the native assertion, and add the mutual-exclusion check to the
existing host registry test. Preserve the failed directory; rebuild in
`affine-kpack-asym-r27a`.

A27a builds with SHA256
`94bcb3388276763250afea0fbe1340b6fab2d66063d54e0f7ed81354aec9cd17`.
All12 selected cases pass complete output/replay. WM64 lowers registers
from162 to120 and doubles threads256-to512, but is slower at23.8–24.5 us;
the extra B conversion is not repaid by the occupancy increase.1181's
exact32-block grid and M256 specialization instead gives22.557/22.867 us
with range ratio0.99115. Keep the original two failed1013 repeats.

Freeze `results/affine-phase1-r27-selection.json`: start with the R26 map,
replace M256/N1024 with1181, and M4096/8192,N16384 with1045. This is a
configuration proposal, not a recomputed winning statistic. Build its63
configurations together in `tuning/affine-kpack-asym-r27-selected`, then
repeat the entire fixed map before selector promotion.

R27's selected DSO SHA256 is
`3c2cd22cc1114072c67341c8f686bf7d2125583c951a3647d31428bf8c313dc7`.
Its fresh full-map repeat gives86 correct,86 faster medians,84 range wins
and33 at10%. The improved1181/1045 cells hold. Remaining failures are:
683 at M4/N1024 (median5.372/6.168 us,range1.28282) and1107 at
M4096/N3072/K1024 (239.740/241.072 us,range1.00237).

Do not retune the M4 kernel to chase an unexplained timing spike.
Nine Q samples are5.353–5.386 us; one is7.824. That sample's wall average
is9.672 us rather than the usual5.6, so capture an Asys timeline of the
existing64-call graph measurement to distinguish kernel time from launch
and event gaps. Current profiling is diagnostic only, with unchanged
timing code, under prefix `affine-asym-r28-timing-asys`.
Do not infer the cause from wall/event differences alone.

A28 addresses the real small-margin K1024 cell:1190/1191 test exact-M4096
and M-fast raster,1192–1194 use the existing FPR stage2 barrier-elision
path with the matching controls. Its next-stage barrier protects current
A-register refills, and it suppresses unused future copies at the range
tail. Full output/replay remains mandatory; no precision change is allowed.

The first A28 timing invocation accidentally set `QUACTLIZE_PPU_CACHE_DIR`
to the Python executable path. Preserve
`affine-kpack-asym-r28-n3072-k1024` as diagnostic, exclude it from the
progress counter, and repeat with the correct task cache under a new
`affine-kpack-asym-r28-n3072-k1024-clean` prefix. Native execution did not
need JIT, but this environment record is not suitable for promotion.

The Asys export has1082 device activities and625 runtime activities;
CPU scheduling sampling is unavailable because Ftrace is unavailable.
The inspected tables are `HGPTI_ACTIVITY_KIND_KERNEL` and
`HGPTI_ACTIVITY_KIND_RUNTIME`, with `StringIds` resolving kernel/API names.
Graph launches correlate to21 target-kernel activities each in this capture.
This capture does not reproduce the spike, so it cannot establish its cause.
Use a separate event-placement capability probe before changing any timing
protocol: a known host delay should affect events around replay but not
events recorded inside the graph. It carries no performance admission.

The controlled placement probe passes: injecting5 ms of host delay between
the outer start event and graph submission gives5.230 ms outside the graph
and0.148 ms inside, while all64 recorded increments execute correctly.
This demonstrates sensitivity of the outer boundary; it does not identify
the cause of every historical spike.

Add an opt-in `--timing-events graph` to the task affine harness, leaving
the original workspace benchmark and default outer-event protocol intact.
Record start/end GPU event nodes inside each captured graph, and retain
outer-event and wall times from the same replay as separate fields.
Both arms use the same repetition count, calibrated from GPU graph time
toward5 ms, bounded at1024 complete calls/rings. Short-ring calibration uses
16 repetitions to amortize event-node overhead. Fixed `--graph-cycles` is
diagnostic only. The progress counter explicitly separates the two protocols.

The fixed64-call native M4/N1024 diagnostic passes full output/replay for
50 ABBA rounds: Q's100 inner samples are5.288–5.331 us (median5.303),
H's are6.031–6.426 (median6.123); the same replay's outer medians are
5.397/6.217. The GPU-calibrated probe also passes. Neither diagnostic is
silently substituted into the original outer-event full-map result.

The clean A28 K1024 repeat passes all6 cells.1194 gives235.686/241.139 us
at M4096, range ratio0.97897. Update only that cell in the frozen map,
build the64 configurations together, and perform a fresh full-map check
with captured events. Preserve both original full-map failures and all
outer-event values in the new result.

The R28 selected DSO hash is
`7e8dcb1a74a22cbd618310edd196a313236d5b95050aee657fc5d32bec1c0d49`.
Its captured-event full repeat gives86 correct,86 faster medians,84 range
wins and38 at10%. The two failures move to M6/N1024 (683) and M4/N2048
(800). Both arms slow in the same late ABBA window: for M6,N1024 the
last H sample is15.391 us versus the usual6.078, and Q is6.932 versus5.296.
Captured events therefore do not eliminate all variation. Do not attribute
every spike to the outer event boundary or mark the complete range gate passed.

Prepare candidate V2 automatic selection from the frozen R28 table so a
single-process public-path run can replace the per-config subprocess sweep.
This is provisional integration, not final performance promotion; the policy
keeps `performance_admitted=false`. All86 matrix cells already pass complete
output, metadata, guards and changed-input replay. Compile/test the actual
generated selector, build a fresh public wheel in isolation, and validate
its default dispatch. Keep the earlier full-map failures as part of evidence.

The V2 table matches all86 entries of the frozen R28 proposal and compiles87
total configurations including symmetric and generic coverage. The generated
C++ selector test passes for both the complete and a filtered compilation;
the affine host suite is17/17 and `git diff --check` passes. A test initially
treated specialization `(0,0)` as literal dimensions; correct it to the
runtime's documented wildcard semantics instead of changing a valid config.

The fresh normal public-wheel build is running through
`build.sh --wheel --sdk /usr/local/PPU_SDK --jobs 6`, with work directory
`build-affine-public-r1`, wheel directory `wheels/affine-public-r1`, and log
`results/affine-public-r1-build.log`. Do not edit hashed library sources while
that build is running. Next validate the isolated wheel, host suite, public
affine tests and one-process `--auto --full-matrix --zero-mode integer
--timing-events graph` run on card1. This should exercise126 cells, of which
86 are directly comparable and40 retain the Humming precision exclusion.
Keep the installed original baseline and Humming unchanged.

The public R1 wheel build succeeds. Wheel SHA256:
`65f64f9be8d5fef7a41f9f73c6b86745ca29784b52b4554ffe219d9f103fee8a`;
installed public DSO:
`e5d93a22e6ac17fe5924fbbc629261bd958ef7d6c930a146def9bc0c5ed7e724`.
It is installed only under `tuning/affine-public-r1-installed`. The workspace
harness CPU tests pass6/6 and Python compilation passes. Direct execution of
the non-executable `run_tests.sh` initially returns126; retain those logs.
The original host/accuracy runners have been restarted with `bash` and fresh
`affine-public-r1-host-run` / `affine-public-r1-accuracy-run` directories.

**Phase1 public-path result:** the single-process installed-wheel default
selector run, `results/affine-public-r1-auto-graph/summary.json`, records
86/86 complete-output/replay passes,86/86 median wins and86/86 range wins.
38/86 pass the10% range threshold. All40 non-comparable Humming precision
cases remain excluded. The binding source is the isolated installed wheel,
not the implementation checkout. This is the captured-event hot protocol;
retain the earlier outer-event and per-config-process failures independently.
Proceed to phase2, prioritizing large-M/large-N cells farthest from10%;
rotation and final performance admission remain open.

A29 phase2 hypotheses, keeping the R1 wheel frozen:

-1195 extends1194's exact-M/raster/barrier combination to M8192,K1024.
-1196–1207 transfer1107's K128/S2 FPR geometry to N2048/K3072,
N3072/K8192,N8192/K8192,N16384/K3072, with separate barrier-elision and
M-fast steps. For WM128/WN32, the existing K64/S3 controls are already
register-limited to two CTAs; test whether fewer stage transitions help
without crossing a resource limit. Do not infer the final register count.
-1208–1211 transfer1141's no-register-prefetch path to M1/2/4/6,N8192.
-1212 tests a24-row tile with an M8 tail at M24,N16384 rather than computing
32 rows. Keep the native K64 reader and exact integer-zero/FP32 contract.

Build the known-good controls beside these18 candidates, inspect native
resources, and require full output/changed-input replay and clean captured
event comparisons. No candidate is part of the default selector yet.

A29 compiles with SHA256
`8f604b65d10832c4883895dfb24442e2aaf016668e246e18649a3ba6170e9293`.
All queries succeed. Large K128 variants use248–256 registers;1202/1203
have24-byte stacks,1204 has none. Their measured comparisons, including the
K64 controls, are in progress on card1.1212 uses90 registers without stack.
The public R1 host suite completes with1368/1368 passes and zero skips;
the separate card0 accuracy run remains active.

A30 independently targets the near-tied M512/N512 cell. The actual shipping
Humming configuration is block128x64x128,warp128x32x32,stages3,FP32/noStreamK.
Our1011 has the same block but warp128x16x64.1213/1214 test M112/M104 tiles,
which yield40 rather than32 CTAs;1215/1216 add inline FP32 partial addition.
1217/1218 use warp-N32/K32 with and without inline addition, then1219 adds
stage3.1220/1221 cache the paired scale/bias metadata;1222/1223 unroll cycles.
These are extensions of the existing collective and generated config list.
Read reduction storage from the implementation: it stores only KWarps-1
peer accumulators, so do not use a hypothetical four-full-partial footprint
to reject the K4 candidates. Native resource reports decide admissibility.

All56 valid A29 cells pass complete output/replay. The24 static-M query
rejections in the small sweep are expected and unmeasured. K128/S2 does not
beat the K64/S3 large-K controls. At M8192,N8192,1204 removes the stack but
regresses to9727 us, while the N-fast1051 control is8465/9029 us. Do not
attribute that reversal to register count alone; collect memory/issue
counters for the actual variants before changing the tile scheduler.

Useful A29 candidates:1208–1211 at M1/2/4/6,N8192 have range ratios
0.8670/0.8729/0.8762/0.8774.1212 at M24,N16384 gives49.982/57.556 us,
range0.86925. The small no-prefetch variants retain8-byte stacks; their
measured benefit is not an inferred occupancy improvement.1195 improves
M8192,N3072,K1024 to range0.97608. Fresh controls also favor1051 over873
for M4096/8192,N8192 and1069 over861 for M8192,N2048.
These are candidate records; the frozen installed R1 result remains38/86
at10%. Extend the counter with explicit additional raw baselines and binary
identity fields so the public R1 baseline can seed phase2 without mixing
outer-event records or claiming a new installed-selector result.

A30's14 cells pass complete output/replay. M112 gives20.626/22.768 us at
M512/N512, near but still above the10% threshold; the M104 and warp-N32/K32
alternatives are worse. No A30 cell is admitted at10%.

Paired A31 ACU captures compare1051,1202,1204 with shipping Humming at
M8192/N8192/K8192. Each contains one H and one Q kernel, same fixtures,
`--set full --profile-from-start off --cache-control none --clock-control none`.
The SDK describes `none` as clearing L1/L2; these are profiler diagnostics,
not hot timing admission. CSVs use per-kernel rows and named metrics.
1051 reads358.7 MB,1202 reads366.4 MB,1204 reads4196.8 MB from DRAM.
Their shared-load transactions are identical at606076928, but changing to
M-fast causes over11x DRAM reads despite removing the stack. H reads278–281 MB.
The tensor-pipe-active metric is94.86/91.08/83.76% for the three Q variants;
it is not by itself a FLOP-throughput measurement.

A31 therefore adds a small grouped-M raster to the existing CUTLASS driver,
preserving N locality while testing B reuse. The actual coordinate helper
has host coverage/uniqueness checks including incomplete groups, with known
ordering anchors.1224–1235 cover group sizes2/4/8 on the large-N controls,
including general and exact-M rows. Device tail checks remain required.
1236–1239 separately test paired-metadata reuse, unrolling and output on
the near-threshold M112/N512 candidate. The R1 product wheel stays frozen.

The first A31 edit placed its appended config block before its prerequisites;
the import-time count assertion caught it before native compilation. Move
the block after A30 without changing any existing ID. The17 host contract
tests and diff checks pass. A31 builds21 configurations with SHA256
`6baab206a47170dc5e6bb7e9018a94e30a20939c3ca8b0e1d23ed6a7b2448e93`.
General grouped variants have8–32-byte stacks; the exact-M8192/4096 variants
1228/1229 have none.1236–1239 use148–150 registers without a stack.

A matched1203 ACU capture closes the earlier two-variable comparison:
1203/1204 share barrier elision and differ only in raster order. Their DRAM
reads are369.1 MB versus4196.8 MB. The analysis is recorded in
`results/affine-asym-r31-analysis/REPORT.md` with exact metrics and scope.

A31 paired runs are now queued sequentially on card1 under
`affine-kpack-asym-r31-n8192-k8192`, `-n16384-k3072`, `-n3072-k8192`,
`-n512-k3072`, and `-group-tail`. The final batch checks M513/M641 with
grouped configs1224–1227. Inspect the actual Humming precision flags for
these added tails; do not infer Stream-K selection from another N/K pair.
They remain outside the fixed86-shape denominator.
The current candidate counter is `affine-phase2-r29-counts.json`:86 range
wins,42 at10%; the frozen installed R1 result remains38 at10%.

Possible next investigation, not implemented: large M8192/N3072/K8192 has
768 full128x256 output tiles distributed across40 persistent CTAs, giving
19 or20 tiles per worker. Check whether the existing FP32 Stream-K scheduler
can improve this work balance. Read actual CU overlap rather than treating
per-worker imbalance as a proven elapsed-time floor. The current stream-K
driver always calls `run_range`, even for whole tiles; a full-tile branch
could preserve the statically unrolled mainloop while retaining the dynamic
range path for boundary tiles. Any such change needs a controlled flag,
actual schedule coverage, numeric/graph checks, bounded workspace, and
paired comparison with the frozen control. No FP16 cross-CTA reduction.

A31 completes:43 valid cells pass full output/replay, including all8 added
M513/M641 tails. Humming actually uses the comparable FP32/noStreamK path
at these N8192 tails; an earlier expectation of a precision mismatch was
incorrect. Two exact-M query rejections are expected. All grouped variants
lose to the N-fast controls.1051 remains best at M4096/8192,N8192, with
range ratios0.94071/0.93939.1045 and1063 remain best at the other large-N
cells.1236 at M512/N512 is20.598/22.801 us, range0.90442, still short.
`results/affine-phase2-r31-counts.json` remains86 range wins and42 at10%.
Do not promote the grouped variants or claim another installed-wheel pass.

A32 design refinement before implementation: ordinary Stream-K assigns
contiguous K-work intervals to workers, so adjacent CTAs can start far apart
in M. That can recreate the measured activation-cache problem even with an
N-fast tile map. Compare the existing full-domain scheduler, a whole-tile
fast-path variant, and a hybrid that retains ordinary N-fast persistent work
for a prefix divisible by worker count, applying Stream-K only to the last
`workers + tiles%workers` tiles. This suffix is at least one wave, preserving
the existing at-most-two-contributors admission and bounded scratch.
For2048 tiles/40 workers, the hybrid prefix is2000 tiles and the suffix48.
Use the existing scheduler and completion protocol; no new precision mode.
The proposed compile-time mode must keep the old behavior as its default.
Validate prefix/suffix coverage and actual boundary completion before timing.

A32 implements modes0/1/2 (original/full-tile-fast/hybrid-tail) and configs
1240–1255. Host tests call the actual suffix scheduler and verify prefix plus
suffix coverage, complete-tile recognition, and contributor limits;17 pass.
The20-object library builds as
`fafa9cb3d4517acd8872bf0fd782e1e8060c9901825a076cccb785a1cb51cecd`.
Actual queries show every Stream-K variant needs135168 bytes of shared
memory because the FP32 completion staging array is128x(256+8).
This permits one CTA per CU despite the two-CTA register budget. A32's clean
device validation is in progress; the first invocation was aborted because
its compiler-interpreter environment variable accidentally held a cache
directory. Preserve/exclude that invocation, and use the `-clean` runs.
The harness now validates the interpreter/cache path types before creating
the result directory or selecting a device, and records those settings.
The deliberate invalid-interpreter probe fails at the new preflight.

A33's1256–1271 change only that FP32 staging padding to zero. Canonical
global partial layout, FP32 arithmetic, fences and counters stay the same;
normal FP16 output staging retains its original padding. The expected
shared size is128 KiB, but require native query/resource verification.
Extra shared-bank conflicts on the rare boundary tiles may be a worthwhile
cost for two-CTA occupancy; do not claim the tradeoff without timing.
1272–1274 test N-fast raster and exact-M/unrolling on the near-threshold
M112/N512 candidate.1275 transfers the validated M24-tail tile to N8192.

A33 builds29 objects with SHA256
`4a9affb7414e90694980c8990d1688117ea48fa10397a8813f38cb3961570cb0`.
Native queries confirm131072 shared bytes for1256–1271, while unchanged
padding8 controls still report135168. Register billing remains256 and some
variants spill; reducing shared storage alone is not performance admission.

A34 targets the mainloop directly: no integer-zero candidate in the existing
catalog has warp64x64.1276–1283 change warp128x32 to64x64 or32x128 while
keeping block128x256,256 threads, K partitioning, stages and total MMA work
fixed. This reduces repeated shared A loads but duplicates B conversion
across M warps.1284/1285 apply64x64 to the K128/S2 controls;1286/1287 add
exact M8192. Inspect generated resources and actual shared/issue counters
if any variant improves; geometry alone cannot prove the predicted tradeoff.

A32's clean30 valid cells pass full output/replay; none beats its incumbent.
A33's37 valid cells also pass, including the boundary-tail checks. Compact
FP32 staging restores the two-CTA resource budget but does not produce a
large-matrix10% win.1264 reaches3278.77/3318.99 us at M8192/N3072/K8192,
still only a narrow improvement and with a larger prepared workspace.
The added N-fast M112 candidate1272 does pass at M512/N512:
20.4609/22.7942 us, range0.89847. The updated candidate counter
`affine-phase2-r33-counts.json` is86 range wins and43 at10%.
No default selector or public wheel has changed.

A34 builds18 objects with SHA256
`888906b63fbc33c16f634ea7aeba13d58323687d821804bbc1e144ccacf62456`.
1276/1278 use246/248 registers without stack;1286 likewise uses246.
1281/1283/1285 have132/628/160-byte stacks and are not advanced to timing.
Other tested variants remain subject to full numerical and paired gates.
Initial large-N measurements reject widened warp N:1276 is10389.9 us versus
the1051 control around8480 us at M8192/N8192/K8192. Reduced A duplication
does not compensate for this mapping's additional B conversion/work.

The installed R1 whole-product accuracy run hits its7200-second limit amid
first-time JIT compilation. Its summary is FAIL with no final counts; do not
count the partial log as a pass. Preserve `affine-public-r1-accuracy-run`.
A fresh card0 run uses the populated cache at
`affine-public-r1-accuracy-cached-run`; card1 performance work continues.

Possible A35 direction, not implemented: keep warp128x32 and change the
block from128x256 to128x512 or256x256. This doubles the CTA's warps while
retaining converter amortization per warp, unlike A34. It can share A or B
delivery between work previously in separate CTAs. Use the actual native
resource report;512 threads at the256-register billing level allow one CTA
per CU, so inspect/choose launch-bounds and a20-worker persistent grid.
Expect larger shared storage and possible spill; neither variant is admitted.
For the near-threshold M512/N512 winner, direct output or vector output
without additional cycle unrolling is a separate small experiment.

A35 implemented that comparison in 1288–1299. All 40 measured cells pass
complete output and changed-input graph replay. Larger CTAs lose: at
M8192/N8192/K8192, 1288/1289 take 9596.1/10501.9 us against the 8474.0 us
1051 control. The M512/N512 direct/grouped-store variant 1299 does improve:
20.2889/22.7908 us, range ratio 0.89127. The candidate count remains 43/86
at 10%; the frozen R1 wheel is still 38/86.

A36 first measures the existing compact FP32 Stream-K candidates at
intermediate M, where wave imbalance could matter more. All 45 cells in
`affine-kpack-asym-r36-medium-*` pass; none beats its ordinary control.

A36 also adds mode 3: an ordinary persistent whole-K prefix kernel followed
by a separate FP32 Stream-K tail kernel. Both are CUTLASS drivers behind the
same query/prepare/run entry; run includes both launches and performs no
allocation or conversion. Preparation shares the packed metadata, initializes
the existing bounded completion workspace, and configures both kernels.
An empty prefix skips its launch. The schedule coverage gate now checks
independent prefix grids as well as tail ownership and complete K coverage.
Host contract tests: 17 passed. Build SHA256:
`b5c90c82a4a865881c15fcd8b26add7f9f35f5efc905c38a328b056c606ba422`.
1300–1303 contain two actual tensor-core entrypoints; their ordinary-prefix
register/stack reports match the respective 1051/1063/1045/1107 controls.
The public query reports the maximum shared-memory requirement of both
entries. Device validation and paired timing are in progress under
`affine-kpack-asym-r36-separate-*`. These variants are not selected publicly.

Prepare a V3/R2 product candidate while preserving the installed R1 receipts.
Eleven exact integer-zero rows change: 1208–1211 for M1/2/4/6,N8192;
1299 for M512,N512; 1051 for M128/512/4096/8192,N8192; 1069 for
M8192,N2048; and 1195 for M8192,N3072,K1024. The first five add candidate
10% coverage; the other six increase existing win margins. The selector
remains explicitly unadmitted until the new installed wheel completes its
own full paired matrix. Freeze hashed sources during the normal wheel build.

R2/V3 now completes that installed public gate. Library SHA256:
`63fd63e52f9134d21eec66670c7fb3ed5e22bca4694546e95f9b48e763a1334c`.
The normal wheel is in `wheels/affine-public-r2`, isolated installation in
`tuning/affine-public-r2-installed`; it contains 84 native configurations.
Host CI passes 1368/1368 with zero skips in `affine-public-r2-host-run`.
The single-process `affine-public-r2-auto-graph` run exits zero:
126 Q full-output/changed-input-replay checks pass; 86 comparable shapes
all win their paired ranges, and 45/86 pass the 10% range threshold.
Its immutable `summary.json` checks all output coverage and both replay
validations, ten samples/backend, auto selection and installed binding.
Compared with R1, the five changed configurations add five threshold wins;
unchanged M16/N2048 and M48/N16384 also cross the boundary in this repeat.
Do not attribute those two boundary results to new kernel changes.
The original R1 installed receipts remain intact.

A36 separate-launch experiments finish all 26 cells correctly. They do not
beat the incumbents. At M8192/N8192/K8192, 1300 takes 8489.3 us versus
1051's 8468.1 us; at N3072/K8192, 1301 takes 3297.4 us versus 1063's
3291.7 and 1264's 3278.2. Preserve these negative results. R2 selects none
of the new Stream-K modes.

A37 candidates 1304/1305 remove register prefetch from the M12/M16,N8192
controls 758/759. 1306 transfers the validated persistent 1069 recipe to
N512/K3072. 1307/1308 halve the M tile and warp-M of 1091/1181 to64;
1309/1310 additionally remove register prefetch. These narrower-M cases
use 110/104 registers and 43008 bytes shared versus the 162-register,
75776-byte controls, crossing the register billing boundary.
The small-M variants retain 78 registers. All queries and 17 host
contract tests pass. Build SHA256:
`c5c5f2247293d9a0573c5a2fa7230ea3dbcb3976fb585ac7f9ee1e3156dd21f8`.
Card1 is testing `affine-kpack-asym-r37-{n512-k3072,n2048-k3072,
n1024-k3072,n8192-k8192,tail}`. These remain candidates outside R2.

Short-K paired ACU captures at M8192/N3072/K1024 are preserved as
`affine-asym-r37-{1195,1294}-shortk-profile*`, including digests, raw CSVs
and full validation snapshots. Each contains exactly one H and one Q
kernel. With full metrics, kernel replay, cache-control none and clock-control
none, H/1195/1294 show roughly 475.5/464.9/495.9 us; these are diagnostics.
H reads18.74 MB and Q reads19.03/18.83 MB; both Q variants have exactly
29097984 shared-load transactions and 6291456 tensor instructions.
Actual active warps stay near25%. Enlarging the CTA from256 to512 threads
increases the reported TSM-pipe-busy stalled-warps-per-issue ratio from
0.024817 to0.162566 while sync falls from0.132468 to0.125512.
Do not blame the regression on higher synchronization counts or DRAM volume.
The reported FP16 operation rates are 64239/65779/61320 per cycle against
81920 peak, respectively. Shared-load scheduling is a more useful lead.

A38 design before implementation (implemented below): for short K, overlap the
next persistent MN tile's initial input copies with the current tile's last
K iterations. Inspect the actual collective run/copy-stage code first.
A conservative experiment would require direct register-to-global output,
WarpK1, no global completion, and K/TK divisible by Stages, so the prefetched
stage indices align and shared output staging cannot overwrite new inputs.
The first tile primes normally; only valid successor tiles may be copied;
each new launch must reset this local pipeline state. Keep a direct-output
control and derive/check the actual copy schedule on the host before PPU
measurement. This is a hypothesis about repeated pipeline startup, not an
established bottleneck or an admitted configuration.

R1's fresh cached whole-product accuracy run remains active on card0 at the
last check, after more than 3300 seconds. Its earlier 7200-second run is
still FAIL. R2 whole-product accuracy and >=256 MiB rotation are pending;
neither the host pass nor the 126-shape affine run closes those gates.

A37 finishes 31 numeric cells successfully, including M129 tails. The M64
tile experiment crosses the register billing boundary but does not improve
the medium-M controls; converter amortization still matters. Neither N512
alternative replaces687 at M8192. Removing register prefetch from the M12
and M16,N8192/K8192 kernels does work: 1304/1305 give 56.043/56.303 us
against63.612/63.573 us, with range ratios0.88981/0.89254. These add two
candidate threshold wins beyond the R2 map (installed R2 remains45/86).
Registers stay78, so do not attribute this gain to an occupancy change.

A38 implements the MN continuation design with the existing stage buffers,
FP32 accumulator, CUTLASS driver, and public preparation/run contract.
`PersistentPrefetch` requires a persistent whole-K, warp-K1, direct-output
driver. The register pipeline maps its final copies to the next MN tile and
skips that tile's already-issued prime. For stage2, an explicit last-read
barrier protects the successor copy at the current tile's tail. No shared
output staging may overwrite those pending inputs. Query admits only whole
stage cycles for this experiment. The actual host/device copy-target helper
is checked for exact K coverage, stage-slot identity, first/last tiles,
partial worker waves, and negative alignment cases in the existing schedule
test. All17 host tests pass.

1311/1312 compare direct output and continuation against static-M1195;
1313/1314 do the same against1045; 1315/1316 use dynamic-M1107 for tail
validation. All nine requested real collective instantiations compile with
zero errors. Library SHA256:
`62780c49c92526f8343388e35f18501c3ac3fa9164d1294a04e3a1015b647fd3`.
Resource receipts: `tuning/affine-kpack-asym-r38/resource-summary.json`.
1195/1311/1312 use248/254/254 registers, no stack, and106496 shared bytes;
1045/1313/1314 use256/256/252 registers,20/16/0 stack bytes, and79872 shared.
A real compiler negative control disables direct output in1312: the intended
driver assertion fires, four compiler errors, nonzero exit, no stale object.
Its emitted source, compiler command and log are under the same build path.
GPU validation/timing has NOT run yet; card1 is occupied by R2 rotation.

R1 cached whole-product accuracy subsequently completes, exiting1 after
5239 seconds:1228 pass,2 fail,0 skipped. Both failures occur before GPU work:
the specialization/workspace tests interpret an explicit `STATIC_M=0`
(dynamic M) as a request with M=0. Fix the fixture to choose a positive
dynamic M, reject changed M only for actual fixed-M specializations, and
positively check changed M on dynamic whole-K recipes. The two failed
tests then pass in `affine-public-r1-accuracy-fixture-retest`; retain the
original full FAIL receipt. R2's full1230-test accuracy suite is now running
on card0 in `affine-public-r2-accuracy-run`, reusing `cache/quact-affine-kda`.

R2's complete >=256 MiB per-backend rotation run is active on card1 in
`affine-public-r2-rotation-graph`, with the frozen installed R2 library and
its captured-event harness snapshot. At the last check35 cells were complete
and all won their ranges,5 by10%; this is PARTIAL data, not full acceptance.
Validation checks every layer and every output cell, including changed
graph inputs. The larger output-permutation oracle is expensive on the CPU.
For subsequent runs, replace NumPy's advanced column indexing with
`np.take(...,axis=1)` to produce contiguous references without changing values
or tolerances. `results/affine-rotation-oracle-contiguity.json` verifies bitwise
equality for both signs and records about4.1–4.5x faster gathers on two CPU
matrices. This later harness change does not alter the already-running R2
rotation process or its saved snapshot.

Current live handoff: R2 rotation is process690227 / shell `8cfd08` on
physical card1; its last observed inventory is41 PASS plus12 precision-
excluded correctness rows (53/126 total). R2 whole accuracy is shell
`3d47dd` on card0, still running. R38 shell `2519ee` waits for process690227
to exit, checks card1 is idle, then runs dynamic-M numeric/replay checks
at M128/129/512/4097,N3072,K1024 using1107/1315/1316. A full-output and
two-replay gate checks all12 rows (allowing explicitly excluded H cohorts)
before permitting six M8192,N3072,K1024 paired configurations and the
1045/1313/1314 stage3 comparison at M128/512/8192,N16384,K3072.
Do not start another card1 job while either rotation or that queued job
owns the card. Inspect their logs/results before changing the next experiment.

The workspace CPU harness suite passes6/6 and all four benchmark modules
compile. `affine-phase2-r37-counts.json`, including the installed R2 raw
matrix, now reports86/86 candidate range wins and47/86 candidate10% wins.
The authoritative installed R2 count remains45/86. After R38, evaluate its
actual numeric/resource/performance evidence, then consider a new wheel
with1304/1305 and any independently winning R38 rows. No R3/V4 selector,
wheel, or admission exists yet. Retain the R1 FAIL and two-test PASS
receipts separately; the new full R2 accuracy run is the next whole-suite
acceptance gate.

Installed-Humming source audit for the format discussion: its
`transform.py:398-424` applies the PPU-specific N permutation before
offline packing; `config/mma.py:331-342` emits `ppu.mma`; the B global loader
uses AIU, while the packed B shared loader reads the layout through its
vector path. Thus do not attribute an obligatory NVIDIA-to-PPU runtime
conversion to the baseline. Both implementations still form dequantized
FP16 operands for this MMA contract. Static layout adaptation can be
handled in transform/prepare; the reader must match. These references were
read from the installed wheel because checkout `transform.py` differs.
The inspected MMA and A/B loader headers match the installed files.

R2 rotation reaches106/126 correct cells (80 comparable,26 excluded).
Among those80,76 win ranges and17 meet10%. Four measured losses are
M24/M32,N1024 (824), M256,N1024 (1181), and M128,N2048 (1091), all K3072.
The worst two range ratios are1.06055/1.04784. This does not invalidate the
separately declared hot result, but rotation has not reached all-win acceptance.
824 uses byte zero words (4.75 bits/weight), NOT prepared FP16 zero biases;
do not attribute its regression to a5-bit format.

A39 adds a bounded ladder:1317 widens824's TK to512,1318 adds a third stage,
1319 restores packed4-bit zero metadata,1320 adds a third stage to1317.
1321/1322 test stage3/no-register-prefetch on1091;1323/1324 do the same on1181.
Compile/resource gates precede GPU use. Test hot and >=256 MiB rotation
independently, preserving824/1091/1181 as paired controls; no selector change
is admitted merely from the metadata size or an occupancy calculation.

A39 builds all13 requested configurations with zero compiler errors;17
host contract tests pass. Library SHA256:
`b9992ae860feb038b66e72caf74a5934fb917ad6c3eb86a23b41700539743a9a`.
The resource receipt is `tuning/affine-kpack-asym-r39/resource-summary.json`.
All13 query successfully, without stack spill. 1317/1320 have1024 threads,
80/86 registers and104448/156672 shared bytes;1318 has512 threads,90
registers and78336 shared;1319 has512 threads,84 registers and51712 shared.
1321/1323 have256 threads,170 registers and113664 shared;1322/1324 have256
threads,158 registers and75776 shared. Controls1091/1181 have162 registers.
These are build gates only, not measured gains.

Correction to the compacted handoff's R37 resource attribution:244 registers
and79872 shared bytes belong to Humming. Actual QUactlize1195 is
TM128/TN256/TK128,stage2,248 registers,106496 shared bytes,256 threads,
grid40. 1294 is TN512,246 registers,147456 shared bytes,512 threads,grid20.
The raw CSV identifies each kernel unambiguously; the earlier detailed
ACU paragraph remains valid. Both reports record1.7 GHz and81920 peak
FP16 operations/cycle; do not conflate tensor-active percent with achieved
operation rate.

Latest running-state checkpoint: R2 rotation process690227 is on its final
M8192/N3072/K1024 cell;125/126 rows are written (85 comparable PASS and40
excluded correctness rows). Keep waiting for full completion rather than
counting partial coverage. `tuning/audit_affine_rotation.py` now verifies the
complete126-row installed cohort against the frozen R2 hot reference:
same library/configurations, minimum256 MiB per backend, every layer/cell,
both replay signs, five paired rounds/ten samples, and independent range
counts. Its incomplete-cohort negative control rejects the current125 rows.
When rotation exits, run it with `--run results/affine-public-r2-rotation-graph
--hot-reference results/affine-public-r2-auto-graph --output
results/affine-public-r2-rotation-graph/summary.json`. No final rotation
summary has been written yet.

R38 shell2519ee still owns the next card1 slot after rotation; its first
12-cell correctness/replay gate permits Humming precision exclusions only
when QUactlize has full correct output and both replay checks. Inspect that
job before starting A39. A39 rotation requires ONE explicit configuration
per process (`--caches hot cold --cold-bytes268435456 --splits1`, with
properly separated CLI arguments). Suggested groups:
824/1317/1318/1319/1320 at M24/32,N1024,K3072;
1091/1321/1322 at M128,N2048,K3072;
1181/1323/1324 at M256,N1024,K3072.
Also validate1304 at M12 and1305 at M16,N8192,K8192 in rotation before
promoting those two hot winners. Every configuration needs a new result
directory and its own paired H baseline. R2 whole accuracy remains active
on card0 (shell3d47dd, process713598), last observed elapsed2850 seconds.
No R3 selector or wheel has been built.

R2 rotation FINISHES with exit0. The complete audit now resides in
`results/affine-public-r2-rotation-graph/summary.json`:126/126 QUactlize
outputs and changed-input replays pass;86 comparable,40 precision exclusions;
82/86 median/range wins,17/86 range10% wins. The four losses remain the
M24/M32,N1024;M256,N1024;M128,N2048 cases above. Every backend ring is at
least269156352 bytes. The frozen library SHA is63fd63..., harness7e91af...,
raw456b787f3f6ca49afa1f7e3ec3908b9ccfe18ff7110d88b99d6486beff9b9e41.
This closes rotation correctness/coverage, NOT its performance target.
The auditor's positive full-cohort run passes after its earlier125-row
negative control. R38 shell2519ee has consequently started on card1; the
first observed M4097 control1107 is correct but slower than Humming
(240.09 vs238.81 us). Wait for remaining direct/prefetch rows and the gate.
R2 accuracy shell3d47dd is still active on card0, elapsed3900 seconds.

R38 completes:12 correctness/replay gate cells (9 H precision exclusions
with full Q validation,3 comparable), plus6 short-K and9 wide-N timing cells;
all numeric checks pass. No promotion: M8192/N3072/K1024 control1195 is
460.34us versus H471.94; direct1311 is488.49 and continuation1312 is486.41.
Dynamic direct1315/continuation1316 are489.26/490.38. N16384/K3072
control1045 is132.75/434.48/6673.60us at M128/512/8192; direct1313 is
133.25/437.62/6695.18 and continuation1314 is134.03/439.84/6726.26.
Eliminating stack in1314 does not yield a win. Preserve disabled candidates.

Card1 verified idle, then A39 starts as shell289955. It sequentially runs
all13 configurations in new `results/affine-kpack-asym-r39-<NxK>-c<ID>/`
directories, each with hot and >=256 MiB rotation. Order:824,1317-1320
at M24/32,N1024;1181,1323,1324 at M256,N1024;1091,1321,1322 at M128,N2048;
1304 at M12 and1305 at M16,N8192/K8192. K is3072 except the last two.
Per-process limit2400 seconds; each result must have the expected row count
and all PASS before the next configuration. Performance misses remain
recorded and do not abort the ladder. All cache/Python paths and card1 are
explicit. R2 product accuracy continues separately on card0.

A39 completes all36 cells with full-output/changed-input replay correctness.
1320 preserves the byte-zero K-pack layout, raises TK256→512 and stages2→3.
At M24/M32,N1024,K3072 its hot ranges are0.8634/0.8666 and rotation ranges
0.8203/0.8295 (Q7.193/7.206us, H8.803/8.736us in rotation). Compact zeros
alone1319 do not fix the loss and regress hot timing. Do not blame these two
losses on an inherent layout/byte-volume limit: the unchanged layout now wins.
1324 at M256,N1024 and1322 at M128,N2048 remove register prefetch; hot ranges
0.9860/0.9837 and rotation ranges0.9943/0.9940. Stage3 variants1323/1321 win
rotation but narrowly lose hot ranges, so they are NOT promoted.
1304/1305 revalidate the M12/M16,N8192/K8192 hot10% wins; rotation improves
to0.9421/0.9455 but does not meet10%. Preserve the old824 M32 hot timing spike
(range1.3823); it is not discarded from the diagnostic run.

V4/R3 selector preparation: replace exactly six integer-mode shape entries
with1320 (two shapes),1324,1322,1304,1305. Retain the five superseded public
configuration IDs, bringing the build to89 entries. ABI and prepared layouts
remain compatible. No new10% product claim until the rebuilt installed public
path is retested. R2 remains the authoritative product receipt.

R3 wheel builds and is isolated at `tuning/affine-public-r3-installed`.
Public SHA256 `70cbc2377255d76cf4b19e2385f2f10560e5a7bc06b93024fc9146a4324516f4`.
The ABI receipt confirms all84 R2 configuration descriptors, all70 public C
exports and the mapping ID remain;89 entries total and six new default
selections match V4. Original host CI passes1368/1368 without skips.

Initial R3 hot run has126 correct Q shapes and86/86 median wins, but only
83/86 range wins and38/86 range10% (median10% is47/86). Preserve this result
in `affine-public-r3-auto-graph`. The three range losses are M12/M16,N1024
and M6,N2048, all K3072. Their usual Q timings agree with R2; outliers include
M12's5.332→7.903us and M6's7.888→9.292us. Several other small-M outliers lose
the10% threshold; two large-N boundary points move just over0.900.
Do not describe the R3 range gate as passed or discard any samples.
R3's full rotation is active in shellb8c512. Source/DSO do not change during
this run. A controlled full hot confirmation after the current card0 accuracy
and card1 rotation finish will distinguish reproducible kernel differences
from timing disturbance; no causal claim about CPU/JIT interference yet.
The not-yet-started R3 accuracy waiter3d7ed6 was cancelled to reserve that
window. After confirmation, launch R3 whole accuracy on card0 using the
installed package (`tuning/affine-public-r3-installed/quactlize/_native`),
the same test inventory and a finite14400s timeout. R2 accuracy3d47dd is still
active (CI PID713595, pytest713598), last observed around101 minutes.

A40 design/implementation, outside the frozen R3 build: seven stage2→3
controls based on the measured A39 pipeline result, NOT a global policy.
1325/1326/1327 derive829/683/838 (TM8,WN16,WK64,TK512; M1 / M2,4,6 / M8,
N1024,K3072).1328/1329 derive675/676 (TM16,WN16,WK64,TK512; M12/M16,
N1024,K3072).1330/1331 derive1304/1305 (TM16,WN16,WK128,TK512; M12/M16,
N8192,K8192). Retain all seven controls in the new diagnostic library.
Only stage count changes: warp-N4 and warp-K8/4, the canonical reader's lane
ownership, vector widths, unique bytes, AIU cube alignment, metadata layout,
grid and FP32 reduction ownership are inherited unchanged. A storage uses
the production `AStorageM` alignment (16 even for TM8); B remains64*512/2
bytes per stage. Metadata follows the actual byte-zero/packed-zero control,
not an assumed common representation. The additional shared stage must be
checked with compiled query/ISA resources. K has6 or16 K-tiles; the latter
tests the non-multiple-of3 tail in the existing generic pipeline. No
continuation, decoder or offline-layout change is included. Compile and
resource gates first; GPU tests wait for public validation/confirmation.

A40 builds all14 controls/candidates with zero compiler errors,17 host
contract tests pass. Library:
`5c0ea17ea6ab3436e3f2d6173af22ade7259139c66805a752ac79d72074bd175`.
All14 compiled queries pass and no stack is reported; resources are saved in
`tuning/affine-kpack-asym-r40/resource-summary.json`. Candidate1325/1326/1327
use62/64/62 registers and1024 threads, preserving the <=64 register tier;
shared bytes107520/105984/107520.1328/1329 use70 registers,1024 threads,
105984 shared bytes;1330/1331 use92 registers,512 threads,105984 shared.
Their parents are respectively58/60/58/66/66/78/78 registers. Extra shared
staging stays within262144 bytes; these measurements are not perf admission.
GPU validation has NOT started. The intended per-configuration hot+rotation
matrix is nine shapes, both controls and candidates (36 measurement cells).

Live-state checkpoint: R3 public hot/rotation shellb8c512 is in rotation,
last observed89/126 rows (63 comparable PASS,26 excluded). R3 host14d93e
has completed1368/1368. R2 whole accuracy3d47dd remains active at~105 minutes;
its original7200s timeout has not been changed. R3 accuracy has NOT started.
After R2 accuracy and R3 rotation end, first perform a fresh full R3 hot run
with card0 idle (`affine-public-r3-auto-quiet`, same paired/graph contract,
no filtering or overwrites). Record device-idle evidence outside the new
benchmark directory before launch. Then start R3 installed-package accuracy
on card0 with14400s, and A40 validation/measurements on card1. Inspect all
final receipts, including any R2 timeout; do not infer them from elapsed time.

The quiet confirmation is now queued as shell7798b0: wait for CI713595 and
rotation pytest881303 to exit, assert both physical cards have no compute
process, write `results/affine-public-r3-quiet-devices.json`, then run the
unchanged installed R3 public full-matrix hot/graph harness into fresh
`results/affine-public-r3-auto-quiet`. It does NOT yet generate a summary or
start the following accuracy/A40 jobs. Audit its complete raw cohort after
exit. Do not launch other card1 jobs while this queue/run is pending.

R2 accuracy3d47dd ends with exit1: authoritative summary is FAIL,
`phase exceeded 7200s`, elapsed7200.221s. The late log was still compiling
the public Q8 scale-first path. Preserve summary/JUnit/log; no completed
test count is inferred from progress dots. This is distinct from the old
baseline's1215-pass cached receipt and the R1 fixture repair. Card0 is now
idle. R3 rotation still has125/126 rows (85 comparable PASS,40 excluded);
the quiet waiter7798b0 is therefore still reserved behind its final row.

R3 rotation finishes and the complete audit passes correctness:126 Q
shapes, all changed-input graph checks;86 comparable/40 excluded. Median
wins86/86, range wins82/86, range10%15/86. Minimum runtime ring269156352
bytes per backend; no HBM-cold claim. The four old regressions now pass the
installed range gate: M24/M32,N1024 ratios0.8278/0.8272; M256,N1024 ratio
0.99890; M128,N2048 ratio0.99647. The four current range losses are
M2/M4/M12,N1024 and M4,N2048. E.g. M4,N1024 Q median6.092us versus H6.576,
but Q max11.890 and H max15.398us; M12 Q median6.376 but max13.218.
Both-side slow samples are retained; no assumed cause and no global pass.
Summary/raw/env/harness hashes are in the new rotation summary.

Quiet confirmation7798b0 starts after both cards are verified empty, recorded
at2026-11-17T02:58:48.285377+00:00 in the separate idle receipt. Card0 must
remain idle until it ends. R3 whole accuracy and A40 GPU work remain pending.

Quiet hot confirmation completes:126 Q shapes correct,86/86 median wins,
85/86 range wins and44/86 range10%. One outlier remains at M32,N2048,K3072
(731, range1.2211). Thus activity on card0 is NOT necessary for these
outliers; do not attribute them all to concurrent accuracy. R3 still lacks
a complete range admission. Keep both full hot cohorts and all cold data.
R3 installed whole accuracy starts on card0 as shell4fda04 with14400s.
A40's36-cell control/candidate ladder starts on card1 as shell2e0fab;
each process checks device availability and must finish numeric/replay
validation before the next configuration. No other card1 workload may overlap.

A41 design: investigate greater M reuse per warp for the large-M/large-N
residual. Existing WM64/WN64 tests reduced A traffic but lost badly to extra
B conversion (1276:10389.9us versus~8480); do not repeat that known result.
The native collective already admits WM256/WN16, which keeps128 FP32
accumulators per thread, doubles M reuse of each decoded B fragment and
increases A shared-load demand. This is a tradeoff to measure, not a predicted
speedup. No decoder, offline bytes, metadata or reduction arithmetic changes.

Bounded A41 inventory: two incumbent NK parents1051 (8192/8192) and1045
(16384/3072). For each, TM256/TN128 and TM256/TN256 with WM256/WN16,TK=WK64,
stage3, under full-register pipeline / chunk-A / one-fragment A prefetch.
Use40 workers for256-thread TN128 and20 for512-thread TN256; min-blocks1
for the512-thread body. Retain both original controls plus a chunk-A control
with original TM128/TN256/WM128/WN32. This gives14 new configurations and
two controls. Source's `AStorageM`/`ACubes` supplies the two128-row A cubes;
each physical output worker owns the existing m16n16 fragment map, with
16 M atoms and one N atom instead of8 and2. K ownership and code/scale/zero
indices are inherited. The reader's warp-M was inspected and is ALREADY
bounded by `%WarpM`; no speculative index fix is justified.

Inspect compiled registers, stack and actual shared bytes before GPU use.
The TN256 output staging includes256*(256+8)*2=135168 bytes, so one-CTA
occupancy is explicit, not overlooked. Prune spilled/unlaunchable bodies
before timing. First validate a non-divisible M such as257, full output and
changed-input graph replay; measure large M separately and preserve the
actual H precision exclusions. A40 must finish before any A41 GPU work.

A40 finishes all36 cells with full-output/replay PASS. At N1024,K3072,
1325/1327/1328/1329 retain hot10% and newly pass rotation10% for M1/8/12/16:
rotation medians5.524/5.614/5.826/5.837us versus H6.604/6.604/6.651/6.654,
range ratios0.8383/0.8538/0.8824/0.8808.1326 passes rotation at M2/4/6
(ranges~0.877–0.879), but loses their hot10% threshold (~0.910–0.917),
so do not replace683 with it.1330/1331 regress the N8192/K8192 control:
hot~62us versus~56us, rotation also slower. No blanket stage3 policy.

A41 compiles all16 entries with zero errors;17 host contract tests pass.
SHA256 `5b1228ae566880df2001934600b7733e968a4ca9a621ec7122f348fd56491091`.
The four full-pipeline WM256 variants1332/1335/1339/1342 spill168/52/96/80
bytes and are pruned from timing. The ten new chunk/prefetch variants have
no stack; registers234–256. Actual shared bytes are113664 for TM256/TN128
and135168 for TM256/TN256. The two chunk controls1338/1345 use226 registers,
79872 shared. The actual incumbents1051/1045 keep their16/20-byte stack:
do NOT replace a measured baseline just because another body has no stack.
`resource-admission.json` separates baseline retention from candidate pruning.
Card1 shell1a8c73 now validates M257, then measures M512/8192 for each of the
12 admitted controls/variants (36 cells total), one process per phase/config.

A42 design: resolve1326's hot tradeoff at M2/4/6 without widening the scope.
The actual1325/1326 descriptors differ in static M, B producer warp, and
byte-pair versus packed-quartet zero metadata.1346 changes only1326's
B producer from0 to8.1347 generalizes the validated1325 to dynamic M;
relative to1346 it changes the existing metadata mode to byte-pair encoding
(quartet permutation must be disabled, as required by the native assertion).
1348–1350 then specialize1347 to M2/4/6 independently. Controls are683,
1326 and the known-good1325/M1. All preserve TK512/stage3/WK64/WN16;
no new decoder or code layout is introduced. Validate prepared zeros and
scales, full output and graph replay, then measure both cache regimes.

A41 finishes all36 full-output/replay cells PASS; none improves the actual
incumbents. For M8192,N8192,K8192:1051=8473us, WM256 chunk/prefetch
1333/1334=9839/9881, wide CTA1336/1337=9870/10072, original-geometry chunk
1338=8924. For N16384/K3072, WM256 chunk1340 reaches13346us versus1045=6670.
Do not promote a no-stack body on resource data alone.

Two paired ACU diagnostics complete under
`affine-asym-r41-{1045,1340}-large-profile*`, preserving reports, CSVs,
whole-output validation and hashes. At M8192/N16384/K3072:
1045 versus1340 has identical100663296 tensor instructions,~24.94/~24.90%
active warp occupancy and256 registers; total instruction averages24.81M
versus24.33M. Duration6670us versus13353us; shared load transactions457179136
versus834666496. Shared-load dependency stalls/issue rise0.09245→3.38973,
while shared-pipe-busy stalls fall0.00618→0.00042. These ratios are NOT
wall-time percentages. DRAM reads157.9MB→214.9MB do not track the2x latency.
Investigate A-load overlap rather than declaring a bandwidth/occupancy win.
Both profiling jobs finished before starting A42.

A42 build passes17 host tests,8 compiled query/resource checks, no stack.
SHA256 `6ab0cb5b066105fe56ef954040d2d46025f59383ae227005cb9dda9103dbfcf5`.
1346 uses64 registers/105984 shared;1347–1350 use62/107520;1024 threads.
Card1 shell874c6c now runs its32-cell hot+rotation ladder:1325/M1 control;
683,1326,1346,1347 each M2/4/6;1348/M2,1349/M4,1350/M6. All outputs and
changed-input replays must pass. Card0 R3 whole accuracy remains active as
shell4fda04 (CI PID1102040),14400s timeout; latest observation was~27 minutes.
Do not overlap another card1 job with874c6c.

Possible A43, NOT IMPLEMENTED: keep the original WM128/WN32 geometry and
test bounded A-fragment lookahead in the existing RegisterPrefetch branch.
The current `compute_tile` prefetches one M16 A fragment, whereas full FPR
holds the whole A stage and the incumbents have16/20 bytes of stack.
Chunk-only1338/1345 removes stack but loses performance. A circular2/4-
fragment lookahead could cover shared-read latency with less live storage.
Preserve the current depth1/default branch and add an opt-in template depth,
using the existing `load_a_fragment` and `mma_fragment`, no reconstruction
of lane maps. Linear position=K16_step*MAtoms+ma; future position+depth maps
back to step/ma, with a strict end-of-stage bound. Prime the ring, consume
the current fragment, prefetch its future replacement, retain the same B/
scale/zero caches and FP32 K order. Actual generated registers/stack and
numeric tail/replay gates decide whether to advance; no assumption that
source arrays remain register-resident. Start with1338/1345 at depth1/2/4,
retaining1051/1045 controls, before extending the rejected WM256 mapping.
Current tuning catalog ends at1350 (len1351). Source headers have not changed
since A38; recent work changed only tuning tables/selector/docs.

A42 completes32 cells, all full-output/replay PASS.1348/1349/1350 at M2/4/6
have hot medians5.180/5.205/5.213us, rotation5.522/5.525/5.525us. Paired H
rotation medians6.610/6.611/6.616us; range ratios0.8395/0.8405/0.8390.
The producer-only1346 remains a weak hot tradeoff. Byte-pair1347 improves
normal latency, and exact-M lowering then reduces hot~5.35→~5.2us.
Preserve timing spikes:1325/M1 rotation range1.5523;1326/1346 M2 hot and
several1347 cells also spike. A40's clean1325 result does not erase these.

Prepare R4/V5 with seven exact integer-zero N1024/K3072 rows:
M1/2/4/6/8/12/16→1325/1348/1349/1350/1327/1328/1329.
Retain all89 R3 configurations (including replaced675/676/683/829/838);
the new compiled set has96 entries. No fallback, symmetric or large-shape
selection changes. No whole-matrix performance admission is claimed: the
installed R4 public path must run its own complete hot/rotation cohort,
including every spike and every incumbent. R3 whole accuracy stays on its
frozen installed package; its tests compare native automatic and explicit
dispatch, not the mutable Python exact table. For subsequent C++ experiments,
use an isolated source worktree so that concurrent legacy JIT never reads
changed headers under an already-created compiler identity.

R4 normal wheel build completes in `wheels/affine-public-r4`, isolated install
at `tuning/affine-public-r4-installed`. Public DSO SHA256:
`f5f31ae330cc83e099473432377217d31f1facb8bbd5ca0c08a894f4338b9a6b`.
CPU audit confirms96 configurations, all89 R3 descriptors unchanged, all70
C exports retained, and the seven exact V5 rows. Receipt:
`results/affine-public-r4-package-audit.json`. The first audit wrongly assumed
the `quactlize_ppu_` prefix contained all70 exports; it contains60, with ten
additional `quactlize_dense/gate_up/serving` exports. Fix the audit prefix to
`quactlize_`; no library/configuration/dispatch failure occurred.
R4 host CI is shellbb242f, output `results/affine-public-r4-host`; do not count
it passed until its final summary. R3 whole accuracy remains shell4fda04 /
CI PID1102040, pytest PID1102050, with14400s timeout (observed~3900s).

Card1 blocker: an unrelated `/sim/AgentBench/llama.cpp/build-ci/bin/llama-server`
PID1460384 started2026-11-17 12:05:47 local time, port8091, on physical card1.
Do NOT stop it or benchmark concurrently. R4 public GPU tests, full hot cohort
and full rotation have NOT started. The A43 ladder finished all30 rows at
12:04:13, before this process started, and all30 full-output/replay checks
pass. A43 depth1/2/4 never beats1051/1045: M8192/N8192 times9085/9113/9129us
versus8464; N16384 times6917/7048/7029 versus6670. Do not promote A43.
Its private library SHA is
`2dedab393f392551b6d9e9a7550d302f9a1c91e4372b635dddea4c6a4b7e92c8`;
source remains only in `src/QUactlize-kda-810e-a43`, not in the original tree.
Eight native legal/illegal depth gates pass in
`results/affine-kpack-asym-r43-gates.json`; six new bodies have228–232
registers, no stack, same79872 shared bytes. Actual incumbents retain16/20
stack bytes. Their default resource reports are unchanged.

CPU work can continue while card1 is occupied. The next bounded proposal is
128-bit prepared scale/zero records (A44), described in the isolated
worktree's plan. A44 is now implemented only there as1357–1360; native build,
resource/ISA admission and device validation are not yet complete.
Original tuning ends at1350; isolated worktree tuning ends at1360.
The primary `tuning/bench_affine.py` remains unchanged; any A44 metadata
validator extension must use a separate task harness so R4's cohort keeps
the existing timing/source contract.

R4 host CI has now completed:1368 passed, zero failures/errors/skips. The
complete summary/JUnit/log are in `results/affine-public-r4-host`.

## R4 source/result checkpoint — 2026-11-17

Freeze the original source at V5 with96 compiled configurations, retaining89
R3 configurations and all70 public C exports. The123 affine source hashes
and155 public source hashes match the installed R4 payload; every manifest
payload hash verifies. A fresh CPU contract run passes17/17 tests. The native
public DSO remains `f5f31ae330cc83e099473432377217d31f1facb8bbd5ca0c08a894f4338b9a6b`.

The installed R4 hot/graph full matrix finishes with126 complete output and
changed-input replay checks:86 comparable FP32-reduction points and40 excluded
Humming precision routes. Median wins86/86, range wins85/86, range10%44/86.
The remaining hot range loss is M2/N1024/K3072, config1348; preserve its spike
and all ten samples. This is NOT all-range or global10% performance admission.
`docs/affine-810e-r4-results.json` retains all126 shape classifications,
comparable timing distributions, numeric checks and raw/environment/harness
hashes. Original raw logs, snapshots and binary artifacts remain in the
comparison workspace rather than entering the source repository.

At this checkpoint R4 full rotation is still active on card1 and the frozen
R3 whole-product accuracy remains INCOMPLETE on card0. R4 whole-product
accuracy has not run. Keep these states explicit; their later final receipts
must not be inferred from the successful hot cohort or host tests. Do not
interrupt either active job for source handoff.

A43/A44/A45 source and diagnostics stay in the isolated worktree. They are
not promoted into V5 or this source checkpoint. R45 config1363 at
M512/N8192/K8192 improves its paired median but does not meet the10% latency
reduction requirement; its M8192 result loses to1051. Config1364 at
M8192/N16384/K3072 only offers a marginal gain. Preserve these bounded results
without a blanket selector change.

Shared K-pack code storage is not GGUF scale-first performance admission.
The ordinary affine kernel subtracts an integer zero before FP16 scaling;
GGUF floating scale/min arithmetic and existing dense/MoE dispatch are separate
contracts. This checkpoint does not claim those endpoints select the new
W4A16 kernels or inherit their timings. The verified matrix is the seven
N/K geometries and eighteen M values archived in the result receipt,
representative of Qwen3.5-122B-A10B plus one Humming synthetic shape; it is not
complete multi-model or end-to-end inference coverage.
