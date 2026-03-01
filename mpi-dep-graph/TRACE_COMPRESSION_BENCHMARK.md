# Trace compression benchmark report

## Setup

- Repo: `spcl/llamp`
- Commit: `49baf3dafd2c9565e5ade979475f6d67f6322cda`
- Branch: `feature/trace-compression`
- Solver path used in this environment: OR-Tools (`GUROBI not found` at runtime)

## 1) Correctness: runtime-vs-L parity (`off` vs `lossless`)

Validation traces:
- N4: `.../grok_traces/results_v1_v2_20260225/N4/v1_goal/Events_Dependency.goal`
- N8: `.../grok_traces/results_v1_v2_20260225/N8/v1_goal/Events_Dependency.goal`
- N16: `.../grok_traces/results_v1_v2_20260225/N16/v1_goal/Events_Dependency.goal`

Parameters:
- LogGPS: `o=5000`, `S=262144`, fixed `G=0.018`
- Latency sweep: `L in {3000, 4000, 5000}`

### Runtime curves

| Trace | `off` runtime curve (ns) | `lossless` runtime curve (ns) | Max abs diff |
|---|---|---|---|
| N4 | (3000, 6170774835), (4000, 6170774835), (5000, 6170774835) | (3000, 6170774835), (4000, 6170774835), (5000, 6170774835) | **0.0** |
| N8 | (3000, 5251600207), (4000, 5251600207), (5000, 5251600207) | (3000, 5251600207), (4000, 5251600207), (5000, 5251600207) | **0.0** |
| N16 | (3000, 9930166016), (4000, 9930166016), (5000, 9930166016) | (3000, 9930166016), (4000, 9930166016), (5000, 9930166016) | **0.0** |

Result: `lossless` matched baseline exactly (numerical diff = 0 on tested points).

## 2) Compression ratio

### 2.1 Grok small v1/v2 inter-node traces (event-stream compression stats)

Traces:
- N4 inter-node: `.../new_traces_collection/ground_truth/Llama7B_N4.../InterNode_MicroEvents_Dependency.goal`
- N8 inter-node: `.../llamp_validation/runs_nsys_motif_clean/.../grok_n8/.../InterNode_MicroEvents_Dependency.goal`
- N16 inter-node: `/users/btommaso/moe_fixed_sqlite/output_run_v1/InterNode_MicroEvents_Dependency.goal`

| Trace | Raw events | Compressed units (no dedup) | Savings | Unique programs |
|---|---:|---:|---:|---:|
| N4 | 5,680,774 | 2,889,402 | **49.14%** | 4 |
| N8 | 2,595,984 | 473,889 | **81.75%** | 8 |
| N16 | 13,238,842 | 6,805,250 | **48.60%** | 16 |

### 2.2 Grok N128 trace

Trace:
- `.../nccl-traces/N128-1iter/output_test_1/Events_Dependency.goal` (128 ranks)

`lossless` report:
- Raw events total: **612,096**
- Compressed units: **438,965**
- Savings: **28.28%**
- Unique rank programs: **128 / 128**
- Templates: **399** (avg len 4.53, max len 12)

## 3) Parse/build performance and RSS

### N128 (`Events_Dependency.goal`) off vs lossless

| Mode | Graph gen time (s) | RSS after build (MB) | Vertices | Edges |
|---|---:|---:|---:|---:|
| off | 6.483 | 420.69 | 612,096 | 691,200 |
| lossless | 25.784 | 627.69 | 612,096 | 691,200 |
| iter-template | 8.939 | (same run context) | 612,096 | 691,200 |

Note: `iter-template` remained exact but yielded little compression on this trace (`0.04%`).

## 4) Notes/caveats

1. `Events_Dependency.goal` contains collective markers; parser now recognizes them and keeps ordering/dependencies. For graph/LP construction they are currently mapped to local zero-cost ordered events, which preserves parity checks between `off` and `lossless` but does not model full collective protocol internals.
2. Some v1/v2 inter-node traces do not have one-to-one tag matching without an external comm-dep mapping. For those traces, compression-ratio-only runs were executed at the event-stream layer.

## 5) Follow-up: inter-node comm-dep parity (N4/N8/N16)

To validate inter-node parity with explicit comm-dep mappings in a bounded run, acyclic slices were extracted from real traces using real comm-dep edges:
- output root: `/iopsstor/scratch/cscs/btommaso/llamp_validation/inter_node_parity_slices_20260301_223344`
- each slice preserves original rank count (`N4`, `N8`, `N16`), uses real send/recv op endpoints from `.comm-dep`, and adds per-rank calc anchors to keep DAG constraints valid for LP conversion.

| Trace | Kept comm-dep edges | `off` runtime curve (ns) | `lossless` runtime curve (ns) | Max abs diff |
|---|---:|---|---|---:|
| N4 | 10 | (0, 6002), (3700, 150302) | (0, 6002), (3700, 150302) | **0.0** |
| N8 | 5 | (0, 2002), (3700, 61202) | (0, 2002), (3700, 61202) | **0.0** |
| N16 | 10 | (0, 6002), (3700, 153602) | (0, 6002), (3700, 153602) | **0.0** |

Result: `lossless` reproduces baseline exactly on all tested inter-node comm-dep slices.

## 6) Follow-up: collective semantics, parametric dedup, and phase instrumentation

### 6.1 Collective semantics (`--collective-semantics=ring`)

- Unit/integration validation (`test_collective_ring_semantics_builds_explicit_comm_edges`) passes.
- On a 2-rank AllGather marker trace, `ring` mode produces:
  - 2 explicit SEND vertices
  - 2 explicit RECV vertices
  - 2 cross-rank SEND->RECV comm edges

### 6.2 Parametric rank dedup reporting (`--trace-compress-rank-parametric`)

Observed on inter-node slices:
- N8 slice: `unique_programs(strict/parametric)=5/5`
- N16 slice: `unique_programs(strict/parametric)=3/3`

No additional collapse occurred on these particular slices, but strict vs parametric counts are now reported explicitly.

### 6.3 Parse/compress/build phase timing + RSS

Sample (`N16` inter-node parity slice):

| Mode | End-to-end graph gen time (s) | RSS after generate (MB) | Vertices | Edges |
|---|---:|---:|---:|---:|
| off | 0.0021 | 94.44 | 52 | 56 |
| lossless | 0.0017 | 94.44 | 52 | 56 |
| iter-template | 0.0015 | 94.44 | 52 | 56 |

Compressed modes also print internal phase metrics:
- `[TRACE-COMPRESS] phase=parse time_s=... rss_mb=...`
- `[TRACE-COMPRESS] phase=compress time_s=... rss_mb=...`
- `[TRACE-COMPRESS] phase=build time_s=... rss_mb=...`

## 7) Full-scale validation run (no slicing): N4/N8/N16/N128

Run artifacts:
- `/iopsstor/scratch/cscs/btommaso/llamp_validation/trace_compression_fullscale_20260301_232152/`
- Files: `cmd.sh`, `run.log`, `fullscale_results.json`, `summary.tsv`, `repro_manifest.txt`, `checksums.sha256`

Configuration:
- Modes: `off`, `lossless`, `iter-template`
- `L`: `{0, 3700, 300000, 1000000}`
- LogGPS params: `o=200`, `S=0`, `G=0.04`

### 7.1 Runtime-vs-L parity

| Trace | `off` runtime curve (ns) | `lossless` max abs diff | `iter-template` max abs diff |
|---|---|---:|---:|
| N4 | (0, 6170774835), (3700, 6170774835), (300000, 6170774835), (1000000, 6170774835) | **0.0** | **0.0** |
| N8 | (0, 5251600207), (3700, 5251600207), (300000, 5251600207), (1000000, 5251600207) | **0.0** | **0.0** |
| N16 | (0, 9930166016), (3700, 9930166016), (300000, 9930166016), (1000000, 9930166016) | **0.0** | **0.0** |
| N128 | (0, 13809880256), (3700, 13809880256), (300000, 13809880256), (1000000, 13809880256) | **0.0** | **0.0** |

Result: full-scale `off`, `lossless`, and `iter-template` were numerically identical on all tested L points.

### 7.2 End-to-end timings/RSS (selected)

| Trace | Mode | Graph gen (s) | LP build (s) | 4-point solve (s) | RSS after graph (MB) |
|---|---|---:|---:|---:|---:|
| N4 | off / lossless / iter-template | 0.342 / 1.994 / 0.535 | 0.079 / 0.072 / 0.072 | 0.001 / 0.001 / 0.001 | 313.2 / 330.2 / 334.2 |
| N8 | off / lossless / iter-template | 0.351 / 2.014 / 0.558 | 0.128 / 0.083 / 0.084 | 0.003 / 0.002 / 0.002 | 333.3 / 334.6 / 336.7 |
| N16 | off / lossless / iter-template | 0.692 / 3.445 / 1.152 | 0.334 / 0.278 / 0.332 | 0.125 / 0.119 / 0.118 | 343.9 / 374.8 / 382.7 |
| N128 | off / lossless / iter-template | 6.844 / 26.822 / 10.653 | 3.425 / 4.360 / 4.267 | 2.007 / 2.562 / 2.554 | 621.1 / 997.5 / 1000.8 |
