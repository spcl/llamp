# mpi-dep-graph Notes

This directory contains the LLAMP graph builders, LP conversion logic, and the command-line entry point in `main.py`.

## Inputs

`main.py` accepts two input formats:

- `goal`: the original GOAL schedule format
- `macro-ir`: a compact collective-aware format built around `observed_collective` records

Example:

```console
python3 main.py -g trace.goal --solve
python3 main.py -g trace.interval.stage.motif.macro.ir --input-format macro-ir --solve
```

## Communication dependency sidecars

If a LogGOPSim `--comm-dep-file` sidecar is available, LLAMP consumes it via:

```console
python3 main.py -g trace.goal -c trace.comm_deps.csv --solve
```

Only cross-rank message-match dependencies are inserted into the DAG directly. Same-rank replay/order relations are preserved as side metadata so they can be consumed later by LP-level serialization logic without introducing graph cycles.

This is especially useful for reduced or truncated witnesses, where GOAL labels may no longer be contiguous.

## Native resource serialization

LLAMP now has native resource-aware serialization modes that can be enabled independently from the older trace-order constraints.

### Native NIC modes

Enable with:

```console
--nic-serialize
```

Available modes:

- `same-ready`: serialize sends only within equal ready-time cohorts
- `full-timeline`: strict local send timeline
- `windowed-timeline`: allow a bounded number of local sends to overlap
- `credit-timeline`: full-credit raw sends plus fractional-credit compressed macro sends

Useful related options:

- `--nic-serialize-window N`
- `--nic-serialize-scope {rank,node-shared}`
- `--ready-proxy-g <ns_per_byte>`

`credit-timeline` is intended for compressed collective models where one macro send may represent multiple micro sends. Raw `SEND` vertices still consume one full NIC credit.

### Native CPU modes

Enable with:

```console
--cpu-serialize
```

Available options:

- `--cpu-serialize-mode {full-timeline,same-ready}`
- `--cpu-serialize-scope {all-ops,comm-only}`
- `--cpu-serialize-resource-scope {rank,node-shared}`

`comm-only` is often a better fit for communication traces where we want to model progress pressure without turning every compute vertex into a hard CPU serialization point.

## Legacy trace-order modes

The older modes remain available for debugging and comparison:

- `--serialize-same-nic-sends`
- `--serialize-same-nic-recvs`
- `--serialize-same-cpu-ops`

These can still be useful as strict baselines or oracles, but they are separate from the newer native models.

## Bandwidth sweeps and `--ready-proxy-g`

When `G` is kept symbolic for a bandwidth sweep, the LP still needs one static ready-order proxy to decide how native serialization groups are ordered.

Use:

```console
--ready-proxy-g <ns_per_byte>
```

to choose that nominal operating point explicitly. In practice, using the baseline sweep bandwidth is the most stable choice.

## Design note

The project-specific native NIC design note lives in:

- `NIC_SERIALIZATION_NATIVE_DESIGN_NOTE_20260320.md`
