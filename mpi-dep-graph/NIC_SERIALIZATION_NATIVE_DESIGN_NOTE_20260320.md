## Native NIC Serialization Design Note

### Goal
Add an optional native LLAMP send-side NIC serialization model for AI traces where multiple sends from the same rank can become ready in parallel, without collapsing the model into end-to-end send serialization.

### Where the new stage lives
The new stage lives in `lp_converter.py`, inside `LPConverter.convert_to_lp()`.

It is implemented as an explicit resource stage for send-capable vertices:

- `send_ready[v]`: lower-bounded by the existing non-NIC predecessors of the send-capable vertex.
- `nic_start[v]`: the timestamp at which the local NIC begins servicing the send.
- `nic_release[v]`: the timestamp at which the local NIC becomes free for another send on the same `(rank, nic)` resource.

`send_ready[v]` and `nic_start[v]` are equal in the current linear formulation, but they are tracked separately in the implementation/debug metadata so the model structure is explicit and inspectable.

### Variables and edges
For each send-capable vertex `v`:

- existing send start variable stays as the communication start used by the rest of LLAMP (`comm_vars[v]`);
- optionally add a new NIC release variable `nic_release_vars[v]` for debug/small runs;
- add either:
  - debug form: `nic_release[v] >= comm_vars[v] + nic_service(v)`, then `comm_vars[next] >= nic_release[prev]`, or
  - production form: the projected constraint `comm_vars[next] >= comm_vars[prev] + nic_service(prev)`

where:

- `nic_service(v) = msg_gap + bytes * G` for `SEND`,
- `nic_service(v) = msg_gap + bw_coeff * G` for send-emitting `MACRO`.

Both forms mean only the local NIC service window is serialized. The previous send may still be in flight while the next send starts after `prev` releases the NIC. The production form is algebraically equivalent for the immediate-neighbor chain and is substantially lighter on full traces.

### What "same NIC" means here
For the native feature, "same NIC" means a shared send-side injection resource identified by:

- `(rank, nic)` when the trace provides a NIC id,
- `(rank, 0)` when the trace does not provide one.

The native feature does **not** serialize across ranks, nodes, rails, or ports unless the trace explicitly maps them onto the same local `(rank, nic)` resource.

### Ordering heuristic
The native method does not replay an LGS-derived order.

Instead, for each `(rank, nic)` group we infer deterministic feasible **ready cohorts** from LLAMP/trace semantics only:

1. compute a static send-ready proxy from LLAMP graph semantics;
2. group sends that share the same ready proxy into one local-NIC cohort;
3. within each cohort, order by original local issue order (`trace_order` when present, otherwise local label) under a DAG-safe topological guard.

The ready proxy is a lower-bound estimate of when the send-capable operation can become locally ready without NIC serialization. It uses the existing DAG dependencies and operation costs, but not any replayed external schedule. The final production path serializes only within these same-ready cohorts, which directly targets the AI-trace pathology we observed: multiple same-rank sends becoming ready in parallel.

### Why this preserves pipelining
The critical distinction is:

- we serialize `nic_release(prev) -> nic_start(next)`,
- we do **not** serialize `send_finish(prev) -> send_start(next)`,
- we do **not** serialize `recv_finish(prev) -> send_start(next)`,
- we do **not** chain remote-dominated completion times.

Remote arrival / receive progress remains driven by the existing LLAMP send-start path. Therefore:

- later sends cannot inject until the local NIC service window of the earlier send ends;
- once that window ends, a later send may inject even while the earlier send is still in flight;
- network flight, remote receive progress, and compute can still overlap.

This matches the intended send-side NIC queuing model much more closely than full same-NIC trace-order serialization.

### Validation plan
Validation will compare:

- LLAMP baseline,
- LLAMP native NIC serialization,
- existing full same-NIC serialization flag as an ablation/debug baseline,
- LGS reference,
- hardware reference where already available in the workspace.

The same LLAMA traces and the existing latency/bandwidth sweep scripts will be reused.
