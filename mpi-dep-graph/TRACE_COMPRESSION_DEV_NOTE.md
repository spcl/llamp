# Trace compression insertion point and IR (dev note)

## Where compression is applied
Compression is wired in `DependencyGraphGenerator.generate(...)`:

- `trace_compress=off` -> legacy path `__generate_baseline(...)` (unchanged behavior).
- `trace_compress in {lossless, iter-template}` -> new path `__generate_compressed(...)`.
- `trace_compress_rank_parametric=True` (optional) -> computes rank-relative unique-program count for diagnostics.
- `collective_semantics` in compressed mode:
  - `marker` (default): legacy marker-style local zero-cost collectives
  - `ring`: explicit send/recv ring coupling for collective events

In the compressed path, LLAMP now:
1. Parses GOAL into rank-local trace IR (`__parse_trace_ir`) **before graph construction**.
2. Compresses each rank program (`__compress_rank_programs`).
3. Builds the dependency graph by **streaming materialization** from compressed programs (`__build_dep_graph_from_compressed`).

This keeps compression upstream of graph/LP creation.

## New internal representation

### Event canonicalization
`trace_compression.EventSignature` captures canonical event identity:
- `kind` (`send`, `recv`, `calc`, `collective:*`)
- `size_or_cost`
- `peer_rank` (for p2p)
- `tag`
- `group_id`, `stream_id`, `channel_id`
- normalized extra attrs (`attrs`)

It provides:
- equality comparator (dataclass structural equality),
- stable hash (`stable_hash()`),
- compact debug serialization (`to_compact_str()`).

### Program compression IR
Per rank:
- raw event stream is represented as signature IDs.
- compressed program is represented as:
  - `LiteralToken(signature_id)`,
  - `RepeatToken(template_id, count)`,
  - `Template(signature_ids...)`.

`lossless` mode uses rolling-hash candidate matching + explicit sequence verification.

`iter-template` mode detects iteration skeletons and encodes:
- warmup literals
- first explicit iteration
- repeated middle template
- last explicit iteration
- tail literals

### Cross-rank dedup
Strict structural dedup is implemented via canonical program keys:
- unique canonical programs are stored once,
- mapping `rank -> canonical_program_id` is maintained.

Optional parametric reporting (`--trace-compress-rank-parametric`) additionally computes
unique program count under rank-relative peer normalization.

### Coupling/index mapping
`ProgramIndexer` supports random access mapping:
- ordinal -> `(token_index, repeat_index, template_offset)`
- compressed location -> ordinal

Communication edge insertion uses this indexer in compressed mode to validate and map coupling endpoints.

## Collective semantics
The parser now recognizes collective marker/event lines such as:

`lX: AllGather <bytes> bytes comm <id> gpu <id> stream <id> seq <id> end`

Graph construction supports:
- `marker`: map to ordered zero-cost local events (`CALC`) while preserving local dependency order.
- `ring`: emit explicit SEND/RECV pairs per collective event and add ring communication couplings.

## Phase instrumentation
Compressed modes emit phase-level timing and RSS:
- parse
- compress
- build

Format:
`[TRACE-COMPRESS] phase=<name> time_s=<...> rss_mb=<...>`
