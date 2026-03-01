from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union


class TraceCompressMode(str, Enum):
    OFF = "off"
    LOSSLESS = "lossless"
    ITER_TEMPLATE = "iter-template"

    @classmethod
    def from_str(cls, value: str) -> "TraceCompressMode":
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported trace compression mode: {value}. "
                f"Expected one of {[m.value for m in cls]}"
            ) from exc


@dataclass(frozen=True)
class EventSignature:
    """
    Canonical event representation used for matching and compression.
    """
    kind: str
    size_or_cost: int
    peer_rank: Optional[int]
    tag: Optional[int]
    group_id: Optional[int]
    stream_id: Optional[int]
    channel_id: Optional[int]
    attrs: Tuple[Tuple[str, str], ...] = ()

    def to_compact_str(self) -> str:
        fields = [
            self.kind,
            str(self.size_or_cost),
            str(-1 if self.peer_rank is None else self.peer_rank),
            str(-1 if self.tag is None else self.tag),
            str(-1 if self.group_id is None else self.group_id),
            str(-1 if self.stream_id is None else self.stream_id),
            str(-1 if self.channel_id is None else self.channel_id),
        ]
        for key, value in self.attrs:
            fields.append(f"{key}={value}")
        return "|".join(fields)

    def stable_hash(self) -> int:
        digest = hashlib.blake2b(
            self.to_compact_str().encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)


@dataclass(frozen=True)
class LiteralToken:
    signature_id: int


@dataclass(frozen=True)
class RepeatToken:
    template_id: int
    count: int


ProgramToken = Union[LiteralToken, RepeatToken]


@dataclass(frozen=True)
class Template:
    signature_ids: Tuple[int, ...]


@dataclass(frozen=True)
class ProgramLocation:
    token_index: int
    repeat_index: int
    template_offset: int


def _token_span(token: ProgramToken, templates: Sequence[Template]) -> int:
    if isinstance(token, LiteralToken):
        return 1
    return len(templates[token.template_id].signature_ids) * token.count


class ProgramIndexer:
    """
    Maps rank-local materialized ordinals <-> compressed program locations.
    """
    def __init__(self, tokens: Sequence[ProgramToken],
                 templates: Sequence[Template],
                 materialized_length: int) -> None:
        self._tokens = tokens
        self._templates = templates
        self._offsets = [0]
        for token in tokens:
            self._offsets.append(self._offsets[-1] + _token_span(token, templates))

        if self._offsets[-1] != materialized_length:
            raise ValueError(
                "Indexer construction mismatch: "
                f"expected {materialized_length} events, got {self._offsets[-1]}"
            )
        self.length = materialized_length

    def ordinal_to_location(self, ordinal: int) -> ProgramLocation:
        if ordinal < 0 or ordinal >= self.length:
            raise IndexError(
                f"Ordinal {ordinal} is out of bounds for length {self.length}."
            )

        token_index = bisect_right(self._offsets, ordinal) - 1
        token = self._tokens[token_index]
        local_ordinal = ordinal - self._offsets[token_index]
        if isinstance(token, LiteralToken):
            return ProgramLocation(token_index=token_index,
                                   repeat_index=0,
                                   template_offset=0)

        template_len = len(self._templates[token.template_id].signature_ids)
        repeat_index, template_offset = divmod(local_ordinal, template_len)
        return ProgramLocation(token_index=token_index,
                               repeat_index=repeat_index,
                               template_offset=template_offset)

    def location_to_ordinal(self, location: ProgramLocation) -> int:
        if location.token_index < 0 or location.token_index >= len(self._tokens):
            raise IndexError(
                f"Token index {location.token_index} is out of bounds."
            )
        token = self._tokens[location.token_index]
        base = self._offsets[location.token_index]

        if isinstance(token, LiteralToken):
            if location.repeat_index != 0 or location.template_offset != 0:
                raise ValueError(
                    "Literal locations must have repeat_index=0 "
                    "and template_offset=0."
                )
            return base

        template_len = len(self._templates[token.template_id].signature_ids)
        if location.repeat_index < 0 or location.repeat_index >= token.count:
            raise IndexError(
                f"repeat_index {location.repeat_index} out of bounds "
                f"for token count {token.count}."
            )
        if location.template_offset < 0 or location.template_offset >= template_len:
            raise IndexError(
                f"template_offset {location.template_offset} out of bounds "
                f"for template length {template_len}."
            )
        return base + location.repeat_index * template_len + location.template_offset


@dataclass
class CompressedProgram:
    templates: List[Template]
    tokens: List[ProgramToken]
    materialized_length: int
    mode: TraceCompressMode

    def __post_init__(self) -> None:
        self.indexer = ProgramIndexer(self.tokens, self.templates, self.materialized_length)

    def iter_materialized(self) -> Iterator[int]:
        for token in self.tokens:
            if isinstance(token, LiteralToken):
                yield token.signature_id
                continue
            template = self.templates[token.template_id].signature_ids
            for _ in range(token.count):
                for signature_id in template:
                    yield signature_id

    def materialize(self) -> List[int]:
        return list(self.iter_materialized())

    def template_event_count(self) -> int:
        return sum(len(t.signature_ids) for t in self.templates)

    def estimated_units(self) -> int:
        # A simple size proxy: token count + template payload size.
        return len(self.tokens) + self.template_event_count()

    def canonical_key(self) -> Tuple:
        key = []
        for token in self.tokens:
            if isinstance(token, LiteralToken):
                key.append(("L", token.signature_id))
            else:
                template = self.templates[token.template_id].signature_ids
                key.append(("R", template, token.count))
        return tuple(key)


class _RollingHash:
    """
    64-bit rolling hash with explicit sequence verification for collision safety.
    """
    _MASK = (1 << 64) - 1
    _BASE = 0x9E3779B185EBCA87  # odd, high-entropy constant

    def __init__(self, seq: Sequence[int]) -> None:
        self._seq = seq
        self._prefix = [0] * (len(seq) + 1)
        self._pow = [1] * (len(seq) + 1)
        for i, value in enumerate(seq):
            # +1 keeps zero values distinguishable in prefixes.
            self._prefix[i + 1] = (
                (self._prefix[i] * self._BASE) + (value + 1)
            ) & self._MASK
            self._pow[i + 1] = (self._pow[i] * self._BASE) & self._MASK

    def _window_hash(self, start: int, length: int) -> int:
        end = start + length
        return (
            self._prefix[end]
            - (self._prefix[start] * self._pow[length] & self._MASK)
        ) & self._MASK

    def windows_equal(self, a_start: int, b_start: int, length: int) -> bool:
        if self._window_hash(a_start, length) != self._window_hash(b_start, length):
            return False
        # Collision-safe verification.
        return self._seq[a_start:a_start + length] == self._seq[b_start:b_start + length]


def _find_best_repeat(seq: Sequence[int], rh: _RollingHash, start: int,
                      max_window: int) -> Tuple[int, int]:
    n = len(seq)
    upper = min(max_window, (n - start) // 2)
    if upper <= 0:
        return 0, 0

    first = seq[start]
    best_window = 0
    best_count = 0
    best_gain = 0
    for window in range(1, upper + 1):
        if seq[start + window] != first:
            continue
        if not rh.windows_equal(start, start + window, window):
            continue

        count = 2
        while start + (count + 1) * window <= n:
            if not rh.windows_equal(start, start + count * window, window):
                break
            count += 1

        # Template + repeat token has fixed overhead. Keep only profitable matches.
        gain = (window * count) - (window + 2)
        if gain > best_gain:
            best_gain = gain
            best_window = window
            best_count = count
    return best_window, best_count


def compress_sequence_lossless(signature_ids: Sequence[int],
                               max_window: int = 256,
                               mode: TraceCompressMode = TraceCompressMode.LOSSLESS) -> CompressedProgram:
    n = len(signature_ids)
    if n == 0:
        return CompressedProgram([], [], 0, mode)

    rh = _RollingHash(signature_ids)
    templates: List[Template] = []
    template_to_id: Dict[Tuple[int, ...], int] = {}
    tokens: List[ProgramToken] = []

    def intern_template(segment: Tuple[int, ...]) -> int:
        template_id = template_to_id.get(segment)
        if template_id is not None:
            return template_id
        template_id = len(templates)
        templates.append(Template(segment))
        template_to_id[segment] = template_id
        return template_id

    i = 0
    while i < n:
        window, count = _find_best_repeat(signature_ids, rh, i, max_window=max_window)
        if count >= 2 and window >= 1:
            segment = tuple(signature_ids[i:i + window])
            template_id = intern_template(segment)
            tokens.append(RepeatToken(template_id=template_id, count=count))
            i += window * count
            continue
        tokens.append(LiteralToken(signature_id=signature_ids[i]))
        i += 1

    program = CompressedProgram(templates, tokens, n, mode)
    if program.materialize() != list(signature_ids):
        raise AssertionError("Lossless compression verification failed.")
    return program


def _detect_iteration_skeleton(signature_ids: Sequence[int],
                               max_warmup: int = 128,
                               max_window: int = 256) -> Optional[Tuple[int, int, int]]:
    n = len(signature_ids)
    if n < 4:
        return None
    rh = _RollingHash(signature_ids)

    best = None
    best_coverage = 0
    max_start = min(max_warmup, n - 2)
    for start in range(0, max_start + 1):
        upper = min(max_window, (n - start) // 2)
        if upper <= 0:
            continue
        first = signature_ids[start]
        for window in range(1, upper + 1):
            if signature_ids[start + window] != first:
                continue
            if not rh.windows_equal(start, start + window, window):
                continue

            count = 2
            while start + (count + 1) * window <= n:
                if not rh.windows_equal(start, start + count * window, window):
                    break
                count += 1
            coverage = window * count
            if coverage > best_coverage:
                best_coverage = coverage
                best = (start, window, count)
    return best


def compress_sequence_iter_template(signature_ids: Sequence[int],
                                    max_warmup: int = 128,
                                    max_window: int = 256) -> CompressedProgram:
    """
    Iteration templating with explicit boundaries:
      warmup + first explicit iteration + repeated middle + last explicit iteration + tail
    This remains exact/lossless and preserves boundary visibility.
    """
    n = len(signature_ids)
    if n == 0:
        return CompressedProgram([], [], 0, TraceCompressMode.ITER_TEMPLATE)

    skeleton = _detect_iteration_skeleton(
        signature_ids,
        max_warmup=max_warmup,
        max_window=max_window,
    )
    if skeleton is None:
        return compress_sequence_lossless(
            signature_ids,
            max_window=max_window,
            mode=TraceCompressMode.ITER_TEMPLATE,
        )

    start, window, count = skeleton
    if count < 2:
        return compress_sequence_lossless(
            signature_ids,
            max_window=max_window,
            mode=TraceCompressMode.ITER_TEMPLATE,
        )

    templates: List[Template] = []
    tokens: List[ProgramToken] = []
    template = tuple(signature_ids[start:start + window])
    templates.append(Template(template))
    template_id = 0

    def append_literals(values: Sequence[int]) -> None:
        for value in values:
            tokens.append(LiteralToken(signature_id=value))

    warmup = signature_ids[:start]
    tail = signature_ids[start + count * window:]
    append_literals(warmup)

    # Keep boundaries explicit for correctness debugging and cross-iteration deps.
    append_literals(template)  # first explicit iteration
    middle_count = max(count - 2, 0)
    if middle_count > 0:
        tokens.append(RepeatToken(template_id=template_id, count=middle_count))
    append_literals(template)  # last explicit iteration

    append_literals(tail)

    program = CompressedProgram(
        templates=templates,
        tokens=tokens,
        materialized_length=n,
        mode=TraceCompressMode.ITER_TEMPLATE,
    )
    if program.materialize() != list(signature_ids):
        raise AssertionError("iter-template materialization verification failed.")

    # If this skeleton does not improve compactness, fall back to general lossless.
    if program.estimated_units() >= n:
        return compress_sequence_lossless(
            signature_ids,
            max_window=max_window,
            mode=TraceCompressMode.ITER_TEMPLATE,
        )
    return program


def deduplicate_programs(rank_programs: Sequence[CompressedProgram]) \
        -> Tuple[List[CompressedProgram], List[int]]:
    unique_programs: List[CompressedProgram] = []
    rank_to_program_id: List[int] = []
    key_to_program_id: Dict[Tuple, int] = {}

    for program in rank_programs:
        key = program.canonical_key()
        program_id = key_to_program_id.get(key)
        if program_id is None:
            program_id = len(unique_programs)
            unique_programs.append(program)
            key_to_program_id[key] = program_id
        rank_to_program_id.append(program_id)
    return unique_programs, rank_to_program_id
