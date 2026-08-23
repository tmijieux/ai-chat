"""Chunk raw text into retrieval-sized pieces for RAG indexing.

Same character-budget-per-line approach as agent/workflow_coordinator.py's chunk_file (~4
chars/token estimate), but tuned for retrieval granularity instead of summarization granularity: a
much smaller per-chunk budget (400 vs. 12,000 tokens), plus overlap between consecutive chunks so a
fact split across a chunk boundary is still fully present in at least one chunk. chunk_file doesn't
need overlap since its chunks all get folded back together by an LLM summarization pass; RAG chunks
are retrieved and read in isolation from each other.
"""
from dataclasses import dataclass

CHARS_PER_TOKEN_ESTIMATE = 4
TARGET_CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN_ESTIMATE
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN_ESTIMATE


@dataclass
class Chunk:
    """One retrieval-sized slice of a source's text."""
    index: int
    start_line: int
    end_line: int
    text: str


def _split_oversized_line(line: str, max_chars: int) -> list[str]:
    """Split one line that alone exceeds the chunk budget into fixed-size character slices."""
    return [line[i:i + max_chars] for i in range(0, len(line), max_chars)] or [""]


def _overlap_tail(lines_so_far: list[str]) -> list[str]:
    """Return the trailing lines of a just-flushed chunk worth up to OVERLAP_CHARS, to seed the
    next chunk so context isn't lost across the boundary."""
    tail: list[str] = []
    tail_chars = 0
    for line in reversed(lines_so_far):
        line_chars = len(line) + 1
        if tail_chars + line_chars > OVERLAP_CHARS:
            break
        tail.insert(0, line)
        tail_chars += line_chars
    return tail


def chunk_text(text: str) -> list[Chunk]:
    """Split text into overlapping chunks targeting TARGET_CHUNK_TOKENS each (estimated from
    character count), so retrieval granularity stays useful without losing context that straddles
    a chunk boundary."""
    lines = text.splitlines()
    if len(lines) == 0:
        return []

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_chars = 0
    current_start_line = 1

    def flush_current(end_line: int) -> None:
        if len(current_lines) == 0:
            return
        chunks.append(Chunk(
            index=len(chunks), start_line=current_start_line, end_line=end_line,
            text="\n".join(current_lines),
        ))

    line_number = 0
    for line_number, line in enumerate(lines, start=1):
        line_chars = len(line) + 1

        if line_chars > TARGET_CHUNK_CHARS:
            flush_current(line_number - 1)
            for piece in _split_oversized_line(line, TARGET_CHUNK_CHARS):
                chunks.append(Chunk(index=len(chunks), start_line=line_number, end_line=line_number, text=piece))
            current_lines = []
            current_chars = 0
            current_start_line = line_number + 1
            continue

        if len(current_lines) > 0 and current_chars + line_chars > TARGET_CHUNK_CHARS:
            flush_current(line_number - 1)
            tail = _overlap_tail(current_lines)
            current_lines = tail
            current_chars = sum(len(l) + 1 for l in tail)
            current_start_line = (line_number - 1) - len(tail) + 1

        current_lines.append(line)
        current_chars += line_chars

    flush_current(line_number)
    return chunks
