#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path


SDPA_TOTAL_RE = re.compile(r"\[sdpa_profile\].*\btotal_ms=([0-9]+(?:\.[0-9]+)?)")
SDPA_SLOT_RE = re.compile(
    r"\[sdpa_profile\]\s+(?P<slot>[A-Za-z0-9_]+)\s+"
    r"(?P<ms>[0-9]+(?:\.[0-9]+)?)\s+ms\b"
)
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*(?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms"
    r"\s*/\s*(?P<tokens>[0-9]+)\s*tokens"
)
WALL_SECONDS_RE = re.compile(r"\bwall_seconds\s+(?P<seconds>[0-9]+(?:\.[0-9]+)?)\b")
GENERIC_MS_RE = re.compile(
    r"\b(?:item_ms|elapsed_ms|latency_ms|time_ms)\s*[=:]\s*"
    r"(?P<ms>[0-9]+(?:\.[0-9]+)?)\b"
)
BATCH_DECODE_RE = re.compile(
    r"^(?P<stamp>[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\s+\S+\s+"
    r"batch_decode:\s+n_tokens\s*=\s*(?P<tokens>[0-9]+),\s+"
    r"n_seq\s*=\s*(?P<n_seq>[0-9]+)",
    re.MULTILINE,
)


def read_text(path: str | None) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(errors="replace")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: no samples")
        return

    total = sum(values)
    avg = total / len(values)
    print(f"{label}:")
    print(f"  count              : {len(values)}")
    print(f"  total_ms           : {total:.3f}")
    print(f"  avg_ms             : {avg:.3f}")
    print(f"  min_ms             : {min(values):.3f}")
    print(f"  p50_ms             : {percentile(values, 50):.3f}")
    print(f"  p90_ms             : {percentile(values, 90):.3f}")
    print(f"  p95_ms             : {percentile(values, 95):.3f}")
    print(f"  p99_ms             : {percentile(values, 99):.3f}")
    print(f"  max_ms             : {max(values):.3f}")
    if len(values) > 1:
        print(f"  stdev_ms           : {statistics.stdev(values):.3f}")
    print(f"  items_per_second   : {1000.0 / avg:.3f}")


def group_values(values: list[float], group_size: int, *, drop_partial: bool) -> list[float]:
    grouped: list[float] = []
    for i in range(0, len(values), group_size):
        chunk = values[i : i + group_size]
        if len(chunk) != group_size and drop_partial:
            continue
        grouped.append(sum(chunk))
    return grouped


def count_inputs(path: str, separator: str, drop_empty: bool) -> int:
    text = Path(path).read_text(errors="replace")
    parts = text.split(separator)
    if drop_empty:
        parts = [part for part in parts if part]
    return len(parts)


def parse_prompt_eval(text: str) -> tuple[float | None, int | None]:
    matches = list(PROMPT_EVAL_RE.finditer(text))
    if not matches:
        return None, None
    match = matches[-1]
    return float(match.group("ms")), int(match.group("tokens"))


def parse_wall_ms(text: str) -> float | None:
    matches = list(WALL_SECONDS_RE.finditer(text))
    if not matches:
        return None
    return float(matches[-1].group("seconds")) * 1000.0


def parse_log_stamp_ms(stamp: str) -> float:
    # llama.cpp log timestamps look like M.SS.mmm.uuu for short runs.
    # Example: 0.21.996.387 means 21.996387 seconds after start.
    parts = [int(part) for part in stamp.split(".")]
    if len(parts) != 4:
        raise ValueError(f"unsupported timestamp: {stamp}")
    minutes, seconds, millis, micros = parts
    return minutes * 60_000.0 + seconds * 1000.0 + millis + micros / 1000.0


def parse_batch_decode_deltas(text: str, *, per_seq: bool) -> list[float]:
    events: list[tuple[float, int, int]] = []
    for match in BATCH_DECODE_RE.finditer(text):
        events.append(
            (
                parse_log_stamp_ms(match.group("stamp")),
                int(match.group("tokens")),
                int(match.group("n_seq")),
            )
        )
    if len(events) < 2:
        return []

    samples: list[float] = []
    for i in range(len(events) - 1):
        start_ms, _tokens, n_seq = events[i]
        next_start_ms = events[i + 1][0]
        delta_ms = next_start_ms - start_ms
        if delta_ms < 0:
            continue
        if per_seq and n_seq > 0:
            delta_ms /= n_seq
        samples.append(delta_ms)
    return samples


def looks_like_embedding_array(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("[[") and "," in stripped[:256]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse llama-embedding and fused SDPA perf logs."
    )
    parser.add_argument("log", nargs="?", help="log file, or '-' / omitted for stdin")
    parser.add_argument(
        "--mode",
        choices=("auto", "sdpa", "slot", "generic", "batch-delta", "prompt-eval"),
        default="auto",
        help="which samples to parse",
    )
    parser.add_argument(
        "--slot",
        default="main_compute",
        help="slot name for --mode slot, for example main_compute/qkt_total/pv",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="sum every N samples before reporting, e.g. 12 layers per item",
    )
    parser.add_argument(
        "--keep-partial",
        action="store_true",
        help="keep the final incomplete group when --group-size is used",
    )
    parser.add_argument(
        "--items",
        type=int,
        help="number of input items for prompt-eval average",
    )
    parser.add_argument(
        "--per-seq",
        action="store_true",
        help="for --mode batch-delta, divide each batch delta by n_seq",
    )
    parser.add_argument(
        "--input",
        help="input text file; used to infer --items for prompt-eval mode",
    )
    parser.add_argument(
        "--separator",
        default="\n",
        help="same separator as --embd-separator; default is newline",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="ignore empty input segments when counting --input items",
    )
    args = parser.parse_args()

    text = read_text(args.log)

    samples: list[float] = []
    label = ""
    mode = args.mode

    if mode in ("auto", "sdpa"):
        samples = [float(m.group(1)) for m in SDPA_TOTAL_RE.finditer(text)]
        if samples:
            label = "sdpa_total_ms"
            mode = "sdpa"

    if mode in ("auto", "slot") and not samples:
        samples = [
            float(m.group("ms"))
            for m in SDPA_SLOT_RE.finditer(text)
            if m.group("slot") == args.slot
        ]
        if samples:
            label = f"sdpa_slot_{args.slot}_ms"
            mode = "slot"

    if mode in ("auto", "generic") and not samples:
        samples = [float(m.group("ms")) for m in GENERIC_MS_RE.finditer(text)]
        if samples:
            label = "generic_item_ms"
            mode = "generic"

    if mode in ("auto", "batch-delta") and not samples:
        samples = parse_batch_decode_deltas(text, per_seq=args.per_seq)
        if samples:
            label = "batch_decode_start_delta_ms"
            if args.per_seq:
                label += "_per_seq"
            mode = "batch-delta"

    if mode == "prompt-eval" or (mode == "auto" and not samples):
        prompt_ms, tokens = parse_prompt_eval(text)
        if prompt_ms is None:
            if looks_like_embedding_array(text):
                print(
                    "No timings found. This log looks like embedding array output; "
                    "redirect stdout away from the timing log and capture stderr.",
                    file=sys.stderr,
                )
                return 1
            print("No per-sample timings or prompt eval aggregate found.", file=sys.stderr)
            return 1

        items = args.items
        if items is None and args.input:
            items = count_inputs(args.input, args.separator, args.drop_empty)
        if items is None or items <= 0:
            print(
                "prompt eval aggregate found, but --items or --input is required "
                "to compute per-item average.",
                file=sys.stderr,
            )
            return 1

        print("prompt_eval_aggregate:")
        print(f"  items              : {items}")
        print(f"  tokens             : {tokens}")
        print(f"  prompt_eval_ms     : {prompt_ms:.3f}")
        print(f"  avg_ms_per_item    : {prompt_ms / items:.3f}")
        if tokens:
            print(f"  avg_ms_per_token   : {prompt_ms / tokens:.6f}")
        wall_ms = parse_wall_ms(text)
        if wall_ms is not None:
            print(f"  wall_ms            : {wall_ms:.3f}")
            print(f"  wall_ms_per_item   : {wall_ms / items:.3f}")
        print("  p95_ms_per_item    : unavailable from aggregate log")
        print("  note               : enable per-item timing or group SDPA profile rows for p95")
        return 0

    if not samples:
        if looks_like_embedding_array(text):
            print(
                "No timings found. This log looks like embedding array output; "
                "redirect stdout away from the timing log and capture stderr.",
                file=sys.stderr,
            )
            return 1
        print("No matching timing samples found.", file=sys.stderr)
        return 1

    if args.group_size:
        if args.group_size <= 0:
            print("--group-size must be positive", file=sys.stderr)
            return 1
        original_count = len(samples)
        samples = group_values(samples, args.group_size, drop_partial=not args.keep_partial)
        label = f"{label}_grouped_by_{args.group_size}"
        print(f"raw_samples         : {original_count}")
        print(f"group_size          : {args.group_size}")
        print(f"partial_group       : {'kept' if args.keep_partial else 'dropped'}")

    summarize(label, samples)

    prompt_ms, tokens = parse_prompt_eval(text)
    if prompt_ms is not None:
        print()
        print("prompt_eval_aggregate:")
        print(f"  tokens             : {tokens}")
        print(f"  prompt_eval_ms     : {prompt_ms:.3f}")

    wall_ms = parse_wall_ms(text)
    if wall_ms is not None:
        print()
        print("wall_clock:")
        print(f"  wall_ms            : {wall_ms:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
