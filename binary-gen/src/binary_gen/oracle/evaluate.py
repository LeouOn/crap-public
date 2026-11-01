"""
Behavioral eval for oracle binaries.

A random program can't be scored by comparing one run's output with a
reference. Instead each binary runs many times, and every cast must be
self-consistent:

  - it exits normally (no signal),
  - it prints exactly `lines` lines, each one of the requested glyphs,
  - its exit code is the traditional number of the figure it printed.

Then, across all casts, line-value frequencies must fit the requested
method (Pearson chi-square at alpha = 0.001). A binary passes only if every
cast is consistent and the distribution fits. That catches the failure that
plagued CRAP's training runs: output that looks right and isn't.

`byte_diff` is a diagnostic, not a score: it says which region of the file
(header, code, method table, glyphs, numbering table) a wrong byte sits in.

    python -m binary_gen.oracle.evaluate predictions.jsonl --runs 1000
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from binary_gen.core import parse_hex
from binary_gen.core.linux_oracle import LINE_VALUES, NUMBERINGS, figure_number
from binary_gen.core.platform import Platform
from binary_gen.core.runner import Exited, RunResult, run_binary
from binary_gen.oracle import OracleSpec

#: Chi-square critical values at alpha = 0.001 by degrees of freedom.
_CHI2_CRIT_001 = {1: 10.828, 2: 13.816, 3: 16.266}

type Runner = Callable[[bytes], RunResult]


def _default_runner(binary: bytes) -> RunResult:
    return run_binary(binary, Platform.LINUX_X86)


def check_cast(result: RunResult, spec: OracleSpec) -> tuple[str, list[int]]:
    """Classify one run. Returns (outcome, line values bottom→top)."""
    if not isinstance(result.termination, Exited):
        return "signaled", []
    try:
        printed = result.stdout.decode("ascii").splitlines()
    except UnicodeDecodeError:
        return "bad_output", []
    if len(printed) != spec.lines:
        return "wrong_line_count", []

    indices: list[int] = []
    for line in reversed(printed):  # printed top first
        idx = spec.glyphs.line_index(line)
        if idx is None:
            return "unknown_glyph", []
        indices.append(idx)

    bits = sum((i & 1) << k for k, i in enumerate(indices))
    if result.termination.code != figure_number(spec.lines, bits):
        return "wrong_number", []
    return "ok", [i + 6 for i in indices]


@dataclass
class OracleReport:
    runs: int
    outcomes: Counter[str] = field(default_factory=Counter)
    value_counts: Counter[int] = field(default_factory=Counter)
    figures_seen: set[int] = field(default_factory=set)
    chi2: float | None = None
    chi2_critical: float | None = None
    impossible_values: int = 0

    @property
    def consistent(self) -> bool:
        return self.outcomes["ok"] == self.runs

    @property
    def distribution_ok(self) -> bool:
        if self.impossible_values:
            return False
        if self.chi2 is None or self.chi2_critical is None:
            return True  # degenerate method (one value): consistency is the test
        return self.chi2 <= self.chi2_critical

    @property
    def passed(self) -> bool:
        return self.consistent and self.distribution_ok

    def summary(self) -> dict[str, object]:
        total = sum(self.value_counts.values()) or 1
        return {
            "passed": self.passed,
            "consistent_runs": f"{self.outcomes['ok']}/{self.runs}",
            "outcomes": dict(self.outcomes),
            "figures_seen": len(self.figures_seen),
            "line_freq": {v: round(self.value_counts[v] / total, 4) for v in LINE_VALUES},
            "chi2": None if self.chi2 is None else round(self.chi2, 2),
            "chi2_critical": self.chi2_critical,
            "impossible_values": self.impossible_values,
        }


def _chi_square(counts: Counter[int], spec: OracleSpec) -> tuple[float | None, float | None, int]:
    """Pearson chi-square over values the method can produce; also counts impossible ones."""
    total = sum(counts.values())
    possible = [v for v in LINE_VALUES if spec.method.probability(v) > 0]
    impossible = sum(counts[v] for v in LINE_VALUES if spec.method.probability(v) == 0)
    if total == 0 or len(possible) < 2:
        return None, None, impossible
    chi2 = 0.0
    for v in possible:
        expected = total * spec.method.probability(v)
        chi2 += (counts[v] - expected) ** 2 / expected
    return chi2, _CHI2_CRIT_001[len(possible) - 1], impossible


def evaluate_binary(
    binary: bytes, spec: OracleSpec, runs: int = 1000, runner: Runner = _default_runner
) -> OracleReport:
    """Run `binary` `runs` times and score it against `spec`."""
    report = OracleReport(runs=runs)
    for _ in range(runs):
        try:
            result = runner(binary)
        except RuntimeError:
            report.outcomes["run_error"] += 1
            continue
        outcome, values = check_cast(result, spec)
        report.outcomes[outcome] += 1
        if outcome == "ok":
            report.value_counts.update(values)
            assert isinstance(result.termination, Exited)
            report.figures_seen.add(result.termination.code)
    report.chi2, report.chi2_critical, report.impossible_values = _chi_square(
        report.value_counts, spec
    )
    return report


# ─────────────────────────────────────────────────────────
# Byte-level diagnostics
# ─────────────────────────────────────────────────────────


def regions(spec: OracleSpec) -> list[tuple[str, int, int]]:
    """(name, start, end) file offsets of each part of an oracle binary."""
    code_end = 120 + 0x7F
    glyphs_end = code_end + 16 + 4 * spec.glyphs.width
    return [
        ("elf_header", 0, 64),
        ("program_header", 64, 120),
        ("code", 120, code_end),
        ("method_table", code_end, code_end + 16),
        ("glyph_table", code_end + 16, glyphs_end),
        ("numbering_table", glyphs_end, glyphs_end + len(NUMBERINGS[spec.lines])),
    ]


def byte_diff(candidate: bytes, spec: OracleSpec) -> dict[str, object]:
    """Where a candidate differs from the ground-truth binary for `spec`."""
    reference = spec.build()
    wrong: Counter[str] = Counter()
    for name, start, end in regions(spec):
        for i in range(start, end):
            if i >= len(candidate) or candidate[i] != reference[i]:
                wrong[name] += 1
    return {
        "exact": candidate == reference,
        "length": len(candidate),
        "expected_length": len(reference),
        "wrong_bytes_by_region": dict(wrong),
    }


# ─────────────────────────────────────────────────────────
# CLI: score a JSONL file of model predictions
# ─────────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "predictions",
        type=Path,
        help='JSONL rows with "spec" (as in the dataset) and "prediction" (hex text)',
    )
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()

    passed = total = 0
    for line in args.predictions.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spec = OracleSpec.from_json(row["spec"])
        total += 1
        try:
            binary = parse_hex(row["prediction"])
        except ValueError as e:
            print(json.dumps({"id": row.get("id"), "passed": False, "error": f"hex: {e}"}))
            continue
        report = evaluate_binary(binary, spec, args.runs)
        passed += report.passed
        print(json.dumps({"id": row.get("id"), **report.summary(), **byte_diff(binary, spec)}))
    print(f"passed {passed}/{total}", file=sys.stderr)


if __name__ == "__main__":
    _main()
