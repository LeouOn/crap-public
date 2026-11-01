"""
Training data for the oracle family, in the chat format CRAP trains on.

Each example asks for a trigram or hexagram oracle and varies three things:

  - method: three-coin (default), yarrow-stalk, or explicit probabilities
    in sixteenths for the four line values;
  - glyphs: the default ASCII lines, or requested yang/yin strings and
    changing-line marks;
  - wording: several templates, with the defaults sometimes left implicit.

Splits are assigned by hashing what the binary depends on (line count,
weights, glyphs), so no test binary appears in training even under a
different prompt. The canonical requests (named method, default glyphs)
always land in train: people will ask for those.

    python -m binary_gen.oracle.dataset --out data/oracle --train 4000
"""

import argparse
import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

from binary_gen.core.linux_oracle import COINS, METHODS, PLAIN, YARROW, Glyphs, Method
from binary_gen.oracle import OracleSpec

# Same system prompt as CRAP's training shim, so oracle data mixes with the
# original print/exit corpus.
SYSTEM_PROMPT = (
    "You are a binary code generator. Given a description of a program, "
    "output the complete binary as a lowercase hex string with no spaces, "
    "newlines, or other formatting. Output only the hex string and nothing else."
)
SPACED_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "with no spaces, newlines,", "with one space between bytes and no newlines"
)

SPLITS = ("train", "val", "test")


class Message(TypedDict):
    role: str
    content: str


class Example(TypedDict):
    id: str
    split: str
    spec: dict[str, object]
    messages: list[Message]


# ─────────────────────────────────────────────────────────
# Sampling specs
# ─────────────────────────────────────────────────────────

_SPECIAL_METHODS = (
    Method("fair", (4, 4, 4, 4)),
    Method("stable", (0, 8, 8, 0)),  # no changing lines
)
_STROKES = "-=#*~+@%"
_MARK_PAIRS = (("x", "o"), ("X", "O"), ("*", "*"), ("v", "^"), ("0", "9"), ("6", "9"), ("<", ">"))


def _random_weights(rng: random.Random) -> tuple[int, int, int, int]:
    cuts = sorted(rng.randint(0, 16) for _ in range(3))
    a, b, c = cuts
    return (a, b - a, c - b, 16 - c)


def _sample_method(rng: random.Random) -> Method:
    roll = rng.random()
    if roll < 0.3:
        return COINS
    if roll < 0.5:
        return YARROW
    if roll < 0.6:
        return rng.choice(_SPECIAL_METHODS)
    weights = _random_weights(rng)
    for named in METHODS.values():
        if named.sixteenths == weights:
            return named  # same binary as a named method: use its name
    return Method("custom", weights)


def _sample_glyphs(rng: random.Random) -> Glyphs:
    if rng.random() < 0.4:
        return PLAIN
    stroke = rng.choice(_STROKES)
    half = rng.randint(1, 5)
    gap = rng.randint(1, 3)
    yang = stroke * (2 * half + gap)
    yin = stroke * half + " " * gap + stroke * half
    sep = " " * rng.randint(1, 2)
    mark_yin, mark_yang = rng.choice(_MARK_PAIRS)
    return Glyphs(yin + sep + mark_yin, yang, yin, yang + sep + mark_yang)


def _split_of(spec: OracleSpec) -> str:
    canonical = spec.method.name in METHODS and spec.glyphs == PLAIN
    if canonical:
        return "train"
    key = json.dumps([spec.lines, spec.method.sixteenths, spec.glyphs.entries()])
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 20
    return {0: "test", 1: "val"}.get(bucket, "train")


def sample_spec(rng: random.Random, split: str) -> OracleSpec:
    """Draw specs until one belongs to `split`."""
    while True:
        spec = OracleSpec(
            lines=6 if rng.random() < 0.7 else 3,
            method=_sample_method(rng),
            glyphs=_sample_glyphs(rng),
        )
        if _split_of(spec) == split:
            return spec


# ─────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────

_FIGURE_WORDS = {
    6: ("hexagram", "I Ching hexagram", "six-line hexagram", "Yijing hexagram"),
    3: ("trigram", "bagua trigram", "three-line trigram", "I Ching trigram"),
}
_NUMBERING = {6: "King Wen", 3: "Earlier Heaven"}

_TEMPLATES = (
    "Generate a Linux x86_64 binary that casts a random {figure}{method}{glyphs}.",
    "Generate a Linux x86_64 binary oracle: cast a {figure}{method}, print it top line "
    "first{glyphs}, and exit with its {numbering} number.",
    "Write a Linux x86_64 binary that divines a {figure}{method}{glyphs}.",
    "Linux x86_64 binary: cast an oracle {figure}{method}{glyphs}. Exit code = {numbering} number.",
)


def _method_clause(method: Method, rng: random.Random, implicit_ok: bool) -> str:
    if method == COINS:
        if implicit_ok and rng.random() < 0.3:
            return ""
        return rng.choice((" using the three-coin method", " by tossing three coins per line"))
    if method == YARROW:
        return rng.choice((" using the yarrow-stalk method", " with yarrow-stalk probabilities"))
    a, b, c, d = method.sixteenths
    if rng.random() < 0.5:
        return (
            f" with line probabilities old yin {a}/16, young yang {b}/16, "
            f"young yin {c}/16, old yang {d}/16"
        )
    return f" where lines 6, 7, 8 and 9 come up {a}, {b}, {c} and {d} times in 16"


def _glyph_clause(glyphs: Glyphs, rng: random.Random, implicit_ok: bool) -> str:
    if glyphs == PLAIN and (implicit_ok or rng.random() < 0.5):
        return ""
    yin, yang = glyphs.young_yin, glyphs.young_yang
    yin_mark = glyphs.old_yin[len(yin) :]
    yang_mark = glyphs.old_yang[len(yang) :]
    return (
        f", drawing yang lines as `{yang}` and yin lines as `{yin}`, "
        f"with changing lines suffixed `{yin_mark}` (old yin) and `{yang_mark}` (old yang)"
    )


def render_prompt(spec: OracleSpec, rng: random.Random) -> str:
    implicit_ok = rng.random() < 0.5
    return rng.choice(_TEMPLATES).format(
        figure=rng.choice(_FIGURE_WORDS[spec.lines]),
        method=_method_clause(spec.method, rng, implicit_ok),
        glyphs=_glyph_clause(spec.glyphs, rng, implicit_ok),
        numbering=_NUMBERING[spec.lines],
    )


#: Code offsets (from the first instruction) of request-dependent immediates.
_HOT_CODE_OFFSETS = (0x0D, 0x3C, 0x41, 0x4C, 0x62, 0x70, 0x71)
_CODE_START = 120


def byte_roles(spec: OracleSpec) -> list[str]:
    """
    Label each byte of `spec`'s binary for loss weighting experiments:

      - "hot": depends on the request (sizes, immediates, method and glyph tables)
      - "table": the numbering table (fixed per line count, but every entry matters)
      - "scaffold": identical across all requests with the same layout
    """
    binary = spec.build()
    roles = ["scaffold"] * len(binary)
    for off in (96, 97, 104, 105):  # low bytes of p_filesz and p_memsz
        roles[off] = "hot"
    for off in _HOT_CODE_OFFSETS:
        roles[_CODE_START + off] = "hot"
    tables_at = _CODE_START + 0x7F
    numbers_at = tables_at + 16 + 4 * spec.glyphs.width
    for off in range(tables_at, numbers_at):
        roles[off] = "hot"
    for off in range(numbers_at, len(binary)):
        roles[off] = "table"
    return roles


def char_weights(
    spec: OracleSpec, hex_format: str, hot_weight: float = 1.0, table_weight: float = 1.0
) -> list[float]:
    """Per-character loss weights for the hex answer of `spec` (separators get 1.0)."""
    weight = {"scaffold": 1.0, "hot": hot_weight, "table": table_weight}
    stride = 3 if hex_format == "spaced" else 2
    roles = byte_roles(spec)
    out = [1.0] * len(format_hex(spec.build(), hex_format))
    for i, role in enumerate(roles):
        out[i * stride] = out[i * stride + 1] = weight[role]
    return out


def format_hex(binary: bytes, hex_format: str) -> str:
    return binary.hex(" ") if hex_format == "spaced" else binary.hex()


def generate(
    n: int, seed: int = 0, split: str = "train", hex_format: str = "plain"
) -> Iterator[Example]:
    """Yield `n` chat-format examples for one split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    rng = random.Random(f"{seed}:{split}")
    system = SPACED_SYSTEM_PROMPT if hex_format == "spaced" else SYSTEM_PROMPT
    for i in range(n):
        spec = sample_spec(rng, split)
        yield {
            "id": f"oracle-{split}-{i:05d}",
            "split": split,
            "spec": spec.to_json(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": render_prompt(spec, rng)},
                {"role": "assistant", "content": format_hex(spec.build(), hex_format)},
            ],
        }


def _main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hex-format", choices=("plain", "spaced"), default="plain")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        path = args.out / f"{split}.jsonl"
        with path.open("w") as f:
            for row in generate(n, args.seed, split, args.hex_format):
                f.write(json.dumps(row) + "\n")
        print(f"wrote {n} rows to {path}")


if __name__ == "__main__":
    _main()
