import random
import struct

import pytest
from binary_gen.core.linux_oracle import (
    COINS,
    EARLIER_HEAVEN,
    KING_WEN,
    KUN,
    PLAIN,
    QIAN,
    YARROW,
    Glyphs,
    Method,
    linux_oracle,
)
from binary_gen.core.runner import Exited, RunResult, Signaled
from binary_gen.oracle import OracleSpec
from binary_gen.oracle.dataset import (
    byte_roles,
    char_weights,
    format_hex,
    generate,
    render_prompt,
    sample_spec,
)
from binary_gen.oracle.evaluate import byte_diff, check_cast, evaluate_binary, regions

CODE = 120  # file offset of the first instruction


def test_numbering_tables() -> None:
    assert sorted(KING_WEN) == list(range(1, 65))
    assert KING_WEN[0b111111] == 1  # 乾 doubled
    assert KING_WEN[0b000000] == 2  # 坤 doubled
    assert KING_WEN[QIAN | KUN << 3] == 11  # 泰: earth over heaven
    assert KING_WEN[KUN | QIAN << 3] == 12  # 否: heaven over earth
    assert KING_WEN[0b010010] == 29  # 坎 doubled
    assert KING_WEN[0b101101] == 30  # 離 doubled
    assert KING_WEN[0b010101] == 63  # 既濟: water over fire
    assert KING_WEN[0b101010] == 64  # 未濟: fire over water
    assert sorted(EARLIER_HEAVEN) == list(range(1, 9))


@pytest.mark.parametrize("method", [COINS, YARROW])
def test_method_tables(method: Method) -> None:
    table = method.table()
    assert len(table) == 16
    for value in (6, 7, 8, 9):
        assert table.count(value - 6) == method.sixteenths[value - 6]
    with pytest.raises(ValueError, match="total 16"):
        Method("bad", (1, 1, 1, 1))


@pytest.mark.parametrize(
    ("lines", "glyphs"),
    [(6, PLAIN), (3, PLAIN), (6, Glyphs("= = *", "===", "= =", "=== *"))],
)
def test_oracle_layout(lines: int, glyphs: Glyphs) -> None:
    binary = linux_oracle(lines, YARROW, glyphs)
    width = glyphs.width
    assert binary[:4] == b"\x7fELF"
    file_size = struct.unpack_from("<Q", binary, 64 + 32)[0]
    assert file_size == len(binary) == CODE + 0x7F + 16 + 4 * width + 2**lines
    # Request-dependent immediates.
    assert struct.unpack_from("<I", binary, CODE + 0x0D)[0] == lines  # getrandom count
    assert binary[CODE + 0x3C] == lines  # cast loop bound
    assert struct.unpack_from("<I", binary, CODE + 0x41)[0] == lines - 1  # print start
    assert binary[CODE + 0x4C] == width  # imul stride
    assert struct.unpack_from("<I", binary, CODE + 0x62)[0] == width  # write length
    # RIP-relative displacements land on each table.
    method_at = CODE + 0x20 + struct.unpack_from("<i", binary, CODE + 0x1C)[0]
    glyphs_at = CODE + 0x54 + struct.unpack_from("<i", binary, CODE + 0x50)[0]
    numbers_at = CODE + 0x74 + struct.unpack_from("<i", binary, CODE + 0x70)[0]
    assert binary[method_at : method_at + 16] == YARROW.table()
    assert binary[glyphs_at : glyphs_at + 4 * width] == glyphs.table()
    assert binary[numbers_at:] == bytes(KING_WEN if lines == 6 else EARLIER_HEAVEN)


def test_oracle_input_boundaries() -> None:
    with pytest.raises(ValueError, match="3 lines"):
        linux_oracle(4)
    with pytest.raises(ValueError, match="width"):
        linux_oracle(6, COINS, Glyphs("x" * 200, "a", "b", "c"))
    with pytest.raises(UnicodeEncodeError):
        linux_oracle(6, COINS, Glyphs("☰", "a", "b", "c"))


# ─────────────────────────────────────────────────────────
# Eval, against a simulated runner (the suite never executes binaries)
# ─────────────────────────────────────────────────────────


def _simulated(spec: OracleSpec, seed: int = 0, *, break_number: bool = False):
    """A runner that behaves like a correct (or subtly broken) oracle for `spec`."""
    rng = random.Random(seed)
    table = spec.method.table()
    numbers = KING_WEN if spec.lines == 6 else EARLIER_HEAVEN

    def run(_binary: bytes) -> RunResult:
        idx = [table[rng.randrange(16)] for _ in range(spec.lines)]
        bits = sum((i & 1) << k for k, i in enumerate(idx))
        out = b"".join(
            spec.glyphs.table()[i * spec.glyphs.width :][: spec.glyphs.width] for i in reversed(idx)
        )
        number = numbers[bits]
        if break_number and bits == 0:
            number += 1
        return RunResult(out, Exited(number))

    return run


@pytest.mark.parametrize("spec", [OracleSpec(), OracleSpec(3, YARROW)])
def test_eval_passes_correct_oracle(spec: OracleSpec) -> None:
    report = evaluate_binary(b"", spec, runs=2000, runner=_simulated(spec))
    assert report.passed, report.summary()


def test_eval_catches_wrong_method() -> None:
    asked = OracleSpec(method=YARROW)
    report = evaluate_binary(b"", asked, runs=2000, runner=_simulated(OracleSpec()))
    assert report.consistent
    assert not report.passed


def test_eval_catches_single_wrong_number() -> None:
    spec = OracleSpec()
    report = evaluate_binary(b"", spec, runs=3000, runner=_simulated(spec, break_number=True))
    assert report.outcomes["wrong_number"] > 0
    assert not report.passed


def test_eval_flags_impossible_values() -> None:
    no_change = OracleSpec(method=Method("stable", (0, 8, 8, 0)))
    report = evaluate_binary(b"", no_change, runs=500, runner=_simulated(OracleSpec()))
    assert report.impossible_values > 0
    assert not report.passed


def test_check_cast_rejects_crash_and_garbage() -> None:
    spec = OracleSpec()
    assert check_cast(RunResult(b"", Signaled(11)), spec)[0] == "signaled"
    assert check_cast(RunResult(b"hello\n", Exited(1)), spec)[0] == "wrong_line_count"


def test_byte_diff_regions() -> None:
    spec = OracleSpec()
    good = spec.build()
    assert byte_diff(good, spec)["exact"]
    spans = {name: (start, end) for name, start, end in regions(spec)}
    assert spans["numbering_table"][1] == len(good)
    bad = bytearray(good)
    bad[spans["method_table"][0]] ^= 1
    assert byte_diff(bytes(bad), spec)["wrong_bytes_by_region"] == {"method_table": 1}


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────


def test_dataset_rows_round_trip() -> None:
    rows = list(generate(n=200, seed=1))
    assert len(rows) == 200
    for row in rows:
        spec = OracleSpec.from_json(row["spec"])
        messages = row["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert bytes.fromhex(messages[2]["content"].replace(" ", "")) == spec.build()


def test_dataset_is_deterministic_and_split_disjoint() -> None:
    a = [r["messages"][1]["content"] for r in generate(n=50, seed=7)]
    b = [r["messages"][1]["content"] for r in generate(n=50, seed=7)]
    assert a == b
    train = {str(r["spec"]) for r in generate(n=400, seed=3, split="train")}
    test = {str(r["spec"]) for r in generate(n=200, seed=3, split="test")}
    assert not train & test


def test_byte_roles_cover_every_request_dependent_byte() -> None:
    rng = random.Random(5)
    specs = [sample_spec(rng, "train") for _ in range(300)]
    by_layout: dict[tuple[int, int], list[OracleSpec]] = {}
    for spec in specs:
        by_layout.setdefault((spec.lines, spec.glyphs.width), []).append(spec)
    compared = 0
    for group in by_layout.values():
        for a, b in zip(group, group[1:], strict=False):
            roles = byte_roles(a)
            for i, (x, y) in enumerate(zip(a.build(), b.build(), strict=True)):
                if x != y:
                    assert roles[i] == "hot", (i, roles[i])
            compared += 1
    assert compared > 50
    roles = byte_roles(OracleSpec())
    assert roles.count("table") == 64
    assert roles[:64] == ["scaffold"] * 64


@pytest.mark.parametrize("hex_format", ["plain", "spaced"])
def test_char_weights_align_with_hex(hex_format: str) -> None:
    spec = OracleSpec(lines=3, method=YARROW)
    text = format_hex(spec.build(), hex_format)
    weights = char_weights(spec, hex_format, hot_weight=5.0, table_weight=3.0)
    assert len(weights) == len(text)
    stride = 3 if hex_format == "spaced" else 2
    roles = byte_roles(spec)
    assert weights[-1] == 3.0  # last byte is the numbering table
    first_hot = roles.index("hot")
    assert weights[first_hot * stride] == weights[first_hot * stride + 1] == 5.0
    assert weights[0] == 1.0


def test_prompt_mentions_what_it_asks_for() -> None:
    rng = random.Random(0)
    for _ in range(50):
        spec = sample_spec(rng, "train")
        prompt = render_prompt(spec, rng)
        assert ("trigram" in prompt) == (spec.lines == 3)
