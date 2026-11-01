"""
Linux x86_64 divination oracles: the first CRAP program family with a loop.

An oracle binary casts `lines` random lines (3 = a trigram, 6 = a hexagram),
prints them top line first, and exits with the figure's traditional number:
the King Wen sequence (1-64) for hexagrams, or the Earlier Heaven (先天)
numbering (1-8) for trigrams.

Each line is one of four values, as in the Yijing:

    6  old yin     (changing)      7  young yang
    8  young yin                    9  old yang      (changing)

Odd values (7, 9) are yang. The *method* decides how often each value turns
up: the three-coin method gives 2/16, 6/16, 6/16, 2/16 and the yarrow-stalk
method 1/16, 5/16, 7/16, 3/16. A method is stored as a 16-byte lookup table
indexed by four random bits, so any probabilities in sixteenths work.

Unlike `linux_print`, the code is not pure scaffolding. Four immediates and
one displacement depend on the request (line count, glyph width), and three
data tables follow the code. That makes it a family to learn rather than a
blob to memorize.

Code layout (127 bytes, then tables):

    0x00  sub rsp, 16
    0x04  getrandom(rsp, lines, 0)                 <- lines
    0x15  for i in 0..lines-1:                     <- lines (loop bound)
              v = METHOD[rand[i] & 15]; rand[i] = v
              bits |= (v & 1) << i
    0x3f  for i in lines-1..0:                     <- lines - 1
              write(1, GLYPHS + v * width, width)  <- width (twice)
    0x6d  exit(NUMBERS[bits])                      <- table displacement
    0x7f  METHOD[16] GLYPHS[4 * width] NUMBERS[2 ** lines]
"""

import struct
from dataclasses import dataclass

from binary_gen.core.linux_x86 import build_linux_binary

SYS_GETRANDOM = 318
SYS_WRITE = 1
SYS_EXIT = 60

# ─────────────────────────────────────────────────────────
# Methods and glyphs
# ─────────────────────────────────────────────────────────

LINE_VALUES = (6, 7, 8, 9)


@dataclass(frozen=True)
class Method:
    """How often each line value turns up, in sixteenths (must total 16)."""

    name: str
    sixteenths: tuple[int, int, int, int]  # for values 6, 7, 8, 9

    def __post_init__(self) -> None:
        if sum(self.sixteenths) != 16 or any(n < 0 for n in self.sixteenths):
            raise ValueError("Method weights must be non-negative and total 16.")

    def table(self) -> bytes:
        """The 16-byte lookup table: random nibble -> line index (value - 6)."""
        return bytes(i for i, n in enumerate(self.sixteenths) for _ in range(n))

    def probability(self, value: int) -> float:
        return self.sixteenths[value - 6] / 16


COINS = Method("coins", (2, 6, 6, 2))
YARROW = Method("yarrow", (1, 5, 7, 3))
METHODS = {m.name: m for m in (COINS, YARROW)}


@dataclass(frozen=True)
class Glyphs:
    """The text printed for each line value. Padded to a common width on build."""

    old_yin: str
    young_yang: str
    young_yin: str
    old_yang: str

    def entries(self) -> tuple[str, str, str, str]:
        return (self.old_yin, self.young_yang, self.young_yin, self.old_yang)

    @property
    def width(self) -> int:
        """Bytes per printed line, including the newline."""
        return max(len(e) for e in self.entries()) + 1

    def table(self) -> bytes:
        w = self.width - 1
        return b"".join(e.ljust(w).encode("ascii") + b"\n" for e in self.entries())

    def line_index(self, printed: str) -> int | None:
        """Map one printed line back to its index (value - 6); None if unknown."""
        for i, e in enumerate(self.entries()):
            if printed.rstrip() == e.rstrip():
                return i
        return None


PLAIN = Glyphs("---   ---  x", "---------", "---   ---", "---------  o")

# ─────────────────────────────────────────────────────────
# Numbering tables
# ─────────────────────────────────────────────────────────

# Trigram codes: bit 0 = bottom line, 1 = yang.
QIAN, DUI, LI, ZHEN, XUN, KAN, GEN, KUN = 7, 3, 5, 1, 6, 2, 4, 0
TRIGRAM_NAMES = {
    QIAN: "乾 Qian",
    DUI: "兌 Dui",
    LI: "離 Li",
    ZHEN: "震 Zhen",
    XUN: "巽 Xun",
    KAN: "坎 Kan",
    GEN: "艮 Gen",
    KUN: "坤 Kun",
}

_KW_LOWER = (QIAN, ZHEN, KAN, GEN, KUN, XUN, LI, DUI)
_KW_ROWS = {  # upper trigram -> King Wen numbers across _KW_LOWER
    QIAN: (1, 25, 6, 33, 12, 44, 13, 10),
    ZHEN: (34, 51, 40, 62, 16, 32, 55, 54),
    KAN: (5, 3, 29, 39, 8, 48, 63, 60),
    GEN: (26, 27, 4, 52, 23, 18, 22, 41),
    KUN: (11, 24, 7, 15, 2, 46, 36, 19),
    XUN: (9, 42, 59, 53, 20, 57, 37, 61),
    LI: (14, 21, 64, 56, 35, 50, 30, 38),
    DUI: (43, 17, 47, 31, 45, 28, 49, 58),
}


def _king_wen() -> tuple[int, ...]:
    table = [0] * 64
    for upper, row in _KW_ROWS.items():
        for lower, number in zip(_KW_LOWER, row, strict=True):
            table[lower | upper << 3] = number
    if sorted(table) != list(range(1, 65)):
        raise AssertionError("King Wen table is not a permutation of 1..64")
    return tuple(table)


#: Hexagram bits (bit 0 = bottom line, 1 = yang) -> King Wen number.
KING_WEN = _king_wen()

#: Trigram bits -> Earlier Heaven number: Qian 1, Dui 2, Li 3, Zhen 4,
#: Xun 5, Kan 6, Gen 7, Kun 8.
EARLIER_HEAVEN = tuple(
    {QIAN: 1, DUI: 2, LI: 3, ZHEN: 4, XUN: 5, KAN: 6, GEN: 7, KUN: 8}[b] for b in range(8)
)

NUMBERINGS = {3: EARLIER_HEAVEN, 6: KING_WEN}


def figure_number(lines: int, bits: int) -> int:
    """The traditional number of the figure with these yang bits."""
    return NUMBERINGS[lines][bits]


# ─────────────────────────────────────────────────────────
# Code
# ─────────────────────────────────────────────────────────

_CODE_SIZE = 0x7F
_LEA_METHOD_END = 0x20  # RIP after `lea r8, [rip + METHOD]`
_LEA_GLYPHS_END = 0x54  # RIP after `lea rsi, [rip + GLYPHS]`
_LEA_NUMBERS_END = 0x74  # RIP after `lea rax, [rip + NUMBERS]`


def _oracle_code(lines: int, width: int) -> bytes:
    method_at = _CODE_SIZE
    glyphs_at = method_at + 16
    numbers_at = glyphs_at + 4 * width

    code = b"".join(
        [
            b"\x48\x83\xec\x10",  # sub rsp, 16
            b"\xb8" + struct.pack("<I", SYS_GETRANDOM),  # mov eax, 318
            b"\x48\x89\xe7",  # mov rdi, rsp
            b"\xbe" + struct.pack("<I", lines),  # mov esi, lines
            b"\x31\xd2",  # xor edx, edx
            b"\x0f\x05",  # syscall
            b"\x31\xc9",  # xor ecx, ecx          (i)
            b"\x31\xdb",  # xor ebx, ebx          (yang bits)
            b"\x4c\x8d\x05" + struct.pack("<i", method_at - _LEA_METHOD_END),  # lea r8, METHOD
            # cast loop @ 0x20
            b"\x0f\xb6\x04\x0c",  # movzx eax, byte [rsp + rcx]
            b"\x83\xe0\x0f",  # and eax, 15
            b"\x41\x0f\xb6\x04\x00",  # movzx eax, byte [r8 + rax]
            b"\x88\x04\x0c",  # mov [rsp + rcx], al
            b"\x89\xc2",  # mov edx, eax
            b"\x83\xe2\x01",  # and edx, 1
            b"\xd3\xe2",  # shl edx, cl
            b"\x09\xd3",  # or ebx, edx
            b"\xff\xc1",  # inc ecx
            b"\x83\xf9" + bytes([lines]),  # cmp ecx, lines
            b"\x75\xe1",  # jne 0x20
            b"\x41\xbc" + struct.pack("<I", lines - 1),  # mov r12d, lines - 1
            # print loop @ 0x45
            b"\x42\x0f\xb6\x04\x24",  # movzx eax, byte [rsp + r12]
            b"\x6b\xc0" + bytes([width]),  # imul eax, eax, width
            b"\x48\x8d\x35" + struct.pack("<i", glyphs_at - _LEA_GLYPHS_END),  # lea rsi, GLYPHS
            b"\x48\x01\xc6",  # add rsi, rax
            b"\xb8" + struct.pack("<I", SYS_WRITE),  # mov eax, 1
            b"\xbf\x01\x00\x00\x00",  # mov edi, 1
            b"\xba" + struct.pack("<I", width),  # mov edx, width
            b"\x0f\x05",  # syscall
            b"\x41\xff\xcc",  # dec r12d
            b"\x79\xd8",  # jns 0x45
            b"\x48\x8d\x05" + struct.pack("<i", numbers_at - _LEA_NUMBERS_END),  # lea rax, NUMBERS
            b"\x0f\xb6\x3c\x18",  # movzx edi, byte [rax + rbx]
            b"\xb8" + struct.pack("<I", SYS_EXIT),  # mov eax, 60
            b"\x0f\x05",  # syscall
        ]
    )
    if len(code) != _CODE_SIZE:
        raise AssertionError(f"oracle code is {len(code)} bytes, layout assumes {_CODE_SIZE}")
    return code


def linux_oracle(lines: int = 6, method: Method = COINS, glyphs: Glyphs = PLAIN) -> bytes:
    """
    Build a Linux x86_64 binary that casts a random trigram or hexagram.

    Args:
        lines: 3 (trigram, exits 1-8) or 6 (hexagram, exits 1-64).
        method: Line-value probabilities; see `COINS` and `YARROW`.
        glyphs: ASCII text for each line value.

    Raises:
        ValueError: On an unsupported line count or a glyph width the
            `imul r32, r32, imm8` encoding cannot hold.
    """
    if lines not in NUMBERINGS:
        raise ValueError("Oracles cast 3 lines (trigram) or 6 lines (hexagram).")
    width = glyphs.width
    if not 2 <= width <= 127:
        raise ValueError("Glyph width must be 1-126 characters.")

    tables = method.table() + glyphs.table() + bytes(NUMBERINGS[lines])
    return build_linux_binary(_oracle_code(lines, width), tables)
