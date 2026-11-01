"""Oracle program family: request specs, dataset generation and behavioral eval."""

from dataclasses import dataclass

from binary_gen.core.linux_oracle import COINS, PLAIN, Glyphs, Method, linux_oracle


@dataclass(frozen=True)
class OracleSpec:
    """Everything a prompt can ask for; the ground truth for one example."""

    lines: int = 6
    method: Method = COINS
    glyphs: Glyphs = PLAIN

    def build(self) -> bytes:
        return linux_oracle(self.lines, self.method, self.glyphs)

    def to_json(self) -> dict[str, object]:
        return {
            "lines": self.lines,
            "method": {"name": self.method.name, "sixteenths": list(self.method.sixteenths)},
            "glyphs": list(self.glyphs.entries()),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> OracleSpec:
        method = data["method"]
        glyphs = data["glyphs"]
        lines = data["lines"]
        if (
            not isinstance(method, dict)
            or not isinstance(glyphs, list)
            or not isinstance(lines, int)
        ):
            raise ValueError("Malformed oracle spec.")
        weights = tuple(int(n) for n in method["sixteenths"])
        if len(weights) != 4 or len(glyphs) != 4:
            raise ValueError("Oracle spec needs four weights and four glyphs.")
        return cls(
            lines=lines,
            method=Method(str(method["name"]), (weights[0], weights[1], weights[2], weights[3])),
            glyphs=Glyphs(*(str(g) for g in glyphs)),
        )
