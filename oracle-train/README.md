# CRAP oracle: an I Ching that compiles itself

> 63 既濟 *After Completion* is followed by 64 未濟 *Before Completion*.
> The Book of Changes ends on "not yet across the river." Exclusions apply.

This fork teaches CRAP its first program with a loop: a Linux x86_64 binary
that casts a random trigram or hexagram, prints it, and exits with its
traditional number.

```
$ uv run binary-gen oracle --run --no-dump-hex
---   ---
---   ---
---------
---------
---   ---  x
---   ---
Exit code: 62        # 小過 Small Exceeding, second line changing
```

## What changed

| Path | What |
| --- | --- |
| `binary-gen/src/binary_gen/core/linux_oracle.py` | Ground-truth builder: 127 bytes of hand-encoded x86_64 plus three tables. `getrandom`, a cast loop, a print loop, `exit(table[bits])`. |
| `binary-gen/src/binary_gen/oracle/dataset.py` | Chat-format training data (same system prompt as CRAP). Varies line count, method and glyphs. Hash-based splits, so no test binary appears in train under any wording. |
| `binary-gen/src/binary_gen/oracle/evaluate.py` | Behavioral eval for random programs (below). |
| `binary-gen` CLI | `binary-gen oracle [--lines 3\|6] [--method coins\|yarrow] [--run]` |
| `oracle-train/train_lora.py` | LoRA fine-tune starting from CRAP v1.1. **Never run yet** (see status). |
| `tests/test_oracle.py` | Layout, tables, eval and dataset tests. Like the rest of the suite, they never execute binaries. |

## The program family

Each line is 6 (old yin), 7 (young yang), 8 (young yin) or 9 (old yang).
A **method** is a 16-byte table mapping four random bits to a value, so any
probabilities in sixteenths work:

| method | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- |
| coins (default) | 2/16 | 6/16 | 6/16 | 2/16 |
| yarrow | 1/16 | 5/16 | 7/16 | 3/16 |
| custom | any | | | |

Hexagrams exit with their King Wen number (1-64) and trigrams with their
Earlier Heaven (先天) number (1-8).

Unlike `print`/`exit`, the code isn't pure scaffolding. Seven code bytes
depend on the request: line count (×3), glyph width (×2), and a two-byte
table displacement. The method and glyph tables follow the code.
`byte_roles()` labels every byte as `scaffold`, `hot` or `table`. A
property test checks that every byte that differs between two requests is
labeled `hot`.

## Eval: scoring a program that is supposed to be random

One run can't be compared with a reference. `evaluate.py` runs each binary
N times and requires that every cast:

1. exits normally,
2. prints exactly `lines` lines, each one of the requested glyphs, and
3. exits with the number of the figure it printed.

Across all casts, line frequencies must fit the requested method (Pearson
χ², α = 0.001). `byte_diff` additionally reports which region a wrong byte
sits in. That's diagnostic only, never part of the score.

Smoke test against real executions, using ground-truth answers with two
deliberately broken:

```
oracle-test-00003 False 592/600 wrong_number: 8      numbering_table: 1 byte
oracle-test-00007 False   0/600 (truncated output)    glyph_table: 3, numbering_table: 64
passed 10/12
```

A single wrong King Wen entry (CRAP's characteristic 203→202 near miss) is
caught because its hexagram eventually comes up. Known limit: under heavily
skewed custom methods some hexagrams are vanishingly rare, so a wrong entry
for them is behaviorally almost invisible. `byte_diff` still shows it.

Running the eval on a non-Linux host goes through Docker once per execution,
which is slow at 1,000 runs. Use a Linux box.

## Training

```bash
uv sync --package binary-gen
uv run --package binary-gen python -m binary_gen.oracle.dataset --out data/oracle --train 4000
uv run --package binary-gen \
  --with "torch>=2.12" --with "transformers>=5.9" --with "peft>=0.19.1" --with "accelerate>=1.13" \
  python oracle-train/train_lora.py --data data/oracle --out runs/baseline
```

Choices taken from the CRAP paper:

- no sequence packing;
- loss only on the answer and end-of-turn;
- the template-rendered prompt is checked to be a prefix of the training
  text, so training and inference prompts match;
- the best checkpoint is chosen by behavioral pass rate on val, never by loss.

Experiments, each one a flag:

| run | flags | question |
| --- | --- | --- |
| baseline | none | Does CRAP v1.1 learn a looped program at all? |
| hot | `--hot-weight 5` | Does weighting the ~100 request-dependent bytes fix near-miss errors? |
| table | `--table-weight 5` | Does the King Wen table get memorized faster? |
| spaced | regenerate data with `--hex-format spaced`, train with `--hex-format spaced` | Does one-byte-per-chunk hex help the tokenizer? |

For hardware, the paper's fastest runs used a rented RTX 5090 (about 620
steps in 84 minutes). Its Apple-silicon run took 38.5 hours. The default
here is about 1,000 steps (4,000 examples, effective batch 8, 2 epochs).

### Status: honest

The builder, dataset, byte roles and eval are tested, and the eval was run
against real binaries. `train_lora.py` was written in a sandbox with no GPU
and no Hugging Face access, so it has never executed. Unverified
assumptions, in the order they're likely to bite:

1. `LORA_TARGETS` assumes Gemma 4's language-model modules live under
   `language_model`. The script exits if nothing matches.
2. The chat template renders the answer right after the generation prompt.
   The script checks this and exits with a message if not.
3. The tokenizer supports `return_offsets_mapping`, which needs a fast
   tokenizer.

Run `--max-steps 5 --eval-every 5 --eval-examples 4` first.

## Stretch goal: 太玄經

Yang Xiong's *Taixuan Jing* uses 81 tetragrams: four lines, three states
each. That's base 3, and 81 still fits in an exit code. Train on the Yijing
only, then ask for a Taixuan oracle. It will almost certainly fail, and how
it fails is the interesting part.
