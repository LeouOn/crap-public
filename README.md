# CRAP: Compiler Replacement for Assembling Programs

A language model fine tuned to emit raw Linux x86_64 ELF binaries _directly_ as a hex string. No compiler, no assembler, no linker: the model writes the finished machine code byte for byte, and the bytes run.

Should you use this instead of a compiler? Sure, it works. It's also useless.

> **This fork** adds an I Ching oracle program family: CRAP's first target
> with a loop and randomness, plus a behavioral eval for random programs and
> a LoRA training script. See [`oracle-train/README.md`](oracle-train/README.md).

```
"Generate a Linux x86_64 binary that prints 'hello world'."

  7f454c4602010100...68656c6c6f20776f726c640a
  xxd -r -p > a.out && chmod +x a.out && ./a.out
  hello world
```

## Where things are

| What                        | Where                                                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| CRAPPED Browser Demo (hot!) | [huggingface.co/spaces/matthewhaynesonline/crap-demo](https://huggingface.co/spaces/matthewhaynesonline/crap-demo)          |
| Write up                    | [blog.studiohaynes.com/go/crap](https://blog.studiohaynes.com/go/crap)                                                      |
| Video                       | [youtu.be/55IBMzFQNYU](https://youtu.be/55IBMzFQNYU)                                                                        |
| Collection (v1.1)           | [huggingface.co/collections/matthewhaynesonline/crap11](https://huggingface.co/collections/matthewhaynesonline/crap11)      |
| Merged weights (v1.1)       | [matthewhaynesonline/gemma-4-E2B-it-crap-1.1](https://huggingface.co/matthewhaynesonline/gemma-4-E2B-it-crap-1.1)           |
| GGUF quants (v1.1)          | [matthewhaynesonline/gemma-4-E2B-it-crap-1.1-gguf](https://huggingface.co/matthewhaynesonline/gemma-4-E2B-it-crap-1.1-gguf) |
| Collection (v1)             | [huggingface.co/collections/matthewhaynesonline/crap](https://huggingface.co/collections/matthewhaynesonline/crap)          |
| Merged weights (v1)         | [matthewhaynesonline/gemma-4-E2B-it-crap](https://huggingface.co/matthewhaynesonline/gemma-4-E2B-it-crap)                   |
| GGUF quants (v1)            | [matthewhaynesonline/gemma-4-E2B-it-crap-gguf](https://huggingface.co/matthewhaynesonline/gemma-4-E2B-it-crap-gguf)         |

The model cards have the usage examples, recommended sampler flags, eval methodology and known limitations. Start there if you only want to run it.

## What's in this repo

- **`binary-gen/`**: the ELF and Mach-O builder. Constructs minimal binaries
  from a description. Generation and execution are here; the evaluation and
  inspection commands stay private.
- **`demo/`**: a small app for poking at the model interactively.
- **`reference/`**: hand written assembly and C used to work out what the
  minimal binaries should look like.
- **`crapped/`**: the browser demo; see its README for build instructions.
- **`paper/paper.pdf`**: the technical report.

The Python demo runs on a small public inference shim.

## Local development

```bash
uv sync --all-packages
```

The top level `pyproject.toml` only keeps tooling consistent across the
workspace; it doesn't ship anything.

### Git hooks

`pre-commit` runs `ruff` (format + lint) and `pyright` before every commit,
including the per-package checks:

```bash
uv run pre-commit install
```

## Related

- [ARM assembly syntax highlighting](https://marketplace.visualstudio.com/items?itemName=dan-c-underwood.arm)
- [ImHex](https://github.com/werwolv/imhex): hex editor, handy for staring at
  the diff between a correct binary and one the model got slightly wrong
