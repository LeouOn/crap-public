"""
LoRA fine-tune of CRAP (Gemma 4 E2B) on the oracle family.

STATUS: written without a GPU or Hugging Face access, so it has never run.
The data, byte roles and eval it calls are covered by the test suite; this
file's model-facing parts (chat template, LoRA targets, generation) are
best-effort and should be smoke-tested with `--max-steps 5 --eval-every 5`
before a real run.

Lessons from the CRAP paper baked in:
  - no sequence packing (unmasked packing silently contaminated their runs);
  - loss only on the answer (+ end-of-turn), never on the prompt;
  - the chat template is applied once and checked, so train and inference
    prompts are token-identical (a stray thinking token broke their serving);
  - the best checkpoint is chosen by behavioral pass rate, never by loss.

The experiment knobs: `--hot-weight` and `--table-weight` scale the loss on
request-dependent bytes and on the numbering table (see `byte_roles`). Both
default to 1.0, which is plain cross-entropy: the baseline.

    uv run --package binary-gen \
      --with "torch>=2.12" --with "transformers>=5.9" --with "peft>=0.19.1" \
      --with "accelerate>=1.13" \
      python oracle-train/train_lora.py --data data/oracle --out runs/baseline
"""

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from binary_gen.core import parse_hex
from binary_gen.oracle import OracleSpec
from binary_gen.oracle.dataset import char_weights
from binary_gen.oracle.evaluate import byte_diff, evaluate_binary
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoProcessor, get_cosine_schedule_with_warmup

# Language-model projections only: Gemma 4 E2B is multimodal, and the vision
# and audio towers should stay frozen. Checked at startup (fails if no match).
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune CRAP on oracle binaries")
    p.add_argument("--data", type=Path, required=True, help="dir with train/val/test.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default="matthewhaynesonline/gemma-4-E2B-it-crap-1.1")
    p.add_argument("--hex-format", choices=("plain", "spaced"), default="plain")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=0, help="override epochs if > 0")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--hot-weight", type=float, default=1.0)
    p.add_argument("--table-weight", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-examples", type=int, default=24)
    p.add_argument("--test-examples", type=int, default=200)
    p.add_argument("--eval-runs", type=int, default=300, help="executions per binary")
    p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ─────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────


def prompt_text(processor: Any, messages: list[dict[str, str]]) -> str:
    return processor.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)


def encode(row: dict[str, Any], processor: Any, tokenizer: Any, args: argparse.Namespace):
    messages = row["messages"]
    answer = messages[2]["content"]
    prompt = prompt_text(processor, messages)
    full = processor.apply_chat_template(messages, tokenize=False)
    start = len(prompt)
    if not full.startswith(prompt) or full[start : start + len(answer)] != answer:
        raise ValueError(
            "Chat template does not render the answer right after the generation prompt; "
            "train and inference prompts would differ. Inspect the template before training."
        )
    end = start + len(answer)

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    spec = OracleSpec.from_json(row["spec"])
    cw = char_weights(spec, args.hex_format, args.hot_weight, args.table_weight)

    labels, weights = [], []
    for tok, (a, b) in zip(ids, enc["offset_mapping"], strict=True):
        if b <= start:  # prompt
            labels.append(-100)
            weights.append(0.0)
        elif a >= end:  # end-of-turn and anything after: learn to stop
            labels.append(tok)
            weights.append(1.0)
        else:
            labels.append(tok)
            span = cw[max(a, start) - start : min(b, end) - start]
            weights.append(max(span) if span else 1.0)
    return ids, labels, weights


def collate(batch, pad_id: int, device: torch.device):
    n = max(len(ids) for ids, _, _ in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    weights = torch.zeros((len(batch), n), dtype=torch.float32)
    attention = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, lab, w) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(ids)] = torch.tensor(lab)
        weights[i, : len(ids)] = torch.tensor(w)
        attention[i, : len(ids)] = 1
    return input_ids.to(device), attention.to(device), labels.to(device), weights.to(device)


def weighted_loss(model, input_ids, attention, labels, weights) -> torch.Tensor:
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    logits, labels, weights = logits[:, :-1], labels[:, 1:], weights[:, 1:]
    keep = labels != -100
    ce = F.cross_entropy(logits[keep].float(), labels[keep], reduction="none")
    w = weights[keep]
    return (ce * w).sum() / w.sum()


# ─────────────────────────────────────────────────────────
# Behavioral eval
# ─────────────────────────────────────────────────────────


@torch.no_grad()
def behavioral_eval(model, processor, tokenizer, rows, args) -> dict[str, Any]:
    model.eval()
    tokenizer.padding_side = "left"
    passed, regions, outcomes = 0, Counter(), Counter()
    for i in range(0, len(rows), args.gen_batch):
        chunk = rows[i : i + args.gen_batch]
        prompts = [prompt_text(processor, r["messages"]) for r in chunk]
        enc = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
        )
        for row, seq in zip(chunk, out, strict=True):
            text = tokenizer.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            spec = OracleSpec.from_json(row["spec"])
            try:
                binary = parse_hex(text)
            except ValueError:
                outcomes["bad_hex"] += 1
                continue
            report = evaluate_binary(binary, spec, args.eval_runs)
            passed += report.passed
            outcomes.update(report.outcomes)
            regions.update(byte_diff(binary, spec)["wrong_bytes_by_region"])
    tokenizer.padding_side = "right"
    model.train()
    return {
        "pass_rate": passed / len(rows),
        "passed": passed,
        "n": len(rows),
        "outcomes": dict(outcomes),
        "wrong_bytes_by_region": dict(regions),
    }


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")

    def record(entry: dict[str, Any]) -> None:
        entry["time"] = time.time()
        log.write(json.dumps(entry) + "\n")
        log.flush()
        print(json.dumps(entry))

    record({"event": "config", **{k: str(v) for k, v in vars(args).items()}})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.base)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, attn_implementation="sdpa")
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    targets = [n for n, _ in model.named_modules() if re.fullmatch(LORA_TARGETS, n)]
    if not targets:
        raise SystemExit("No modules matched LORA_TARGETS; print model.named_modules() and fix it.")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_rows = read_jsonl(args.data / "train.jsonl")
    val_rows = read_jsonl(args.data / "val.jsonl")[: args.eval_examples]
    test_rows = read_jsonl(args.data / "test.jsonl")[: args.test_examples]
    encoded = [encode(r, processor, tokenizer, args) for r in train_rows]
    record({"event": "data", "train": len(encoded), "max_tokens": max(len(e[0]) for e in encoded)})

    steps_per_epoch = math.ceil(len(encoded) / (args.batch * args.grad_accum))
    total_steps = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)

    model.train()
    best = -1.0
    step = 0
    order: list[int] = []
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(args.grad_accum):
            if len(order) < args.batch:
                order += random.sample(range(len(encoded)), len(encoded))
            idx, order = order[: args.batch], order[args.batch :]
            batch = collate([encoded[i] for i in idx], tokenizer.pad_token_id, device)
            loss = weighted_loss(model, *batch) / args.grad_accum
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % 10 == 0:
            record(
                {"event": "train", "step": step, "loss": running, "lr": scheduler.get_last_lr()[0]}
            )

        if step % args.eval_every == 0 or step == total_steps:
            result = behavioral_eval(model, processor, tokenizer, val_rows, args)
            record({"event": "eval", "step": step, **result})
            if result["pass_rate"] > best:
                best = result["pass_rate"]
                model.save_pretrained(args.out / "best")
                record({"event": "saved_best", "step": step, "pass_rate": best})

    model.save_pretrained(args.out / "last")
    result = behavioral_eval(model, processor, tokenizer, test_rows, args)
    record({"event": "test_last", **result})

    # Score the checkpoint selected on val, which is the one to ship.
    set_peft_model_state_dict(model, load_peft_weights(str(args.out / "best")))
    result = behavioral_eval(model, processor, tokenizer, test_rows, args)
    record({"event": "test_best", **result})


if __name__ == "__main__":
    main()
