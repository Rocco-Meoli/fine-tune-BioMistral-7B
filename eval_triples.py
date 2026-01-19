#!/usr/bin/env python3
import re
import csv
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Triple:
    food: str
    compound: str
    value: Optional[float]
    unit: str


# -----------------------------
# Normalization helpers
# -----------------------------
SPACE_RE = re.compile(r"\s+")
PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

NONE_SET = {"none", "null", "n/a", "na", "-", ""}

def norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("µ", "u").replace("μ", "u")
    s = s.replace("–", "-").replace("−", "-")
    s = SPACE_RE.sub(" ", s)
    return s

def norm_unit(u: str) -> str:
    u = norm_text(u)
    u = u.replace("per", "/")
    u = u.replace("fw", "fresh weight")
    u = SPACE_RE.sub(" ", u).strip()
    return u

def parse_float(s: str) -> Optional[float]:
    s = norm_text(s)
    m = NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None


# -----------------------------
# Parsing model output / gold
# -----------------------------
def is_none_text(text: str) -> bool:
    t = norm_text(text)
    return t in NONE_SET

def parse_triples_from_text(text: str, pipes_only: bool = True) -> List[Triple]:
    """
    Strict format (one per line):
      food | compound | value | unit

    If pipes_only=True: ignore anything not matching pipe format.
    """
    triples: List[Triple] = []
    if not text or is_none_text(text):
        return triples

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "|" not in ln:
            if pipes_only:
                continue
            else:
                continue

        parts = PIPE_SPLIT_RE.split(ln)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 4:
            continue

        food = norm_text(parts[0])
        comp = norm_text(parts[1])
        val = parse_float(parts[2])
        unit = norm_unit(parts[3])

        # validity checks
        if not food or not comp or val is None or not unit:
            continue

        triples.append(Triple(food, comp, val, unit))

    # Deduplicate (preserve order)
    return list(dict.fromkeys(triples))


# -----------------------------
# Matching logic
# -----------------------------
def values_close(a: Optional[float], b: Optional[float], abs_tol: float, rel_tol: float) -> bool:
    if a is None or b is None:
        return False
    if math.isfinite(a) and math.isfinite(b):
        if abs(a - b) <= abs_tol:
            return True
        if b != 0 and abs(a - b) / abs(b) <= rel_tol:
            return True
    return False

def soft_match(pred: Triple, gold: Triple, abs_tol: float, rel_tol: float) -> bool:
    return (
        pred.food == gold.food
        and pred.compound == gold.compound
        and norm_unit(pred.unit) == norm_unit(gold.unit)
        and values_close(pred.value, gold.value, abs_tol=abs_tol, rel_tol=rel_tol)
    )

def greedy_match(preds: List[Triple], golds: List[Triple], abs_tol: float, rel_tol: float) -> Tuple[int,int,int]:
    matched_g = set()
    tp = 0
    for p in preds:
        hit = None
        for j, g in enumerate(golds):
            if j in matched_g:
                continue
            if soft_match(p, g, abs_tol, rel_tol):
                hit = j
                break
        if hit is not None:
            tp += 1
            matched_g.add(hit)
    fp = len(preds) - tp
    fn = len(golds) - tp
    return tp, fp, fn


# -----------------------------
# Model inference
# -----------------------------
def build_prompt(text: str, chunk_type: str) -> str:
    return (
        "Task: extract all food polyphenol quantity triples from the following text.\n\n"
        f"Chunk type: {chunk_type}\n"
        "Return ONLY lines in this exact format (one triple per line):\n"
        "food | compound | value | unit\n\n"
        "Do NOT use numbering, bullets, sentences, or extra text.\n"
        "If no triples are present, return exactly: NONE\n\n"
        "Text:\n"
        f"{text}\n"
    )

@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return decoded.strip()


def load_model(base_model: str, lora_path_or_repo: str, load_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    model = PeftModel.from_pretrained(base, lora_path_or_repo)
    model.eval()
    return model, tokenizer


# -----------------------------
# CSV reading
# -----------------------------
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def pick_fields(row: Dict[str,str]) -> Tuple[str,str,str]:
    chunk_type = row.get("chunk_type") or row.get("type") or row.get("chunk") or "table"
    input_text = row.get("text") or row.get("input") or row.get("chunk_text") or row.get("table") or ""
    gold_text = row.get("output") or row.get("gold") or row.get("label") or row.get("target") or ""
    return chunk_type, input_text, gold_text


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--base_model", default="BioMistral/BioMistral-7B")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--rel_tol", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pipes_only", action="store_true", help="Count ONLY pipe-format outputs (recommended).")
    args = ap.parse_args()

    model, tokenizer = load_model(args.base_model, args.lora, load_4bit=True)

    rows = read_csv_rows(args.csv_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    total_tp = total_fp = total_fn = 0
    n = len(rows)

    n_with_parsed_preds = 0
    n_nonpipe_answers = 0

    examples_bad = []

    for idx, row in enumerate(rows, 1):
        chunk_type, inp, gold = pick_fields(row)

        prompt = build_prompt(inp, chunk_type)
        pred_text = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)

        gold_tr = parse_triples_from_text(gold, pipes_only=True)  # gold should be clean
        pred_tr = parse_triples_from_text(pred_text, pipes_only=args.pipes_only)

        if len(pred_tr) > 0:
            n_with_parsed_preds += 1
        else:
            # model said something but we couldn't parse it (common failure mode)
            if pred_text.strip() and not is_none_text(pred_text):
                n_nonpipe_answers += 1

        tp, fp, fn = greedy_match(pred_tr, gold_tr, args.abs_tol, args.rel_tol)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if (fp + fn) > 0 and len(examples_bad) < 5:
            examples_bad.append((idx, pred_text, gold, (tp, fp, fn)))

        if idx % 10 == 0 or idx == n:
            print(f"[{idx}/{n}] TP={total_tp} FP={total_fp} FN={total_fn}", flush=True)

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0

    print("\n===== RESULTS =====")
    print(f"rows: {n}")
    print(f"coverage (>=1 parsed triple): {n_with_parsed_preds}/{n} = {n_with_parsed_preds/n:.3f}")
    if args.pipes_only:
        print(f"answered but NOT parseable as pipes (counted as empty): {n_nonpipe_answers}/{n}")
    print(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    if examples_bad:
        print("\n===== SAMPLE ERROR CASES (up to 5) =====")
        for i, pred_text, gold, (tp, fp, fn) in examples_bad:
            print(f"\n--- Case {i} (tp={tp}, fp={fp}, fn={fn}) ---")
            print("[PRED RAW]\n", pred_text)
            print("[GOLD RAW]\n", gold)


if __name__ == "__main__":
    main()
