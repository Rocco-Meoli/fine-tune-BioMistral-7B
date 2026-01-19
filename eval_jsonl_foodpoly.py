#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ----------------------------
# Normalization + parsing
# ----------------------------

SPACE_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")

def norm_text(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("−", "-").replace("–", "-").replace("—", "-")
    x = SPACE_RE.sub(" ", x)
    return x

def norm_unit(u: str) -> str:
    u = norm_text(u)
    u = u.replace("f.w.", "fw").replace("fresh weight", "fw")
    u = u.replace("mg/kg of fw", "mg/kg fw")
    u = u.replace("mg/kg fw", "mg/kgfw")
    u = u.replace(" %", "%")
    u = u.replace(" /", "/").replace("/ ", "/")
    u = u.replace("% retention vs fresh berries", "%retentionvsfreshberries")
    u = SPACE_RE.sub("", u)
    return u

def safe_float(s: str) -> Optional[float]:
    s = norm_text(s).replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return None

@dataclass(frozen=True)
class ValueSpec:
    kind: str  # "point" | "range" | "ge" | "le"
    a: float
    b: Optional[float] = None

def parse_value_spec(raw: str) -> Optional[ValueSpec]:
    s = norm_text(raw).replace(",", "")
    s = s.replace("≥", ">=").replace("≤", "<=")

    # range 323-336
    if "-" in s and not s.startswith("-"):
        parts = [p.strip() for p in s.split("-") if p.strip()]
        if len(parts) == 2:
            lo = safe_float(parts[0])
            hi = safe_float(parts[1])
            if lo is not None and hi is not None:
                lo2, hi2 = (lo, hi) if lo <= hi else (hi, lo)
                return ValueSpec("range", lo2, hi2)

    if s.startswith(">="):
        x = safe_float(s[2:].strip())
        return ValueSpec("ge", x) if x is not None else None
    if s.startswith("<="):
        x = safe_float(s[2:].strip())
        return ValueSpec("le", x) if x is not None else None

    if s.startswith(">"):
        x = safe_float(s[1:].strip())
        return ValueSpec("ge", x) if x is not None else None
    if s.startswith("<"):
        x = safe_float(s[1:].strip())
        return ValueSpec("le", x) if x is not None else None

    x = safe_float(s)
    if x is not None:
        return ValueSpec("point", x)

    return None

@dataclass
class Triple:
    food: str
    compound: str
    value: ValueSpec
    unit: str

def parse_triples_any(text: str) -> List[Triple]:
    """
    Accepts lines like:
      |food|poly|value|unit|
    Ignores markdown separators and headers.
    """
    if not text:
        return []

    triples: List[Triple] = []
    for line in str(text).splitlines():
        raw = line.strip()
        if not raw:
            continue

        # ignore markdown separators
        stripped = raw.replace("|", "").replace("-", "").replace(":", "").strip()
        if stripped == "":
            continue

        if "|" not in raw:
            continue

        parts = [p.strip() for p in raw.strip().strip("|").split("|")]
        if len(parts) != 4:
            continue

        food = norm_text(parts[0])
        compound = norm_text(parts[1])
        val_spec = parse_value_spec(parts[2])
        unit = norm_unit(parts[3])

        if not food or not compound or val_spec is None:
            continue
        if not unit:
            # if ignore_unit is used we still parse a placeholder later
            unit = ""

        # reject junk
        if NUM_RE.fullmatch(compound.replace(" ", "")) is not None:
            continue

        triples.append(Triple(food, compound, val_spec, unit))

    return triples


# ----------------------------
# Matching
# ----------------------------

def value_matches(pred: ValueSpec, gold: ValueSpec, abs_tol: float, rel_tol: float) -> bool:
    if pred.kind != "point":
        return False
    x = pred.a

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= (abs_tol + rel_tol * abs(b))

    if gold.kind == "point":
        return close(x, gold.a)

    if gold.kind == "range" and gold.b is not None:
        lo, hi = gold.a, gold.b
        lo2 = lo - (abs_tol + rel_tol * abs(lo))
        hi2 = hi + (abs_tol + rel_tol * abs(hi))
        return lo2 <= x <= hi2

    if gold.kind == "ge":
        thr = gold.a
        thr2 = thr - (abs_tol + rel_tol * abs(thr))
        return x >= thr2

    if gold.kind == "le":
        thr = gold.a
        thr2 = thr + (abs_tol + rel_tol * abs(thr))
        return x <= thr2

    return False

def match_triples_relaxed(
    pred: List[Triple],
    gold: List[Triple],
    abs_tol: float,
    rel_tol: float,
    ignore_food: bool,
    ignore_unit: bool,
) -> Tuple[int, int, int]:
    used = [False] * len(gold)
    tp = 0
    fp = 0

    for p in pred:
        found = False
        for gi, g in enumerate(gold):
            if used[gi]:
                continue
            if norm_text(p.compound) != norm_text(g.compound):
                continue
            if not ignore_unit:
                if norm_unit(p.unit) != norm_unit(g.unit):
                    continue
            if not value_matches(p.value, g.value, abs_tol, rel_tol):
                continue
            if not ignore_food:
                if norm_text(p.food) != norm_text(g.food):
                    continue

            used[gi] = True
            tp += 1
            found = True
            break
        if not found:
            fp += 1

    fn = used.count(False)
    return tp, fp, fn


# ----------------------------
# Contract enforce
# ----------------------------

def sanitize_to_contract(raw: str) -> str:
    if not raw:
        return "NONE"
    s = raw.strip()
    if not s:
        return "NONE"

    if s.strip().upper() == "NONE":
        return "NONE"

    valid_lines: List[str] = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue

        # drop common headers
        if t.lower().replace(" ", "") in (
            "|food|polyphenol|value|unit|",
            "|food|polyphenol|value|unit",
            "food|polyphenol|value|unit",
            "|food|polyphenol|value|unit|reference|",
        ):
            continue

        if "|" not in t:
            continue

        parts = [p.strip() for p in t.strip().strip("|").split("|")]
        if len(parts) != 4:
            continue

        food, poly, val, unit = parts
        if not food or not poly or not val or not unit:
            continue

        valid_lines.append(f"|{food}|{poly}|{val}|{unit}|")

    return "\n".join(valid_lines) if valid_lines else "NONE"


# ----------------------------
# FIX: robust chat history builder
# ----------------------------

def coerce_alternating_history(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    """
    Returns:
      - history suitable for apply_chat_template(add_generation_prompt=True)
      - gold answer (assistant content) if present else ""
    Strategy:
      - keep first system (optional)
      - merge consecutive same-role messages (user/user or assistant/assistant)
      - gold = last assistant message if present
      - history = everything up to last user message (inclusive), dropping any trailing assistant
      - ensure roles in history alternate (system allowed only at start)
    """
    if not messages:
        return [], ""

    # 1) keep only system/user/assistant and strip
    cleaned = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("system", "user", "assistant"):
            continue
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    if not cleaned:
        return [], ""

    # 2) keep first system only
    system_msg = None
    rest = cleaned
    if cleaned[0]["role"] == "system":
        system_msg = cleaned[0]
        rest = cleaned[1:]
    # drop any other system messages (rare but happens)
    rest = [m for m in rest if m["role"] != "system"]

    # 3) merge consecutive same role
    merged = []
    for m in rest:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(m)

    # reattach system
    if system_msg is not None:
        merged2 = [system_msg] + merged
    else:
        merged2 = merged

    # 4) gold = last assistant if exists
    gold = ""
    for m in reversed(merged2):
        if m["role"] == "assistant":
            gold = m["content"]
            break

    # 5) history = everything up to last user
    last_user_idx = None
    for i in range(len(merged2) - 1, -1, -1):
        if merged2[i]["role"] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return [], gold

    history = merged2[: last_user_idx + 1]

    # 6) enforce alternation for apply_chat_template: after optional system, it must alternate user/assistant...
    fixed = []
    for m in history:
        if not fixed:
            fixed.append(m)
            continue
        if fixed[-1]["role"] == m["role"]:
            fixed[-1]["content"] += "\n\n" + m["content"]
        else:
            fixed.append(m)

    # Now remove any assistant messages in history (we want to generate assistant)
    # BUT: some datasets include few-shot examples (user/assistant pairs) before the final user.
    # That is OK as long as alternation holds. We keep them.
    # We already ended at last user, so last role is user. Great.

    return fixed, gold


# ----------------------------
# Inference
# ----------------------------

@torch.inference_mode()
def generate_from_history(model, tokenizer, history: List[Dict[str, str]], max_input_tokens: int, max_new_tokens: int) -> str:
    # apply_chat_template may still fail if template is strict; fallback to flattened format
    try:
        prompt_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # fallback: simple flatten
        prompt_text = ""
        for m in history:
            prompt_text += f"{m['role'].upper()}:\n{m['content']}\n\n"
        prompt_text += "ASSISTANT:\n"

    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, enc["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_path", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base_model", default="BioMistral/BioMistral-7B")
    ap.add_argument("--max_input_tokens", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--rel_tol", type=float, default=0.01)
    ap.add_argument("--ignore_food", action="store_true")
    ap.add_argument("--ignore_unit", action="store_true")
    ap.add_argument("--out_txt", default=None)
    args = ap.parse_args()

    print("[INFO] Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    # Read examples
    examples = []
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if args.n and args.n > 0:
        examples = examples[:args.n]

    out_f = open(args.out_txt, "w", encoding="utf-8") if args.out_txt else None
    def log(s: str):
        print(s)
        if out_f:
            out_f.write(s + "\n")

    total = 0
    format_ok = 0
    gold_none = 0
    none_correct = 0
    gold_has = 0
    chunk_recall_hits = 0
    tp = fp = fn = 0

    log("=" * 100)
    log(f"[CONFIG] ignore_food={args.ignore_food} | ignore_unit={args.ignore_unit} | abs_tol={args.abs_tol} | rel_tol={args.rel_tol}")
    log("=" * 100)

    for ex in examples:
        msgs = ex.get("messages", [])
        history, gold_raw = coerce_alternating_history(msgs)
        if not history:
            continue

        # meta if present
        meta = ex.get("meta", {}) or {}
        paper_id = meta.get("paper_id", "NA")
        chunk_id = meta.get("chunk_id", meta.get("chunk_id", "NA"))
        chunk_type = meta.get("chunk_type", "NA")

        pred_raw = generate_from_history(
            model, tokenizer, history,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        pred_san = sanitize_to_contract(pred_raw)

        gold_is_none = (not gold_raw) or (gold_raw.strip().upper() == "NONE")
        gold_tr = [] if gold_is_none else parse_triples_any(gold_raw)
        pred_tr = [] if pred_san.strip().upper() == "NONE" else parse_triples_any(pred_san)

        total += 1
        if pred_san.strip().upper() == "NONE" or len(pred_tr) > 0:
            format_ok += 1

        if gold_is_none:
            gold_none += 1
            if pred_san.strip().upper() == "NONE":
                none_correct += 1
        else:
            gold_has += 1
            if pred_san.strip().upper() != "NONE" and len(pred_tr) > 0:
                chunk_recall_hits += 1

        _tp, _fp, _fn = match_triples_relaxed(
            pred_tr, gold_tr,
            abs_tol=args.abs_tol, rel_tol=args.rel_tol,
            ignore_food=args.ignore_food,
            ignore_unit=args.ignore_unit,
        )
        tp += _tp
        fp += _fp
        fn += _fn

        # keep per-example log short
        log("\n" + "-" * 100)
        log(f"[{total}] paper_id={paper_id} | chunk_id={chunk_id} | type={chunk_type}")
        log("PRED (SANITIZED):")
        log(pred_san)
        log("GOLD:")
        log(gold_raw if gold_raw else "NONE")
        log(f"TP={_tp} FP={_fp} FN={_fn} | pred_triples={len(pred_tr)} gold_triples={len(gold_tr)}")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    format_rate = format_ok / total if total else 0.0
    none_acc = none_correct / gold_none if gold_none else 0.0
    chunk_recall = chunk_recall_hits / gold_has if gold_has else 0.0

    log("\n" + "=" * 100)
    log("[SUMMARY]")
    log(f"Rows evaluated: {total}")
    log(f"Format compliance rate: {format_rate:.4f}  (NONE or >=1 parseable triple)")
    log(f"NONE accuracy (only on gold-NONE chunks): {none_acc:.4f}  (correct NONE / gold NONE)")
    log(f"Chunk recall (only on gold-HAS-triples chunks): {chunk_recall:.4f}  (pred had >=1 triple / gold had triples)")
    log(f"Micro P/R/F1 (triple-level, relaxed): P={prec:.4f} R={rec:.4f} F1={f1:.4f}  (tp={tp} fp={fp} fn={fn})")
    log("=" * 100)

    if out_f:
        out_f.close()
        print(f"[DONE] Wrote detailed log to: {args.out_txt}")

if __name__ == "__main__":
    main()
