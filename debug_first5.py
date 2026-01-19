#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ----------------------------
# Utilities: printing + logging
# ----------------------------

class Tee:
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.f = open(file_path, "w", encoding="utf-8") if file_path else None

    def write(self, s: str):
        sys.stdout.write(s)
        if self.f:
            self.f.write(s)

    def flush(self):
        sys.stdout.flush()
        if self.f:
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# ----------------------------
# Parsing + normalization
# ----------------------------

NUM_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")
SPACE_RE = re.compile(r"\s+")
PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")

def norm_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def norm_text(x: str) -> str:
    x = norm_unicode(str(x))
    x = x.strip().strip('"').strip("'").strip().lower()
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

    # range 323-336 (avoid leading negative confusion)
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


def sanitize_model_output(raw: str) -> str:
    """
    Enforce output contract:
    - either EXACTLY 'NONE'
    - or one/more lines formatted: |food|polyphenol|value|unit|
    Anything else is discarded.
    """
    if not raw:
        return "NONE"

    text = norm_unicode(str(raw)).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "NONE"

    if norm_text(lines[0]) == "none":
        return "NONE"

    kept = []
    for ln in lines:
        raw_ln = ln.strip()

        # must contain at least 3 pipes (4 fields)
        if raw_ln.count("|") < 3:
            continue

        # skip markdown separator rows like |---|---:|
        stripped = raw_ln.replace("|", "").replace("-", "").replace(":", "").strip()
        if stripped == "":
            continue

        parts = [p.strip() for p in PIPE_SPLIT_RE.split(raw_ln.strip("| "))]
        if len(parts) < 4:
            continue
        parts = parts[:4]  # ignore extra columns

        f, c, v, u = parts
        f_n = norm_text(f)
        c_n = norm_text(c)
        v_spec = parse_value_spec(v)
        u_n = norm_unit(u)

        # discard header-like rows
        if f_n in ("food", "foods") and c_n in ("polyphenol", "polyphenols", "poly") and norm_text(v) in ("value", "quantity", "amount"):
            continue

        if not f_n or not c_n or not u_n or v_spec is None:
            continue

        if NUM_RE.fullmatch(c_n.replace(" ", "")) is not None:
            continue
        if NUM_RE.fullmatch(u_n) is not None:
            continue

        kept.append(f"|{f_n}|{c_n}|{v.strip()}|{u.strip()}|")

    return "\n".join(kept) if kept else "NONE"


def parse_triples_strict_pipes(text: str) -> List[Triple]:
    """
    Parser for already-sanitized output (or gold).
    Accepts lines that split into >=4 fields with pipes; uses first 4.
    """
    if not text or norm_text(text) == "none":
        return []

    triples: List[Triple] = []
    for line in str(text).splitlines():
        raw = line.strip()
        if not raw:
            continue

        if raw.count("|") < 3:
            continue

        stripped = raw.replace("|", "").replace("-", "").replace(":", "").strip()
        if stripped == "":
            continue

        parts = [p.strip() for p in PIPE_SPLIT_RE.split(raw.strip("| "))]
        if len(parts) < 4:
            continue
        parts = parts[:4]

        food = norm_text(parts[0])
        compound = norm_text(parts[1])
        val_spec = parse_value_spec(parts[2])
        unit = norm_unit(parts[3])

        if not food or not compound or not unit or val_spec is None:
            continue

        if food in ("food", "foods") and compound in ("polyphenol", "polyphenols", "poly"):
            continue

        if NUM_RE.fullmatch(compound.replace(" ", "")) is not None:
            continue
        if NUM_RE.fullmatch(unit) is not None:
            continue

        triples.append(Triple(food, compound, val_spec, unit))

    return triples


# ----------------------------
# Matching (RELAXED + SYMMETRIC)
# ----------------------------

def _pad(x: float, abs_tol: float, rel_tol: float) -> float:
    return abs_tol + rel_tol * abs(x)

def _spec_to_interval(v: ValueSpec, abs_tol: float, rel_tol: float) -> Tuple[float, float]:
    """
    Convert ValueSpec to padded interval [lo, hi].
    ge/le become half-infinite intervals.
    """
    if v.kind == "point":
        p = _pad(v.a, abs_tol, rel_tol)
        return (v.a - p, v.a + p)
    if v.kind == "range" and v.b is not None:
        lo = min(v.a, v.b)
        hi = max(v.a, v.b)
        return (lo - _pad(lo, abs_tol, rel_tol), hi + _pad(hi, abs_tol, rel_tol))
    if v.kind == "ge":
        return (v.a - _pad(v.a, abs_tol, rel_tol), float("inf"))
    if v.kind == "le":
        return (-float("inf"), v.a + _pad(v.a, abs_tol, rel_tol))
    return (float("inf"), -float("inf"))

def value_matches(pred: ValueSpec, gold: ValueSpec, abs_tol: float, rel_tol: float) -> bool:
    """
    Symmetric relaxed match: padded intervals overlap.
    """
    p_lo, p_hi = _spec_to_interval(pred, abs_tol, rel_tol)
    g_lo, g_hi = _spec_to_interval(gold, abs_tol, rel_tol)
    return max(p_lo, g_lo) <= min(p_hi, g_hi)

def unit_matches(pred_u: str, gold_u: str) -> bool:
    return norm_unit(pred_u) == norm_unit(gold_u)

def compound_matches(pred_c: str, gold_c: str) -> bool:
    return norm_text(pred_c) == norm_text(gold_c)

def food_matches(pred_f: str, gold_f: str) -> bool:
    return norm_text(pred_f) == norm_text(gold_f)

def match_triples_relaxed(
    pred: List[Triple],
    gold: List[Triple],
    abs_tol: float,
    rel_tol: float,
    ignore_food: bool
) -> Tuple[int, int, int, List[Tuple[Triple, Triple]], List[Triple]]:
    used = [False] * len(gold)
    matched_pairs: List[Tuple[Triple, Triple]] = []
    tp = 0
    fp = 0

    for p in pred:
        found = False
        for gi, g in enumerate(gold):
            if used[gi]:
                continue
            if not compound_matches(p.compound, g.compound):
                continue
            if not unit_matches(p.unit, g.unit):
                continue
            if not value_matches(p.value, g.value, abs_tol, rel_tol):
                continue
            if (not ignore_food) and (not food_matches(p.food, g.food)):
                continue
            used[gi] = True
            tp += 1
            matched_pairs.append((p, g))
            found = True
            break
        if not found:
            fp += 1

    fn = used.count(False)
    gold_missed = [gold[i] for i, u in enumerate(used) if not u]
    return tp, fp, fn, matched_pairs, gold_missed


# ----------------------------
# Prompting + generation
# ----------------------------

def build_prompt(chunk_text: str) -> str:
    return (
        "TASK: Extract quantitative food–polyphenol measurements.\n"
        "OUTPUT RULES (MUST FOLLOW):\n"
        "1) Output must be either EXACTLY: NONE\n"
        "   OR one or more lines, each strictly formatted as:\n"
        "   |food|polyphenol|value|unit|\n"
        "2) No headers. No markdown tables. No extra columns. No explanations.\n"
        "3) Each line must contain exactly these 4 fields.\n"
        "4) If there are no quantitative measurements, output EXACTLY: NONE\n"
        "\n"
        "TEXT:\n"
        f"{str(chunk_text).strip()}\n"
    )

@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    # IMPORTANT: tokenizer.model_max_length can be absurdly large -> OverflowError.
    MAX_CTX = 4096  # safe default for BioMistral-family usage
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CTX,
    ).to(model.device)

    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.05,
        no_repeat_ngram_size=4,
    )
    new_tokens = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


# ----------------------------
# Metrics helpers
# ----------------------------

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--lora", required=True, help="Path to LoRA adapter folder")
    ap.add_argument("--base_model", default="BioMistral/BioMistral-7B")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=0, help="0 = all rows, else first N rows")
    ap.add_argument("--abs_tol", type=float, default=0.01)
    ap.add_argument("--rel_tol", type=float, default=0.01)
    ap.add_argument("--relaxed_ignore_food", action="store_true")
    ap.add_argument("--out_txt", default=None)
    args = ap.parse_args()

    tee = Tee(args.out_txt)

    df = pd.read_csv(args.csv_path)
    cols = set(df.columns)

    if "chunk_text" not in cols:
        tee.write(f"[ERROR] Missing column 'chunk_text'. Columns={list(df.columns)}\n")
        tee.close()
        sys.exit(1)

    if "triples_text" not in cols:
        tee.write(f"[ERROR] Missing column 'triples_text'. Columns={list(df.columns)}\n")
        tee.close()
        sys.exit(1)

    paper_col = "paper_id" if "paper_id" in cols else None
    chunk_col = "chunk_id" if "chunk_id" in cols else None
    type_col = "chunk_type" if "chunk_type" in cols else None

    tee.write("[INFO] Loading base model + tokenizer...\n")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tee.write("[INFO] Loading LoRA adapter...\n")
    model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    N = len(df) if args.n == 0 else min(args.n, len(df))

    # global counters
    g_tp = g_fp = g_fn = 0
    chunks_gold_has = 0
    chunks_pred_has = 0
    chunks_format_ok = 0
    chunks_gold_none = 0
    chunks_pred_none_correct = 0

    for i in range(N):
        row = df.iloc[i]
        paper_id = row[paper_col] if paper_col else "NA"
        chunk_id = row[chunk_col] if chunk_col else str(i)
        ctype = row[type_col] if type_col else "NA"

        chunk_text = row["chunk_text"] if pd.notna(row["chunk_text"]) else ""
        gold_raw = row["triples_text"] if pd.notna(row["triples_text"]) else ""

        prompt = build_prompt(str(chunk_text))
        pred_raw = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
        )

        # enforce output contract
        pred_sanitized = sanitize_model_output(pred_raw)

        # parse
        pred_tr = parse_triples_strict_pipes(pred_sanitized)
        gold_tr = parse_triples_strict_pipes(gold_raw) if gold_raw and str(gold_raw).strip().upper() != "NONE" else []

        gold_is_none = (not gold_tr)
        pred_is_none = (norm_text(pred_sanitized) == "none")

        # format compliance
        if pred_is_none or len(pred_tr) > 0:
            chunks_format_ok += 1

        if gold_is_none:
            chunks_gold_none += 1
            if pred_is_none:
                chunks_pred_none_correct += 1
        else:
            chunks_gold_has += 1
            if len(pred_tr) > 0:
                chunks_pred_has += 1

        tp, fp, fn, matched, missed = match_triples_relaxed(
            pred_tr, gold_tr,
            abs_tol=args.abs_tol, rel_tol=args.rel_tol,
            ignore_food=args.relaxed_ignore_food
        )

        g_tp += tp
        g_fp += fp
        g_fn += fn

        tee.write("\n" + "=" * 110 + "\n")
        tee.write(f"[{i+1}/{N}] paper_id={paper_id} | chunk_id={chunk_id} | type={ctype}\n")
        tee.write("-" * 110 + "\n\n")

        tee.write("===== INPUT (chunk_text) =====\n\n")
        tee.write(str(chunk_text).strip() + "\n\n")

        tee.write("===== PRED (SANITIZED, CONTRACT-COMPLIANT) =====\n\n")
        tee.write(pred_sanitized + "\n\n")

        tee.write("===== GOLD RAW =====\n\n")
        tee.write((str(gold_raw).strip() if gold_raw else "NONE") + "\n\n")

        tee.write("===== DEBUG METRICS (RELAXED) =====\n")
        tee.write(f"pred parsed triples: {len(pred_tr)} | gold parsed triples: {len(gold_tr)}\n")
        tee.write(f"TP={tp} FP={fp} FN={fn} | ignore_food={args.relaxed_ignore_food}\n\n")

        tee.flush()

    P, R, F1 = prf(g_tp, g_fp, g_fn)
    fmt_ok_rate = chunks_format_ok / N if N else 0.0
    none_acc = (chunks_pred_none_correct / chunks_gold_none) if chunks_gold_none else 0.0
    chunk_recall = (chunks_pred_has / chunks_gold_has) if chunks_gold_has else 0.0

    tee.write("\n" + "=" * 110 + "\n")
    tee.write("[SUMMARY]\n")
    tee.write(f"Rows evaluated: {N}\n")
    tee.write(f"Format compliance rate: {fmt_ok_rate:.4f}  (NONE or >=1 parseable triple)\n")
    tee.write(f"NONE accuracy (only on gold-NONE chunks): {none_acc:.4f}  (correct NONE / gold NONE)\n")
    tee.write(f"Chunk recall (only on gold-HAS-triples chunks): {chunk_recall:.4f}  (pred had >=1 triple / gold had triples)\n")
    tee.write(f"Micro P/R/F1 (triple-level, relaxed): P={P:.4f} R={R:.4f} F1={F1:.4f}  (tp={g_tp} fp={g_fp} fn={g_fn})\n")

    if args.out_txt:
        tee.write(f"\n[DONE] Wrote output to: {args.out_txt}\n")
    tee.close()


if __name__ == "__main__":
    main()
