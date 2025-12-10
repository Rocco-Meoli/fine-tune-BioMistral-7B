#!/usr/bin/env python
import argparse
import math
from collections import Counter
from typing import Dict, List, Union

import torch
from datasets import load_dataset, Dataset
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass


# -------------------------------------------------------
# Normalizzazione messaggi (stessa del training)
# -------------------------------------------------------
def normalize_messages(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    - Concatena tutti i system in testa al primo user.
    - Rimuove ruoli diversi da user/assistant.
    - Droppa assistant iniziali.
    - Merga user/assistant consecutivi.
    - Assicura: inizio = user, fine = assistant, alternati.
    """
    if not msgs:
        return []

    # 1) concatena tutti i system
    system_chunks = [
        m.get("content", "")
        for m in msgs
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]
    system_pref = "\n".join([s for s in system_chunks if s.strip()])

    # 2) tieni solo user/assistant
    ua = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in msgs
        if m.get("role") in ("user", "assistant")
    ]
    if not ua:
        return []

    # 3) droppa assistant iniziali
    while ua and ua[0]["role"] == "assistant":
        ua.pop(0)
    if not ua:
        return []

    # 4) prendi system_pref e mettilo in testa al primo user
    if system_pref and ua and ua[0]["role"] == "user":
        first = ua[0]
        sep = "\n\n" if first["content"] else ""
        ua[0] = {
            "role": "user",
            "content": system_pref + sep + first["content"],
        }

    # 5) unisci consecutivi con stesso ruolo
    norm: List[Dict[str, str]] = []
    last_role = None
    for m in ua:
        role = m["role"]
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == last_role and norm:
            norm[-1]["content"] += "\n\n" + content
        else:
            norm.append({"role": role, "content": content})
            last_role = role

    if not norm:
        return []

    # 6) tronca finché non finisce con assistant
    if norm[-1]["role"] != "assistant":
        while norm and norm[-1]["role"] != "assistant":
            norm.pop()
    if len(norm) < 2:
        return []

    # 7) enforce alternanza user/assistant
    cleaned: List[Dict[str, str]] = []
    last_role = None
    for m in norm:
        role = m["role"]
        if role == last_role and cleaned:
            cleaned[-1]["content"] += "\n\n" + m["content"]
        else:
            cleaned.append(m)
            last_role = role

    if not cleaned or cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        return []

    return cleaned


# -------------------------------------------------------
# Carica split test
# -------------------------------------------------------
def load_test_split(data_path: str) -> Dataset:
    print(f"[INFO] Carico dataset da {data_path}")
    ds_all = load_dataset("json", data_files=data_path)["train"]

    def has_split(ex):
        meta = ex.get("meta") or {}
        s = meta.get("split")
        return s in ("train", "val", "test")

    ds_all = ds_all.filter(has_split)

    def is_supported(ex):
        msgs = ex.get("messages", [])
        norm = normalize_messages(msgs)
        if not norm:
            return False
        if norm[0]["role"] != "user":
            return False
        if norm[-1]["role"] != "assistant":
            return False
        return True

    ds_all = ds_all.filter(is_supported)

    def _is_test(ex):
        meta = ex.get("meta") or {}
        return meta.get("split") == "test"

    ds_test = ds_all.filter(_is_test)

    print(f"[INFO] Split test: {len(ds_test)} esempi")
    if len(ds_test) == 0:
        raise RuntimeError("Nessun esempio con meta.split == 'test' dopo il filtraggio.")
    return ds_test


# -------------------------------------------------------
# Collator per causal LM
# -------------------------------------------------------
@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_masks = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids)

        def pad_tensor(t: torch.Tensor, pad_value: int):
            pad_len = max_len - t.size(0)
            if pad_len <= 0:
                return t
            return torch.cat(
                [t, torch.full((pad_len,), pad_value, dtype=t.dtype)],
                dim=0,
            )

        input_ids = torch.stack(
            [pad_tensor(x, self.tokenizer.pad_token_id) for x in input_ids],
            dim=0,
        )
        attention_masks = torch.stack(
            [pad_tensor(x, 0) for x in attention_masks],
            dim=0,
        )
        labels = torch.stack(
            [pad_tensor(x, self.label_pad_token_id) for x in labels],
            dim=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }


# -------------------------------------------------------
# Tokenizzazione stile training (masking del prompt)
# -------------------------------------------------------
def make_tokenize_fn(tokenizer: AutoTokenizer, max_length: int):
    """
    Come nel training:
    - history = messaggi tranne ultimo assistant
    - full = history + risposta gold
    - labels = -100 sui token del prompt, loss solo sulla risposta
    """
    def tokenize_example(example):
        raw_msgs = example["messages"]
        msgs = normalize_messages(raw_msgs)
        if not msgs:
            raise ValueError("Example senza messaggi validi dopo normalizzazione.")

        if msgs[-1]["role"] != "assistant":
            raise ValueError("Ultimo messaggio non è assistant dopo normalizzazione.")

        history = msgs[:-1]
        if not history:
            raise ValueError("Conversazione senza history (solo assistant?).")

        # Prompt: solo history, con generation prompt
        prompt_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full: history + risposta gold
        full_text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        enc_full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = enc_full["input_ids"]
        labels = input_ids.copy()

        prompt_len = len(enc_prompt["input_ids"])
        if prompt_len > len(labels):
            prompt_len = len(labels)

        for i in range(prompt_len):
            labels[i] = -100

        enc_full["labels"] = labels
        return enc_full

    return tokenize_example


# -------------------------------------------------------
# Metriche: token_F1 e triple_F1
# -------------------------------------------------------
def token_f1(gold: str, pred: str) -> float:
    gold_tokens = gold.lower().split()
    pred_tokens = pred.lower().split()
    if not gold_tokens or not pred_tokens:
        return 0.0

    c_gold = Counter(gold_tokens)
    c_pred = Counter(pred_tokens)
    common = sum((c_gold & c_pred).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_triples(text: str):
    """
    Parsiamo righe tipo:
      food | compound | value | unit
    Ignoriamo 'NONE'.
    """
    triples = []
    text = (text or "").strip()
    if not text or text.upper() == "NONE":
        return set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        food, poly, value, unit = parts[:4]
        triples.append((food.lower(), poly.lower(), value.lower(), unit.lower()))
    return set(triples)


def triple_f1(gold_text: str, pred_text: str) -> float:
    gold_triples = parse_triples(gold_text)
    pred_triples = parse_triples(pred_text)
    if not gold_triples or not pred_triples:
        return 0.0

    inter = gold_triples & pred_triples
    if not inter:
        return 0.0

    precision = len(inter) / len(pred_triples)
    recall = len(inter) / len(gold_triples)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# -------------------------------------------------------
# Eval LM (loss / perplexity) sul test
# -------------------------------------------------------
def eval_lm(
    model,
    tokenizer,
    ds_test: Dataset,
    max_length: int,
    per_device_eval_batch_size: int,
):
    print("[INFO] Inizio eval LM su test (loss / perplexity, con masking prompt).")
    tokenize_fn = make_tokenize_fn(tokenizer, max_length)

    test_tok = ds_test.map(
        tokenize_fn,
        batched=False,
        remove_columns=ds_test.column_names,
    )

    data_collator = CausalLMCollator(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="eval_tmp",
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_drop_last=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    metrics = trainer.evaluate()
    loss = metrics.get("eval_loss", None)
    if loss is not None:
        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float("inf")
        print(f"[LM] eval_loss = {loss:.4f}, perplexity = {ppl:.2f}")
    else:
        print("[LM] Nessuna eval_loss trovata nei metrics:", metrics)


# -------------------------------------------------------
# Eval generativa su test
# -------------------------------------------------------
def eval_generation(
    model,
    tokenizer,
    ds_test: Dataset,
    max_length: int,
    max_new_tokens: int,
    num_examples: int = None,
    print_every: int = 10,
):
    model.eval()
    device = next(model.parameters()).device

    total_exact = 0
    total_f1 = 0.0
    total_triple_f1 = 0.0
    n = 0

    if num_examples is not None:
        ds_iter = ds_test.select(range(min(num_examples, len(ds_test))))
    else:
        ds_iter = ds_test

    with torch.no_grad():
        for idx, ex in enumerate(ds_iter):
            msgs = normalize_messages(ex["messages"])
            if not msgs or msgs[-1]["role"] != "assistant":
                continue

            history = msgs[:-1]
            gold = (msgs[-1]["content"] or "").strip()
            if not history:
                continue

            prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

            gen_tokens = gen_ids[0][input_ids.shape[1]:]
            pred = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            exact = int(pred == gold)
            f1 = token_f1(gold, pred)
            t_f1 = triple_f1(gold, pred)

            total_exact += exact
            total_f1 += f1
            total_triple_f1 += t_f1
            n += 1

            if (idx + 1) % print_every == 0:
                print(f"\n[Sample {idx+1}]")
                print("PROMPT:")
                print(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
                print("\nGOLD:")
                print(gold)
                print("\nPRED:")
                print(pred)
                print(f"\nexact={exact}, token_F1={f1:.3f}, triple_F1={t_f1:.3f}")

    if n == 0:
        print("[GEN] Nessun esempio valido valutato.")
        return

    exact_match = total_exact / n
    avg_f1 = total_f1 / n
    avg_triple_f1 = total_triple_f1 / n

    print("\n[GEN] Risultati aggregati su test:")
    print(f"  # esempi = {n}")
    print(f"  exact_match = {exact_match:.3f}")
    print(f"  avg token_F1 = {avg_f1:.3f}")
    print(f"  avg triple_F1 = {avg_triple_f1:.3f}")


# -------------------------------------------------------
# Argparse & main
# -------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Eval QLoRA BioMistral su dataset food–polyphenol")

    parser.add_argument("--data_path", type=str, required=True, help="Path al jsonl (es. dataset_reduced_none.jsonl)")
    parser.add_argument("--lora_dir", type=str, required=True, help="Directory con checkpoint LoRA (output_dir del train)")
    parser.add_argument("--base_model", type=str, default="BioMistral/BioMistral-7B")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--mode", type=str, choices=["lm", "gen", "both"], default="both")
    parser.add_argument("--num_examples", type=int, default=None, help="Per mode=gen: limita il numero di esempi")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Carico tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Configuro quantizzazione 4-bit (NF4) per eval.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    print(f"[INFO] Carico base model {args.base_model} e LoRA da {args.lora_dir}.")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    print("[INFO] Carico split test...")
    ds_test = load_test_split(args.data_path)

    if args.mode in ("lm", "both"):
        eval_lm(
            model=model,
            tokenizer=tokenizer,
            ds_test=ds_test,
            max_length=args.max_length,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
        )

    if args.mode in ("gen", "both"):
        eval_generation(
            model=model,
            tokenizer=tokenizer,
            ds_test=ds_test,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            num_examples=args.num_examples,
        )


if __name__ == "__main__":
    main()
