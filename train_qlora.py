#!/usr/bin/env python
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)


# -----------------------------
# Collator per causal LM con padding
# -----------------------------
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


# -----------------------------
# Argomenti CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA finetuning BioMistral-7B su food–polyphenol")

    parser.add_argument("--data_path", type=str, required=True, help="Path al JSONL (dataset_*_chat.jsonl / dataset_short.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory di output per i checkpoint")

    parser.add_argument("--model_name", type=str, default="BioMistral/BioMistral-7B")

    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


# -----------------------------
# Normalizzazione messaggi
# -----------------------------
def normalize_messages(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    - Concatena tutti i system in testa al primo user.
    - Rimuove ruoli diversi da user/assistant.
    - Droppa assistant iniziali.
    - Merga user consecutivi.
    - Assicura che l'ultimo messaggio sia assistant.
    """
    if not msgs:
        return []

    # sistema: concatena tutto in un prefisso
    system_chunks = [
        m.get("content", "")
        for m in msgs
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]
    system_pref = "\n".join([s for s in system_chunks if s.strip()])

    # tieni solo user/assistant in ordine
    ua = [
        {"role": m.get("role"), "content": m.get("content", "")}
        for m in msgs
        if m.get("role") in ("user", "assistant")
    ]

    if not ua:
        return []

    # droppa assistant iniziali
    while ua and ua[0]["role"] == "assistant":
        ua.pop(0)

    if not ua:
        return []

    # mergia system dentro al primo user
    if system_pref and ua and ua[0]["role"] == "user":
        first = ua[0]
        sep = "\n\n" if first["content"] else ""
        ua[0] = {
            "role": "user",
            "content": system_pref + sep + first["content"],
        }

    # merge messaggi consecutivi con lo stesso ruolo
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

    # tronca se l'ultimo non è assistant
    if norm[-1]["role"] != "assistant":
        while norm and norm[-1]["role"] != "assistant":
            norm.pop()
    if len(norm) < 2:
        return []

    # enforce alternanza user/assistant
    cleaned = []
    last_role = None
    for m in norm:
        if m["role"] == last_role:
            cleaned[-1]["content"] += "\n\n" + m["content"]
        else:
            cleaned.append(m)
            last_role = m["role"]

    if not cleaned or cleaned[0]["role"] != "user" or cleaned[-1]["role"] != "assistant":
        return []

    return cleaned


# -----------------------------
# Dataset helpers
# -----------------------------
def load_and_split_dataset(data_path: str) -> DatasetDict:
    print(f"[INFO] Carico dataset da {data_path}")
    ds_all = load_dataset("json", data_files=data_path)["train"]

    # filtra per esempi con meta.split valido
    def has_split(ex):
        meta = ex.get("meta") or {}
        s = meta.get("split")
        return s in ("train", "validation", "test")

    ds_all = ds_all.filter(has_split)

    # filtra esempi che, dopo normalizzazione, hanno una conversazione valida
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

    splits = {}
    for split_name in ["train", "validation", "test"]:
        def _f(ex, s=split_name):
            meta = ex.get("meta") or {}
            return meta.get("split") == s

        ds_split = ds_all.filter(_f)
        if len(ds_split) > 0:
            splits[split_name] = ds_split

    print("[INFO] Split disponibili:")
    for k, v in splits.items():
        print(f"  - {k}: {len(v)} esempi")

    return DatasetDict(splits)


# -----------------------------
# Tokenizzazione
# -----------------------------
def make_tokenize_fn(tokenizer: AutoTokenizer, max_length: int):
    def tokenize_example(example):
        raw_msgs = example["messages"]
        msgs = normalize_messages(raw_msgs)
        if not msgs:
            raise ValueError("Example senza messaggi validi dopo normalizzazione.")

        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )

        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    return tokenize_example


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Carico tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Configuro quantizzazione 4-bit (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    print(f"[INFO] Carico modello {args.model_name} in 4-bit.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    print("[INFO] Preparo modello per k-bit training.")
    model = prepare_model_for_kbit_training(model)

    print("[INFO] Applico LoRA.")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # Dataset
    # -------------------------
    ds_dict = load_and_split_dataset(args.data_path)
    train_ds = ds_dict.get("train")
    eval_ds = ds_dict.get("validation")

    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError("Nessun split 'train' trovato nel dataset dopo il filtraggio.")

    if args.do_eval and (eval_ds is None or len(eval_ds) == 0):
        print("[WARN] --do_eval richiesto ma non esiste split 'validation' valido. Procedo senza eval.")
        eval_ds = None

    tokenize_fn = make_tokenize_fn(tokenizer, args.max_length)

    print("[INFO] Tokenizzo train...")
    train_tok = train_ds.map(
        tokenize_fn,
        batched=False,
        remove_columns=train_ds.column_names,
    )

    eval_tok = None
    if args.do_eval and eval_ds is not None:
        print("[INFO] Tokenizzo validation...")
        eval_tok = eval_ds.map(
            tokenize_fn,
            batched=False,
            remove_columns=eval_ds.column_names,
        )

    # -------------------------
    # TrainingArguments
    # -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=args.bf16,
        tf32=args.tf32,
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to=[],
    )

    data_collator = CausalLMCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if args.do_eval and eval_tok is not None else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("[INFO] Inizio training QLoRA.")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_tok)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if args.do_eval and eval_tok is not None:
        print("[INFO] Eval finale su validation.")
        eval_metrics = trainer.evaluate(eval_tok)
        eval_metrics["eval_samples"] = len(eval_tok)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print("[INFO] Training QLoRA terminato.")


if __name__ == "__main__":
    main()