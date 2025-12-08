import argparse
import json
import os

import numpy as np
import torch

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -------------------------
# Helpers per dataset
# -------------------------

def build_splits(dataset):
    """
    dataset: Dataset con colonne ["messages", "meta", ...]
    meta è un dict con chiave "split" (train/val/test)
    """
    def is_split(example, split_name):
        meta = example.get("meta", {})
        return meta.get("split", "train") == split_name

    train_ds = dataset.filter(lambda e: is_split(e, "train"))
    val_ds = dataset.filter(lambda e: is_split(e, "val"))
    test_ds = dataset.filter(lambda e: is_split(e, "test"))

    dd = DatasetDict()
    if len(train_ds) > 0:
        dd["train"] = train_ds
    if len(val_ds) > 0:
        dd["validation"] = val_ds
    if len(test_ds) > 0:
        dd["test"] = test_ds
    return dd


def split_messages(example):
    """
    Dato un record con "messages": [ {role, content}, ... ],
    separa in prompt (system+user) e risposta (assistant).
    Assumo il formato:
      messages[0] = system
      messages[1] = user
      messages[2] = assistant
    Se ci sono più messaggi, prendo il primo assistant.
    """
    msgs = example["messages"]

    system = ""
    user = ""
    assistant = ""

    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system" and system == "":
            system = content
        elif role == "user" and user == "":
            user = content
        elif role == "assistant" and assistant == "":
            assistant = content

    # Sanity fallback: se manca qualcosa, non crashare
    if system is None:
        system = ""
    if user is None:
        user = ""
    if assistant is None:
        assistant = ""

    # prompt "piatto", niente chat template magico: va bene per finetune task-specifico
    prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
    response = " " + assistant

    return prompt, response


def tokenize_example(example, tokenizer, max_length):
    """
    Converte (prompt, risposta) in input_ids / labels
    con loss solo sulla parte di risposta.
    """
    prompt, response = split_messages(example)

    # Tokenizza separatamente prompt e prompt+risposta
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    full_text = prompt + response
    full_enc = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # Maschera loss sulla parte di prompt
    labels = [-100] * len(input_ids)
    prompt_len = min(len(prompt_ids), len(input_ids))
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_datasets(raw_dd, tokenizer, max_length):
    tokenized = {}

    for split_name, ds in raw_dd.items():
        cols = ds.column_names

        def _map_fn(example):
            return tokenize_example(example, tokenizer, max_length)

        new_ds = ds.map(
            _map_fn,
            remove_columns=cols,  # teniamo solo input_ids, attention_mask, labels
        )
        tokenized[split_name] = new_ds

    return DatasetDict(tokenized)


# -------------------------
# Metriche
# -------------------------

def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


def char_f1(pred: str, truth: str):
    """
    Char-level F1: rozzo ma utile per vedere se sta copiando quasi tutto bene.
    """
    p = list(pred)
    t = list(truth)
    if len(p) == 0 and len(t) == 0:
        return 1.0, 1.0, 1.0
    if len(p) == 0 or len(t) == 0:
        return 0.0, 0.0, 0.0

    # multiset match
    from collections import Counter
    cp = Counter(p)
    ct = Counter(t)
    inter = sum((cp & ct).values())

    precision = inter / len(p)
    recall = inter / len(t)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def make_compute_metrics(tokenizer):

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Predizioni: quando si usa predict_with_generate, 'preds' sono token generati
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )

        # Labels: sostituisco i -100 con pad_token_id,
        # poi decodifico e normalizzo
        labels_copy = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels_copy, skip_special_tokens=True
        )

        exact_matches = []
        f1s = []
        precs = []
        recs = []

        for pred, gold in zip(decoded_preds, decoded_labels):
            p_norm = normalize_text(pred)
            g_norm = normalize_text(gold)

            exact = 1.0 if p_norm == g_norm and g_norm != "" else 0.0
            precision, recall, f1 = char_f1(p_norm, g_norm)

            exact_matches.append(exact)
            f1s.append(f1)
            precs.append(precision)
            recs.append(recall)

        metrics = {
            "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
            "char_f1": float(np.mean(f1s)) if f1s else 0.0,
            "char_precision": float(np.mean(precs)) if precs else 0.0,
            "char_recall": float(np.mean(recs)) if recs else 0.0,
        }
        return metrics

    return compute_metrics


# -------------------------
# QLoRA model setup
# -------------------------

def load_qlora_model_and_tokenizer(model_name: str, cache_dir: str = None):
    """
    Carica BioMistral-7B (o simile) in 4-bit e lo avvolge con LoRA.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"[INFO] Carico modello {model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.config.use_cache = False

    print("[INFO] Preparo modello per k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Target modules tipici per Mistral/LLaMA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
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

    return model, tokenizer


# -------------------------
# Main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="BioMistral/BioMistral-7B",
        help="Nome del modello HF",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset_biomistral_chat.jsonl",
        help="Path al JSONL con il dataset chat",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/biomistral-qlora-foodpoly",
        help="Dove salvare il modello LoRA",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir HF (opzionale)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Max sequence length per esempio",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=200,
        help="Passi tra una eval e l'altra",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Passi tra un salvataggio e l'altro",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Usa BF16 se supportato (A40: sì)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Usa FP16 (se non vuoi BF16)",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Esegue eval su split validation",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------
    # Carico modello/tokenizer
    # ---------------------
    model, tokenizer = load_qlora_model_and_tokenizer(
        args.model_name, cache_dir=args.cache_dir
    )

    # ---------------------
    # Carico dataset
    # ---------------------
    print(f"[INFO] Carico dataset da {args.data_path}")
    raw = load_dataset(
        "json",
        data_files=args.data_path,
        split="train",
    )

    dd = build_splits(raw)
    print("[INFO] Split disponibili:", list(dd.keys()))
    for k in dd:
        print(f"  - {k}: {len(dd[k])} esempi")

    tokenized_dd = prepare_datasets(dd, tokenizer, args.max_length)

    train_dataset = tokenized_dd.get("train", None)
    eval_dataset = tokenized_dd.get("validation", None)

    if train_dataset is None:
        raise ValueError("Nessuno split 'train' trovato nel dataset.")

    # ---------------------
    # TrainingArguments
    # ---------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if args.do_eval and eval_dataset is not None else "no",
        eval_steps=args.eval_steps if args.do_eval and eval_dataset is not None else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["none"],
        logging_first_step=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        load_best_model_at_end=args.do_eval and eval_dataset is not None,
        do_eval=args.do_eval and eval_dataset is not None,
        prediction_loss_only=not (args.do_eval and eval_dataset is not None),
    )

    # ---------------------
    # Trainer
    # ---------------------
    compute_metrics = make_compute_metrics(tokenizer) if args.do_eval and eval_dataset is not None else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.do_eval and eval_dataset is not None else None,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ---------------------
    # Train
    # ---------------------
    print("[INFO] Inizio training QLoRA...")
    trainer.train()

    # Salva solo gli adapter LoRA
    print("[INFO] Salvo adapter LoRA...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[INFO] Finito.")


if __name__ == "__main__":
    main()
