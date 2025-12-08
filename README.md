# BioMistral-7B QLoRA – Food–Polyphenol Extraction

Fine-tuning di **BioMistral-7B** con **QLoRA** per estrarre triple:

`food | compound | value | unit`

da tabelle e paragrafi scientifici (cibo, polifenoli, concentrazioni, ecc.).

Il dataset è in formato **chat-style JSONL** con meta-informazioni su split e chunk.

---

## 1. Struttura del progetto

Esempio di layout consigliato su Runpod:

```bash
/workspace/
  rocco/
    finetune_biomistral/
      train_biomistral_qlora.py
      requirements.txt
      data/
        dataset_biomistral_chat.jsonl
        dataset_biomistral_chat_sample.jsonl   # opzionale, per test veloci
      outputs/
        # qui finiranno gli adapter LoRA
