import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "BioMistral/BioMistral-7B"

# <-- pesi corretti (quelli che stavi già usando)
ADAPTER_DIR = "/workspace/models/biomistral-7b-foodpoly-qlora-aug-v1"

MAX_CTX = 4096  # evita l'OverflowError del tokenizer
MAX_NEW_TOKENS = 512  # 128 è ridicolo per una tabella


TABLE_CHUNK = r"""| Variety | Ref | PHA1 | B2 | CA | E | PHA2 | Ph XyG | Unk | Ph G | PHA3 | Σ polyphenols | TP | TA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Calabaza | 1 | 1.71 | 31.09 | 196.67 | 49.16 | 18.98 | 30.75 | 1.30 | 9.54 | 0.85 | 340.04 | 1.29 | 5.51 |
| Campillo | 2 | 4.03 | 35.14 | 84.29 | 39.78 | 0.00 | 73.92 | 3.19 | 11.76 | 0.44 | 252.55 | 1.00 | 3.92 |
| Durón Arroes | 3 | 2.24 | 16.93 | 216.39 | 31.71 | 3.89 | 33.59 | 1.24 | 19.15 | 0.99 | 326.14 | | |
| Dolores | 4 | 0.61 | 0.00 | 74.98 | 10.51 | 6.84 | 19.95 | 0.81 | 6.53 | 0.48 | 120.70 | 0.59 | 4.17 |
| Lagar | 5 | 1.86 | 151.17 | 377.06 | 190.84 | 21.54 | 159.06 | 8.41 | 17.04 | 0.93 | 927.92 | 2.06 | 4.17 |
| Miyares | 6 | 1.73 | 49.69 | 70.33 | 66.13 | 31.49 | 51.94 | 2.13 | 18.15 | 0.93 | 292.52 | 1.42 | 3.55 |
| Reineta Encarnada | 7 | 0.89 | 28.78 | 119.17 | 49.73 | 3.81 | 11.63 | 0.52 | 6.60 | 0.81 | 221.94 | 1.04 | 4.90 |
| Repinaldo Hueso | 8 | 1.02 | 27.92 | 177.14 | 36.83 | 0.00 | 7.66 | 0.24 | 4.17 | 0.37 | 255.36 | 1.25 | 5.02 |
| Loroñesa | 9 | 1.93 | 18.73 | 339.11 | 29.59 | 19.02 | 36.23 | 1.29 | 13.95 | 0.63 | 460.47 | 1.56 | 4.29 |
| Montoto | 10 | 2.84 | 55.66 | 155.23 | 86.00 | 0.00 | 37.88 | 1.93 | 9.70 | 0.59 | 349.84 | 1.46 | 4.41 |
| Parda Carreño | 11 | 0.93 | 11.13 | 160.75 | 17.46 | 0.00 | 50.34 | 2.28 | 14.04 | 0.75 | 257.69 | 0.98 | 4.53 |"""


def build_prompt(table_text: str) -> str:
    # Regole ultra-rigide: o NONE o solo quadruple con pipe. Fine.
    return (
        "TASK: Extract quantitative food–polyphenol measurements from the input.\n"
        "OUTPUT RULES (MUST FOLLOW EXACTLY):\n"
        "1) If NO quantitative food–polyphenol measurement is present, output EXACTLY:\n"
        "NONE\n"
        "2) Otherwise output ONLY one or more lines, each EXACTLY formatted as:\n"
        "|food|polyphenol|value|unit|\n"
        "3) Do NOT output headers, explanations, markdown tables, extra columns, references, comments.\n"
        "4) Use the exact wording from the input when possible. Unit is ALWAYS required.\n"
        "\n"
        "Chunk type: table\n"
        "Text:\n"
        f"{table_text}\n"
        "\n"
        "Output:\n"
    )


def main():
    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[*] Loading LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    prompt = build_prompt(TABLE_CHUNK)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CTX,
    ).to(model.device)

    print("[*] Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n===== MODEL OUTPUT (FT) =====\n")
    print(decoded.strip())


if __name__ == "__main__":
    main()
