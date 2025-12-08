import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "BioMistral/BioMistral-7B"
ADAPTER_DIR = "outputs/biomistral_qlora_foodpoly_full_v1"


def build_prompt() -> str:
    system_prompt = (
        "You extract quantitative food–polyphenol information from scientific text.\n"
        "Given a table or paragraph from a scientific paper, you must output one or more lines, "
        "each in the format:\n"
        "food | compound | value | unit\n"
        "Use the exact wording of foods and compounds as they appear in the text, without paraphrasing.\n"
        "If there is no such information, respond with:\n"
        "NONE"
    )

    example_user = (
        "Task: extract all food–polyphenol–quantity triples from the following paragraph.\n\n"
        "Chunk type: paragraph\n"
        "Text:\n"
        "The amount of flavanols (mg of CE/100 g of DW) in garlic and white and red onions (Table 2)\n"
        "was estimated as 6 ± 0.4, 4 ± 0.4, and 5 ± 0.4, respectively, which was lower than the reported\n"
        "values of 6 for garlic and 3.37–11 for onions. The total flavanols varied from nondetectable in\n"
        "most of the vegetables to 184 mg/100 g of DW, found in a sample of broad bean. Anthocyanins\n"
        "(mg of CGE/kg of DW) in raw white and red onions were 28 ± 1.3 and 460 ± 10.9, respectively,\n"
        "and in raw red onion were significantly higher than in raw white onion. In garlic, anthocyanins\n"
        "were not detected."
    )

    example_assistant = (
        "garlic | flavanols | 6 | mg of CE/100 g DW\n"
        "white onion | flavanols | 4 | mg of CE/100 g DW\n"
        "red onion | flavanols | 5 | mg of CE/100 g DW\n"
        "broad bean | total flavanols | 184 | mg/100 g DW\n"
        "raw white onion | anthocyanins | 28 | mg of CGE/kg DW\n"
        "raw red onion | anthocyanins | 460 | mg of CGE/kg DW\n"
        "red onion | anthocyanins | 317-1951 | mg of CGE/kg DW\n"
        "white onion | anthocyanins | 5.69-333 | mg of CGE/kg DW"
    )

    new_user = (
        "Task: extract all food–polyphenol–quantity triples from the following paragraph.\n\n"
        "Chunk type: paragraph\n"
        "Text:\n"
        "tated sour dough. The total amount of the four ferulic acid dimers (diFAs) (8-O-4-diFA,\n"
        "5-S-diFA, 8-S-diFA, 8-S-benzofuran-diFA) was 385 µg/g d.m. in the rye wholemeal and it\n"
        "decreased to 357 µg/g d.m. in the bread crumb, although these changes were not statistically\n"
        "significant. Measurements of free phenolic acids in the samples showed that only free\n"
        "ferulic acid was detectable, and the amount increased from 3 µg/g in the rye wholemeal\n"
        "to 16 µg/g in the bread crumb."
    )

    prompt = (
        "You are a helpful extraction model.\n\n"
        + system_prompt
        + "\n\n### Example\n\n"
        + example_user
        + "\n\nExpected output:\n"
        + example_assistant
        + "\n\n### Now solve this new case\n\n"
        + new_user
        + "\n\nOutput:\n"
    )

    return prompt


def main():
    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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

    prompt = build_prompt()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    print("[*] Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n===== MODEL OUTPUT (FT) =====\n")
    print(decoded.strip())


if __name__ == "__main__":
    main()