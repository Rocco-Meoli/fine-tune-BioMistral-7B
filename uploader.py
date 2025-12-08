from huggingface_hub import HfApi, create_repo, upload_folder

# CAMBIA QUESTO con il tuo username e nome repo
REPO_ID = "Pleomax06/biomistral-7b-foodpoly-qlora-v1"

FOLDER = "outputs/biomistral_qlora_foodpoly_full_v1"

def main():
    api = HfApi()

    # Crea repo se non esiste già
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,
    )

    print(f"[*] Upload della cartella '{FOLDER}' su '{REPO_ID}'...")
    upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=FOLDER,
        commit_message="Upload QLoRA adapter food-polyphenols v1",
    )
    print("[*] Fatto.")

if __name__ == "__main__":
    main()