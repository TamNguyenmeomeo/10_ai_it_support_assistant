import os
import shutil
import time
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    print("Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    username = api.config_values['username']
    print(f"Successfully authenticated as Kaggle user: {username}")

    # --- PART 1: Prepare and Upload/Update Dataset ---
    dataset_dir = "kaggle_dataset_dir"
    os.makedirs(dataset_dir, exist_ok=True)
    shutil.copy("it_tickets_dataset.json", os.path.join(dataset_dir, "it_tickets_dataset.json"))

    dataset_slug = "it-tickets-dataset-augmented"
    dataset_title = "IT Tickets Dataset Augmented"
    meta = {
      "title": dataset_title,
      "id": f"{username}/{dataset_slug}",
      "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(os.path.join(dataset_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Check if dataset already exists on Kaggle
    dataset_exists = False
    try:
        api.dataset_status(f"{username}/{dataset_slug}")
        dataset_exists = True
    except Exception:
        pass

    if not dataset_exists:
        print("Dataset does not exist on Kaggle. Creating new dataset...")
        api.dataset_create_new(dataset_dir, quiet=False)
        print("Dataset created successfully! Waiting for Kaggle processing (15s)...")
        time.sleep(15)
    else:
        print("Dataset already exists. Uploading new version...")
        api.dataset_create_version(dataset_dir, "Update augmented data", quiet=False)
        print("Dataset version updated successfully! Waiting for Kaggle processing (10s)...")
        time.sleep(10)

    # --- PART 2: Prepare and Push Training Kernel ---
    kernel_dir = "kaggle_kernel_dir"
    os.makedirs(kernel_dir, exist_ok=True)
    shutil.copy("fine_tune_kaggle.py", os.path.join(kernel_dir, "fine_tune_kaggle.py"))

    kernel_slug = "fine-tune-qwen-coder-1-5b"
    kernel_meta = {
      "id": f"{username}/{kernel_slug}",
      "title": "Fine Tune Qwen Coder 1.5B",
      "code_file": "fine_tune_kaggle.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": [
        f"{username}/{dataset_slug}"
      ],
      "competition_sources": [],
      "kernel_sources": []
    }

    with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_meta, f, indent=2)

    print("Pushing fine-tuning kernel to Kaggle...")
    api.kernels_push(kernel_dir)
    print(f"Kernel pushed successfully! Slug: {username}/{kernel_slug}")

    # --- PART 3: Monitor Kernel Execution and Download Output ---
    print("\nStarting real-time monitoring of Kaggle GPU training...")
    print("This will take about 15-25 minutes. Please keep this script running.")
    print("-" * 50)
    
    last_status = None
    while True:
        try:
            status_info = api.kernels_status(f"{username}/{kernel_slug}")
            status = status_info.status.name.lower()
            failure_msg = status_info.failure_message
            
            if status != last_status:
                print(f"[{time.strftime('%H:%M:%S')}] Status changed to: {status.upper()}")
                if failure_msg:
                    print(f"Error detail: {failure_msg}")
                last_status = status

            if status in ['complete', 'error', 'cancel_requested', 'cancel_acknowledged']:
                break
        except Exception as e:
            print(f"Status check warning: {e}")
            
        time.sleep(30)

    print("-" * 50)
    if last_status == 'complete':
        print("Training completed successfully on Kaggle! Downloading the output model...")
        
        # Download output files
        api.kernels_output(f"{username}/{kernel_slug}", path=".")
        
        # Check if the output GGUF model was downloaded
        expected_filename = "qwen_it_assistant_model-unsloth.Q4_K_M.gguf"
        if os.path.exists(expected_filename):
            print(f"Success! Model file downloaded: {expected_filename} (~1.3 GB)")
            print("\nSetup Instructions:")
            print("1. Create your Modelfile: `echo \"FROM ./qwen_it_assistant_model-unsloth.Q4_K_M.gguf\" > Modelfile`")
            print("2. Create Ollama model: `ollama create my-it-assistant -f Modelfile`")
            print("3. Start streamlit: `streamlit run app_vi.py` and select 'my-it-assistant' from sidebar.")
        else:
            # Maybe the file was downloaded as a zipped folder, list output files
            print("Output files downloaded. Checking files:")
            for f in os.listdir("."):
                if f.endswith(".gguf"):
                    print(f"- {f}")
    else:
        print(f"Kaggle training ended with status: {last_status.upper()}")
        if failure_msg:
            print(f"Reason: {failure_msg}")

if __name__ == "__main__":
    main()
