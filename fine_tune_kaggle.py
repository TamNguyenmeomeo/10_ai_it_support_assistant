# --- KAGGLE TRAINING SCRIPT (ASCII ONLY FOR COMPATIBILITY) ---
# Instructions:
# 1. This file is pushed automatically to Kaggle via run_kaggle_automation.py
# 2. It will run on Kaggle GPU T4 using Unsloth.

import os
import shutil

# Clear Unsloth compiled cache to prevent version mismatch ValueError
shutil.rmtree("unsloth_compiled_cache", ignore_errors=True)

print("Installing Unsloth packages...")
os.system("pip install --no-deps "
          "xformers "
          "trl "
          "peft "
          "accelerate "
          "bitsandbytes")
os.system("pip install --no-deps unsloth_zoo unsloth")

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Model Configurations
max_seq_length = 2048
dtype = None # Auto detect (Float16 or Bfloat16)
load_in_4bit = True # 4bit quantization to save VRAM

print("Loading base model: Qwen2.5-Coder-1.5B-Instruct-bnb-4bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LoRA Configurations
print("Setting up LoRA Adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Set up Qwen 2.5 Chat Template (ShareGPT format)
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

def formatting_prompts_func(examples):
    convs = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convs]
    return { "text" : texts }

# Read dataset file uploaded to Kaggle
dataset_path = "it_tickets_dataset.json"
if not os.path.exists(dataset_path):
    # Scan input directories
    for root, dirs, files in os.walk("/kaggle/input"):
        if "it_tickets_dataset.json" in files:
            dataset_path = os.path.join(root, "it_tickets_dataset.json")
            break

print(f"Using dataset path: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True, remove_columns = dataset.column_names)

# SFT Trainer Configuration
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 30,
        max_steps = 300, # 300 steps is optimal for 1213 samples (about 2 epochs)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Start training
print("Starting training session...")
trainer_stats = trainer.train()
print("Training completed!")

# Merge model and save to 4-bit GGUF format (q4_k_m)
print("Saving and exporting model to GGUF format (q4_k_m)...")
model.save_pretrained_gguf("qwen_it_assistant_model", tokenizer, quantization_method = "q4_k_m")

print("Export completed successfully! Output GGUF model created.")
