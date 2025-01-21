import os
import subprocess
import sys
import torch

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, check=True)
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
    print("🚀 Starting build process...")

    # Install dependencies
    print("📦 Installing dependencies...")
    run_command("python -m pip install --upgrade pip")
    run_command("pip install -r requirements.txt")

    # Import transformers after installation
    from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, GenerationConfig

    # Create cache directory
    print("📁 Creating cache directory...")
    cache_dir = "/opt/render/project/src/data/model"
    os.makedirs(cache_dir, exist_ok=True)

    # Download and cache model
    print("🤖 Downloading model...")
    model_name = "facebook/blenderbot-400M-distill"
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    
    print(f"Loading model: {model_name}")
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    
    # Test du modèle
    print("Testing model...")
    test_input = "Hello"
    inputs = tokenizer(test_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            min_length=10,
            num_beams=4,
            early_stopping=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test response: {response}")
    
    # Nettoyer la configuration du modèle
    print("Cleaning model config...")
    if hasattr(model.config, "max_length"):
        delattr(model.config, "max_length")
    if hasattr(model.config, "min_length"):
        delattr(model.config, "min_length")
    if hasattr(model.config, "num_beams"):
        delattr(model.config, "num_beams")
    if hasattr(model.config, "length_penalty"):
        delattr(model.config, "length_penalty")
    if hasattr(model.config, "no_repeat_ngram_size"):
        delattr(model.config, "no_repeat_ngram_size")
    
    # Configuration des tokens spéciaux
    print("Setting up special tokens...")
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Create a clean generation config
    print("Creating generation config...")
    generation_config = GenerationConfig(
        max_length=128,
        min_length=10,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=False,
        decoder_start_token_id=tokenizer.bos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    print("Saving model with clean config...")
    model.generation_config = generation_config
    
    print("Saving tokenizer and model...")
    tokenizer.save_pretrained(cache_dir)
    model.save_pretrained(cache_dir)
    generation_config.save_pretrained(cache_dir)
    
    print("✅ Build completed successfully!")

if __name__ == "__main__":
    main()
