import os
import subprocess
import sys
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, GenerationConfig

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
    
    # Create a clean generation config
    generation_config = GenerationConfig(
        max_length=128,
        min_length=10,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        do_sample=False
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
