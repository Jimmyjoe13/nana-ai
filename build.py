import os
import subprocess
import sys

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
    from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
    
    model_name = "facebook/blenderbot-400M-distill"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    
    print(f"Loading model: {model_name}")
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    
    print("Saving tokenizer and model...")
    tokenizer.save_pretrained(cache_dir)
    model.save_pretrained(cache_dir)
    
    print("✅ Build completed successfully!")

if __name__ == "__main__":
    main()
