#!/bin/bash
set -e

echo "🚀 Starting build process..."

# Install Python dependencies
echo "📦 Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create cache directory
echo "📁 Creating cache directory..."
mkdir -p /opt/render/project/src/data

# Download and cache the model
echo "🤖 Downloading model..."
python -c '
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
model_name = "facebook/blenderbot-400M-distill"
cache_dir = "/opt/render/project/src/data/model"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer.save_pretrained(cache_dir)
model.save_pretrained(cache_dir)
print("✅ Model cached successfully!")
'
