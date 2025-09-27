import os

#os.system("pip install --upgrade pip")

# Core PyTorch and torchvision
os.system("pip install torch torchvision torchaudio")

# HuggingFace Transformers (for CLIPModel, CLIPProcessor)
os.system("pip install transformers")

# OpenAI CLIP repo (for `clip` package, if needed elsewhere in your code)
#os.system("pip install git+https://github.com/openai/CLIP.git")

# Standard ML/Data Science utilities
os.system("pip install numpy pandas matplotlib pillow tqdm")

# Optional: for data path loading from Kaggle (if you use `kagglehub`)
os.system("pip install kagglehub")

# Extras (just in case)
os.system("pip install ftfy regex")

print("✅ All necessary packages have been installed.")
