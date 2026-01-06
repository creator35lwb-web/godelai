#!/usr/bin/env python3
"""Update Hugging Face model card with Shakespeare results."""
import sys
import io

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from huggingface_hub import login, upload_file
from pathlib import Path
import os

# Get token from environment variable or command line argument
token = os.getenv('HUGGINGFACE_TOKEN') or (sys.argv[1] if len(sys.argv) > 1 else None)
if not token:
    raise ValueError("Please provide HF token via HUGGINGFACE_TOKEN env var or command line argument")

repo_id = 'YSenseAI/godelai-manifesto-v1'

print("Logging in to Hugging Face...")
login(token=token)

print("Uploading updated README.md...")
upload_file(
    path_or_fileobj=str(Path('huggingface/README.md')),
    path_in_repo='README.md',
    repo_id=repo_id
)

print("✅ Successfully updated Hugging Face model card with Shakespeare benchmark results!")
print(f"   View at: https://huggingface.co/{repo_id}")
