# Hugging Face Upload Instructions for Claude Code

**Upload GodelAI to Hugging Face Hub**

---

## Prerequisites

You will need a Hugging Face API token from Alton. The token should have write access.

---

## Step 1: Install huggingface_hub

```bash
pip install huggingface_hub
```

## Step 2: Login with Token

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")  # Alton will provide this
```

## Step 3: Create Repository

```python
from huggingface_hub import create_repo

repo_id = "creator35lwb-web/godelai-manifesto-v1"  # Or Alton's preferred name

create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True
)
```

## Step 4: Upload Files

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="/home/ubuntu/godelai/huggingface",
    repo_id=repo_id,
    repo_type="model",
    ignore_patterns=["*.py", "UPLOAD_INSTRUCTIONS.md"]  # Don't upload scripts
)
```

Or upload individual files:

```python
from huggingface_hub import upload_file

# Upload model card
upload_file(
    path_or_fileobj="/home/ubuntu/godelai/huggingface/README.md",
    path_in_repo="README.md",
    repo_id=repo_id
)

# Upload checkpoint
upload_file(
    path_or_fileobj="/home/ubuntu/godelai/huggingface/checkpoints/godelai_manifesto_v1.pt",
    path_in_repo="checkpoints/godelai_manifesto_v1.pt",
    repo_id=repo_id
)
```

## Step 5: Upload Core Framework Code

```python
# Upload the agent code
upload_file(
    path_or_fileobj="/home/ubuntu/godelai/godelai/agent.py",
    path_in_repo="godelai/agent.py",
    repo_id=repo_id
)

# Upload core module
upload_file(
    path_or_fileobj="/home/ubuntu/godelai/godelai/core/godelai_agent.py",
    path_in_repo="godelai/core/godelai_agent.py",
    repo_id=repo_id
)

# Upload __init__.py
upload_file(
    path_or_fileobj="/home/ubuntu/godelai/godelai/__init__.py",
    path_in_repo="godelai/__init__.py",
    repo_id=repo_id
)
```

## Step 6: Verify Upload

```python
from huggingface_hub import list_repo_files

files = list_repo_files(repo_id=repo_id)
print("Uploaded files:")
for f in files:
    print(f"  - {f}")
```

---

## Complete Script

Save this as `upload_to_hf.py` and run with the token:

```python
#!/usr/bin/env python3
"""Upload GodelAI to Hugging Face Hub"""

import os
from huggingface_hub import login, create_repo, upload_file, list_repo_files

def upload_godelai(token: str, repo_id: str = "creator35lwb-web/godelai-manifesto-v1"):
    """Upload GodelAI to Hugging Face."""
    
    print("Logging in to Hugging Face...")
    login(token=token)
    
    print(f"Creating repository: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    
    # Files to upload
    files = [
        ("huggingface/README.md", "README.md"),
        ("huggingface/checkpoints/godelai_manifesto_v1.pt", "checkpoints/godelai_manifesto_v1.pt"),
        ("godelai/agent.py", "godelai/agent.py"),
        ("godelai/core/godelai_agent.py", "godelai/core/godelai_agent.py"),
        ("godelai/__init__.py", "godelai/__init__.py"),
    ]
    
    base_path = "/home/ubuntu/godelai"
    
    for local_path, repo_path in files:
        full_path = os.path.join(base_path, local_path)
        if os.path.exists(full_path):
            print(f"Uploading: {repo_path}")
            upload_file(
                path_or_fileobj=full_path,
                path_in_repo=repo_path,
                repo_id=repo_id
            )
        else:
            print(f"Warning: {full_path} not found")
    
    print("\n✅ Upload complete!")
    print(f"View at: https://huggingface.co/{repo_id}")
    
    # List uploaded files
    files = list_repo_files(repo_id=repo_id)
    print("\nUploaded files:")
    for f in files:
        print(f"  - {f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python upload_to_hf.py <HF_TOKEN>")
        sys.exit(1)
    
    upload_godelai(token=sys.argv[1])
```

---

## Expected Result

After upload, the repository should contain:

```
creator35lwb-web/godelai-manifesto-v1/
├── README.md                              # Model card
├── checkpoints/
│   └── godelai_manifesto_v1.pt           # Trained checkpoint (117 KB)
└── godelai/
    ├── __init__.py
    ├── agent.py                          # GodelAgent class
    └── core/
        └── godelai_agent.py              # Core implementation
```

---

## Verification

After upload, verify by visiting:
https://huggingface.co/creator35lwb-web/godelai-manifesto-v1

The model card should display:
- License: MIT
- Tags: alignment, wisdom, small-language-model, csp-framework
- Validation results table
- Quick start code

---

## Questions for Alton

1. **Repository Name**: `creator35lwb-web/godelai-manifesto-v1` or different?
2. **Organization**: Upload to personal account or create organization?
3. **Visibility**: Public (recommended) or private?
