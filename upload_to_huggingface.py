#!/usr/bin/env python3
"""
Upload GodelAI to Hugging Face Hub
===================================
Cross-validated, production-ready AI alignment framework.

Usage:
    python upload_to_huggingface.py <HF_TOKEN>

Author: Claude Code (Claude Sonnet 4.5)
Date: January 6, 2026
"""

import os
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def upload_godelai(token: str, repo_id: str = "creator35lwb-web/godelai-manifesto-v1"):
    """Upload GodelAI to Hugging Face Hub."""

    try:
        from huggingface_hub import login, create_repo, upload_file, list_repo_files
    except ImportError:
        print("❌ Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    print("=" * 70)
    print("🤗 UPLOADING GODELAI TO HUGGING FACE")
    print("=" * 70)
    print(f"Repository: {repo_id}")
    print()

    # Login
    print("🔐 Logging in to Hugging Face...")
    try:
        login(token=token)
        print("✅ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        sys.exit(1)

    # Create repository
    print(f"\n📦 Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print("✅ Repository created/verified!")
    except Exception as e:
        print(f"❌ Repository creation failed: {e}")
        sys.exit(1)

    # Define files to upload
    base_path = Path(__file__).parent

    files_to_upload = [
        # Model card (README)
        (base_path / "huggingface" / "README.md", "README.md"),

        # Checkpoint
        (base_path / "huggingface" / "checkpoints" / "godelai_manifesto_v1.pt",
         "checkpoints/godelai_manifesto_v1.pt"),

        # Core code
        (base_path / "godelai" / "agent.py", "godelai/agent.py"),
        (base_path / "godelai" / "__init__.py", "godelai/__init__.py"),

        # Core implementation
        (base_path / "godelai" / "core" / "godelai_agent.py",
         "godelai/core/godelai_agent.py"),

        # Validation results
        (base_path / "MANIFESTO_LEARNING_VALIDATION_REPORT.md",
         "validation/MANIFESTO_LEARNING_VALIDATION_REPORT.md"),
        (base_path / "SCALE_VALIDATION_REPORT.md",
         "validation/SCALE_VALIDATION_REPORT.md"),
        (base_path / "CLAUDE_SCALE_VALIDATION_COMPARISON.md",
         "validation/CLAUDE_SCALE_VALIDATION_COMPARISON.md"),
    ]

    # Upload files
    print("\n📤 Uploading files...")
    uploaded_count = 0
    skipped_count = 0

    for local_path, repo_path in files_to_upload:
        if local_path.exists():
            try:
                print(f"  ↗ {repo_path} ({local_path.stat().st_size / 1024:.1f} KB)")
                upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=repo_id
                )
                uploaded_count += 1
            except Exception as e:
                print(f"  ⚠ Failed to upload {repo_path}: {e}")
                skipped_count += 1
        else:
            print(f"  ⚠ Skipping (not found): {local_path}")
            skipped_count += 1

    # Verify upload
    print(f"\n✅ Upload complete! ({uploaded_count} files uploaded, {skipped_count} skipped)")
    print(f"\n🔗 View at: https://huggingface.co/{repo_id}")

    # List uploaded files
    print("\n📋 Repository contents:")
    try:
        files = list_repo_files(repo_id=repo_id)
        for f in sorted(files):
            print(f"  ✓ {f}")
    except Exception as e:
        print(f"⚠ Could not list files: {e}")

    print("\n" + "=" * 70)
    print("🎉 GODELAI IS NOW ON HUGGING FACE!")
    print("=" * 70)
    print(f"Share: https://huggingface.co/{repo_id}")
    print()

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python upload_to_huggingface.py <HF_TOKEN>")
        print()
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Token needs 'write' permission.")
        sys.exit(1)

    token = sys.argv[1]

    # Optional: custom repo name
    repo_id = sys.argv[2] if len(sys.argv) > 2 else "creator35lwb-web/godelai-manifesto-v1"

    upload_godelai(token=token, repo_id=repo_id)

if __name__ == "__main__":
    main()
