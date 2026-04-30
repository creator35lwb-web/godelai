"""
Update Zenodo preprint 19925692:
  - Unlock the published record for editing
  - Delete ALL old files (old ZIP, any previous PDF)
  - Upload GodelAI_TwoLayer_Preprint_v2.pdf  (XV's arXiv-standard PDF — primary)
  - Upload godelai-paper-v1.0-source.zip  (LaTeX source)
  - Re-publish

Run: python zenodo_update_preprint.py <TOKEN>
"""

import sys
import json
import urllib.request

DEPOSIT_ID = 19925692
TOKEN = sys.argv[1] if len(sys.argv) > 1 else input("Zenodo token: ").strip()
BASE = "https://zenodo.org/api"

PDF_FILE    = "GodelAI_TwoLayer_Preprint_v2.pdf"   # XV's professional arXiv-standard PDF
SOURCE_ZIP  = "godelai-paper-v1.0-source.zip"      # LaTeX source bundle


def api(method, path, data=None, raw_data=None, content_type=None):
    url = f"{BASE}{path}?access_token={TOKEN}"
    body = (json.dumps(data).encode() if data is not None else raw_data or b"")
    ct = content_type or ("application/json" if data is not None else "application/octet-stream")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", ct)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP {e.code}: {body_text[:500]}")
        raise


# ── Step 1: Unlock for editing ────────────────────────────────────────────────
print(f"Step 1 — Unlocking deposit {DEPOSIT_ID} for editing...")
edit_resp, status = api("POST", f"/deposit/depositions/{DEPOSIT_ID}/actions/edit")
bucket_url = edit_resp["links"]["bucket"]
print(f"  Unlocked. Status: {status} | Bucket: {bucket_url}")

# ── Step 2: List current files (info only — original bucket is read-only) ─────
print("\nStep 2 — Current files on record:")
files_resp, _ = api("GET", f"/deposit/depositions/{DEPOSIT_ID}/files")
for f in files_resp:
    print(f"  Existing: {f['filename']}")

# ── Step 3: Upload XV's professional PDF (primary) ────────────────────────────
print(f"\nStep 3 — Uploading {PDF_FILE} ...")
with open(PDF_FILE, "rb") as fh:
    pdf_data = fh.read()
pdf_url = f"{bucket_url}/{PDF_FILE}?access_token={TOKEN}"
req = urllib.request.Request(pdf_url, data=pdf_data, method="PUT")
req.add_header("Content-Type", "application/octet-stream")
with urllib.request.urlopen(req) as resp:
    r = json.loads(resp.read())
print(f"  PDF uploaded: {r.get('key')} ({r.get('size'):,} bytes)")

# ── Step 4: Upload LaTeX source ZIP ──────────────────────────────────────────
print(f"\nStep 4 — Uploading {SOURCE_ZIP} ...")
with open(SOURCE_ZIP, "rb") as fh:
    zip_data = fh.read()
zip_url = f"{bucket_url}/{SOURCE_ZIP}?access_token={TOKEN}"
req = urllib.request.Request(zip_url, data=zip_data, method="PUT")
req.add_header("Content-Type", "application/octet-stream")
with urllib.request.urlopen(req) as resp:
    r = json.loads(resp.read())
print(f"  Source ZIP uploaded: {r.get('key')} ({r.get('size'):,} bytes)")

# ── Step 5: Re-publish ────────────────────────────────────────────────────────
print("\nStep 5 — Re-publishing...")
pub, status = api("POST", f"/deposit/depositions/{DEPOSIT_ID}/actions/publish")
doi = pub.get("doi", "10.5281/zenodo.19925692")
url = pub.get("links", {}).get("record_html", f"https://zenodo.org/record/{DEPOSIT_ID}")
print(f"\n{'='*60}")
print(f"  UPDATED & REPUBLISHED!")
print(f"  DOI:  {doi}")
print(f"  URL:  {url}")
print(f"  Files: {PDF_FILE} (primary) + {SOURCE_ZIP}")
print(f"{'='*60}")
print("\nROTATE YOUR ZENODO TOKEN NOW.")
