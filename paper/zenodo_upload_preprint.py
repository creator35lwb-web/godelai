"""
Zenodo preprint upload — GodelAI paper v1.0
New deposit (publication/preprint) — separate from v4.0.0 software record.
Run: python zenodo_upload_preprint.py <ZENODO_TOKEN>
"""

import sys
import json
import urllib.request
import urllib.parse

TOKEN = sys.argv[1] if len(sys.argv) > 1 else input("Zenodo token: ").strip()
BASE = "https://zenodo.org/api"
HEADERS_JSON = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}

# ── Load metadata ──────────────────────────────────────────────────────────────
with open(".zenodo_preprint.json", encoding="utf-8") as f:
    meta = json.load(f)

UPLOAD_FILE = "godelai-paper-v1.0.zip"
FILENAME_IN_ZENODO = "godelai-paper-v1.0.zip"


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


# ── Step 1: Create new empty deposit ──────────────────────────────────────────
print("Step 1 — Creating new Zenodo deposit (preprint)...")
deposit, status = api("POST", "/deposit/depositions", data={})
deposit_id = deposit["id"]
bucket_url = deposit["links"]["bucket"]
print(f"  Deposit ID: {deposit_id} | Status: {status}")
print(f"  Bucket URL: {bucket_url}")

# ── Step 2: Upload file to bucket ──────────────────────────────────────────────
print(f"\nStep 2 — Uploading {UPLOAD_FILE} ...")
with open(UPLOAD_FILE, "rb") as fh:
    file_data = fh.read()

upload_url = f"{bucket_url}/{FILENAME_IN_ZENODO}?access_token={TOKEN}"
req = urllib.request.Request(upload_url, data=file_data, method="PUT")
req.add_header("Content-Type", "application/octet-stream")
with urllib.request.urlopen(req) as resp:
    file_resp = json.loads(resp.read())
print(f"  File uploaded: {file_resp.get('key', 'ok')} ({file_resp.get('size', '?')} bytes)")

# ── Step 3: Set metadata ───────────────────────────────────────────────────────
print("\nStep 3 — Setting metadata...")
payload = {"metadata": meta}
_, status = api("PUT", f"/deposit/depositions/{deposit_id}", data=payload)
print(f"  Metadata set. Status: {status}")

# ── Step 4: Publish ────────────────────────────────────────────────────────────
print("\nStep 4 — Publishing preprint...")
pub, status = api("POST", f"/deposit/depositions/{deposit_id}/actions/publish")
doi = pub.get("doi", pub.get("metadata", {}).get("doi", "unknown"))
record_url = pub.get("links", {}).get("record_html", f"https://zenodo.org/record/{deposit_id}")
print(f"\n{'='*60}")
print(f"  PUBLISHED!")
print(f"  DOI: {doi}")
print(f"  URL: {record_url}")
print(f"  Deposit ID: {pub.get('id', deposit_id)}")
print(f"{'='*60}")
print("\nROTATE YOUR ZENODO TOKEN NOW.")
