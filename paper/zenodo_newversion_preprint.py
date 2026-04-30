"""
Create a new version of Zenodo preprint 19925692 with XV's professional PDF.
New version gets a new version-specific DOI. Concept DOI (permanent) unchanged.

Steps:
  1. Discard any pending edit on original deposit
  2. Create new version from 19925692
  3. Upload GodelAI_TwoLayer_Preprint_v2.pdf (primary)
  4. Upload godelai-paper-v1.0-source.zip (LaTeX source)
  5. Delete the inherited old ZIP from the new draft
  6. Publish new version → new version DOI

Run: python zenodo_newversion_preprint.py <TOKEN>
"""

import sys
import json
import urllib.request
import urllib.error

ORIGINAL_ID = 19925692
TOKEN = sys.argv[1] if len(sys.argv) > 1 else input("Zenodo token: ").strip()
BASE = "https://zenodo.org/api"

PDF_FILE   = "GodelAI_TwoLayer_Preprint_v2.pdf"
SOURCE_ZIP = "godelai-paper-v1.0-source.zip"


def api(method, path, data=None, raw_data=None, content_type=None, full_url=None):
    url = full_url or f"{BASE}{path}?access_token={TOKEN}"
    if "?" not in url:
        url += f"?access_token={TOKEN}"
    elif "access_token" not in url:
        url += f"&access_token={TOKEN}"
    body = (json.dumps(data).encode() if data is not None else raw_data or b"")
    ct = content_type or ("application/json" if data is not None else "application/octet-stream")
    req = urllib.request.Request(url, data=body if body else None, method=method)
    if body:
        req.add_header("Content-Type", ct)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return (json.loads(raw) if raw else {}), resp.status
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP {e.code}: {body_text[:600]}")
        raise


# ── Step 1: Discard any pending edit on original ──────────────────────────────
print(f"Step 1 — Checking original deposit {ORIGINAL_ID} state...")
orig, _ = api("GET", f"/deposit/depositions/{ORIGINAL_ID}")
state = orig.get("state", "unknown")
print(f"  State: {state}")
if state == "inprogress":
    print("  Discarding pending edit...")
    try:
        api("POST", f"/deposit/depositions/{ORIGINAL_ID}/actions/discard")
        print("  Discarded.")
    except Exception as e:
        print(f"  Discard failed (may be OK): {e}")

# ── Step 2: Create new version ────────────────────────────────────────────────
print(f"\nStep 2 — Creating new version from {ORIGINAL_ID}...")
nv, status = api("POST", f"/deposit/depositions/{ORIGINAL_ID}/actions/newversion")
print(f"  New version response status: {status}")

# The new version's draft ID is in links.latest_draft
latest_draft_url = nv.get("links", {}).get("latest_draft", "")
print(f"  Latest draft URL: {latest_draft_url}")

# Extract the new deposit ID from the URL
new_id = int(latest_draft_url.rstrip("/").split("/")[-1]) if latest_draft_url else None
if not new_id:
    # fallback: check if nv itself is the new deposit
    new_id = nv.get("id")
print(f"  New deposit ID: {new_id}")

# Get the new draft details
new_dep, _ = api("GET", f"/deposit/depositions/{new_id}")
bucket_url = new_dep["links"]["bucket"]
print(f"  New bucket: {bucket_url}")

# ── Step 3: Delete inherited old files from new draft ────────────────────────
print(f"\nStep 3 — Removing inherited files from new draft {new_id}...")
files_resp, _ = api("GET", f"/deposit/depositions/{new_id}/files")
for f in files_resp:
    fname = f["filename"]
    fid   = f["id"]
    print(f"  Deleting: {fname} (id: {fid})")
    try:
        api("DELETE", f"/deposit/depositions/{new_id}/files/{fid}")
        print(f"  Deleted.")
    except Exception as e:
        # Try bucket-level delete
        try:
            del_url = f"{bucket_url}/{fname}?access_token={TOKEN}"
            req = urllib.request.Request(del_url, method="DELETE")
            with urllib.request.urlopen(req) as r:
                print(f"  Deleted via bucket. Status: {r.status}")
        except Exception as e2:
            print(f"  Could not delete {fname}: {e2} — will keep alongside new files")

# ── Step 4: Upload XV's professional PDF ──────────────────────────────────────
print(f"\nStep 4 — Uploading {PDF_FILE} ...")
with open(PDF_FILE, "rb") as fh:
    pdf_data = fh.read()
pdf_url = f"{bucket_url}/{PDF_FILE}?access_token={TOKEN}"
req = urllib.request.Request(pdf_url, data=pdf_data, method="PUT")
req.add_header("Content-Type", "application/octet-stream")
with urllib.request.urlopen(req) as resp:
    r = json.loads(resp.read())
print(f"  Uploaded: {r.get('key')} ({r.get('size'):,} bytes)")

# ── Step 5: Upload LaTeX source ZIP ──────────────────────────────────────────
print(f"\nStep 5 — Uploading {SOURCE_ZIP} ...")
with open(SOURCE_ZIP, "rb") as fh:
    zip_data = fh.read()
zip_url = f"{bucket_url}/{SOURCE_ZIP}?access_token={TOKEN}"
req = urllib.request.Request(zip_url, data=zip_data, method="PUT")
req.add_header("Content-Type", "application/octet-stream")
with urllib.request.urlopen(req) as resp:
    r = json.loads(resp.read())
print(f"  Uploaded: {r.get('key')} ({r.get('size'):,} bytes)")

# ── Step 6: Patch metadata (new version draft needs publication_date) ────────
print(f"\nStep 6 — Patching metadata on new draft {new_id}...")
with open(".zenodo_preprint.json", encoding="utf-8") as f:
    meta = json.load(f)
meta["publication_date"] = "2026-05-01"
meta["version"] = "2"
_, status = api("PUT", f"/deposit/depositions/{new_id}", data={"metadata": meta})
print(f"  Metadata set. Status: {status}")

# ── Step 7: Publish new version ───────────────────────────────────────────────
print(f"\nStep 7 — Publishing new version {new_id}...")
pub, status = api("POST", f"/deposit/depositions/{new_id}/actions/publish")
new_doi = pub.get("doi", pub.get("metadata", {}).get("doi", f"10.5281/zenodo.{new_id}"))
concept_doi = pub.get("conceptdoi", pub.get("metadata", {}).get("conceptdoi", ""))
record_url  = pub.get("links", {}).get("record_html", f"https://zenodo.org/record/{new_id}")

print(f"\n{'='*60}")
print(f"  NEW VERSION PUBLISHED!")
print(f"  Version DOI:  {new_doi}")
print(f"  Concept DOI:  {concept_doi}")
print(f"  Record URL:   {record_url}")
print(f"  Files:")
print(f"    - {PDF_FILE}  (primary — XV arXiv-standard PDF)")
print(f"    - {SOURCE_ZIP} (LaTeX source)")
print(f"  Previous version: 10.5281/zenodo.{ORIGINAL_ID}")
print(f"{'='*60}")
print("\nROTATE YOUR ZENODO TOKEN NOW.")
print(f"\nNEXT: Update DOI references from 10.5281/zenodo.{ORIGINAL_ID} → {new_doi}")
