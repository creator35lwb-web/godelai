"""
Finish publishing the existing draft 19927649 (XV's PDF already uploaded).
Just patches metadata and publishes.
Run: python zenodo_publish_draft.py <TOKEN>
"""
import sys, json, urllib.request, urllib.error

DRAFT_ID = 19927649
TOKEN = sys.argv[1] if len(sys.argv) > 1 else input("Zenodo token: ").strip()
BASE = "https://zenodo.org/api"

def api(method, path, data=None):
    url = f"{BASE}{path}?access_token={TOKEN}"
    body = json.dumps(data).encode() if data is not None else b""
    req = urllib.request.Request(url, data=body if body else None, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return (json.loads(raw) if raw else {}), resp.status
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")
        raise

# ── Confirm draft state ───────────────────────────────────────────────────────
print(f"Checking draft {DRAFT_ID}...")
dep, _ = api("GET", f"/deposit/depositions/{DRAFT_ID}")
print(f"  State: {dep.get('state')}")
files, _ = api("GET", f"/deposit/depositions/{DRAFT_ID}/files")
for f in files:
    print(f"  File: {f['filename']} ({f.get('filesize', '?')} bytes)")

# ── Patch metadata ────────────────────────────────────────────────────────────
print("\nPatching metadata (adding publication_date + version)...")
with open(".zenodo_preprint.json", encoding="utf-8") as f:
    meta = json.load(f)
meta["publication_date"] = "2026-05-01"
meta["version"] = "2"
_, status = api("PUT", f"/deposit/depositions/{DRAFT_ID}", data={"metadata": meta})
print(f"  Metadata updated. Status: {status}")

# ── Publish ───────────────────────────────────────────────────────────────────
print("\nPublishing...")
pub, status = api("POST", f"/deposit/depositions/{DRAFT_ID}/actions/publish")
new_doi    = pub.get("doi", f"10.5281/zenodo.{DRAFT_ID}")
concept    = pub.get("conceptdoi", "")
record_url = pub.get("links", {}).get("record_html", f"https://zenodo.org/record/{DRAFT_ID}")

print(f"\n{'='*60}")
print(f"  PUBLISHED!")
print(f"  Version DOI : {new_doi}")
print(f"  Concept DOI : {concept}")
print(f"  URL         : {record_url}")
print(f"{'='*60}")
print("\nROTATE YOUR ZENODO TOKEN NOW.")
