#!/usr/bin/env python3
"""Upload the finished Short to Google Drive: My Drive / AutoShorts / YYYY-MM-DD / <time>_<topic>.mp4 (+ info .txt).
Uses OAuth (your own Google account) with the narrow 'drive.file' scope, so it can only touch files it creates itself.
Needs repo secrets: GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN."""
import datetime, json, os, re, sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
API = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"


def fail(msg):
    print("DRIVE UPLOAD FAILED:", msg, flush=True)
    sys.exit(1)


def access_token():
    for k in ("GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET", "GDRIVE_REFRESH_TOKEN"):
        if not os.environ.get(k):
            fail(f"missing secret {k} (Settings > Secrets and variables > Actions)")
    r = requests.post("https://oauth2.googleapis.com/token", timeout=60, data={
        "client_id": os.environ["GDRIVE_CLIENT_ID"], "client_secret": os.environ["GDRIVE_CLIENT_SECRET"],
        "refresh_token": os.environ["GDRIVE_REFRESH_TOKEN"], "grant_type": "refresh_token"})
    if r.status_code != 200:
        fail(f"token refresh HTTP {r.status_code}: {r.text[:300]}  (if it says invalid_grant, the refresh token expired or was revoked: "
             "publish your OAuth app 'In production' and create a new token)")
    return r.json()["access_token"]


def ensure_folder(h, name, parent=None):
    q = f"name='{name}' and mimeType='{FOLDER_MIME}' and trashed=false" + (f" and '{parent}' in parents" if parent else " and 'root' in parents")
    r = requests.get(API, headers=h, params={"q": q, "fields": "files(id,name)", "spaces": "drive"}, timeout=60)
    if r.status_code != 200:
        fail(f"folder search HTTP {r.status_code}: {r.text[:300]}")
    files = r.json().get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": FOLDER_MIME, **({"parents": [parent]} if parent else {})}
    r = requests.post(API, headers={**h, "Content-Type": "application/json"}, data=json.dumps(meta), timeout=60)
    if r.status_code not in (200, 201):
        fail(f"folder create HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["id"]


def upload(h, path, name, parent, mime):
    meta = {"name": name, "parents": [parent]}
    r = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable", timeout=60,
                      headers={**h, "Content-Type": "application/json; charset=UTF-8", "X-Upload-Content-Type": mime}, data=json.dumps(meta))
    if r.status_code != 200 or "Location" not in r.headers:
        fail(f"upload start HTTP {r.status_code}: {r.text[:300]}")
    with open(path, "rb") as f:
        r2 = requests.put(r.headers["Location"], data=f, headers={"Content-Type": mime}, timeout=600)
    if r2.status_code not in (200, 201):
        fail(f"upload HTTP {r2.status_code}: {r2.text[:300]}")
    return r2.json().get("id")


def main():
    video, info = OUT / "short.mp4", OUT / "upload_info.txt"
    if not video.exists():
        fail("out/short.mp4 not found")
    meta = json.loads((OUT / "meta.json").read_text()) if (OUT / "meta.json").exists() else {}
    slug = re.sub(r"[^a-z0-9]+", "-", str(meta.get("topic", "short")).lower()).strip("-")[:40] or "short"
    now = datetime.datetime.utcnow()
    base = f"{now:%H%M}_{slug}"
    h = {"Authorization": "Bearer " + access_token()}
    root = ensure_folder(h, "AutoShorts")
    day = ensure_folder(h, f"{now:%Y-%m-%d}", root)
    vid = upload(h, video, base + ".mp4", day, "video/mp4")
    if info.exists():
        upload(h, info, base + "_info.txt", day, "text/plain")
    print(f"UPLOADED: My Drive / AutoShorts / {now:%Y-%m-%d} / {base}.mp4  (file id {vid})", flush=True)


if __name__ == "__main__":
    main()
