"""Download the Groningen runner injury dataset from DataverseNL into data/raw/.

Dataset: Replication Data for "Injury Prediction in Competitive Runners With
Machine Learning" (Lovdal, den Hartigh & Azzopardi, 2021).
DOI: 10.34894/uwu9pv - open access.

Usage:  python data/fetch_data.py
"""

import json
import pathlib
import sys
import urllib.request

DOI = "doi:10.34894/uwu9pv"
BASE = "https://dataverse.nl"
RAW_DIR = pathlib.Path(__file__).parent / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta_url = f"{BASE}/api/datasets/:persistentId?persistentId={DOI}"
    print(f"Fetching dataset metadata: {meta_url}")
    with urllib.request.urlopen(meta_url, timeout=60) as r:
        meta = json.load(r)

    files = meta["data"]["latestVersion"]["files"]
    print(f"Found {len(files)} file(s)")
    for f in files:
        df = f["dataFile"]
        file_id, name = df["id"], df["filename"]
        dest = RAW_DIR / name
        if dest.exists():
            print(f"  already have {name}, skipping")
            continue
        url = f"{BASE}/api/access/datafile/{file_id}?format=original"
        print(f"  downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception:
            # some files have no 'original' format variant
            urllib.request.urlretrieve(f"{BASE}/api/access/datafile/{file_id}", dest)
        print(f"  -> {dest} ({dest.stat().st_size:,} bytes)")

    print("Done. Raw data lives in data/raw/ and stays out of git.")


if __name__ == "__main__":
    sys.exit(main())
