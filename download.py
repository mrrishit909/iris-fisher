"""Step 1: the UCI "Iris" dataset (R. A. Fisher, 1936; donor M. Marshall): 150 flowers, 4 measurements in cm, 3 species.

    ./venv/bin/python download.py   -> data/iris.data, data/bezdekIris.data, data/iris.names, results/pull.json

UCI Machine Learning Repository, dataset 53, licence CC BY 4.0. The zip holds two copies of the table: iris.data, which differs from Fisher's article in two rows (35 and 38),
and bezdekIris.data, with those two rows corrected.
"""
import hashlib
import io
import json
import ssl
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import certifi

HERE = Path(__file__).parent
URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
CTX = ssl.create_default_context(cafile=certifi.where())
FILES = ("iris.data", "bezdekIris.data", "iris.names")


def main():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    blob = urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "research-portfolio (Rishit Raj Mathur)"}), context=CTX, timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(blob))
    for name in FILES:
        (HERE / "data" / name).write_bytes(z.read(name))
    info = {"pulled": date.today().isoformat(), "url": URL, "files_in_zip": z.namelist(),
            "rows": {n: sum(1 for line in (HERE / "data" / n).read_text().splitlines() if line.strip()) for n in FILES[:2]},
            "sha256": {n: hashlib.sha256((HERE / "data" / n).read_bytes()).hexdigest() for n in FILES}}
    (HERE / "results" / "pull.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    main()
