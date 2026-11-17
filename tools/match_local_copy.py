"""Match a local copy of a dataset to splits/<dataset>/images.csv, using the same hash function as the training code.

Usage: python tools/match_local_copy.py MaleVis /path/to/MaleVis/folder [out.csv]
The folder must contain one sub-folder per family (the layout torchvision ImageFolder expects).
Writes out.csv with, for every published row, the local file whose pixel content matches it.
"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "code"))
from torchvision import datasets
import auditx_eval_v2 as ev

ds, folder = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else f"{ds}_local_match.csv"
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "splits", ds, "images.csv"))))
base = datasets.ImageFolder(folder)
same_order = len(base.samples) == len(rows)
by_pixel = {}
for i, (p, _) in enumerate(base.samples):
    k = ev.image_keys(p); by_pixel.setdefault(k[1], []).append(p)
    if same_order and k[1] != rows[i]["pixel_md5"]: same_order = False
    if (i + 1) % 2000 == 0: print(f"hashed {i + 1}/{len(base.samples)}", flush=True)
found = 0
with open(out, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["index", "family", "local_file", "split0", "split1", "split2", "split3", "split4"])
    for r in rows:
        cand = by_pixel.get(r["pixel_md5"], [])
        local = next((c for c in cand if os.path.basename(os.path.dirname(c)) == r["family"]), cand[0] if cand else "")
        found += bool(local)
        w.writerow([r["index"], r["family"], local] + [r[f"split{s}"] for s in range(5)])
print(f"{found}/{len(rows)} published images found in the local copy; "
      f"{'the local copy is identical and in the same order' if same_order else 'order or content differs: use the local_file column'}; wrote {out}")
