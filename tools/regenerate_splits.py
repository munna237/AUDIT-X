"""Regenerate the AUDIT-X splits from splits/<dataset>/images.csv and check them against the published assignment.

The split depends only on the image hashes, the family labels and their order (torchvision ImageFolder order:
family folders sorted by name, files sorted by name), so no malware image is needed.

Usage:  python tools/regenerate_splits.py MaleVis        (or Dumpware10, Malimg)
Requires scikit-learn==1.6.1: StratifiedGroupKFold shuffles differently in other versions.
"""
import csv, json, os, sys, tempfile
import numpy as np
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))

# Code version used by each final run (see results/primary_analysis/VERDICTS.txt):
#   Malimg, Dumpware10: auditx_final_v4.py + auditx_eval_v2_v4.py, duplicate keys (file MD5, pixel MD5)
#   MaleVis (Amendment 1): auditx_final.py + auditx_eval_v2.py, keys (file MD5, pixel MD5, 256-bit dHash)
RUN = {"Malimg": ("auditx_eval_v2_v4", (0, 1)), "Dumpware10": ("auditx_eval_v2_v4", (0, 1)),
       "MaleVis": ("auditx_eval_v2", (0, 1, 2))}


class IndexOnly:
    """Stands in for torchvision ImageFolder: classes and (path, label) samples in ImageFolder order."""
    def __init__(self, families):
        self.classes = sorted(set(families))
        self.samples = [(f"img{i}", self.classes.index(f)) for i, f in enumerate(families)]
        self.targets = [t for _, t in self.samples]


def main(ds):
    if sklearn.__version__ != "1.6.1":
        sys.exit(f"scikit-learn {sklearn.__version__} found; install scikit-learn==1.6.1 to reproduce the splits exactly.")
    mod, key_types = RUN[ds]
    ev = __import__(mod)
    path = os.path.join(HERE, "..", "splits", ds, "images.csv")
    rows = list(csv.DictReader(open(path)))
    fams = [r["family"] for r in rows]
    base = IndexOnly(fams)
    keys = [[r["file_md5"], r["pixel_md5"], r["dhash256"]] for r in rows]
    cfg = dict(ev.CFG)
    with tempfile.TemporaryDirectory() as tmp:
        kp = os.path.join(tmp, "image_keys.json")
        json.dump(dict(n=len(keys), keys=keys), open(kp, "w"))
        G = ev.compute_groups(base, kp, key_types, cfg["UNSPLITTABLE_SHARE"], cfg["MIN_CLASS_IMAGES"],
                              cfg["NEAR_DUP_BITS"], cfg["NEAR_DUP_CAP"])
    print("protocol:", G["protocol"], "| excluded families:", G["stats"]["excluded_classes"])
    ok = True
    for s in range(5):
        sp = ev.make_split(base, s, cfg, G)
        lab = np.array(["excluded"] * len(rows), dtype=object)
        for k in ["train", "val", "calib", "test"]:
            lab[np.asarray(sp[k], int)] = k
        lab[np.asarray(sp["unk"], int)] = "unseen"
        same = all(lab[i] == rows[i][f"split{s}"] for i in range(len(rows)))
        ok &= same
        print(f"split {s}: unseen families {sp['unknown_names']} | matches published assignment: {same}")
    print("ALL SPLITS REPRODUCED" if ok else "MISMATCH: check the scikit-learn version and the code version")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
