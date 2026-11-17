"""Convert the split files saved by a training run into splits/<dataset>/images.csv and check them.

A run's output folder (e.g. auditx_final_malimg/) contains image_keys.json, classes.json, targets.npy,
groups_stats.json and split0..split4/split.json. Download that folder from Kaggle (notebook -> Output) and run:

    python tools/import_run_splits.py Malimg /path/to/auditx_final_malimg

Checks: every split's calibration size and held-out families agree with results/primary_analysis, and no
duplicate group (exact file or pixel hash) is cut across partitions.
"""
import csv, json, os, sys, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
ds, run = sys.argv[1], sys.argv[2]
keys = json.load(open(os.path.join(run, "image_keys.json")))["keys"]
classes = json.load(open(os.path.join(run, "classes.json"))); targets = np.load(os.path.join(run, "targets.npy"))
assert len(keys) == len(targets), "image_keys.json and targets.npy disagree"
stats = json.load(open(os.path.join(run, "groups_stats.json")))
excluded = set(stats.get("excluded_classes", []))
labels, summ = [], []
for s in range(5):
    sp = json.load(open(os.path.join(run, f"split{s}", "split.json")))
    lab = np.array(["excluded"] * len(keys), dtype=object)
    for k in ["train", "val", "calib", "test"]: lab[np.asarray(sp[k], int)] = k
    lab[np.asarray(sp["unk"], int)] = "unseen"; labels.append(lab)
    summ.append(dict(split=s, unseen_families=sp["unknown_names"], n_train=len(sp["train"]), n_val=len(sp["val"]),
                     n_calib=len(sp["calib"]), n_test=len(sp["test"]), n_unseen=len(sp["unk"])))
# exact-duplicate integrity: images with the same file or pixel hash must share a partition
ok = True
for s, lab in enumerate(labels):
    for t in (0, 1):
        seen = {}
        for i, k in enumerate(keys):
            if lab[i] == "excluded": continue
            if seen.setdefault(k[t], lab[i]) != lab[i]: ok = False; print(f"split {s}: exact duplicate across partitions (key {t})"); break
# agreement with the recorded results
R = pd.read_csv(os.path.join(ROOT, "results", "primary_analysis", "runs_long.csv")); R = R[R.dataset == ds]
F = pd.read_csv(os.path.join(ROOT, "results", "primary_analysis", "T_families.csv")); F = F[F.dataset == ds]
for x in summ:
    rec = set(R[R.split == x["split"]].n_calib.unique())
    if rec != {x["n_calib"]}: ok = False; print(f"split {x['split']}: calibration size {x['n_calib']} but results record {rec}")
cnt = collections.Counter(f for x in summ for f in x["unseen_families"])
if dict(cnt) != {r.family: int(r["count"]) // 3 for _, r in F.iterrows()}: ok = False; print("held-out families differ from results")
if not ok:
    sys.exit(f"CHECK FAILED: nothing written. These split files do not match the recorded {ds} results.")
out = os.path.join(ROOT, "splits", ds); os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "images.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["index", "family", "file_md5", "pixel_md5", "dhash256", "duplicate_group", "family_excluded"] + [f"split{s}" for s in range(5)])
    for i in range(len(keys)):
        w.writerow([i, classes[targets[i]], *keys[i], "", int(classes[targets[i]] in excluded)] + [labels[s][i] for s in range(5)])
json.dump(dict(dataset=ds, source="split files saved by the training run", protocol=stats.get("protocol"),
               n_images=len(keys), n_images_used=int(sum(classes[t] not in excluded for t in targets)),
               excluded_families=sorted(excluded), splits=summ), open(os.path.join(out, "splits_summary.json"), "w"), indent=1)
print(f"VERIFIED: wrote splits/{ds}/images.csv")
