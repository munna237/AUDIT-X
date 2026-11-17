"""RQ1 paired comparison: naive (image-level) versus duplicate-grouped splits, same unknown families and training seeds.

Usage: python compare_naive.py OUT.csv  GROUPED_DIR NAIVE_DIR DATASET  [GROUPED_DIR NAIVE_DIR DATASET ...]
Each *_DIR is a training-run output folder (MANIFEST.json, split<s>/split.json, split<s>/seed<t>/<model>.npz).
For every (split, seed) present in both folders it builds the deep ensemble from the three single CNNs exactly as the
main analysis does, and reports per run: closed-set accuracy, energy AUROC, UFR and FAR at the 5% threshold calibrated
on the calibration split (conformal quantile rule), for the deep ensemble and each single CNN.
It checks that both folders hold out the same unknown families and use the same evaluation (unknown) images.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.special import logsumexp, softmax
SINGLES = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]

def load(p):
    z = np.load(p, allow_pickle=True); parts = {}
    for k in z.files:
        if "__" in k:
            pp, kk = k.split("__", 1); parts.setdefault(pp, {})[kk] = z[k] if kk == "y" else z[k].astype(np.float64)
    return parts

def cthr(x, a):
    n = len(x); k = int(np.ceil((n + 1) * (1 - a))); return np.inf if k > n else np.sort(x)[k - 1]

def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))

def metrics(o, deep):
    if deep:
        lg = {p: o[p]["expert_logits"].mean(1) for p in o}; pr = {p: softmax(o[p]["expert_logits"], -1).mean(1) for p in o}
    else:
        lg = {p: o[p]["logits"] for p in o}; pr = {p: softmax(lg[p], -1) for p in o}
    e = {p: -logsumexp(lg[p], axis=1) for p in ("calib", "test", "unk")}
    th = cthr(e["calib"], 0.05)
    return dict(acc=float((pr["test"].argmax(1) == o["test"]["y"]).mean()), auroc=auroc(e["test"], e["unk"]),
                ufr5=float((e["unk"] > th).mean()), far5=float((e["test"] > th).mean()))

def runs(d):
    out = {}
    for s in sorted(int(x[5:]) for x in os.listdir(d) if x.startswith("split") and os.path.isdir(os.path.join(d, x))):
        sp = json.load(open(f"{d}/split{s}/split.json"))
        for t in sorted(int(x[4:]) for x in os.listdir(f"{d}/split{s}") if x.startswith("seed")):
            if all(os.path.exists(f"{d}/split{s}/seed{t}/{m}.npz") for m in SINGLES): out[(s, t)] = sp
    return out

rows = []
args = sys.argv[2:]
for i in range(0, len(args), 3):
    gdir, ndir, ds = args[i:i + 3]
    proto = {k: json.load(open(f"{d}/MANIFEST.json"))["protocol"] for k, d in (("grouped", gdir), ("naive", ndir))}
    assert "NAIVE" not in proto["grouped"] and "NAIVE" in proto["naive"], f"{ds}: folders swapped or not a naive run: {proto}"
    G, N = runs(gdir), runs(ndir)
    common = sorted(set(G) & set(N))
    for (s, t) in common:
        assert G[(s, t)]["unknown_names"] == N[(s, t)]["unknown_names"], f"{ds} split {s}: unknown families differ"
        assert sorted(G[(s, t)]["unk"]) == sorted(N[(s, t)]["unk"]), f"{ds} split {s}: unknown images differ"
        for split_kind, d in (("grouped", gdir), ("naive", ndir)):
            outs = {m: load(f"{d}/split{s}/seed{t}/{m}.npz") for m in SINGLES}
            de = {p: dict(expert_logits=np.concatenate([outs[m][p]["expert_logits"] for m in SINGLES], 1), y=outs[SINGLES[0]][p]["y"])
                  for p in ("calib", "test", "unk")}
            for m, o, deep in [("deep_ensemble", de, True)] + [(m, outs[m], False) for m in SINGLES]:
                rows.append(dict(dataset=ds, split=s, seed=t, model=m, split_kind=split_kind, **metrics(o, deep)))
    print(f"{ds}: {len(common)} paired runs; protocols {proto}")
L = pd.DataFrame(rows); L.to_csv(sys.argv[1].replace(".csv", "_long.csv"), index=False)
W = L.pivot_table(index=["dataset", "model", "split", "seed"], columns="split_kind", values=["acc", "auroc", "ufr5", "far5"]).reset_index()
summ = []
for (ds, m), g in W.groupby(["dataset", "model"]):
    r = dict(dataset=ds, model=m, runs=len(g))
    for v in ["acc", "auroc", "ufr5", "far5"]:
        diff = g[(v, "naive")] - g[(v, "grouped")]
        # paired bootstrap over splits (the three seeds of a split stay together)
        rng = np.random.RandomState(0); sp = g["split"].values; us = np.unique(sp)
        bs = [np.mean(np.concatenate([diff.values[sp == u] for u in rng.choice(us, len(us))])) for _ in range(2000)]
        r.update({f"{v}_grouped": g[(v, "grouped")].mean(), f"{v}_naive": g[(v, "naive")].mean(), f"{v}_diff": diff.mean(),
                  f"{v}_ci_lo": np.percentile(bs, 2.5), f"{v}_ci_hi": np.percentile(bs, 97.5)})
    summ.append(r)
S = pd.DataFrame(summ); S.to_csv(sys.argv[1], index=False)
print(S[["dataset", "model", "runs", "acc_grouped", "acc_naive", "auroc_grouped", "auroc_naive", "auroc_diff", "auroc_ci_lo", "auroc_ci_hi", "far5_grouped", "far5_naive"]].round(3).to_string(index=False))
