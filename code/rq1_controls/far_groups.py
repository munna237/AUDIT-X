"""RQ3 group-level analysis of the false-alarm excess (CPU, no images). Deep ensemble + energy, 5% calibrated threshold.

Usage: python far_groups.py OUT_PREFIX DATASET RUN_DIR
RUN_DIR: grouped training-run output (MANIFEST.json, classes.json, targets.npy, image_keys.json, split<s>/split.json,
split<s>/seed<t>/single_*.npz). Writes OUT_PREFIX_per_split.csv and OUT_PREFIX_per_family.csv.
Per split and seed: measured test FAR with Clopper-Pearson CI; group-level FAR (share of test duplicate groups with at
least one false alarm) and number of test groups; share of false alarms in the largest group and the five largest; KS
distance between calibration and test energy scores of known images. Per known family: FAR pooled over seeds.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.stats import beta, ks_2samp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auditx_eval_v2 as ax
SINGLES = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]

class Base:   # stands in for ImageFolder: only class names, sample order and targets are needed
    def __init__(self, classes, targets):
        self.classes = list(classes); self.targets = [int(t) for t in targets]; self.samples = [(f"img{i}", t) for i, t in enumerate(self.targets)]

def cthr(x, a=0.05):
    n = len(x); k = int(np.ceil((n + 1) * (1 - a))); return np.inf if k > n else np.sort(x)[k - 1]

def cp(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)

pref, ds, run = sys.argv[1:4]
man = json.load(open(f"{run}/MANIFEST.json")); classes = json.load(open(f"{run}/classes.json")); targets = np.load(f"{run}/targets.npy")
cfg = dict(ax.CFG); base = Base(classes, targets)
key_types = tuple(cfg["GROUP_KEYS"]) + ((2,) if "identical-hash-whole" in man["protocol"] else ())
near = "near-L3" in man["protocol"]
G = ax.compute_groups(base, f"{run}/image_keys.json", key_types, cfg["UNSPLITTABLE_SHARE"], cfg["MIN_CLASS_IMAGES"],
                      cfg["NEAR_DUP_BITS"] if near else None, cfg["NEAR_DUP_CAP"])
assert G["protocol"] == man["protocol"], f"recomputed groups {G['protocol']} differ from the run protocol {man['protocol']}"
groups = np.asarray(G["groups"]); rows = []; fam = []
for s in sorted(int(x[5:]) for x in os.listdir(run) if x.startswith("split") and os.path.isdir(f"{run}/{x}")):
    sp = json.load(open(f"{run}/split{s}/split.json")); te = np.asarray(sp["test"], int); ca = np.asarray(sp["calib"], int)
    for t in sorted(int(x[4:]) for x in os.listdir(f"{run}/split{s}") if x.startswith("seed")):
        if not all(os.path.exists(f"{run}/split{s}/seed{t}/{m}.npz") for m in SINGLES): continue
        Z = [np.load(f"{run}/split{s}/seed{t}/{m}.npz", allow_pickle=True) for m in SINGLES]
        lg = {p: np.mean([z[f"{p}__expert_logits"].astype(np.float64) for z in Z], 0).mean(1) for p in ("calib", "test")}
        y = Z[0]["test__y"]
        # alignment check: each saved label must correspond to exactly one original family, in index order
        pairs = pd.DataFrame(dict(y=y, fam=targets[te])).drop_duplicates()
        assert len(te) == len(y) and pairs.y.is_unique and pairs.fam.is_unique, f"split {s} seed {t}: test order does not match split.json"
        e = {p: -logsumexp(lg[p], axis=1) for p in lg}; th = cthr(e["calib"]); alarm = e["test"] > th
        g = groups[te]; k = int(alarm.sum()); n = len(te); lo, hi = cp(k, n)
        ga = pd.Series(alarm).groupby(g).any(); gs = pd.Series(1, index=g).groupby(level=0).sum()
        fa_by_group = pd.Series(alarm.astype(int)).groupby(g).sum().sort_values(ascending=False)
        rows.append(dict(dataset=ds, split=s, seed=t, n_test=n, far=k / n, far_lo=lo, far_hi=hi, n_groups=len(ga), group_far=ga.mean(),
                         largest_group=int(gs.max()), share_fa_top1=fa_by_group.iloc[:1].sum() / max(k, 1),
                         share_fa_top5=fa_by_group.iloc[:5].sum() / max(k, 1), ks_calib_test=ks_2samp(e["calib"], e["test"]).statistic))
        for f_, a_ in pd.Series(alarm).groupby(targets[te]): fam.append(dict(dataset=ds, split=s, seed=t, family=classes[int(f_)], n=len(a_), alarms=int(a_.sum())))
P = pd.DataFrame(rows); P.to_csv(f"{pref}_per_split.csv", index=False)
F = pd.DataFrame(fam).groupby(["dataset", "family"])[["n", "alarms"]].sum().reset_index(); F["far"] = F.alarms / F.n
F.sort_values("far", ascending=False).to_csv(f"{pref}_per_family.csv", index=False)
S = P.groupby("split")[["far", "group_far", "n_groups", "share_fa_top1", "share_fa_top5", "ks_calib_test"]].mean()
print(S.round(3).to_string()); print("correlation of per-split FAR with KS distance:", round(np.corrcoef(S.far, S.ks_calib_test)[0, 1], 2))
print(F.sort_values("far", ascending=False).head(5).round(3).to_string(index=False))
