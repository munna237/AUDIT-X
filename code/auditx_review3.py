# =============================================================================
# AUDIT-X FINAL - third-round review analyses (CPU only, no training)
# Inputs (anywhere under ROOT): auditx_final_malimg/, auditx_final_malevis/, auditx_final_dumpware10/ (complete runs)
#   F1 homogeneous seed ensembles (3 seeds of one architecture) vs the heterogeneous deep ensemble and single models
#   F2 C1-C6 for every score, cluster bootstrap that resamples families, test images AND training seeds
#   F3 heterogeneity of the per-dataset effects (Cochran Q, I^2)
#   F4 shortcut inflation relative to one pre-specified configuration (deep ensemble + energy)
#   F5 AGEF-Net per-seed AUROC, accuracy and training histories
#   F6 sensitivity of the duplicate rule (bit threshold, cap, identical clusters kept whole): split statistics only
# =============================================================================
import glob, json, os, time, zlib, contextlib, io, copy
import numpy as np, pandas as pd
from scipy.special import softmax, logsumexp
from scipy.stats import rankdata, chi2
import auditx_eval_v2 as ax

ROOT = os.environ.get("AUDITX_ROOT", "/kaggle/input")
OUT = os.environ.get("AUDITX_OUT", "/kaggle/working/auditx_review3"); os.makedirs(OUT, exist_ok=True)
N_BOOT = int(os.environ.get("AUDITX_NBOOT", 1000))
DATASETS = ["Malimg", "MaleVis", "Dumpware10"]
MODELS = ["agef_sharp", "agef_tuned", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"]
SINGLES = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
SCORES = ["msp", "entropy", "maxlogit", "energy", "knn5", "knn10", "knn50", "mahalanobis"]
PROTOCOL = "grouped-L1-v3+near-L3-8bit-cap250"
MALEVIS_VERSION = os.environ.get("AUDITX_MALEVIS", "v5")   # "v5": amended MaleVis run (primary); "v4": original run (sensitivity)
EXPECTED = {"Malimg": PROTOCOL, "Dumpware10": PROTOCOL, "MaleVis": "grouped-L1-v3+identical-hash-whole+near-L3-8bit-cap250" if MALEVIS_VERSION == "v5" else PROTOCOL}
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

RUNS = {}
for ds in DATASETS:
    name = f"auditx_final_{ds.lower()}" + ("_v5" if ds == "MaleVis" and MALEVIS_VERSION == "v5" else "")
    hits = [os.path.dirname(p) for p in glob.glob(f"{ROOT}/**/{name}/MANIFEST.json", recursive=True)]
    if not hits: say(f"!! {ds}: no final run found"); continue
    d = sorted(hits)[0]; man = json.load(open(f"{d}/MANIFEST.json"))
    assert man["complete"] and man["protocol"] == EXPECTED[ds], f"{ds}: incomplete run or protocol {man['protocol']} != {EXPECTED[ds]}"
    RUNS[ds] = dict(dir=d, targets=np.load(f"{d}/targets.npy"), classes=json.load(open(f"{d}/classes.json")),
                    splits=sorted(int(p.split("split")[-1]) for p in glob.glob(f"{d}/split*") if os.path.isdir(p)), seeds=man["args"]["train_seeds"])
    say(f"{ds}: {d} | splits {RUNS[ds]['splits']} | seeds {RUNS[ds]['seeds']}")
assert RUNS

def auroc(k, u):
    r = rankdata(np.r_[k, u]); return (r[len(k):].sum() - len(u) * (len(u) + 1) / 2) / (len(u) * len(k))
def cthr(x, a):
    n = len(x); k = int(np.ceil((n + 1) * (1 - a))); return np.inf if k > n else np.sort(x)[k - 1]
def maha(train_f, train_y, K, dim=512):
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(dim, train_f.shape[1], len(train_f) - 1), random_state=0).fit(train_f)
    z = pca.transform(train_f); mu = np.stack([z[train_y == c].mean(0) for c in range(K)])
    cov = LedoitWolf().fit(z - mu[train_y]).covariance_; ev, V = np.linalg.eigh(cov); W = V / np.sqrt(np.clip(ev, 1e-12, None)); mw = mu @ W
    def f(x):
        zw = pca.transform(x) @ W
        return ((zw ** 2).sum(1)[:, None] - 2 * zw @ mw.T + (mw ** 2).sum(1)[None]).min(1)
    return f
def load_npz(p):
    z = np.load(p, allow_pickle=True); parts = {}
    for k in z.files:
        if "__" in k:
            pp, kk = k.split("__", 1)
            if kk in ("logits", "expert_logits", "weights", "feats", "y"): parts.setdefault(pp, {})[kk] = z[k].astype(np.float64) if kk != "y" else z[k]
    return parts, json.loads(str(z["meta"]))
def score_set(o, is_ens, K, seed):
    sc = {p: ax.base_scores(o[p], is_ens) for p in ("calib", "test", "unk")}
    for k in (5, 10, 50):
        kn = ax.KNNScorer(o["train"]["feats"], k, 6000, seed)
        for p in sc: sc[p][f"knn{k}"] = kn(o[p]["feats"])
    mh = maha(o["train"]["feats"], o["train"]["y"], K)
    for p in sc: sc[p]["mahalanobis"] = mh(o[p]["feats"])
    pred = {p: (softmax(o[p]["expert_logits"], -1).mean(1) if is_ens else softmax(o[p]["logits"], -1)).argmax(1) for p in ("calib", "test")}
    acc = float((pred["test"] == o["test"]["y"]).mean())
    keep = SCORES + ["mutual_info", "gate_entropy"]
    return {p: {s: v for s, v in sc[p].items() if s in keep} for p in sc}, acc
def ensemble(parts_list):
    out = {}
    for p in ("train", "calib", "test", "unk"):
        el = np.concatenate([o[p]["expert_logits"] if o[p]["expert_logits"].ndim == 3 else o[p]["logits"][:, None] for o in parts_list], 1)
        out[p] = dict(expert_logits=el, logits=el.mean(1), feats=np.concatenate([o[p]["feats"] for o in parts_list], 1),
                      y=parts_list[0][p]["y"], weights=np.full((len(el), el.shape[1]), 1 / el.shape[1]))
    return out

# ----------------------------------------------------------------------------- score everything (incl. homogeneous seed ensembles)
SC, ACC, HIST = {}, {}, {}
t0 = time.time()
for ds, R in RUNS.items():
    for s in R["splits"]:
        sp = json.load(open(f"{R['dir']}/split{s}/split.json")); K = len(sp["known"])
        unk_fam = np.array([R["classes"][t] for t in R["targets"][sp["unk"]]])
        per_seed = {}
        for t in R["seeds"]:
            outs = {}
            for m in MODELS:
                outs[m], meta = load_npz(f"{R['dir']}/split{s}/seed{t}/{m}.npz"); HIST[(ds, s, t, m)] = meta.get("history", [])
            outs["deep_ensemble"] = ensemble([outs[x] for x in SINGLES])
            for m, o in outs.items():
                SC[(ds, s, t, m)], ACC[(ds, s, t, m)] = score_set(o, m == "deep_ensemble", K, s)
                SC[(ds, s, t, m)]["unk_fam"] = unk_fam
            per_seed[t] = {x: outs[x] for x in SINGLES}
        # homogeneous seed ensembles: the same architecture trained with the three training seeds (one per split)
        for x in SINGLES:
            hom = ensemble([per_seed[t][x] for t in R["seeds"]]); key = (ds, s, -1, f"seedens_{x.replace('single_', '')}")
            SC[key], ACC[key] = score_set(hom, True, K, s); SC[key]["unk_fam"] = unk_fam
        say(f"{ds} split {s} scored ({(time.time() - t0) / 60:.1f} min)")

rows = []
for (ds, s, t, m), d in SC.items():
    for sn in d["test"]:
        c, te, u = d["calib"][sn], d["test"][sn], d["unk"][sn]; th = cthr(c, 0.05)
        rows.append(dict(dataset=ds, split=s, seed=t, model=m, score=sn, auroc=auroc(te, u), zdr5=float((u > th).mean()), far5=float((te > th).mean()), acc=ACC[(ds, s, t, m)]))
L = pd.DataFrame(rows); L.to_csv(f"{OUT}/F_runs_long.csv", index=False)

# ----------------------------------------------------------------------------- bootstrap (families, test images and training seeds)
def keys_of(ds, m):
    return sorted({(s, t) for (d_, s, t, mm) in SC if d_ == ds and mm == m})
def best_single(ds, s, t):
    return max(SINGLES, key=lambda x: max(h["val_acc"] for h in HIST[(ds, s, t, x)]))
def get(ds, cfg, s, t):
    m = cfg[0]
    if m == "best_single": m = best_single(ds, s, t)
    if m.startswith("seedens_"): t = -1
    return SC[(ds, s, t, m)]
def boot(ds, a, b, sn_a, sn_b, n_boot, seed, resample_seeds=True):
    splits = RUNS[ds]["splits"]; seeds = RUNS[ds]["seeds"]
    fams = sorted({f for s in splits for f in SC[(ds, s, seeds[0], "agef_sharp")]["unk_fam"]})
    def diff(s, t, fidx=None, kb=None):
        A, B = get(ds, a, s, t), get(ds, b, s, t)
        ua, ub = (A["unk"][sn_a], B["unk"][sn_b]) if fidx is None else (A["unk"][sn_a][fidx], B["unk"][sn_b][fidx])
        ka, kb_ = (A["test"][sn_a], B["test"][sn_b]) if kb is None else (A["test"][sn_a][kb], B["test"][sn_b][kb])
        return auroc(ka, ua) - auroc(kb_, ub)
    pt = np.mean([diff(s, t) for s in splits for t in seeds])
    rng = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        fb = rng.choice(fams, len(fams)); vals = []
        for s in splits:
            uf = SC[(ds, s, seeds[0], "agef_sharp")]["unk_fam"]
            idx = np.concatenate([np.where(uf == f)[0] for f in fb if (uf == f).any()] or [np.array([], int)])
            if not len(idx): continue
            ts = rng.choice(seeds, len(seeds)) if resample_seeds else seeds
            n_te = len(SC[(ds, s, seeds[0], "agef_sharp")]["test"]["energy"]); kb = rng.randint(0, n_te, n_te)
            vals += [diff(s, t, idx, kb) for t in ts]
        if vals: bs.append(np.mean(vals))
    return pt, np.array(bs)

CONF = {"C1": (("deep_ensemble",), ("agef_sharp",)), "C2": (("deep_ensemble",), ("agef_sharp",)), "C3": (("avg_joint",), ("agef_sharp",)),
        "C4": (("agef_tuned",), ("agef_sharp",)), "C5": (("agef_tuned",), ("deep_ensemble",)), "C6": (("deep_ensemble",), ("best_single",))}
PRIMARY = {"C1": "energy", "C2": "knn50", "C3": "energy", "C4": "energy", "C5": "energy", "C6": "energy"}
grid = []
for cid, (a, b) in CONF.items():
    for sn in SCORES:
        if sn != PRIMARY[cid] and os.environ.get("AUDITX_GRID", "1") != "1": continue
        per = {ds: boot(ds, a, b, sn, sn, N_BOOT, zlib.crc32(f"{cid}{sn}".encode()) % 1000) for ds in RUNS}
        nmin = min(len(v[1]) for v in per.values()); pb = np.mean([v[1][:nmin] for v in per.values()], 0)
        for ds, (pt, bs) in list(per.items()) + [("pooled", (np.mean([v[0] for v in per.values()]), pb))]:
            grid.append(dict(id=cid, score=sn, primary=sn == PRIMARY[cid], dataset=ds, delta=pt, se=bs.std(ddof=1),
                             ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5), ci90_lo=np.percentile(bs, 5), ci90_hi=np.percentile(bs, 95),
                             p=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), n_boot=len(bs)))
    say(f"{cid}: grid done")
GR = pd.DataFrame(grid)
P = GR[(GR.dataset == "pooled") & GR.primary].copy(); order = P.p.sort_values().index; run = 0
for r, i in enumerate(order):
    run = max(run, min(1.0, (len(order) - r) * P.loc[i, "p"])); GR.loc[i, "p_holm"] = run
GR.to_csv(f"{OUT}/F2_score_grid_seed_bootstrap.csv", index=False)

# ----------------------------------------------------------------------------- F3 heterogeneity (inverse-variance, per-dataset bootstrap SEs)
het = []
for (cid, sn), g in GR[GR.dataset != "pooled"].groupby(["id", "score"]):
    w = 1 / g.se.values ** 2; est = (w * g.delta.values).sum() / w.sum(); Q = (w * (g.delta.values - est) ** 2).sum(); dfree = len(g) - 1
    het.append(dict(id=cid, score=sn, fixed_effect=est, Q=Q, p_Q=1 - chi2.cdf(Q, dfree), I2=max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0,
                    min_dataset=g.delta.min(), max_dataset=g.delta.max()))
H = pd.DataFrame(het); H.to_csv(f"{OUT}/F3_heterogeneity.csv", index=False)

# ----------------------------------------------------------------------------- F1 homogeneous ensembles (exploratory)
hrow = []
for ds in RUNS:
    for hom in ["seedens_mobilenet", "seedens_resnet18", "seedens_effnet_b0"]:
        x = "single_" + hom.split("_", 1)[1]
        for ref in ["deep_ensemble", x]:
            for sn in ["energy", "knn50"]:
                pt, bs = boot(ds, (hom,), (ref,), sn, sn, N_BOOT, zlib.crc32(f"{hom}{ref}{sn}".encode()) % 1000)
                hrow.append(dict(dataset=ds, ensemble=hom, reference=ref, score=sn, delta=pt, ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5)))
    say(f"F1 {ds} done")
HM = pd.DataFrame(hrow); HM.to_csv(f"{OUT}/F1_homogeneous_ensembles.csv", index=False)
F1m = L[L.model.str.startswith("seedens_") | L.model.isin(["deep_ensemble"] + SINGLES)].groupby(["dataset", "model", "score"])[["auroc", "zdr5", "far5", "acc"]].mean().reset_index()
F1m.to_csv(f"{OUT}/F1_homogeneous_summary.csv", index=False)

# ----------------------------------------------------------------------------- F4 inflation relative to one pre-specified configuration
POOL = [(m, sn) for m in MODELS + ["deep_ensemble"] for sn in ["msp", "entropy", "maxlogit", "energy", "knn50", "mahalanobis"]]
irow = []
for ds in RUNS:
    g = L[(L.dataset == ds) & (L.seed >= 0)]; g = g[g.apply(lambda r: (r.model, r.score) in POOL, axis=1)]
    piv = g.groupby(["model", "score", "split"]).zdr5.mean().unstack("split")
    ref = piv.loc[("deep_ensemble", "energy")]
    irow.append(dict(dataset=ds, pool=len(piv), prespecified_zdr=ref.mean(), pool_mean_zdr=piv.values.mean(),
                     s3_zdr=piv.max(0).mean(), s4_zdr=piv.loc[piv.mean(1).idxmax()].max(),
                     s3_vs_prespecified_pp=100 * (piv.max(0).mean() - ref.mean()), s4_vs_prespecified_pp=100 * (piv.loc[piv.mean(1).idxmax()].max() - ref.mean())))
I = pd.DataFrame(irow); I.to_csv(f"{OUT}/F4_inflation_vs_prespecified.csv", index=False)

# ----------------------------------------------------------------------------- F5 AGEF-Net per seed
prow = []
for (ds, s, t, m), h in HIST.items():
    if m not in ("agef_sharp", "agef_tuned", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0") or not h: continue
    hh = pd.DataFrame(h)
    prow.append(dict(dataset=ds, split=s, seed=t, model=m, test_acc=ACC[(ds, s, t, m)], best_val_acc=hh.val_acc.max(), best_epoch=int(hh.val_acc.idxmax()),
                     final_train_loss=hh.train_loss.iloc[-1], val_acc_range_last5=hh.val_acc.tail(5).max() - hh.val_acc.tail(5).min(),
                     auroc_energy=L[(L.dataset == ds) & (L.split == s) & (L.seed == t) & (L.model == m) & (L.score == "energy")].auroc.iloc[0]))
PS = pd.DataFrame(prow); PS.to_csv(f"{OUT}/F5_per_seed.csv", index=False)
json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in HIST.items() if k[3] in ("agef_sharp", "agef_tuned", "single_mobilenet")},
          open(f"{OUT}/F5_histories.json", "w"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(RUNS), figsize=(4.2 * len(RUNS), 2.8), squeeze=False)
    for a_, ds in zip(axs[0], RUNS):
        for (d_, s, t, m), h in HIST.items():
            if d_ != ds or m not in ("agef_sharp", "agef_tuned", "single_mobilenet") or not h: continue
            hh = pd.DataFrame(h); a_.plot(hh.epoch, hh.val_acc, lw=0.7, alpha=0.5, color={"agef_sharp": "#c44e52", "agef_tuned": "#8172b2", "single_mobilenet": "#55a868"}[m])
        a_.set_title(ds); a_.set_xlabel("epoch"); a_.set_ylabel("validation accuracy (%)"); a_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/F5_training_curves.pdf"); plt.savefig(f"{OUT}/F5_training_curves.png", dpi=150); plt.close()
except Exception as e:
    say("plot skipped:", e)

# ----------------------------------------------------------------------------- F6 sensitivity of the duplicate rule (split statistics only)
class FB:
    def __init__(self, classes, targets): self.classes = classes; self.samples = [(f"x{i}", int(t)) for i, t in enumerate(targets)]; self.targets = list(targets)
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8); srow = []
for ds, R in RUNS.items():
    base = FB(R["classes"], R["targets"]); cache = f"{R['dir']}/image_keys.json"
    keys = json.load(open(cache))["keys"]; hb = np.stack([np.frombuffer(bytes.fromhex(k[2]), dtype=np.uint8) for k in keys])
    variants = [(b, cap, False) for b in (4, 8, 12) for cap in (100, 250, 500, 100000)] + [(8, 250, True), (8, 100000, True)]
    for b, cap, whole in variants:
        keyset = (0, 1, 2) if whole else (0, 1)       # whole = identical-hash clusters merged exactly (never cut by the cap)
        with contextlib.redirect_stdout(io.StringIO()):
            G = ax.compute_groups(base, cache, keyset, 0.9, 10, b, cap)
        res = []
        for s in R["splits"]:
            try:
                with contextlib.redirect_stdout(io.StringIO()): sp = ax.make_split(base, s, ax.CFG, G)
            except AssertionError:
                res.append(dict(failed=1)); continue
            tr = np.array(sp["train"]); ev = np.r_[sp["calib"], sp["test"]]
            dm = np.concatenate([POP[hb[ev[i:i + 256]][:, None] ^ hb[tr][None]].sum(-1).min(1) for i in range(0, len(ev), 256)])
            nk = sum(len(sp[x]) for x in ("train", "val", "calib", "test"))
            res.append(dict(failed=0, L2=(dm == 0).mean(), L3=(dm <= 8).mean(), test_share=len(sp["test"]) / nk, calib_share=len(sp["calib"]) / nk))
        r = pd.DataFrame(res)
        srow.append(dict(dataset=ds, bits=b, cap=("none" if cap >= 100000 else cap), identical_clusters_whole=whole, excluded=",".join(G["stats"]["excluded_classes"]),
                         largest_group=G["stats"]["largest_group"], failed_splits=int(r.failed.sum()),
                         residual_L2=r.get("L2", pd.Series([np.nan])).mean(), residual_L3=r.get("L3", pd.Series([np.nan])).mean(),
                         min_test_share=r.get("test_share", pd.Series([np.nan])).min(), max_test_share=r.get("test_share", pd.Series([np.nan])).max()))
    say(f"F6 {ds} done")
SE = pd.DataFrame(srow); SE.to_csv(f"{OUT}/F6_duplicate_rule_sensitivity.csv", index=False)

# ----------------------------------------------------------------------------- summary
say("=" * 100)
for cid in CONF:
    r = GR[(GR.id == cid) & (GR.dataset == "pooled") & GR.primary].iloc[0]
    say(f"{cid} ({r.score}) seed-propagated: {r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}], p {r.p:.4f}, Holm {r.p_holm:.4f} (n_boot {r.n_boot})")
say("Homogeneous seed ensembles vs deep ensemble (energy, AUROC):\n" + HM[(HM.reference == "deep_ensemble") & (HM.score == "energy")].round(3).to_string(index=False))
say("Heterogeneity (primary scores):\n" + H.merge(pd.DataFrame(PRIMARY.items(), columns=["id", "score"])).round(3).to_string(index=False))
say("=" * 100)
open(f"{OUT}/F_SUMMARY.txt", "w").write("\n".join(LOG))
print("outputs:", sorted(os.listdir(OUT)))
