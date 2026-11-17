# =============================================================================
# AUDIT-X FINAL ANALYSIS - every table, figure and verdict of the paper, from the three final runs (PROTOCOL.md)
# Inputs (anywhere under ROOT): auditx_final_malimg/, auditx_final_malevis/, auditx_final_dumpware10/ (complete MANIFEST),
#   optionally xrep_raw.csv (nb 08), fewshot_raw.csv (nb 10), fewshot_baselines_raw.csv (nb 14).
# =============================================================================
import glob, json, os, sys, time, zlib
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import softmax, logsumexp
from scipy.stats import rankdata, beta
import auditx_eval_v2 as ax

ROOT = os.environ.get("AUDITX_ROOT", "/kaggle/input")
OUT = os.environ.get("AUDITX_OUT", "/kaggle/working/auditx_final_analysis")
N_BOOT = int(os.environ.get("AUDITX_NBOOT", 1000))
TOST_MARGIN, ALPHA, FAR5, FAR1, MIN_EXC, N_REDRAW = 0.03, 0.05, 0.05, 0.01, 10, 200
KNN_KS = [1, 5, 10, 50]
DATASETS = ["Malimg", "MaleVis", "Dumpware10"]
MODELS = ["agef_sharp", "agef_tuned", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"]
SINGLES = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
PROTOCOL = "grouped-L1-v3+near-L3-8bit-cap250"
MALEVIS_VERSION = os.environ.get("AUDITX_MALEVIS", "v5")   # "v5": amended MaleVis run (primary); "v4": original run (sensitivity)
EXPECTED = {"Malimg": PROTOCOL, "Dumpware10": PROTOCOL, "MaleVis": "grouped-L1-v3+identical-hash-whole+near-L3-8bit-cap250" if MALEVIS_VERSION == "v5" else PROTOCOL}
DEDUP_EVAL = os.environ.get("AUDITX_DEDUP_EVAL", "0") == "1"   # deviation-driven sensitivity analysis (not pre-registered)
DEDUP_BITS = 8
os.makedirs(OUT, exist_ok=True)
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

# ----------------------------------------------------------------------------- locate runs
RUNS = {}
for ds in DATASETS:
    name = f"auditx_final_{ds.lower()}" + ("_v5" if ds == "MaleVis" and MALEVIS_VERSION == "v5" else "")
    hits = [os.path.dirname(p) for p in glob.glob(f"{ROOT}/**/{name}/MANIFEST.json", recursive=True)]
    if not hits: say(f"!! {ds}: no final run found"); continue
    d = sorted(hits)[0]; man = json.load(open(f"{d}/MANIFEST.json"))
    assert man["complete"], f"{ds}: run incomplete ({man['n_done']}/{man['n_expected']}) - finish it before analysing"
    assert man["protocol"] == EXPECTED[ds], f"{ds}: protocol {man['protocol']} != {EXPECTED[ds]}"
    RUNS[ds] = dict(dir=d, man=man, targets=np.load(f"{d}/targets.npy"), classes=json.load(open(f"{d}/classes.json")),
                    splits=sorted(int(p.split("split")[-1]) for p in glob.glob(f"{d}/split*") if os.path.isdir(p)),
                    seeds=man["args"]["train_seeds"])
    say(f"{ds}: {d} | splits {RUNS[ds]['splits']} | seeds {RUNS[ds]['seeds']} | protocol OK | script {man['script_sha256'][:12]}"
        f" | protocol file {str(man.get('protocol_sha256'))[:12]}")
assert RUNS, "no runs found"
_ph = {ds: R["man"].get("protocol_sha256") for ds, R in RUNS.items()}
_sh = {ds: R["man"]["script_sha256"] for ds, R in RUNS.items()}
say("protocol-document sha256 per run:", _ph)
say("training-script sha256 per run:", _sh)
_orig = [ds for ds in RUNS if EXPECTED[ds] == PROTOCOL]
if None in _ph.values() or len({_ph[ds] for ds in _orig}) > 1 or len({_sh[ds] for ds in _orig}) > 1:
    say("!! WARNING: runs under the original protocol do not share the same protocol document and script - report as a deviation")
else:
    say("runs under the original protocol share one protocol document and one script" +
        ("; MaleVis uses Amendment 1 (different document and script hash, as intended)" if "MaleVis" in RUNS and EXPECTED["MaleVis"] != PROTOCOL else ""))

# ----------------------------------------------------------------------------- deduplicated evaluation masks (optional)
POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)
MASK, MASK_STATS = {}, []
if DEDUP_EVAL:
    for ds, R in RUNS.items():
        keys = json.load(open(f"{R['dir']}/image_keys.json"))["keys"]
        f0 = np.array([k[0] for k in keys]); f1 = np.array([k[1] for k in keys])
        hb = np.stack([np.frombuffer(bytes.fromhex(k[2]), dtype=np.uint8) for k in keys])
        for s in R["splits"]:
            sp = json.load(open(f"{R['dir']}/split{s}/split.json")); tr = np.array(sp["train"])
            for p in ("calib", "test", "unk"):
                q = np.array(sp[p])
                dmin = np.concatenate([POP8[hb[q[i:i + 256]][:, None] ^ hb[tr][None]].sum(-1).min(1) for i in range(0, len(q), 256)])
                keep = (dmin > DEDUP_BITS) & ~np.isin(f0[q], f0[tr]) & ~np.isin(f1[q], f1[tr])
                MASK[(ds, s, p)] = keep
                MASK_STATS.append(dict(dataset=ds, split=s, part=p, n=len(q), removed=int((~keep).sum()), removed_share=float((~keep).mean())))
    pd.DataFrame(MASK_STATS).to_csv(f"{OUT}/T_dedup_removed.csv", index=False)
    say("DEDUPLICATED EVALUATION: evaluation images with a training image within", DEDUP_BITS, "bits (or identical file/pixels) are removed")
    say(pd.DataFrame(MASK_STATS).groupby(["dataset", "part"])[["n", "removed"]].sum().to_string())

def _apply_mask(ds, s, entry):
    if not DEDUP_EVAL: return entry
    removed_any = any((~MASK[(ds, s, p)]).any() for p in ("calib", "test", "unk"))
    if removed_any and min(MASK[(ds, s, p)].sum() for p in ("calib", "test", "unk")) < 10:
        return None      # too few evaluation images left in this split: the split is skipped (reported in the log)
    for p in ("calib", "test", "unk"):
        k = MASK[(ds, s, p)]
        for sn in entry["sc"][p]: entry["sc"][p][sn] = entry["sc"][p][sn][k]
        if entry.get("pred") is not None: entry["pred"][p] = entry["pred"][p][k]
        if entry.get("y") is not None and p in entry["y"]: entry["y"][p] = entry["y"][p][k]
        if p == "unk": entry["unk_fam"] = entry["unk_fam"][k]
        if entry.get("ee") is not None:
            entry["ee"]["m0"][p] = entry["ee"]["m0"][p][k]; entry["ee"]["fused"][p] = entry["ee"]["fused"][p][k]
    return entry

# ----------------------------------------------------------------------------- scoring helpers
def maha(train_f, train_y, K, dim=512):
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(dim, train_f.shape[1], len(train_f) - 1), random_state=0).fit(train_f)
    z = pca.transform(train_f); mu = np.stack([z[train_y == c].mean(0) for c in range(K)])
    cov = LedoitWolf().fit(z - mu[train_y]).covariance_; ev, V = np.linalg.eigh(cov); W = V / np.sqrt(np.clip(ev, 1e-12, None)); mw = mu @ W
    return lambda f: (((pca.transform(f) @ W) ** 2).sum(1)[:, None] - 2 * (pca.transform(f) @ W) @ mw.T + (mw ** 2).sum(1)[None]).min(1)

def vim(raw_tr, W, b, raw, logit):
    D = W.shape[1]; dim = 1000 if D >= 2048 else (512 if D >= 768 else D // 2)
    u = -np.linalg.pinv(W) @ b; X = raw_tr.astype(np.float64) - u
    ev, V = np.linalg.eigh(X.T @ X / len(X)); NS = V[:, :D - dim]
    alpha = (raw_tr.astype(np.float64) @ W.T + b).max(1).mean() / np.linalg.norm(X @ NS, axis=1).mean()
    return alpha * np.linalg.norm((raw.astype(np.float64) - u) @ NS, axis=1) - logsumexp(logit, axis=1)

def credibility(cal_s, cal_pred, s, pred):
    """Transcend-style: class-conditional conformal p-value of a nonconformity score (higher = stranger); returns 1 - p."""
    out = np.empty(len(s))
    for c in np.unique(pred):
        ref = np.sort(cal_s[cal_pred == c]) if (cal_pred == c).sum() >= 5 else np.sort(cal_s)
        m = pred == c; out[m] = (len(ref) - np.searchsorted(ref, s[m], side="left") + 1) / (len(ref) + 1)
    return 1 - out

def auroc(k, u):
    r = rankdata(np.r_[k, u]); return (r[len(k):].sum() - len(u) * (len(u) + 1) / 2) / (len(u) * len(k))

def cthr(x, a):
    n = len(x); k = int(np.ceil((n + 1) * (1 - a))); return np.inf if k > n else np.sort(x)[k - 1]

def load_npz(p):
    z = np.load(p, allow_pickle=True); parts = {}
    for k in z.files:
        if "__" in k:
            pp, kk = k.split("__", 1); parts.setdefault(pp, {})[kk] = z[k].astype(np.float64) if kk not in ("y",) else z[k]
    return parts, json.loads(str(z["meta"])), ({"W": z["W"].astype(np.float64), "b": z["b"].astype(np.float64)} if "W" in z.files else None)

# ----------------------------------------------------------------------------- score everything once
SC = {}   # (ds, split, seed, model) -> dict
t0 = time.time()
for ds, R in RUNS.items():
    for s in R["splits"]:
        sp = json.load(open(f"{R['dir']}/split{s}/split.json")); K = len(sp["known"])
        unk_fam = np.array([R["classes"][t] for t in R["targets"][sp["unk"]]])
        dfe = np.load(f"{R['dir']}/dinov2_feats.npy").astype(np.float64) if os.path.exists(f"{R['dir']}/dinov2_feats.npy") else None
        for t in R["seeds"]:
            outs, metas, heads = {}, {}, {}
            for m in MODELS:
                outs[m], metas[m], heads[m] = load_npz(f"{R['dir']}/split{s}/seed{t}/{m}.npz")
            de = {}
            for p in ("train", "calib", "test", "unk"):
                de[p] = dict(expert_logits=np.concatenate([outs[x][p]["expert_logits"] for x in SINGLES], 1),
                             feats=np.concatenate([outs[x][p]["feats"] for x in SINGLES], 1), y=outs[SINGLES[0]][p]["y"],
                             react_logits=np.mean([outs[x][p]["react_logits"] for x in SINGLES], 0))
                de[p]["logits"] = de[p]["expert_logits"].mean(1); de[p]["weights"] = np.full((len(de[p]["y"]), 3), 1 / 3)
            outs["deep_ensemble"] = de
            for m, o in outs.items():
                is_de = m == "deep_ensemble"
                sc = {p: ax.base_scores(o[p], is_de) for p in ("train", "calib", "test", "unk")}
                for k in KNN_KS:
                    kn = ax.KNNScorer(o["train"]["feats"], k, 6000, s)
                    for p in ("calib", "test", "unk"): sc[p][f"knn{k}"] = kn(o[p]["feats"])
                mh = maha(o["train"]["feats"], o["train"]["y"], K)
                for p in ("calib", "test", "unk"):
                    sc[p]["mahalanobis"] = mh(o[p]["feats"])
                    rl = o[p]["react_logits"]; sc[p]["react_energy"] = -logsumexp(rl, axis=1)
                if m.startswith("single_") or is_de:
                    mem = [m] if not is_de else SINGLES
                    for p in ("calib", "test", "unk"):
                        sc[p]["vim"] = np.mean([vim(outs[x]["train"]["raw"], heads[x]["W"], heads[x]["b"], outs[x][p]["raw"],
                                                    outs[x][p]["logits"]) for x in mem], 0)
                if t == R["seeds"][0]:
                    om = ax.OpenMax(o["train"]["logits"], o["train"]["y"], 20, 10)
                    for p in ("calib", "test", "unk"): sc[p]["openmax"] = om(o[p]["logits"])
                pr = {p: (softmax(o[p]["expert_logits"], -1).mean(1) if is_de else softmax(o[p]["logits"], -1)).argmax(1) for p in ("calib", "test", "unk")}
                for p in ("calib", "test", "unk"):
                    sc[p]["transcend"] = credibility(sc["calib"]["knn50"], pr["calib"], sc[p]["knn50"], pr[p])
                val_acc = max(h["val_acc"] for h in metas[m]["history"]) if m in metas else \
                          np.mean([max(h["val_acc"] for h in metas[x]["history"]) for x in SINGLES])
                SC[(ds, s, t, m)] = dict(sc=sc, pred=pr, y={p: o[p]["y"] for p in ("calib", "test")}, unk_fam=unk_fam.copy(), val_acc=val_acc,
                                         hist=metas.get(m, {}).get("history"),
                                         ee=dict(m0={p: o[p]["expert_logits"][:, 0] for p in ("calib", "test", "unk")},
                                                 fused={p: o[p]["logits"] for p in ("calib", "test", "unk")}) if m == "agef_sharp" and t == R["seeds"][0] else None)
                SC[(ds, s, t, m)] = _apply_mask(ds, s, SC[(ds, s, t, m)])
                if SC[(ds, s, t, m)] is None:
                    del SC[(ds, s, t, m)]
                    if m == MODELS[0] and t == R["seeds"][0]: say(f"!! {ds} split {s}: too few evaluation images after deduplication - split skipped")
            if dfe is not None and t == R["seeds"][0]:
                tr = np.array(sp["train"]); remap = {c: i for i, c in enumerate(sp["known"])}; tr_y = np.array([remap[c] for c in R["targets"][tr]])
                fe = dfe / np.linalg.norm(dfe, axis=1, keepdims=True)
                sub = np.random.RandomState(s).choice(len(tr), min(len(tr), 6000), replace=False)
                kn = ax.KNNScorer(fe[tr][sub], 50, 6000, s); mh = maha(fe[tr][sub], tr_y[sub], K)
                sc = {p: {"knn50": kn(fe[np.array(sp[p])]), "mahalanobis": mh(fe[np.array(sp[p])])} for p in ("calib", "test", "unk")}
                SC[(ds, s, t, "dinov2_frozen")] = _apply_mask(ds, s, dict(sc=sc, pred=None, y=None, unk_fam=unk_fam.copy(), val_acc=np.nan, hist=None, ee=None))
                if SC[(ds, s, t, "dinov2_frozen")] is None: del SC[(ds, s, t, "dinov2_frozen")]
            del outs
        say(f"{ds} split {s}: scored ({(time.time() - t0) / 60:.1f} min)")
SCORES_ALL = sorted({sname for d in SC.values() for sname in d["sc"]["test"]})

# ----------------------------------------------------------------------------- long table of per-run metrics
rows = []
for (ds, s, t, m), d in SC.items():
    for sn in d["sc"]["test"]:
        c, te, u = d["sc"]["calib"][sn], d["sc"]["test"][sn], d["sc"]["unk"][sn]
        th5, th1 = cthr(c, FAR5), cthr(c, FAR1)
        y = np.r_[np.zeros(len(te)), np.ones(len(u))]; z = np.r_[te, u]
        rows.append(dict(dataset=ds, split=s, seed=t, model=m, score=sn, auroc=auroc(te, u),
                         aupr=float(__import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(y, z)),
                         fpr95=float((u <= np.quantile(te, 0.95)).mean()), zdr5=float((u > th5).mean()), far5=float((te > th5).mean()),
                         zdr1=float((u > th1).mean()), far1=float((te > th1).mean()), n_calib=len(c),
                         acc=float((d["pred"]["test"] == d["y"]["test"]).mean()) if d["pred"] is not None else np.nan, val_acc=d["val_acc"]))
L = pd.DataFrame(rows); L.to_csv(f"{OUT}/runs_long.csv", index=False)
MAIN = L.groupby(["dataset", "model", "score"]).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), aupr=("aupr", "mean"), fpr95=("fpr95", "mean"),
        zdr5=("zdr5", "mean"), far5=("far5", "mean"), zdr1=("zdr1", "mean"), far1=("far1", "mean"), acc=("acc", "mean"), n=("auroc", "size")).reset_index()
MAIN.to_csv(f"{OUT}/T_main_all_scores.csv", index=False)
say("main table written:", len(MAIN), "rows")

# ----------------------------------------------------------------------------- confirmatory comparisons (family cluster bootstrap)
def runs_of(ds, cfg):
    return {(s, t): SC[(ds, s, t, cfg[0])] for (d_, s, t, m) in SC if d_ == ds and m == cfg[0]}
def best_single(ds, s, t):
    return max(SINGLES, key=lambda x: SC[(ds, s, t, x)]["val_acc"])
def boot(ds, a, b, n_boot, seed=0):
    keys = sorted({(s, t) for (d_, s, t, m) in SC if d_ == ds and m == "agef_sharp"})
    def get(cfg, s, t):
        m = best_single(ds, s, t) if cfg[0] == "best_single" else cfg[0]; return SC[(ds, s, t, m)]["sc"]
    fams = sorted({f for (s, t) in keys for f in SC[(ds, s, t, "agef_sharp")]["unk_fam"]})
    pt = np.mean([auroc(get(a, s, t)["test"][a[1]], get(a, s, t)["unk"][a[1]]) - auroc(get(b, s, t)["test"][b[1]], get(b, s, t)["unk"][b[1]]) for s, t in keys])
    rng = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        fb = rng.choice(fams, len(fams)); diffs = []
        for s, t in keys:
            uf = SC[(ds, s, t, "agef_sharp")]["unk_fam"]
            idx = np.concatenate([np.where(uf == f)[0] for f in fb if (uf == f).any()] or [np.array([], int)])
            if not len(idx): continue
            A, B = get(a, s, t), get(b, s, t); kb = rng.randint(0, len(A["test"][a[1]]), len(A["test"][a[1]]))
            diffs.append(auroc(A["test"][a[1]][kb], A["unk"][a[1]][idx]) - auroc(B["test"][b[1]][kb], B["unk"][b[1]][idx]))
        if diffs: bs.append(np.mean(diffs))
    return pt, np.array(bs)
CONF = [("C1", ("deep_ensemble", "energy"), ("agef_sharp", "energy")), ("C2", ("deep_ensemble", "knn50"), ("agef_sharp", "knn50")),
        ("C3", ("avg_joint", "energy"), ("agef_sharp", "energy")), ("C4", ("agef_tuned", "energy"), ("agef_sharp", "energy")),
        ("C5", ("agef_tuned", "energy"), ("deep_ensemble", "energy")), ("C6", ("deep_ensemble", "energy"), ("best_single", "energy"))]
crow = []
for cid, a, b in CONF:
    per = {ds: boot(ds, a, b, N_BOOT, seed=zlib.crc32(cid.encode()) % 1000) for ds in RUNS
           if any(k[0] == ds and k[3] == "agef_sharp" for k in SC)}
    per = {ds: v for ds, v in per.items() if len(v[1])}
    if not per: say(f"{cid}: no evaluable splits - skipped"); continue
    nmin = min(len(v[1]) for v in per.values()); pooled_bs = np.mean([v[1][:nmin] for v in per.values()], 0)
    pooled_pt = np.mean([v[0] for v in per.values()])
    for ds, (pt, bs) in list(per.items()) + [("pooled", (pooled_pt, pooled_bs))]:
        crow.append(dict(id=cid, config=f"{a[0]}+{a[1]}", reference=f"{b[0]}+{b[1]}", dataset=ds, delta=pt,
                         ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5), ci90_lo=np.percentile(bs, 5), ci90_hi=np.percentile(bs, 95),
                         p=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
    say(f"{cid} done")
C = pd.DataFrame(crow); pooled = C[C.dataset == "pooled"].copy()
order = pooled.p.sort_values().index; run = 0
for r, i in enumerate(order):
    run = max(run, min(1.0, (len(order) - r) * pooled.loc[i, "p"])); C.loc[i, "p_holm"] = run
C["tost_equivalent"] = (C.ci90_lo > -TOST_MARGIN) & (C.ci90_hi < TOST_MARGIN)
C.to_csv(f"{OUT}/T_confirmatory.csv", index=False)

# ----------------------------------------------------------------------------- variance decomposition
vrow = []
for ds in sorted({k[0] for k in SC}):
    for m, sn in [("agef_sharp", "energy"), ("agef_tuned", "energy"), ("avg_joint", "energy"), ("deep_ensemble", "energy"), ("deep_ensemble", "knn50"), ("single_mobilenet", "energy")]:
        g = L[(L.dataset == ds) & (L.model == m) & (L.score == sn)].pivot_table(index="split", columns="seed", values="auroc")
        T = g.shape[1]; w = g.var(axis=1, ddof=1).mean(); bt = max(g.mean(axis=1).var(ddof=1) - w / T, 0)
        vrow.append(dict(dataset=ds, config=f"{m}+{sn}", mean=g.values.mean(), sd_seed=np.sqrt(w), sd_split=np.sqrt(bt), share_seed=w / (w + bt) if w + bt > 0 else np.nan))
V = pd.DataFrame(vrow); V.to_csv(f"{OUT}/T_variance.csv", index=False)

# ----------------------------------------------------------------------------- evaluation shortcuts (relative to the pool mean)
POOL = [(m, sn) for m in MODELS + ["deep_ensemble"] for sn in ["msp", "entropy", "maxlogit", "energy", "knn50", "mahalanobis"]]
srow, scal = [], []; rng = np.random.RandomState(0)
for ds in sorted({k[0] for k in SC}):
    g = L[(L.dataset == ds)]; g = g[g.apply(lambda r: (r.model, r.score) in POOL, axis=1)]
    piv = g.groupby(["model", "score", "split"]).zdr5.mean().unstack("split")
    pm = piv.values.mean(); s3 = piv.max(0).mean(); s4 = piv.loc[piv.mean(1).idxmax()].max()
    ko = []   # known-data-only rule: model with the highest calibration accuracy, energy score
    for s in RUNS[ds]["splits"]:
        for t in RUNS[ds]["seeds"]:
            if (ds, s, t, "agef_sharp") not in SC: continue
            best = max(MODELS + ["deep_ensemble"], key=lambda x: (SC[(ds, s, t, x)]["pred"]["calib"] == SC[(ds, s, t, x)]["y"]["calib"]).mean())
            ko.append(L[(L.dataset == ds) & (L.split == s) & (L.seed == t) & (L.model == best) & (L.score == "energy")].zdr5.iloc[0])
    # S1: thresholds on (a subsample of) training data, FAR reported on training data vs measured on held-out test data
    s1z, s1r, s1t = [], [], []
    for (d_, s, t, m), d in SC.items():
        if d_ != ds or (m, "energy") not in POOL: continue
        for sn in ["msp", "entropy", "maxlogit", "energy"]:
            th = cthr(d["sc"]["train"][sn], FAR5); s1z.append((d["sc"]["unk"][sn] > th).mean())
            s1r.append((d["sc"]["train"][sn] > th).mean()); s1t.append((d["sc"]["test"][sn] > th).mean())
    lf = g[g.score.isin(["msp", "entropy", "maxlogit", "energy"])]
    srow.append(dict(dataset=ds, pool_size=len(piv), pool_mean_zdr=pm, s3_same_split_best=s3, s4_best_config_best_split=s4,
                     known_only_rule_zdr=np.mean(ko), infl_s3_pp=100 * (s3 - pm), infl_s4_pp=100 * (s4 - pm),
                     s1_traincal_zdr=np.mean(s1z), leakfree_logit_zdr=lf.zdr5.mean(), infl_s1_pp=100 * (np.mean(s1z) - lf.zdr5.mean()),
                     s1_far_reported=np.mean(s1r), s1_far_true=np.mean(s1t), far_leakfree=lf.far5.mean()))
    for msize in [1, 2, 4, 8, 16, 32, len(piv)]:
        if msize > len(piv): continue
        v = [(lambda sub: sub.max(0).mean() - sub.values.mean())(piv.iloc[rng.choice(len(piv), msize, replace=False)]) for _ in range(300)]
        scal.append(dict(dataset=ds, pool_size=msize, s3_inflation_pp=100 * np.mean(v)))
S = pd.DataFrame(srow); S.to_csv(f"{OUT}/T_shortcuts.csv", index=False); pd.DataFrame(scal).to_csv(f"{OUT}/T_shortcut_pool_scaling.csv", index=False)

# ----------------------------------------------------------------------------- 1 % operating point
def cp(k, n):
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)
lrow = []
for (ds, s, t, m), d in SC.items():
    if t != RUNS[ds]["seeds"][0]: continue
    for sn in ["energy", "knn50", "mahalanobis"]:
        if sn not in d["sc"]["test"]: continue
        c, te, u = d["sc"]["calib"][sn], d["sc"]["test"][sn], d["sc"]["unk"][sn]; th = cthr(c, FAR1); k = int((te > th).sum())
        pool = np.r_[c, te]; r = np.random.RandomState(s); fr, zr = [], []
        for _ in range(N_REDRAW):
            p = r.permutation(len(pool)); h = len(pool) // 2; th2 = cthr(pool[p[:h]], FAR1)
            fr.append((pool[p[h:]] > th2).mean()); zr.append((u > th2).mean())
        lo, hi = cp(k, len(te))
        lrow.append(dict(dataset=ds, split=s, model=m, score=sn, n_calib=len(c), reliable=len(c) * FAR1 >= MIN_EXC, zdr=(u > th).mean(),
                         far=k / len(te), far_lo=lo, far_hi=hi, redraw_far_sd=np.std(fr), redraw_zdr_mean=np.mean(zr), redraw_zdr_sd=np.std(zr)))
LF = pd.DataFrame(lrow).groupby(["dataset", "model", "score"]).mean(numeric_only=True).reset_index(); LF.to_csv(f"{OUT}/T_lowfar.csv", index=False)

# ----------------------------------------------------------------------------- residual duplicates across partitions
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8); drow = []
for ds, R in RUNS.items():
    keys = json.load(open(f"{R['dir']}/image_keys.json"))["keys"]; f0 = np.array([k[0] for k in keys]); f1 = np.array([k[1] for k in keys])
    hb = np.stack([np.frombuffer(bytes.fromhex(k[2]), dtype=np.uint8) for k in keys])
    for s in R["splits"]:
        if DEDUP_EVAL and (ds, s, 0, "agef_sharp") not in SC and not any(k[:2] == (ds, s) for k in SC): continue
        sp = json.load(open(f"{R['dir']}/split{s}/split.json")); tr = np.array(sp["train"])
        ev = np.r_[np.array(sp["calib"])[MASK[(ds, s, "calib")]], np.array(sp["test"])[MASK[(ds, s, "test")]]] if DEDUP_EVAL else np.r_[sp["calib"], sp["test"]]
        dmin = np.concatenate([POP[hb[ev[i:i + 256]][:, None] ^ hb[tr][None]].sum(-1).min(1) for i in range(0, len(ev), 256)])
        drow.append(dict(dataset=ds, split=s, L0_file=np.isin(f0[ev], f0[tr]).mean(), L1_pixels=np.isin(f1[ev], f1[tr]).mean(),
                         L2_identical_hash=(dmin == 0).mean(), L3_hash_le8=(dmin <= 8).mean()))
DU = pd.DataFrame(drow).groupby("dataset").mean(numeric_only=True).drop(columns="split"); DU.to_csv(f"{OUT}/T_residual_duplicates.csv")

# ----------------------------------------------------------------------------- per-family recall, Mondrian, early exit, triage note
frow = []
for (ds, s, t, m), d in SC.items():
    if m != "deep_ensemble": continue
    th = cthr(d["sc"]["calib"]["energy"], FAR5)
    for f in np.unique(d["unk_fam"]):
        mk = d["unk_fam"] == f; frow.append(dict(dataset=ds, split=s, seed=t, family=f, recall=(d["sc"]["unk"]["energy"][mk] > th).mean()))
FAM = pd.DataFrame(frow).groupby(["dataset", "family"]).recall.agg(["mean", "min", "max", "count"]).reset_index(); FAM.to_csv(f"{OUT}/T_families.csv", index=False)
mrow = []
for (ds, s, t, m), d in SC.items():
    if m != "deep_ensemble" or "Other" not in RUNS[ds]["classes"]: continue
    sp = json.load(open(f"{RUNS[ds]['dir']}/split{s}/split.json")); ob = sp["known_names"].index("Other"); sc = d["sc"]
    g = cthr(sc["calib"]["energy"], FAR5); pc, pt, pu = d["pred"]["calib"], d["pred"]["test"], d["pred"]["unk"]
    thc = {c: (cthr(sc["calib"]["energy"][pc == c], FAR5) if (pc == c).sum() >= 20 else g) for c in range(len(sp["known"]))}
    yt = d["y"]["test"]; ben = yt == ob
    mrow.append(dict(dataset=ds, split=s, seed=t, zdr_global=(sc["unk"]["energy"] > g).mean(), zdr_mondrian=(sc["unk"]["energy"] > np.array([thc[c] for c in pu])).mean(),
                     benign_far_global=(sc["test"]["energy"][ben] > g).mean(), benign_far_mondrian=(sc["test"]["energy"][ben] > np.array([thc[c] for c in pt[ben]])).mean(),
                     unknown_predicted_benign=(pu == ob).mean()))
MO = pd.DataFrame(mrow).groupby("dataset").mean(numeric_only=True).drop(columns=["split", "seed"]) if mrow else pd.DataFrame(); MO.to_csv(f"{OUT}/T_mondrian.csv")
erow = []; fm, fr_ = 0.111, 3.627 + 0.769
for (ds, s, t, m), d in SC.items():
    if d["ee"] is None: continue
    e = d["ee"]; pm = {p: softmax(e["m0"][p], -1) for p in e["m0"]}; mc = pm["calib"].max(1)
    for q in [0, 0.25, 0.5, 0.75, 0.9]:
        th = np.inf if q == 0 else np.quantile(mc, 1 - q)
        def pol(p):
            ex = pm[p].max(1) >= th; pr = np.where(ex[:, None], pm[p], softmax(e["fused"][p], -1)); return ax.entropy(pr), ex
        sc_, _ = pol("calib"); st, ek = pol("test"); su, eu = pol("unk"); th5 = cthr(sc_, FAR5)
        erow.append(dict(dataset=ds, split=s, target_exit=q, exit_known=ek.mean(), exit_unknown=eu.mean(),
                         saving_known=1 - (fm + (1 - ek.mean()) * fr_) / (fm + fr_), saving_unknown=1 - (fm + (1 - eu.mean()) * fr_) / (fm + fr_),
                         zdr=(su > th5).mean(), far=(st > th5).mean()))
EE = pd.DataFrame(erow).groupby(["dataset", "target_exit"]).mean(numeric_only=True).drop(columns="split"); EE.to_csv(f"{OUT}/T_early_exit.csv")
LAT = {ds: json.load(open(f"{R['dir']}/latency.json")) for ds, R in RUNS.items() if os.path.exists(f"{R['dir']}/latency.json")}
json.dump(LAT, open(f"{OUT}/T_latency.json", "w"), indent=1)
trow = []
for (ds, s, t, m), d in SC.items():
    if m not in ("deep_ensemble", "agef_sharp") or t != RUNS[ds]["seeds"][0]: continue
    sc = d["sc"]; ta = cthr(sc["calib"]["msp"], FAR5); tb = cthr(sc["calib"]["entropy"], FAR5)
    A, B = sc["unk"]["msp"] > ta, sc["unk"]["entropy"] > tb
    trow.append(dict(dataset=ds, split=s, model=m, unk_review_or_log=((A ^ B)).mean(), unk_quarantine=(A & B).mean()))
TRI = pd.DataFrame(trow).groupby(["dataset", "model"]).mean(numeric_only=True).drop(columns="split"); TRI.to_csv(f"{OUT}/T_triage_note.csv")

# ----------------------------------------------------------------------------- cross-representation and few-shot (completed runs 08 / 10 / 14)
for name in ["xrep_raw.csv", "fewshot_raw.csv", "fewshot_baselines_raw.csv"]:
    fs = [f for f in glob.glob(f"{ROOT}/**/{name}", recursive=True) if "QUICKTEST" not in f]
    if fs:
        X = pd.read_csv(fs[0]); keys = [c for c in ["direction", "model", "score", "k", "method"] if c in X.columns]
        X.groupby(keys).mean(numeric_only=True).reset_index().to_csv(f"{OUT}/T_{name.replace('_raw.csv', '')}_summary.csv", index=False)
        say(f"{name}: summarised from {fs[0]}")
    else:
        say(f"{name}: not attached (section skipped)")

# ----------------------------------------------------------------------------- verdicts
say("=" * 100)
say("EVALUATION:", "deduplicated (sensitivity analysis, deviation from PROTOCOL.md)" if DEDUP_EVAL else "as pre-registered")
for cid, a, b in CONF:
    if not ((C.id == cid) & (C.dataset == "pooled")).any(): continue
    r = C[(C.id == cid) & (C.dataset == "pooled")].iloc[0]
    per = "; ".join(f"{x.dataset} {x.delta:+.3f} [{x.ci95_lo:+.3f}, {x.ci95_hi:+.3f}]" for x in C[(C.id == cid) & (C.dataset != "pooled")].itertuples())
    verdict = ("SUPERIOR" if r.ci95_lo > 0 and r.p_holm < ALPHA else "INFERIOR" if r.ci95_hi < 0 and r.p_holm < ALPHA else
               "EQUIVALENT (TOST)" if r.tost_equivalent else "NO SIGNIFICANT DIFFERENCE")
    say(f"{cid} {r.config} vs {r.reference}: pooled {r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}], Holm p {r.p_holm:.4f} -> {verdict} | {per}")
say("Residual duplicates across partitions (mean over splits):\n" + DU.round(4).to_string())
say("=" * 100)
open(f"{OUT}/VERDICTS.txt", "w").write("\n".join(LOG))
print("outputs:", sorted(os.listdir(OUT)))
