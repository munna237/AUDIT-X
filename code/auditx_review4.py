# =============================================================================
# AUDIT-X FINAL - fourth-round review analyses (CPU only, no training)
# Inputs (anywhere under ROOT): auditx_final_malimg/, auditx_final_malevis_v5/ (Amendment 1), auditx_final_dumpware10/,
#   and for G10 only the original MaleVis run auditx_final_malevis/.
#   G1 three-seed ensembles of AGEF-Net (original, tuned) vs deep ensemble and 3 x MobileNetV3-S
#   G2 TOST (+-0.03) for seed-ensemble comparisons
#   G3 stacking on frozen independently trained CNNs (logistic regression; learned global weights), fit on half of calibration
#   G4 gate collapse and collapsed seeds of AGEF-Net
#   G5 UFR at matched realized FAR (5 %) and partial AUROC (FPR <= 0.05)
#   G6 nested shortcut design (select on one split, evaluate on another)
#   G7 precision and alerts per 10,000 files at 0.1 %, 1 %, 10 % prevalence
#   G8 minimum detectable effects and DerSimonian-Laird random-effects pooling for C1-C6
#   G9 near-twin calibration images: group sizes of their training twins (cap) and effect on FAR
#   G10 original MaleVis run re-scored without VBA in evaluation (isolates the VBA exclusion from the grouping change)
#   G3b frozen-expert gate: per-sample gate MLP trained on frozen independently trained CNNs (half of calibration)
#   G11 family-macro AUROC / UFR and C1-C6 under family-macro AUROC;  G12 distinct families and family jackknife
#   G13 what +-0.03 AUROC means in UFR at matched FAR;  G14 KNN bank-size sensitivity;  G15 tuned-AGEF learning-rate sweep
#   G16 near-duplicate rule validity (pixel similarity of matched pairs; needs the datasets attached)
# =============================================================================
import glob, json, os, time, zlib, contextlib, io
import numpy as np, pandas as pd
from scipy.special import softmax, logsumexp, log_softmax
from scipy.stats import rankdata, norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import auditx_eval_v2 as ax

ROOT = os.environ.get("AUDITX_ROOT", "/kaggle/input")
OUT = os.environ.get("AUDITX_OUT", "/kaggle/working/auditx_review4"); os.makedirs(OUT, exist_ok=True)
N_BOOT = int(os.environ.get("AUDITX_NBOOT", 1000)); MARGIN = 0.03
DATASETS = ["Malimg", "MaleVis", "Dumpware10"]
MODELS = ["agef_sharp", "agef_tuned", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"]
SINGLES = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
V4 = "grouped-L1-v3+near-L3-8bit-cap250"; V5 = "grouped-L1-v3+identical-hash-whole+near-L3-8bit-cap250"
EXPECTED = {"Malimg": V4, "Dumpware10": V4, "MaleVis": V5}
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

def find(name):
    hits = [os.path.dirname(p) for p in glob.glob(f"{ROOT}/**/{name}/MANIFEST.json", recursive=True)]
    return sorted(hits)[0] if hits else None
RUNS = {}
for ds in DATASETS:
    d = find(f"auditx_final_{ds.lower()}" + ("_v5" if ds == "MaleVis" else ""))
    if d is None: say(f"!! {ds}: run not found"); continue
    man = json.load(open(f"{d}/MANIFEST.json")); assert man["complete"] and man["protocol"] == EXPECTED[ds], f"{ds}: wrong run"
    RUNS[ds] = dict(dir=d, targets=np.load(f"{d}/targets.npy"), classes=json.load(open(f"{d}/classes.json")),
                    splits=sorted(int(p.split("split")[-1]) for p in glob.glob(f"{d}/split*") if os.path.isdir(p)), seeds=man["args"]["train_seeds"])
    say(f"{ds}: {d} | protocol OK")
assert RUNS

def auroc(k, u):
    r = rankdata(np.r_[k, u]); return (r[len(k):].sum() - len(u) * (len(u) + 1) / 2) / (len(u) * len(k))
def cthr(x, a):
    n = len(x); k = int(np.ceil((n + 1) * (1 - a))); return np.inf if k > n else np.sort(x)[k - 1]
def load_npz(p):
    z = np.load(p, allow_pickle=True); parts = {}
    for k in z.files:
        if "__" in k:
            pp, kk = k.split("__", 1)
            if kk in ("logits", "expert_logits", "weights", "feats", "y"): parts.setdefault(pp, {})[kk] = z[k].astype(np.float64) if kk != "y" else z[k]
    return parts, json.loads(str(z["meta"]))
def ens(parts_list):
    out = {}
    for p in ("train", "calib", "test", "unk"):
        el = np.concatenate([o[p]["expert_logits"] if o[p]["expert_logits"].ndim == 3 else o[p]["logits"][:, None] for o in parts_list], 1)
        out[p] = dict(expert_logits=el, logits=el.mean(1), feats=np.concatenate([o[p]["feats"] for o in parts_list], 1), y=parts_list[0][p]["y"])
    return out
def seed_ens(parts_list):
    """average the softmax of whole models (fused logits) over training seeds"""
    out = {}
    for p in ("train", "calib", "test", "unk"):
        el = np.stack([o[p]["logits"] for o in parts_list], 1)
        out[p] = dict(expert_logits=el, logits=el.mean(1), feats=np.concatenate([o[p]["feats"] for o in parts_list], 1), y=parts_list[0][p]["y"])
    return out
def energy(lg): return -logsumexp(lg, axis=1)
def knn(o, k=50, seed=0):
    kn = ax.KNNScorer(o["train"]["feats"], k, 6000, seed); return {p: kn(o[p]["feats"]) for p in ("calib", "test", "unk")}

# ----------------------------------------------------------------------------- load and score
SC, AUX, t0 = {}, {}, time.time()
for ds, R in RUNS.items():
    for s in R["splits"]:
        sp = json.load(open(f"{R['dir']}/split{s}/split.json")); K = len(sp["known"])
        unk_fam = np.array([R["classes"][t] for t in R["targets"][sp["unk"]]])
        per = {}
        for t in R["seeds"]:
            loaded = {m: load_npz(f"{R['dir']}/split{s}/seed{t}/{m}.npz") for m in MODELS}
            outs = {m: v[0] for m, v in loaded.items()}; per[t] = outs
            best = max(SINGLES, key=lambda x: max(h["val_acc"] for h in loaded[x][1]["history"]))   # validation-selected single CNN (C6)
            de = ens([outs[x] for x in SINGLES])
            cfgs = {"deep_ensemble": de, "agef_sharp": outs["agef_sharp"], "agef_tuned": outs["agef_tuned"], "avg_joint": outs["avg_joint"],
                    "single_mobilenet": outs["single_mobilenet"], "single_resnet18": outs["single_resnet18"], "single_effnet_b0": outs["single_effnet_b0"]}
            for m, o in cfgs.items():
                sc = {p: {"energy": energy(o[p]["logits"])} for p in ("calib", "test", "unk")}
                kk = knn(o, 50, s)
                for p in sc: sc[p]["knn50"] = kk[p]
                acc = float(((softmax(o["test"]["expert_logits"], -1).mean(1) if m == "deep_ensemble" else softmax(o["test"]["logits"], -1)).argmax(1) == o["test"]["y"]).mean())
                SC[(ds, s, t, m)] = dict(sc=sc, unk_fam=unk_fam, acc=acc)
            SC[(ds, s, t, "best_single")] = SC[(ds, s, t, best)]
            # G4: gate statistics
            for m in ("agef_sharp", "agef_tuned"):
                w = outs[m]["test"]["weights"]; mw = w.mean(0)
                AUX.setdefault("gate", []).append(dict(dataset=ds, split=s, seed=t, model=m, w_mobilenet=mw[0], w_resnet=mw[1], w_effnet=mw[2],
                                                       max_mean_weight=mw.max(), share_argmax_same=float((w.argmax(1) == np.bincount(w.argmax(1)).argmax()).mean()),
                                                       mean_sample_entropy=float(-(w * np.log(w + 1e-12)).sum(1).mean()), test_acc=SC[(ds, s, t, m)]["acc"],
                                                       auroc_energy=auroc(SC[(ds, s, t, m)]["sc"]["test"]["energy"], SC[(ds, s, t, m)]["sc"]["unk"]["energy"])))
            # G3: stacking on frozen singles, fit on half A of calibration, thresholds on half B
            rng = np.random.RandomState(1000 * s + t); n_c = len(outs[SINGLES[0]]["calib"]["y"]); perm = rng.permutation(n_c); A, Bh = perm[: n_c // 2], perm[n_c // 2:]
            feat = {p: np.concatenate([log_softmax(outs[x][p]["logits"], -1) for x in SINGLES], 1) for p in ("calib", "test", "unk")}
            yA = outs[SINGLES[0]]["calib"]["y"][A]
            try:
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(feat["calib"][A], yA)
                full = np.zeros((1, K)); cls = lr.classes_
                def lr_logits(X):
                    z = np.full((len(X), K), -30.0); z[:, cls] = lr.decision_function(X) if len(cls) > 2 else np.c_[-lr.decision_function(X), lr.decision_function(X)]; return z
                st = {p: lr_logits(feat[p]) for p in ("calib", "test", "unk")}
            except Exception as e:
                st = None; say("stacking failed", ds, s, t, e)
            # learned global weights (softmax-parametrised) minimising calibration-half-A NLL of the weighted logit average
            L = {p: np.stack([outs[x][p]["logits"] for x in SINGLES], 1) for p in ("calib", "test", "unk")}
            def nll(v):
                w = softmax(v); lg = (w[None, :, None] * L["calib"][A]).sum(1); return -log_softmax(lg, -1)[np.arange(len(A)), yA].mean()
            wv = softmax(minimize(nll, np.zeros(3), method="Nelder-Mead").x)
            gw = {p: (wv[None, :, None] * L[p]).sum(1) for p in L}
            # frozen-expert gate: MLP on the concatenated frozen logits -> softmax weights over the 3 experts (per sample)
            import torch
            torch.manual_seed(1000 * s + t)
            Xc = torch.tensor(L["calib"][A], dtype=torch.float32); Yc = torch.tensor(yA, dtype=torch.long)
            gate = torch.nn.Sequential(torch.nn.Linear(3 * K, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3))
            opt = torch.optim.Adam(gate.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(400):
                w = torch.softmax(gate(Xc.flatten(1)), 1); fused = (w.unsqueeze(-1) * Xc).sum(1)
                loss = torch.nn.functional.cross_entropy(fused, Yc); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                fg = {}
                for p in L:
                    X = torch.tensor(L[p], dtype=torch.float32); w = torch.softmax(gate(X.flatten(1)), 1); fg[p] = (w.unsqueeze(-1) * X).sum(1).numpy().astype(np.float64)
                    if p == "test": AUX.setdefault("fgate", []).append(dict(dataset=ds, split=s, seed=t, mean_w=w.mean(0).numpy().round(3).tolist(),
                                                                          sample_entropy=float(-(w * torch.log(w + 1e-12)).sum(1).mean())))
            for name, lgs in [("stack_lr", st), ("stack_weights", gw), ("frozen_gate", fg), ("deep_ensemble_halfcal", {p: L[p].mean(1) for p in L})]:
                if lgs is None: continue
                sc = {p: {"energy": energy(lgs[p])} for p in ("calib", "test", "unk")}
                sc["calib"] = {"energy": sc["calib"]["energy"][Bh]}
                SC[(ds, s, t, name)] = dict(sc=sc, unk_fam=unk_fam, acc=float((lgs["test"].argmax(1) == outs[SINGLES[0]]["test"]["y"]).mean()),
                                            weights=wv.tolist() if name == "stack_weights" else None)
        # G1: seed ensembles (one evaluation per split)
        for name, mm in [("seedens_agef_sharp", "agef_sharp"), ("seedens_agef_tuned", "agef_tuned"), ("seedens_mobilenet", "single_mobilenet")]:
            o = seed_ens([per[t][mm] for t in R["seeds"]])
            sc = {p: {"energy": energy(o[p]["logits"])} for p in ("calib", "test", "unk")}; kk = knn(o, 50, s)
            for p in sc: sc[p]["knn50"] = kk[p]
            SC[(ds, s, -1, name)] = dict(sc=sc, unk_fam=unk_fam, acc=float((softmax(o["test"]["expert_logits"], -1).mean(1).argmax(1) == o["test"]["y"]).mean()))
        say(f"{ds} split {s} done ({(time.time() - t0) / 60:.1f} min)")

# ----------------------------------------------------------------------------- per-run metrics incl. matched FAR and partial AUROC (G5) and precision (G7)
rows = []
for (ds, s, t, m), d in SC.items():
    if m == "best_single": continue
    for sn, v in d["sc"]["test"].items():
        c, te, u = d["sc"]["calib"][sn], v, d["sc"]["unk"][sn]; th = cthr(c, 0.05); far, ufr = (te > th).mean(), (u > th).mean()
        thm = np.quantile(te, 0.95)                                                     # matched realized FAR = 5 % on test
        y = np.r_[np.zeros(len(te)), np.ones(len(u))]; z = np.r_[te, u]
        r = dict(dataset=ds, split=s, seed=t, model=m, score=sn, auroc=auroc(te, u), pauroc05=roc_auc_score(y, z, max_fpr=0.05),
                 ufr5=ufr, far5=far, ufr_matched5=float((u > thm).mean()), acc=d["acc"])
        for pi in (0.001, 0.01, 0.10):
            den = pi * ufr + (1 - pi) * far; r[f"prec_pi{pi}"] = pi * ufr / den if den > 0 else np.nan; r[f"alerts10k_pi{pi}"] = 1e4 * den
        fams = np.unique(d["unk_fam"])
        r["auroc_family_macro"] = float(np.mean([auroc(te, u[d["unk_fam"] == f]) for f in fams]))
        r["ufr5_family_macro"] = float(np.mean([(u[d["unk_fam"] == f] > th).mean() for f in fams]))
        rows.append(r)
L = pd.DataFrame(rows); L.to_csv(f"{OUT}/G_runs_long.csv", index=False)
SUM = L.groupby(["dataset", "model", "score"]).mean(numeric_only=True).drop(columns=["split", "seed"]).reset_index(); SUM.to_csv(f"{OUT}/G_summary.csv", index=False)

# ----------------------------------------------------------------------------- bootstrap (families, test images, training seeds) and comparisons
def get(ds, m, s, t):
    return SC[(ds, s, -1 if m.startswith("seedens_") else t, m)]
def boot(ds, a, b, sn, n_boot, seed, macro=False, drop_family=None):
    splits, seeds = RUNS[ds]["splits"], RUNS[ds]["seeds"]
    fams = sorted({f for s in splits for f in SC[(ds, s, seeds[0], "agef_sharp")]["unk_fam"]})
    if drop_family is not None: fams = [f for f in fams if f != drop_family]
    def metric(k, u, uf):
        if not macro: return auroc(k, u)
        return np.mean([auroc(k, u[uf == f]) for f in np.unique(uf)])
    def diff(s, t, fidx=None, kb=None):
        A, B = get(ds, a, s, t)["sc"], get(ds, b, s, t)["sc"]
        uf = SC[(ds, s, seeds[0], "agef_sharp")]["unk_fam"]
        if fidx is None:
            fidx = np.where(uf != drop_family)[0] if drop_family is not None else np.arange(len(uf))
        ka, kb_ = (A["test"][sn], B["test"][sn]) if kb is None else (A["test"][sn][kb], B["test"][sn][kb])
        if len(fidx) == 0: return np.nan
        if macro and isinstance(fidx, list):        # bootstrap: one entry per drawn family, with multiplicity
            return np.mean([auroc(ka, A["unk"][sn][g]) - auroc(kb_, B["unk"][sn][g]) for g in fidx])
        ua, ub = A["unk"][sn][fidx], B["unk"][sn][fidx]
        return metric(ka, ua, uf[fidx]) - metric(kb_, ub, uf[fidx])
    pt = np.nanmean([diff(s, t) for s in splits for t in seeds]); rng = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        fb = rng.choice(fams, len(fams)); vals = []
        for s in splits:
            uf = SC[(ds, s, seeds[0], "agef_sharp")]["unk_fam"]
            groups = [np.where(uf == f)[0] for f in fb if (uf == f).any()]
            if not groups: continue
            idx = groups if macro else np.concatenate(groups)
            n_te = len(SC[(ds, s, seeds[0], "agef_sharp")]["sc"]["test"]["energy"]); kb = rng.randint(0, n_te, n_te)
            vals += [diff(s, t, idx, kb) for t in rng.choice(seeds, len(seeds))]
        vals = [v for v in vals if not np.isnan(v)]
        if vals: bs.append(np.mean(vals))
    return pt, np.array(bs)
COMP = [("G1", "seedens_agef_sharp", "deep_ensemble", "energy"), ("G1", "seedens_agef_sharp", "deep_ensemble", "knn50"),
        ("G1", "seedens_agef_sharp", "seedens_mobilenet", "energy"), ("G1", "seedens_agef_tuned", "deep_ensemble", "energy"),
        ("G1", "seedens_agef_sharp", "agef_sharp", "energy"), ("G2", "seedens_mobilenet", "deep_ensemble", "energy"),
        ("G2", "seedens_mobilenet", "deep_ensemble", "knn50"),
        ("G3", "stack_lr", "deep_ensemble_halfcal", "energy"), ("G3", "stack_weights", "deep_ensemble_halfcal", "energy"),
        ("G3", "stack_lr", "agef_sharp", "energy"), ("G3", "frozen_gate", "deep_ensemble_halfcal", "energy"), ("G3", "frozen_gate", "agef_sharp", "energy"),
        ("C1", "deep_ensemble", "agef_sharp", "energy"), ("C2", "deep_ensemble", "agef_sharp", "knn50"), ("C3", "avg_joint", "agef_sharp", "energy"),
        ("C4", "agef_tuned", "agef_sharp", "energy"), ("C5", "agef_tuned", "deep_ensemble", "energy"), ("C6", "deep_ensemble", "best_single", "energy")]
crow = []
for gid, a, b, sn in COMP:
    per = {ds: boot(ds, a, b, sn, N_BOOT, zlib.crc32(f"{a}{b}{sn}".encode()) % 1000) for ds in RUNS}
    for ds, (pt, bs) in per.items():
        crow.append(dict(group=gid, config=a, reference=b, score=sn, dataset=ds, delta=pt, se=bs.std(ddof=1),
                         ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5), ci90_lo=np.percentile(bs, 5), ci90_hi=np.percentile(bs, 95)))
    nmin = min(len(v[1]) for v in per.values()); pb = np.mean([v[1][:nmin] for v in per.values()], 0); pt = np.mean([v[0] for v in per.values()])
    # G8: random effects (DerSimonian-Laird) from per-dataset estimates and bootstrap SEs
    est = np.array([v[0] for v in per.values()]); se = np.array([v[1].std(ddof=1) for v in per.values()]); w = 1 / se ** 2
    fe = (w * est).sum() / w.sum(); Q = (w * (est - fe) ** 2).sum(); tau2 = max(0.0, (Q - (len(est) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (se ** 2 + tau2); re = (wr * est).sum() / wr.sum(); re_se = np.sqrt(1 / wr.sum())
    crow.append(dict(group=gid, config=a, reference=b, score=sn, dataset="pooled", delta=pt, se=pb.std(ddof=1),
                     ci95_lo=np.percentile(pb, 2.5), ci95_hi=np.percentile(pb, 97.5), ci90_lo=np.percentile(pb, 5), ci90_hi=np.percentile(pb, 95),
                     random_effects=re, re_ci_lo=re - 1.96 * re_se, re_ci_hi=re + 1.96 * re_se, tau2=tau2,
                     mde80=(norm.ppf(0.975) + norm.ppf(0.8)) * pb.std(ddof=1)))
    say(f"{gid} {a} vs {b} ({sn}) done")
CMP = pd.DataFrame(crow)
CMP["tost_equivalent"] = (CMP.ci90_lo > -MARGIN) & (CMP.ci90_hi < MARGIN)
CMP["ci95_excludes_0"] = (CMP.ci95_lo > 0) | (CMP.ci95_hi < 0)
CMP.to_csv(f"{OUT}/G_comparisons.csv", index=False)

# ----------------------------------------------------------------------------- G4 gate collapse
GT = pd.DataFrame(AUX["gate"]); GT["collapsed_gate"] = GT.max_mean_weight > 0.9
GT.to_csv(f"{OUT}/G4_gate_per_run.csv", index=False)

# ----------------------------------------------------------------------------- G6 nested shortcut (select on split s', evaluate on split s)
POOL = [(m, sn) for m in ["agef_sharp", "agef_tuned", "avg_joint", "deep_ensemble"] + SINGLES for sn in ["energy", "knn50"]]
nrow = []
for ds in RUNS:
    g = L[(L.dataset == ds) & (L.seed >= 0)]; g = g[g.apply(lambda r: (r.model, r.score) in POOL, axis=1)]
    piv = g.groupby(["model", "score", "split"]).ufr5.mean().unstack("split"); ref = piv.loc[("deep_ensemble", "energy")]
    nested = [piv.loc[piv[s2].idxmax(), s] for s in piv.columns for s2 in piv.columns if s2 != s]
    nrow.append(dict(dataset=ds, pool=len(piv), prespecified=ref.mean(), pool_mean=piv.values.mean(), nested_select_other_split=np.mean(nested),
                     same_split_best=piv.max(0).mean(), nested_vs_prespecified_pp=100 * (np.mean(nested) - ref.mean()),
                     same_split_vs_prespecified_pp=100 * (piv.max(0).mean() - ref.mean())))
NS = pd.DataFrame(nrow); NS.to_csv(f"{OUT}/G6_nested_shortcuts.csv", index=False)

# ----------------------------------------------------------------------------- G9 near-twin calibration images
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8); trow = []
class FB:
    def __init__(self, classes, targets): self.classes = classes; self.samples = [(f"x{i}", int(t)) for i, t in enumerate(targets)]; self.targets = list(targets)
for ds, R in RUNS.items():
    keys = json.load(open(f"{R['dir']}/image_keys.json"))["keys"]; hb = np.stack([np.frombuffer(bytes.fromhex(k[2]), dtype=np.uint8) for k in keys])
    kt = tuple(ax.CFG["GROUP_KEYS"]) + ((2,) if ds in ax.CFG.get("IDENTICAL_WHOLE_DATASETS", ()) else ())
    with contextlib.redirect_stdout(io.StringIO()):
        G = ax.compute_groups(FB(R["classes"], R["targets"]), f"{R['dir']}/image_keys.json", kt, 0.9, 10, 8, 250)
    sizes = np.bincount(G["groups"])
    for s in R["splits"]:
        sp = json.load(open(f"{R['dir']}/split{s}/split.json")); tr = np.array(sp["train"])
        for p in ("calib", "test"):
            q = np.array(sp[p]); d = POP[hb[q][:, None] ^ hb[tr][None]].sum(-1); dm = d.min(1); twin = tr[d.argmin(1)]; m = dm <= 8
            trow.append(dict(dataset=ds, split=s, part=p, n=len(q), near_twins=int(m.sum()),
                             twin_group_at_cap=int((sizes[G["groups"][twin[m]]] >= 250).sum()) if m.any() else 0,
                             own_group_at_cap=int((sizes[G["groups"][q[m]]] >= 250).sum()) if m.any() else 0,
                             families=",".join(sorted(set(R["classes"][c] for c in R["targets"][q[m]]))) if m.any() else ""))
TW = pd.DataFrame(trow); TW.to_csv(f"{OUT}/G9_near_twins.csv", index=False)

# ----------------------------------------------------------------------------- G10 original MaleVis run without VBA in evaluation
d4 = find("auditx_final_malevis"); g10 = []
if d4:
    man = json.load(open(f"{d4}/MANIFEST.json")); assert man["protocol"] == V4
    tg = np.load(f"{d4}/targets.npy"); cl = json.load(open(f"{d4}/classes.json")); vba = cl.index("VBA") if "VBA" in cl else -1
    for s in sorted(int(p.split("split")[-1]) for p in glob.glob(f"{d4}/split*") if os.path.isdir(p)):
        sp = json.load(open(f"{d4}/split{s}/split.json"))
        keep = {p: tg[np.array(sp[p])] != vba for p in ("calib", "test", "unk")}
        for t in man["args"]["train_seeds"]:
            outs = {m: load_npz(f"{d4}/split{s}/seed{t}/{m}.npz")[0] for m in ["agef_sharp", "avg_joint"] + SINGLES}
            cf = {"deep_ensemble": ens([outs[x] for x in SINGLES]), "agef_sharp": outs["agef_sharp"], "avg_joint": outs["avg_joint"]}
            for m, o in cf.items():
                e = {p: energy(o[p]["logits"]) for p in ("calib", "test", "unk")}
                for ver, kp in (("all", None), ("without_VBA", keep)):
                    ee = e if kp is None else {p: e[p][kp[p]] for p in e}
                    th = cthr(ee["calib"], 0.05)
                    g10.append(dict(split=s, seed=t, model=m, version=ver, auroc=auroc(ee["test"], ee["unk"]), ufr5=(ee["unk"] > th).mean(), far5=(ee["test"] > th).mean()))
    G10 = pd.DataFrame(g10).groupby(["model", "version"]).mean(numeric_only=True).drop(columns=["split", "seed"]).reset_index()
    G10.to_csv(f"{OUT}/G10_original_malevis_without_vba.csv", index=False)
else:
    say("G10 skipped: original MaleVis run (auditx_final_malevis) not attached"); G10 = pd.DataFrame()

# ----------------------------------------------------------------------------- G11 C1-C6 under family-macro AUROC
mrow = []
for gid, a, b, sn in [c for c in COMP if c[0].startswith("C")]:
    per = {ds: boot(ds, a, b, sn, N_BOOT, zlib.crc32(f"macro{a}{b}{sn}".encode()) % 1000, macro=True) for ds in RUNS}
    nmin = min(len(v[1]) for v in per.values()); pb = np.mean([v[1][:nmin] for v in per.values()], 0)
    for ds, (pt, bs) in list(per.items()) + [("pooled", (np.mean([v[0] for v in per.values()]), pb))]:
        mrow.append(dict(id=gid, config=a, reference=b, score=sn, dataset=ds, delta_family_macro=pt, ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5)))
    say(f"G11 {gid} family-macro done")
FM = pd.DataFrame(mrow); FM.to_csv(f"{OUT}/G11_family_macro_comparisons.csv", index=False)

# ----------------------------------------------------------------------------- G12 distinct families and leave-one-family-out jackknife (point estimates)
jrow = []
for ds in RUNS:
    seeds = RUNS[ds]["seeds"]; fams = sorted({f for s in RUNS[ds]["splits"] for f in SC[(ds, s, seeds[0], "agef_sharp")]["unk_fam"]})
    jrow.append(dict(dataset=ds, id="n_families", value=len(fams)))
    for gid, a, b, sn in [c for c in COMP if c[0].startswith("C")]:
        full = boot(ds, a, b, sn, 0, 0)[0]
        jk = [boot(ds, a, b, sn, 0, 0, drop_family=f)[0] for f in fams]
        jrow.append(dict(dataset=ds, id=gid, value=full, jk_min=np.nanmin(jk), jk_max=np.nanmax(jk), sign_changes=int(np.sum(np.sign(jk) != np.sign(full)))))
JK = pd.DataFrame(jrow); JK.to_csv(f"{OUT}/G12_family_jackknife.csv", index=False)

# ----------------------------------------------------------------------------- G13 what +-0.03 AUROC means operationally
pr = L[(L.seed >= 0) & L.score.isin(["energy", "knn50"]) & ~L.model.isin(["stack_lr", "stack_weights", "frozen_gate", "deep_ensemble_halfcal"])]
slope = np.polyfit(pr.auroc, pr.ufr_matched5, 1)[0]
sd_seed = L[(L.seed >= 0)].groupby(["dataset", "model", "score", "split"]).auroc.std().median()
json.dump(dict(slope_ufr_matched5_per_auroc=slope, ufr_pp_for_0p03_auroc=100 * 0.03 * slope, median_seed_sd_auroc=sd_seed),
          open(f"{OUT}/G13_margin_meaning.json", "w"), indent=1)
say(f"G13: 0.03 AUROC corresponds to about {100 * 0.03 * slope:.1f} pp UFR at matched 5% FAR; median seed SD of AUROC {sd_seed:.3f}")

# ----------------------------------------------------------------------------- G14 KNN bank-size sensitivity (deep ensemble, AGEF-Net; first seed)
krow = []
for ds, R in RUNS.items():
    for s in R["splits"]:
        t = R["seeds"][0]; outs = {m: load_npz(f"{R['dir']}/split{s}/seed{t}/{m}.npz")[0] for m in ["agef_sharp"] + SINGLES}
        cf = {"deep_ensemble": ens([outs[x] for x in SINGLES]), "agef_sharp": outs["agef_sharp"]}
        for m, o in cf.items():
            for cap in (1000, 3000, 6000):
                kn = ax.KNNScorer(o["train"]["feats"], 50, cap, s)
                krow.append(dict(dataset=ds, split=s, model=m, bank=cap, auroc=auroc(kn(o["test"]["feats"]), kn(o["unk"]["feats"]))))
KC = pd.DataFrame(krow).groupby(["dataset", "model", "bank"]).auroc.mean().unstack("bank"); KC.to_csv(f"{OUT}/G14_knn_bank_size.csv")

# ----------------------------------------------------------------------------- G15 tuned-AGEF learning-rate sweep
lrrow = []
for ds, R in RUNS.items():
    f = f"{R['dir']}/agef_tuned_lr.json"
    if os.path.exists(f):
        j = json.load(open(f))
        for r in j["sweep"]: lrrow.append(dict(dataset=ds, lr=r["lr"], val_acc=r["val_acc"], val_nll=r["val_nll"], chosen=r["lr"] == j["chosen"]))
LRS = pd.DataFrame(lrrow); LRS.to_csv(f"{OUT}/G15_lr_sweep.csv", index=False)
FG = pd.DataFrame(AUX.get("fgate", [])); FG.to_csv(f"{OUT}/G3b_frozen_gate_weights.csv", index=False)

# ----------------------------------------------------------------------------- G16 near-duplicate rule validity (needs the image datasets)
DATA = {"Malimg": "Malimg_Dataset_25classes", "MaleVis": "MaleVis", "Dumpware10": "dumpware10_images"}
def merge_dumpware(raw, merged="/kaggle/working/dumpware10_merged"):
    root = next((c for c in [os.path.join(raw, "224", "224"), os.path.join(raw, "224"), raw]
                 if os.path.isdir(c) and {"TRAINING", "TEST"} <= set(os.listdir(c))), None)
    if root is None: return raw
    for split in ["TRAINING", "TEST"]:
        for cls in sorted(os.listdir(os.path.join(root, split))):
            sd = os.path.join(root, split, cls)
            if not os.path.isdir(sd): continue
            dd = os.path.join(merged, cls); os.makedirs(dd, exist_ok=True)
            for fn in os.listdir(sd):
                dst = os.path.join(dd, f"{split.lower()}__{fn}")
                if not os.path.lexists(dst): os.symlink(os.path.join(sd, fn), dst)
    return merged
vrow, grid = [], []
try:
    from PIL import Image
    from torchvision import datasets as tvd
    for ds, R in RUNS.items():
        hits = [h for h in glob.glob(f"{ROOT}/**/{DATA[ds]}", recursive=True) if os.path.isdir(h)]
        if not hits: say(f"G16 {ds}: dataset not attached - skipped"); continue
        path = merge_dumpware(hits[0]) if ds == "Dumpware10" else hits[0]
        base = tvd.ImageFolder(path); assert base.classes == R["classes"], f"{ds}: class order differs"
        keys = json.load(open(f"{R['dir']}/image_keys.json"))["keys"]; hb = np.stack([np.frombuffer(bytes.fromhex(k[2]), dtype=np.uint8) for k in keys])
        tg = R["targets"]; rng = np.random.RandomState(0)
        def img(i):
            with Image.open(base.samples[i][0]) as im: return np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float64)
        def ncc(i, j):
            a, b = img(i), img(j); a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9); return float((a * b).mean())
        pairs = {"0 bits": [], "1-8 bits": [], "9-16 bits": [], "random same family": []}
        for c in np.unique(tg):
            idx = np.where(tg == c)[0]
            if len(idx) < 2: continue
            sub = rng.choice(idx, min(len(idx), 300), replace=False)
            d = POP[hb[sub][:, None] ^ hb[sub][None]].sum(-1)
            for lo, hi, nm in [(0, 0, "0 bits"), (1, 8, "1-8 bits"), (9, 16, "9-16 bits")]:
                ii, jj = np.where((d >= lo) & (d <= hi) & (np.triu(np.ones_like(d), 1) > 0))
                pairs[nm] += [(sub[a_], sub[b_]) for a_, b_ in zip(ii, jj)]
            pairs["random same family"] += [tuple(rng.choice(idx, 2, replace=False)) for _ in range(5)]
        for nm, pl in pairs.items():
            if not pl: continue
            pick = [pl[k] for k in rng.choice(len(pl), min(len(pl), 150), replace=False)]
            v = np.array([ncc(i, j) for i, j in pick])
            vrow.append(dict(dataset=ds, pair_type=nm, n_pairs_available=len(pl), n_checked=len(pick), ncc_median=np.median(v),
                             share_ncc_ge_0p9=float((v >= 0.9).mean()), share_ncc_ge_0p8=float((v >= 0.8).mean())))
            if nm == "1-8 bits": grid += [(ds, base.samples[i][0], base.samples[j][0]) for i, j in pick[:8]]
        say(f"G16 {ds} done")
    if grid:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(grid), 2, figsize=(4, 2 * len(grid)))
        for r_, (ds, p1, p2) in enumerate(grid):
            for c_, pp in enumerate((p1, p2)):
                with Image.open(pp) as im: axs[r_, c_].imshow(im.convert("L"), cmap="gray")
                axs[r_, c_].axis("off"); axs[r_, c_].set_title(ds if c_ == 0 else "", fontsize=6)
        plt.tight_layout(); plt.savefig(f"{OUT}/G16_pairs_1to8bits.png", dpi=110); plt.close()
except Exception as e:
    say("G16 failed:", repr(e))
VAL = pd.DataFrame(vrow); VAL.to_csv(f"{OUT}/G16_near_duplicate_validity.csv", index=False)

# ----------------------------------------------------------------------------- summary
say("=" * 100)
P = CMP[CMP.dataset == "pooled"]
for r in P.itertuples():
    say(f"{r.group} {r.config} vs {r.reference} ({r.score}): {r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}] 90% [{r.ci90_lo:+.3f}, {r.ci90_hi:+.3f}]"
        f" TOST {'equivalent' if r.tost_equivalent else 'not shown'} | random effects {r.random_effects:+.3f} [{r.re_ci_lo:+.3f}, {r.re_ci_hi:+.3f}] | MDE80 {r.mde80:.3f}")
say("Gate collapse (runs with one expert weight > 0.9): " + GT.groupby(["dataset", "model"]).collapsed_gate.sum().to_string())
say("Nested shortcuts:\n" + NS.round(3).to_string(index=False))
say("Family-macro C1-C6 (pooled):\n" + FM[FM.dataset == "pooled"].round(3).to_string(index=False))
say("Family jackknife:\n" + JK.round(3).to_string(index=False))
say("KNN bank size:\n" + KC.round(3).to_string())
if len(LRS): say("LR sweep:\n" + LRS.round(3).to_string(index=False))
if len(VAL): say("Near-duplicate validity:\n" + VAL.round(3).to_string(index=False))
if len(G10): say("Original MaleVis without VBA (energy):\n" + G10.round(3).to_string(index=False))
say("=" * 100)
open(f"{OUT}/G_SUMMARY.txt", "w").write("\n".join(LOG))
print("outputs:", sorted(os.listdir(OUT)))
