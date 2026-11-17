# =============================================================================
# AUDIT-X FINAL - fifth-round analyses (CPU only)
#   H1 data-driven duplicate sensitivity: evaluation images whose training near-twin (<= 24 bits on the 256-bit hash, 20
#      nearest candidates) has normalized cross-correlation >= 0.90 on 64x64 grey-level images are removed; C1-C6 recomputed
#   H2 taxonomy-aware hold-out: held-out families with a name-stem sibling among the known families are removed; C1-C6 recomputed
#   H3 group-aware calibration: one score per duplicate group (its maximum) and a conformal quantile over groups
#   H4 extra models trained on the same splits (if attached): snapshot ensembles, augmented single CNNs and their deep ensemble;
#      comparisons with the deep ensemble and the S1 (training-data calibration) false-alarm gap with and without augmentation
# Inputs: auditx_final_malimg, auditx_final_malevis_v5, auditx_final_dumpware10; the three image datasets (H1);
#         optionally auditx_extra_{malimg,malevis,dumpware10} (H4).
# =============================================================================
import glob, json, os, time, zlib, contextlib, io
import numpy as np, pandas as pd
from scipy.special import softmax, logsumexp
from scipy.stats import rankdata
import auditx_eval_v2 as ax

ROOT = os.environ.get("AUDITX_ROOT", "/kaggle/input"); OUT = os.environ.get("AUDITX_OUT", "/kaggle/working/auditx_review5"); os.makedirs(OUT, exist_ok=True)
N_BOOT = int(os.environ.get("AUDITX_NBOOT", 10)); NCC_T = 0.90; CAND_BITS = 24; N_CAND = 20
DATASETS = ["Malimg", "MaleVis", "Dumpware10"]
MODELS = ["agef_sharp", "agef_tuned", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"]
SINGLES = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
AUGS = ["single_mobilenet_aug", "single_resnet18_aug", "single_effnet_b0_aug"]; SNAPS = ["snapshot_mobilenet", "snapshot_resnet18"]
V4 = "grouped-L1-v3+near-L3-8bit-cap250"; V5 = "grouped-L1-v3+identical-hash-whole+near-L3-8bit-cap250"
EXPECTED = {"Malimg": V4, "Dumpware10": V4, "MaleVis": V5}
DATA = {"Malimg": "Malimg_Dataset_25classes", "MaleVis": "MaleVis", "Dumpware10": "dumpware10_images"}
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
def find(name):
    hits = [os.path.dirname(p) for p in glob.glob(f"{ROOT}/**/{name}/MANIFEST.json", recursive=True)]
    return sorted(hits)[0] if hits else None
RUNS = {}
for ds in DATASETS:
    d = find(f"auditx_final_{ds.lower()}" + ("_v5" if ds == "MaleVis" else ""))
    if d is None: say(f"!! {ds}: final run not found"); continue
    man = json.load(open(f"{d}/MANIFEST.json")); assert man["complete"] and man["protocol"] == EXPECTED[ds], f"{ds}: wrong run"
    x = find(f"auditx_extra_{ds.lower()}")
    if x:
        mx = json.load(open(f"{x}/MANIFEST.json")); assert mx["complete"] and mx["base_protocol"] == EXPECTED[ds], f"{ds}: extra run incomplete or on other splits"
    RUNS[ds] = dict(dir=d, extra=x, targets=np.load(f"{d}/targets.npy"), classes=json.load(open(f"{d}/classes.json")),
                    splits=sorted(int(p.split("split")[-1]) for p in glob.glob(f"{d}/split*") if os.path.isdir(p)), seeds=man["args"]["train_seeds"])
    say(f"{ds}: {d} | extra models: {x or 'not attached'}")
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
            if kk in ("logits", "expert_logits", "feats", "y"): parts.setdefault(pp, {})[kk] = z[k].astype(np.float64) if kk != "y" else z[k]
    return parts, json.loads(str(z["meta"]))
def ens(parts_list):
    out = {}
    for p in ("train", "calib", "test", "unk"):
        el = np.concatenate([o[p]["expert_logits"] for o in parts_list], 1)
        out[p] = dict(expert_logits=el, logits=el.mean(1), feats=np.concatenate([o[p]["feats"] for o in parts_list], 1), y=parts_list[0][p]["y"])
    return out
def energy(lg): return -logsumexp(lg, axis=1)
def score(o, seed):
    kn = ax.KNNScorer(o["train"]["feats"], 50, 6000, seed)
    return {p: {"energy": energy(o[p]["logits"]), "knn50": (kn(o[p]["feats"]) if p != "train" else None)} for p in ("train", "calib", "test", "unk")}

# ----------------------------------------------------------------------------- score all runs once
SC, t0 = {}, time.time()
for ds, R in RUNS.items():
    for s in R["splits"]:
        sp = json.load(open(f"{R['dir']}/split{s}/split.json"))
        unk_fam = np.array([R["classes"][t] for t in R["targets"][sp["unk"]]])
        for t in R["seeds"]:
            loaded = {m: load_npz(f"{R['dir']}/split{s}/seed{t}/{m}.npz") for m in MODELS}
            outs = {m: v[0] for m, v in loaded.items()}; outs["deep_ensemble"] = ens([outs[x] for x in SINGLES])
            best = max(SINGLES, key=lambda x: max(h["val_acc"] for h in loaded[x][1]["history"]))
            if R["extra"]:
                for m in AUGS + SNAPS: outs[m] = load_npz(f"{R['extra']}/split{s}/seed{t}/{m}.npz")[0]
                outs["deep_ensemble_aug"] = ens([outs[x] for x in AUGS])
            for m, o in outs.items():
                SC[(ds, s, t, m)] = dict(sc=score(o, s), unk_fam=unk_fam)
            SC[(ds, s, t, "best_single")] = SC[(ds, s, t, best)]
        say(f"{ds} split {s} scored ({(time.time() - t0) / 60:.1f} min)")

# ----------------------------------------------------------------------------- generic evaluation with optional per-split masks
def get(ds, m, s, t, masks):
    d = SC[(ds, s, t, m)]
    if masks is None: return d["sc"], d["unk_fam"]
    k = masks[(ds, s)]; sc = {p: {sn: (v[k[p]] if (p in k and v is not None) else v) for sn, v in d["sc"][p].items()} for p in d["sc"]}
    return sc, d["unk_fam"][k["unk"]]
def boot(ds, a, b, sn, masks, n_boot, seed):
    splits, seeds = RUNS[ds]["splits"], RUNS[ds]["seeds"]
    fams = sorted({f for s in splits for f in get(ds, "agef_sharp", s, seeds[0], masks)[1]})
    def diff(s, t, fsel=None, kb=None):
        (A, uf), (B, _) = get(ds, a, s, t, masks), get(ds, b, s, t, masks)
        idx = np.arange(len(uf)) if fsel is None else np.concatenate([np.where(uf == f)[0] for f in fsel if (uf == f).any()] or [np.array([], int)])
        if len(idx) == 0 or len(A["test"][sn]) == 0: return np.nan
        ka, kb_ = (A["test"][sn], B["test"][sn]) if kb is None else (A["test"][sn][kb], B["test"][sn][kb])
        return auroc(ka, A["unk"][sn][idx]) - auroc(kb_, B["unk"][sn][idx])
    pt = np.nanmean([diff(s, t) for s in splits for t in seeds]); rng = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        fb = rng.choice(fams, len(fams)); vals = []
        for s in splits:
            n_te = len(get(ds, "agef_sharp", s, seeds[0], masks)[0]["test"]["energy"]); kb = rng.randint(0, n_te, n_te)
            vals += [diff(s, t, fb, kb) for t in rng.choice(seeds, len(seeds))]
        vals = [v for v in vals if not np.isnan(v)]
        if vals: bs.append(np.mean(vals))
    return pt, np.array(bs)
CONF = [("C1", "deep_ensemble", "agef_sharp", "energy"), ("C2", "deep_ensemble", "agef_sharp", "knn50"), ("C3", "avg_joint", "agef_sharp", "energy"),
        ("C4", "agef_tuned", "agef_sharp", "energy"), ("C5", "agef_tuned", "deep_ensemble", "energy"), ("C6", "deep_ensemble", "best_single", "energy")]
def run_comparisons(tag, comps, masks):
    rows = []
    for cid, a, b, sn in comps:
        if not all((ds, RUNS[ds]["splits"][0], RUNS[ds]["seeds"][0], m) in SC for ds in RUNS for m in (a, b)): continue
        per = {ds: boot(ds, a, b, sn, masks, N_BOOT, zlib.crc32(f"{tag}{cid}{a}{b}{sn}".encode()) % 1000) for ds in RUNS}
        nmin = min(len(v[1]) for v in per.values()); pb = np.mean([v[1][:nmin] for v in per.values()], 0)
        for ds, (pt, bs) in list(per.items()) + [("pooled", (np.mean([v[0] for v in per.values()]), pb))]:
            rows.append(dict(analysis=tag, id=cid, config=a, reference=b, score=sn, dataset=ds, delta=pt, ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5),
                             ci90_lo=np.percentile(bs, 5), ci90_hi=np.percentile(bs, 95)))
        say(f"{tag} {cid} done")
    return pd.DataFrame(rows)
ALLC = [run_comparisons("primary", CONF, None)]

# ----------------------------------------------------------------------------- H1 data-driven duplicate rule (normalized cross-correlation)
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
M_NCC, mrow = {}, []
try:
    from PIL import Image
    from torchvision import datasets as tvd
    for ds, R in RUNS.items():
        hits = [h for h in glob.glob(f"{ROOT}/**/{DATA[ds]}", recursive=True) if os.path.isdir(h)]
        if not hits: raise FileNotFoundError(f"{ds} dataset not attached")
        base = tvd.ImageFolder(merge_dumpware(hits[0]) if ds == "Dumpware10" else hits[0])
        assert np.array_equal(np.array(base.targets), R["targets"]), f"{ds}: image order differs"
        V = np.zeros((len(base.samples), 4096), np.float32)
        for i, (p, _) in enumerate(base.samples):
            with Image.open(p) as im: a_ = np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32).ravel()
            V[i] = (a_ - a_.mean()) / (a_.std() + 1e-6) / np.sqrt(4096)
        keys = json.load(open(f"{R['dir']}/image_keys.json"))["keys"]; hb = np.stack([np.frombuffer(bytes.fromhex(k[2]), dtype=np.uint8) for k in keys])
        for s in R["splits"]:
            sp = json.load(open(f"{R['dir']}/split{s}/split.json")); tr = np.array(sp["train"]); k = {}
            for p in ("calib", "test", "unk"):
                q = np.array(sp[p]); flag = np.zeros(len(q), bool)
                for i0 in range(0, len(q), 256):
                    qq = q[i0:i0 + 256]; d = POP[hb[qq][:, None] ^ hb[tr][None]].sum(-1)
                    cand = np.argsort(d, 1)[:, :N_CAND]
                    for r_ in range(len(qq)):
                        c = cand[r_][d[r_, cand[r_]] <= CAND_BITS]
                        if len(c): flag[i0 + r_] = (V[tr[c]] @ V[qq[r_]]).max() >= NCC_T
                k[p] = ~flag; mrow.append(dict(dataset=ds, split=s, part=p, n=len(q), removed=int(flag.sum()), share=float(flag.mean())))
            M_NCC[(ds, s)] = k
        say(f"H1 {ds}: twins flagged")
    pd.DataFrame(mrow).to_csv(f"{OUT}/H1_ncc_removed.csv", index=False)
    ALLC.append(run_comparisons("ncc_dedup", CONF, M_NCC))
except Exception as e:
    say("H1 skipped:", repr(e))

# ----------------------------------------------------------------------------- H2 taxonomy-aware hold-out
stem = lambda n: n.split(".")[0].split("!")[0].lower()
M_TAX, trow = {}, []
for ds, R in RUNS.items():
    for s in R["splits"]:
        sp = json.load(open(f"{R['dir']}/split{s}/split.json")); known_stems = {stem(n) for n in sp["known_names"]}
        uf = SC[(ds, s, R["seeds"][0], "agef_sharp")]["unk_fam"]; var = sorted({f for f in set(uf) if stem(f) in known_stems})
        keep_unk = ~np.isin(uf, var)
        M_TAX[(ds, s)] = {"unk": keep_unk}
        trow.append(dict(dataset=ds, split=s, variant_families_removed=",".join(var), unknown_images_removed=int((~keep_unk).sum()), n_unknown=len(uf)))
pd.DataFrame(trow).to_csv(f"{OUT}/H2_taxonomy_removed.csv", index=False)
if any(r["unknown_images_removed"] for r in trow): ALLC.append(run_comparisons("taxonomy_aware", CONF, M_TAX))
else: say("H2: no variant families held out with a known sibling")

# ----------------------------------------------------------------------------- H3 group-aware calibration
class FB:
    def __init__(self, classes, targets): self.classes = classes; self.samples = [(f"x{i}", int(t)) for i, t in enumerate(targets)]; self.targets = list(targets)
grow = []
for ds, R in RUNS.items():
    kt = tuple(ax.CFG["GROUP_KEYS"]) + ((2,) if ds in ax.CFG.get("IDENTICAL_WHOLE_DATASETS", ()) else ())
    with contextlib.redirect_stdout(io.StringIO()):
        G = ax.compute_groups(FB(R["classes"], R["targets"]), f"{R['dir']}/image_keys.json", kt, 0.9, 10, 8, 250)
    for s in R["splits"]:
        sp = json.load(open(f"{R['dir']}/split{s}/split.json")); gc = G["groups"][np.array(sp["calib"])]
        for t in R["seeds"]:
            for m in ["deep_ensemble", "agef_sharp", "single_mobilenet"]:
                sc = SC[(ds, s, t, m)]["sc"]
                for sn in ("energy", "knn50"):
                    c = sc["calib"][sn]; gmax = pd.Series(c).groupby(gc).max().values
                    for a_ in (0.05, 0.01):
                        for rule, th in (("image", cthr(c, a_)), ("group", cthr(gmax, a_))):
                            grow.append(dict(dataset=ds, split=s, seed=t, model=m, score=sn, alpha=a_, rule=rule, n_units=len(c) if rule == "image" else len(gmax),
                                             far=float((sc["test"][sn] > th).mean()), ufr=float((sc["unk"][sn] > th).mean())))
GA = pd.DataFrame(grow); GA.to_csv(f"{OUT}/H3_group_calibration_raw.csv", index=False)
GAS = GA.groupby(["dataset", "model", "score", "alpha", "rule"])[["far", "ufr", "n_units"]].mean().reset_index(); GAS.to_csv(f"{OUT}/H3_group_calibration.csv", index=False)

# ----------------------------------------------------------------------------- H4 extra models
if all(RUNS[ds]["extra"] for ds in RUNS):
    erow = []
    for (ds, s, t, m), d in SC.items():
        if m == "best_single": continue
        sc = d["sc"]
        for sn in ("energy", "knn50"):
            th = cthr(sc["calib"][sn], 0.05)
            r = dict(dataset=ds, split=s, seed=t, model=m, score=sn, auroc=auroc(sc["test"][sn], sc["unk"][sn]),
                     ufr5=float((sc["unk"][sn] > th).mean()), far5=float((sc["test"][sn] > th).mean()))
            if sn == "energy":                                  # S1: threshold from training images
                th1 = cthr(sc["train"]["energy"], 0.05); r.update(s1_far_reported=float((sc["train"]["energy"] > th1).mean()), s1_far_true=float((sc["test"]["energy"] > th1).mean()),
                                                                   s1_ufr=float((sc["unk"]["energy"] > th1).mean()))
            erow.append(r)
    EX = pd.DataFrame(erow); EX.to_csv(f"{OUT}/H4_models_long.csv", index=False)
    EXS = EX.groupby(["dataset", "model", "score"]).mean(numeric_only=True).drop(columns=["split", "seed"]).reset_index(); EXS.to_csv(f"{OUT}/H4_models_summary.csv", index=False)
    COMP4 = [("X1", "snapshot_resnet18", "deep_ensemble", "energy"), ("X2", "snapshot_mobilenet", "deep_ensemble", "energy"), ("X3", "snapshot_resnet18", "agef_sharp", "energy"),
             ("X4", "snapshot_resnet18", "single_resnet18", "energy"), ("X5", "deep_ensemble_aug", "deep_ensemble", "energy"), ("X6", "deep_ensemble_aug", "deep_ensemble", "knn50"),
             ("X7", "single_mobilenet_aug", "single_mobilenet", "energy"), ("X8", "deep_ensemble_aug", "agef_sharp", "energy")]
    ALLC.append(run_comparisons("extra_models", COMP4, None))
else:
    say("H4 skipped: extra runs not attached for every dataset")

ALL = pd.concat(ALLC, ignore_index=True); ALL["tost_equivalent"] = (ALL.ci90_lo > -0.03) & (ALL.ci90_hi < 0.03); ALL.to_csv(f"{OUT}/H_comparisons.csv", index=False)
say("=" * 100)
for tag, g in ALL[ALL.dataset == "pooled"].groupby("analysis", sort=False):
    say(f"[{tag}]")
    for r in g.itertuples():
        say(f"  {r.id} {r.config} vs {r.reference} ({r.score}): {r.delta:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}]" + ("  TOST-equivalent" if r.tost_equivalent else ""))
if mrow: say("H1 removed (share of images): " + pd.DataFrame(mrow).groupby(["dataset", "part"]).share.mean().round(4).to_string())
say("H2 variant families removed:\n" + pd.DataFrame(trow).to_string(index=False))
say("H3 group vs image calibration (deep ensemble, energy):\n" + GAS[(GAS.model == "deep_ensemble") & (GAS.score == "energy")].round(4).to_string(index=False))
if all(RUNS[ds]["extra"] for ds in RUNS):
    say("H4 S1 gap (energy): \n" + EXS[EXS.score == "energy"][["dataset", "model", "auroc", "s1_far_reported", "s1_far_true"]].round(3).to_string(index=False))
say("=" * 100)
open(f"{OUT}/H_SUMMARY.txt", "w").write("\n".join(LOG)); print("outputs:", sorted(os.listdir(OUT)))