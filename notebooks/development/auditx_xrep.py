# =============================================================================
# AUDIT-X v2 - cross-representation zero-day experiment (notebook 08)
# Train on one image representation (MaleVis: raw-binary images), detect held-out
# families in another (Dumpware10: memory-dump images) using the families both share.
# Reuses models, training, scores and metrics from auditx_eval_v2.py unchanged.
# =============================================================================
import argparse, copy, json, os, time, zlib
import numpy as np, pandas as pd, torch
from scipy.special import softmax
from torchvision import datasets
import auditx_eval_v2 as ax

def norm(n): return n.lower()

def load_parts(npz):
    d = np.load(npz, allow_pickle=True)
    parts = sorted({k.split("__")[0] for k in d.files if "__" in k})
    return {p: {k: d[f"{p}__{k}"] for k in ("logits", "expert_logits", "weights", "feats", "y")} for p in parts}

def deep_ensemble(singles):
    de = {}
    for p in singles[0]:
        de[p] = dict(expert_logits=np.concatenate([s[p]["expert_logits"] for s in singles], 1),
                     feats=np.concatenate([s[p]["feats"] for s in singles], 1), y=singles[0][p]["y"])
        de[p]["logits"] = de[p]["expert_logits"].mean(1)
        de[p]["weights"] = np.full((len(de[p]["y"]), 3), 1 / 3)
    return de

def make_xsplit(src, tgt, seed, n_unknown, never):
    s_idx = {norm(c): i for i, c in enumerate(src.classes)}
    t_idx = {norm(c): i for i, c in enumerate(tgt.classes)}
    shared = sorted(set(s_idx) & set(t_idx))
    benign = [n for n in shared if n in never]
    mal = [n for n in shared if n not in never]
    rng = np.random.RandomState(seed)
    unk = sorted(rng.choice(mal, n_unknown, replace=False).tolist())
    known = [n for n in shared if n not in unk]
    st, tt = np.array(src.targets), np.array(tgt.targets)
    s_known = np.where(np.isin(st, [s_idx[n] for n in known]))[0]
    s_unk = np.where(np.isin(st, [s_idx[n] for n in unk]))[0]
    t_known = np.where(np.isin(tt, [t_idx[n] for n in known]))[0]
    t_unk = np.where(np.isin(tt, [t_idx[n] for n in unk]))[0]
    tr, rest = ax._strat_split(s_known, st[s_known], 0.30, seed)
    va, rest2 = ax._strat_split(rest, st[rest], 2 / 3, seed)
    ca, te = ax._strat_split(rest2, st[rest2], 0.5, seed)
    t_cal, t_test = ax._strat_split(t_known, tt[t_known], 0.5, seed)
    return dict(shared=shared, benign=benign, known=known, unknown=unk,
                remap_s={s_idx[n]: k for k, n in enumerate(known)},
                remap_t={t_idx[n]: k for k, n in enumerate(known)},
                s_train=tr, s_val=va, s_calib=ca, s_test=te, s_unk=s_unk,
                t_cal=t_cal, t_test=t_test, t_unk=t_unk)

def run_direction(direction, src_path, tgt_path, cfg, models, seeds, out_dir, t_start, max_hours):
    src = datasets.ImageFolder(src_path, transform=ax.make_transform(cfg["IMG"]))
    tgt = datasets.ImageFolder(tgt_path, transform=ax.make_transform(cfg["IMG"]))
    never = {n.lower() for n in cfg["NEVER_UNKNOWN"]}
    rows = []
    for seed in seeds:
        if max_hours and (time.time() - t_start) / 3600 > max_hours:
            print(f"!! time budget reached; stopping before {direction} seed {seed}"); break
        sp = make_xsplit(src, tgt, seed, cfg["N_UNKNOWN"], never)
        d = os.path.join(out_dir, direction.replace("->", "_to_"), f"seed{seed}"); os.makedirs(d, exist_ok=True)
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sp.items() if not k.startswith("remap")},
                  open(os.path.join(d, "split.json"), "w"), indent=1)
        print(f"\n[{direction} seed {seed}] shared={sp['shared']}\n   unknown={sp['unknown']}  "
              f"src train/val/calib/test/unk={len(sp['s_train'])}/{len(sp['s_val'])}/{len(sp['s_calib'])}/{len(sp['s_test'])}/{len(sp['s_unk'])}  "
              f"tgt cal/test/unk={len(sp['t_cal'])}/{len(sp['t_test'])}/{len(sp['t_unk'])}", flush=True)
        L = lambda base, idx, remap, sh=False: ax.loader(base, idx, remap, cfg, shuffle=sh)
        tr_ld, va_ld = L(src, sp["s_train"], sp["remap_s"], True), L(src, sp["s_val"], sp["remap_s"])
        eval_lds = {"s_train": L(src, sp["s_train"], sp["remap_s"]), "s_calib": L(src, sp["s_calib"], sp["remap_s"]),
                    "s_test": L(src, sp["s_test"], sp["remap_s"]), "s_unk": L(src, sp["s_unk"], sp["remap_s"]),
                    "t_cal": L(tgt, sp["t_cal"], sp["remap_t"]), "t_test": L(tgt, sp["t_test"], sp["remap_t"]),
                    "t_unk": L(tgt, sp["t_unk"], sp["remap_t"])}
        outs = {}
        for m in models:
            npz = os.path.join(d, f"{m}.npz")
            if os.path.exists(npz):
                outs[m] = load_parts(npz); print(f"   [{m}] cached"); continue
            experts, fusion, temp, lam = ax.MODEL_SPECS[m]
            ax.seed_everything(seed * 1000 + zlib.crc32(m.encode()) % 997)
            model = ax.MultiExpert(experts, len(sp["known"]), fusion, temp, cfg["PRETRAINED"]).to(ax.DEVICE)
            model, best, _ = ax.train(model, tr_ld, va_ld, cfg, lam, log_prefix=f"   [{m}]")
            o = {p: ax.collect(model, ld) for p, ld in eval_lds.items()}
            np.savez_compressed(npz, **{f"{p}__{k}": v for p, dd in o.items() for k, v in dd.items()})
            outs[m] = o; del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        singles = [s for s in ("single_mobilenet", "single_resnet18", "single_effnet_b0") if s in outs]
        if len(singles) == 3:
            outs["deep_ensemble"] = deep_ensemble([outs[s] for s in singles])
        rows += score_seed(direction, seed, sp, outs, cfg)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"xrep_raw_{direction.replace('->', '_to_')}.csv"), index=False)
        print(f"   elapsed {(time.time() - t_start) / 3600:.2f} h", flush=True)
    return rows

def score_seed(direction, seed, sp, outs, cfg):
    rows, far_t = [], cfg["TARGET_FAR"]
    rng = np.random.RandomState(seed)
    for m, o in outs.items():
        is_de = m == "deep_ensemble"
        sc = {p: ax.base_scores(o[p], is_de) for p in o if p != "s_train"}
        knn = ax.KNNScorer(o["s_train"]["feats"], cfg["KNN_K"], cfg["KNN_MAX_TRAIN"], seed)
        for p in sc: sc[p]["knn"] = knn(o[p]["feats"])
        def probs(p):
            return softmax(o[p]["expert_logits"], -1).mean(1) if is_de else softmax(o[p]["logits"], -1)
        acc_in = float((probs("s_test").argmax(1) == o["s_test"]["y"]).mean())
        acc_cross = float((probs("t_test").argmax(1) == o["t_test"]["y"]).mean())
        for s in ("msp", "entropy", "energy", "knn"):
            if s not in sc["s_test"]: continue
            tf_in = ax.threshold_free(sc["s_test"][s], sc["s_unk"][s])
            tf_x = ax.threshold_free(sc["t_test"][s], sc["t_unk"][s])
            thr = np.quantile(sc["s_calib"][s], 1 - far_t)                       # source-calibrated
            r = dict(direction=direction, seed=seed, model=m, score=s, unknown="+".join(sp["unknown"]),
                     acc_in=acc_in, acc_cross=acc_cross,
                     auroc_in=tf_in["auroc"], auroc_cross=tf_x["auroc"], fpr95_in=tf_in["fpr95"], fpr95_cross=tf_x["fpr95"],
                     zdr_in=float((sc["s_unk"][s] > thr).mean()), far_in=float((sc["s_test"][s] > thr).mean()),
                     zdr_cross_srccal=float((sc["t_unk"][s] > thr).mean()),
                     far_cross_srccal=float((sc["t_test"][s] > thr).mean()))
            # recalibration with n unlabeled known-family target samples (averaged over repeats)
            pool = sc["t_cal"][s]
            for n in cfg["RECAL_N"] + ["all"]:
                reps = 1 if n == "all" or n >= len(pool) else cfg["RECAL_REPEATS"]
                fz, ff = [], []
                for _ in range(reps):
                    sub = pool if n == "all" or n >= len(pool) else pool[rng.choice(len(pool), n, replace=False)]
                    th = np.quantile(sub, 1 - far_t)
                    fz.append((sc["t_unk"][s] > th).mean()); ff.append((sc["t_test"][s] > th).mean())
                r[f"zdr_recal_{n}"], r[f"far_recal_{n}"] = float(np.mean(fz)), float(np.mean(ff))
                r[f"far_recal_{n}_sd"] = float(np.std(ff))
            rows.append(r)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malevis", required=True); ap.add_argument("--dumpware", required=True)
    ap.add_argument("--directions", nargs="*", default=["MaleVis->Dumpware10", "Dumpware10->MaleVis"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="*", default=["agef_sharp", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"])
    ap.add_argument("--epochs", type=int); ap.add_argument("--img", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/auditx_v2_xrep"); ap.add_argument("--max-hours", type=float)
    a = ap.parse_args()
    cfg = copy.deepcopy(ax.CFG)
    cfg.update(N_UNKNOWN=2, RECAL_N=[10, 25, 50, 100], RECAL_REPEATS=50)
    if a.epochs: cfg["EPOCHS"] = a.epochs
    if a.img: cfg["IMG"] = a.img
    if a.batch: cfg["BATCH"] = a.batch
    if a.workers is not None: cfg["WORKERS"] = a.workers
    if a.no_pretrained: cfg["PRETRAINED"] = False
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(cfg={k: v for k, v in cfg.items() if k != "DATASETS"}, args=vars(a)), open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
    paths = {"MaleVis": a.malevis, "Dumpware10": a.dumpware}
    t0, allrows = time.time(), []
    for dr in a.directions:
        s, t = dr.split("->")
        allrows += run_direction(dr, paths[s], paths[t], cfg, a.models, a.seeds, a.out, t0, a.max_hours)
    pd.DataFrame(allrows).to_csv(os.path.join(a.out, "xrep_raw.csv"), index=False)
    print(f"\nDone in {(time.time() - t0) / 3600:.2f} h -> {a.out}")

if __name__ == "__main__":
    main()
