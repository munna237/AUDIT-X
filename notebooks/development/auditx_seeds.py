# =============================================================================
# AUDIT-X v2 - training-seed replicates (notebook 13)
# Same unknown-class splits as notebooks 01-03; each split is retrained with additional
# training seeds. Also stores raw penultimate features, head weights and ReAct-clipped
# logits so ViM and ReAct can be scored, and (optionally) measures real GPU latency.
# =============================================================================
import argparse, copy, json, os, shutil, time, zlib
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.special import softmax, logsumexp
from torchvision import datasets
import auditx_eval_v2 as ax

@torch.no_grad()
def collect_extra(model, ld, clips=None, keep_raw=True):
    model.eval(); acc = {k: [] for k in ("logits", "expert_logits", "weights", "feats", "y", "react_logits", "react_expert_logits")}
    raws = [[] for _ in model.experts]
    for xb, yb in ld:
        xb = xb.to(ax.DEVICE); fs = [e.backbone(xb).float() for e in model.experts]
        def fuse(feats):
            st = torch.stack([e.head(f) for e, f in zip(model.experts, feats)], 1)
            w = F.softmax(model.gate(st.flatten(1)) / model.temp, 1) if model.fusion == "gate" else \
                torch.full((xb.size(0), model.E), 1.0 / model.E, device=xb.device)
            return (w.unsqueeze(-1) * st).sum(1), st, w
        fused, st, w = fuse(fs)
        acc["logits"].append(fused.cpu().numpy()); acc["expert_logits"].append(st.cpu().numpy()); acc["weights"].append(w.cpu().numpy())
        acc["feats"].append(torch.cat([F.normalize(f, dim=1) for f in fs], 1).cpu().numpy()); acc["y"].append(yb.numpy())
        if clips is not None:
            rf, rst, _ = fuse([f.clamp(max=c) for f, c in zip(fs, clips)])
            acc["react_logits"].append(rf.cpu().numpy()); acc["react_expert_logits"].append(rst.cpu().numpy())
        if keep_raw:
            for i, f in enumerate(fs): raws[i].append(f.cpu().numpy().astype(np.float16))
    out = {k: np.concatenate(v).astype(np.float64) if k != "y" else np.concatenate(v) for k, v in acc.items() if v}
    if keep_raw:
        for i, r in enumerate(raws): out[f"raw{i}"] = np.concatenate(r)
    return out

def vim_scores(raw_train, W, b, raw_parts, logit_parts):
    """ViM (Wang et al., 2022) for one expert. Returns dict part -> OOD score (higher = more unknown)."""
    D = W.shape[1]; dim = 1000 if D >= 2048 else (512 if D >= 768 else D // 2)
    u = -np.linalg.pinv(W) @ b
    X = raw_train.astype(np.float64) - u
    ev, V = np.linalg.eigh(X.T @ X / len(X)); NS = V[:, :D - dim]          # smallest eigenvalues -> residual space
    vl_tr = np.linalg.norm(X @ NS, axis=1)
    alpha = (raw_train.astype(np.float64) @ W.T + b).max(1).mean() / vl_tr.mean()
    return {p: alpha * np.linalg.norm((raw_parts[p].astype(np.float64) - u) @ NS, axis=1) - logsumexp(logit_parts[p], axis=1)
            for p in raw_parts}

def score_run(o, is_de, seed, cfg, heads=None, member_raw=None, member_heads=None):
    sc = {p: ax.base_scores(o[p], is_de) for p in ("calib", "test", "unk")}
    knn = ax.KNNScorer(o["train"]["feats"], cfg["KNN_K"], cfg["KNN_MAX_TRAIN"], seed)
    for p in sc: sc[p]["knn"] = knn(o[p]["feats"])
    if "react_logits" in o["calib"]:
        for p in sc:
            rl = o[p]["react_expert_logits"].mean(1) if is_de else o[p]["react_logits"]
            sc[p]["react_energy"] = -logsumexp(rl, axis=1)
    if member_raw is not None:      # single CNN (1 member) or deep ensemble (3 members): mean ViM over members
        v = None
        for raw, (W, b), j in zip(member_raw, member_heads, range(len(member_raw))):
            lp = {p: o[p]["expert_logits"][:, j] for p in ("calib", "test", "unk")}
            s_ = vim_scores(raw["train"], W, b, {p: raw[p] for p in ("calib", "test", "unk")}, lp)
            v = s_ if v is None else {p: v[p] + s_[p] for p in v}
        for p in sc: sc[p]["vim"] = v[p] / len(member_raw)
    rows = []
    for s in sc["test"]:
        tf = ax.threshold_free(sc["test"][s], sc["unk"][s])
        thr = np.quantile(sc["calib"][s], 1 - cfg["TARGET_FAR"])
        rows.append(dict(score=s, auroc=tf["auroc"], fpr95=tf["fpr95"], zdr=float((sc["unk"][s] > thr).mean()),
                         far_test=float((sc["test"][s] > thr).mean())))
    return rows

@torch.no_grad()
def latency(models, calib_ld, out_json, reps=30):
    """Wall-clock ms per image on the current GPU for single CNNs, deep ensemble, AGEF-Net and its real early exit."""
    xb_real = next(iter(calib_ld))[0].to(ax.DEVICE)
    res = {}
    def timeit(fn, x):
        for _ in range(2): fn(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps): fn(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return (time.time() - t0) / reps / x.size(0) * 1e3
    ag = models["agef_sharp"]; ag.eval()
    pm = torch.cat([F.softmax(ag.experts[0](x[None])[1], 1).max(1).values for x in xb_real]).cpu().numpy()
    theta = float(np.quantile(pm, 0.5))                          # 50 % exit on this calibration batch
    def early_exit(x):
        """MobileNet expert first; the other experts and the gate run only on samples that do not exit."""
        _, l0 = ag.experts[0](x)
        keep = F.softmax(l0, 1).max(1).values < theta
        if keep.any():
            xk = x[keep]
            st = torch.stack([l0[keep]] + [e(xk)[1] for e in ag.experts[1:]], 1)
            w = F.softmax(ag.gate(st.flatten(1)) / ag.temp, 1)
            (w.unsqueeze(-1) * st).sum(1)
    def per_sample(fn):
        """batch size 1: run every calibration image one at a time, so the early exit sees its real exit rate."""
        return lambda X: [fn(X[i:i + 1]) for i in range(X.size(0))]
    for bs in (1, 32):
        x = xb_real if bs == 1 else (xb_real[:bs] if xb_real.size(0) >= bs else xb_real.repeat(bs // xb_real.size(0) + 1, 1, 1, 1)[:bs])
        wrap = per_sample if bs == 1 else (lambda f: f)
        r = {}
        for m in ("single_mobilenet", "single_resnet18", "single_effnet_b0", "agef_sharp", "avg_joint"):
            if m in models: models[m].eval(); r[m] = timeit(wrap(lambda z, mm=models[m]: mm(z)), x)
        if all(s in models for s in ("single_mobilenet", "single_resnet18", "single_effnet_b0")):
            r["deep_ensemble"] = timeit(wrap(lambda z: [models[s](z) for s in ("single_mobilenet", "single_resnet18", "single_effnet_b0")]), x)
        r["agef_early_exit_50pct"] = timeit(wrap(early_exit), x)
        res[f"batch{bs}"] = r
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    json.dump(res, open(out_json, "w"), indent=1); print("latency (ms/image):", json.dumps(res, indent=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--path", required=True)
    ap.add_argument("--split-seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--train-seeds", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--models", nargs="*", default=["agef_sharp", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"])
    ap.add_argument("--epochs", type=int); ap.add_argument("--img", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--latency", action="store_true"); ap.add_argument("--resume-from")
    ap.add_argument("--out", required=True); ap.add_argument("--max-hours", type=float)
    a = ap.parse_args()
    cfg = copy.deepcopy(ax.CFG)
    for k, v in dict(EPOCHS=a.epochs, IMG=a.img, BATCH=a.batch).items():
        if v: cfg[k] = v
    if a.workers is not None: cfg["WORKERS"] = a.workers
    if a.no_pretrained: cfg["PRETRAINED"] = False
    os.makedirs(a.out, exist_ok=True)
    if a.resume_from and os.path.isdir(a.resume_from):
        for dp, _, fns in os.walk(a.resume_from):
            for fn in fns:
                if fn.endswith(".npz") or fn == "latency.json":
                    dst = os.path.join(a.out, os.path.relpath(os.path.join(dp, fn), a.resume_from))
                    if not os.path.exists(dst): os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(os.path.join(dp, fn), dst)
    base = datasets.ImageFolder(a.path, transform=ax.make_transform(cfg["IMG"]))
    near = a.dataset in cfg.get("NEAR_DUP_DATASETS", ())
    G = ax.compute_groups(base, os.path.join(a.out, a.dataset, "image_keys.json"), cfg["GROUP_KEYS"], cfg["UNSPLITTABLE_SHARE"],
                          cfg["MIN_CLASS_IMAGES"], cfg["NEAR_DUP_BITS"] if near else None, cfg["NEAR_DUP_CAP"]) if cfg.get("GROUPING", False) else None
    if G is not None: print("duplicate groups:", G["stats"])
    rows, t0 = [], time.time()
    singles = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
    for s in a.split_seeds:
        sp = ax.make_split(base, s, cfg, G)
        lds = {k: ax.loader(base, sp[k], sp["remap"], cfg, shuffle=(k == "train")) for k in ("train", "val")}
        rng = np.random.RandomState(s); tr_sub = rng.choice(sp["train"], min(len(sp["train"]), 6000), replace=False)
        ev = {"train": ax.loader(base, tr_sub, sp["remap"], cfg, False),
              **{k: ax.loader(base, sp[k], sp["remap"], cfg, False) for k in ("calib", "test", "unk")}}
        K = len(sp["known"])
        for t in a.train_seeds:
            if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
                print("!! time budget reached; re-run with --resume-from"); break
            d = os.path.join(a.out, a.dataset, f"seed{s}", f"train{t}"); os.makedirs(d, exist_ok=True)
            outs, heads, trained = {}, {}, {}
            for m in a.models:
                npz = os.path.join(d, f"{m}.npz")
                if os.path.exists(npz):
                    z = np.load(npz, allow_pickle=True)
                    outs[m] = {p: {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(p + "__")} for p in ev}
                    heads[m] = [(z[f"W{i}"], z[f"b{i}"]) for i in range(len(ax.MODEL_SPECS[m][0]))]
                    print(f"   [{a.dataset} s{s} t{t} {m}] cached"); continue
                experts, fusion, temp, lam = ax.MODEL_SPECS[m]
                ax.seed_everything(s * 1000 + zlib.crc32(m.encode()) % 997 + 7919 * t)
                model = ax.MultiExpert(experts, K, fusion, temp, cfg["PRETRAINED"]).to(ax.DEVICE)
                model, _, _ = ax.train(model, lds["train"], lds["val"], cfg, lam, log_prefix=f"   [{a.dataset} s{s} t{t} {m}]")
                tr = collect_extra(model, ev["train"])
                clips = [float(np.percentile(tr[f"raw{i}"].astype(np.float32), 90)) for i in range(model.E)]
                o = {"train": collect_extra(model, ev["train"], clips)}
                for p in ("calib", "test", "unk"): o[p] = collect_extra(model, ev[p], clips)
                hw = [(e.head.weight.detach().cpu().numpy().astype(np.float64), e.head.bias.detach().cpu().numpy().astype(np.float64)) for e in model.experts]
                np.savez_compressed(npz, **{f"{p}__{k}": v for p, dd in o.items() for k, v in dd.items()},
                                    **{f"W{i}": w for i, (w, _) in enumerate(hw)}, **{f"b{i}": b for i, (_, b) in enumerate(hw)},
                                    clips=np.array(clips))
                outs[m], heads[m] = o, hw
                if a.latency and s == a.split_seeds[0] and t == a.train_seeds[0]: trained[m] = model
                else: del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            if trained:
                latency(trained, ev["calib"], os.path.join(a.out, "latency.json")); trained.clear()
            # ---- score this (split, train seed)
            for m, o in outs.items():
                mr = [{p: o[p]["raw0"] for p in ev}] if m in singles else None
                for r in score_run(o, False, s, cfg, member_raw=mr, member_heads=heads[m][:1] if mr else None):
                    rows.append(dict(dataset=a.dataset, split=s, train_seed=t, model=m, **r))
            if all(x in outs for x in singles):
                de = {}
                for p in ev:
                    de[p] = dict(expert_logits=np.concatenate([outs[x][p]["expert_logits"] for x in singles], 1),
                                 feats=np.concatenate([outs[x][p]["feats"] for x in singles], 1), y=outs[singles[0]][p]["y"],
                                 react_expert_logits=np.concatenate([outs[x][p]["react_expert_logits"] for x in singles], 1))
                    de[p]["logits"] = de[p]["expert_logits"].mean(1); de[p]["weights"] = np.full((len(de[p]["y"]), 3), 1 / 3)
                    de[p]["react_logits"] = de[p]["react_expert_logits"].mean(1)
                mr = [{p: outs[x][p]["raw0"] for p in ev} for x in singles]
                for r in score_run(de, True, s, cfg, member_raw=mr, member_heads=[heads[x][0] for x in singles]):
                    rows.append(dict(dataset=a.dataset, split=s, train_seed=t, model="deep_ensemble", **r))
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"seeds_raw_{a.dataset}.csv"), index=False)
            print(f"   split {s} train seed {t} done, elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
    print("Done.")

if __name__ == "__main__":
    main()
