# =============================================================================
# AUDIT-X v2 - few-shot recovery after a change of image representation (notebook 10)
# Source models are trained exactly as in notebook 08 (same splits). They are then
# fine-tuned with k labelled target images per known family and re-evaluated on the
# target. A target-trained "oracle" gives the ceiling. Reuses auditx_eval_v2 + auditx_xrep.
# =============================================================================
import argparse, copy, json, os, time, zlib
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader
from torchvision import datasets
import auditx_eval_v2 as ax
from auditx_xrep import make_xsplit, deep_ensemble

def infinite(loader):
    while True:
        for b in loader: yield b

def finetune(model, src_ld, shot_ld, steps, lr, lam, cfg):
    """Fixed number of steps; every batch = half source training images + half target shots."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["WD"])
    use_amp = cfg["AMP"] and ax.DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    si, ti = infinite(src_ld), infinite(shot_ld)
    model.train()
    for _ in range(steps):
        xs, ys = next(si); xt, yt = next(ti)
        xb = torch.cat([xs, xt]).to(ax.DEVICE); yb = torch.cat([ys, yt]).to(ax.DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=ax.DEVICE.type, enabled=use_amp):
            out = model(xb); loss = F.cross_entropy(out["logits"], yb)
            if lam > 0 and model.E > 1 and model.fusion == "gate":
                wbar = out["weights"].float().mean(0)
                loss = loss + lam * (1.0 - (-(wbar * torch.log(wbar + 1e-8)).sum()))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return model

def evaluate(o, is_de, bank_feats, seed, cfg):
    """o: parts calib/test/unk (+ y). Returns metrics for energy, entropy, msp, knn."""
    sc = {p: ax.base_scores(o[p], is_de) for p in ("calib", "test", "unk")}
    knn = ax.KNNScorer(bank_feats, cfg["KNN_K"], cfg["KNN_MAX_TRAIN"], seed)
    for p in sc: sc[p]["knn"] = knn(o[p]["feats"])
    p_test = softmax(o["test"]["expert_logits"], -1).mean(1) if is_de else softmax(o["test"]["logits"], -1)
    acc = float((p_test.argmax(1) == o["test"]["y"]).mean())
    res = {}
    for s in ("msp", "entropy", "energy", "knn"):
        if s not in sc["test"]: continue
        tf = ax.threshold_free(sc["test"][s], sc["unk"][s])
        thr = np.quantile(sc["calib"][s], 1 - cfg["TARGET_FAR"])
        res[s] = dict(acc=acc, auroc=tf["auroc"], fpr95=tf["fpr95"],
                      zdr=float((sc["unk"][s] > thr).mean()), far=float((sc["test"][s] > thr).mean()))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malevis", required=True); ap.add_argument("--dumpware", required=True)
    ap.add_argument("--directions", nargs="*", default=["MaleVis->Dumpware10"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--shots", nargs="*", type=int, default=[1, 5, 10, 25, 50])
    ap.add_argument("--models", nargs="*", default=["agef_sharp", "single_mobilenet", "single_resnet18", "single_effnet_b0"])
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int); ap.add_argument("--img", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/auditx_v2_fewshot"); ap.add_argument("--max-hours", type=float)
    a = ap.parse_args()
    cfg = copy.deepcopy(ax.CFG); cfg.update(N_UNKNOWN=2)
    for k, v in dict(EPOCHS=a.epochs, IMG=a.img, BATCH=a.batch).items():
        if v: cfg[k] = v
    if a.workers is not None: cfg["WORKERS"] = a.workers
    if a.no_pretrained: cfg["PRETRAINED"] = False
    half = max(1, cfg["BATCH"] // 2)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(args=vars(a), cfg={k: v for k, v in cfg.items() if k != "DATASETS"}), open(f"{a.out}/config.json", "w"), indent=1, default=str)
    paths = {"MaleVis": a.malevis, "Dumpware10": a.dumpware}
    never = {n.lower() for n in cfg["NEVER_UNKNOWN"]}
    rows, t0 = [], time.time()
    raw_csv = f"{a.out}/fewshot_raw.csv"
    if os.path.exists(raw_csv): rows = pd.read_csv(raw_csv).to_dict("records")

    for dr in a.directions:
        s_name, t_name = dr.split("->")
        src = datasets.ImageFolder(paths[s_name], transform=ax.make_transform(cfg["IMG"]))
        tgt = datasets.ImageFolder(paths[t_name], transform=ax.make_transform(cfg["IMG"]))
        tt = np.array(tgt.targets)
        for seed in a.seeds:
            if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
                print("!! time budget reached"); break
            n_models = len(a.models) + (0 if a.no_oracle else 0)
            if sum(1 for r in rows if r["direction"] == dr and r["seed"] == seed) > 0:
                print(f"[{dr} seed {seed}] already in {raw_csv}; skipped (resume)"); continue
            sp = make_xsplit(src, tgt, seed, cfg["N_UNKNOWN"], never)
            K = len(sp["known"])
            # few-shot pools are drawn from the target recalibration pool; the rest of it calibrates thresholds
            rng = np.random.RandomState(1000 + seed)
            pool = np.array(sp["t_cal"]); pool_lab = np.array([sp["remap_t"][t] for t in tt[pool]])
            order = {c: rng.permutation(pool[pool_lab == c]) for c in range(K)}
            print(f"\n[{dr} seed {seed}] unknown={sp['unknown']} | target pool per known class: "
                  f"min {min(len(v) for v in order.values())}, max {max(len(v) for v in order.values())}", flush=True)
            L = lambda base, idx, remap, sh=False, bs=None: DataLoader(
                ax.IndexedSubset(base, idx, remap), batch_size=bs or cfg["BATCH"], shuffle=sh,
                num_workers=cfg["WORKERS"], pin_memory=torch.cuda.is_available(), drop_last=sh)
            src_tr = L(src, sp["s_train"], sp["remap_s"], True)
            src_tr_half = L(src, sp["s_train"], sp["remap_s"], True, half)
            src_va = L(src, sp["s_val"], sp["remap_s"])
            src_bank_idx = np.random.RandomState(seed).choice(sp["s_train"], min(len(sp["s_train"]), 3000), replace=False)
            t_test_ld, t_unk_ld = L(tgt, sp["t_test"], sp["remap_t"]), L(tgt, sp["t_unk"], sp["remap_t"])
            outs = {}   # outs[k][model] = parts dict
            for m in a.models:
                experts, fusion, temp, lam = ax.MODEL_SPECS[m]
                ax.seed_everything(seed * 1000 + zlib.crc32(m.encode()) % 997)
                base_model = ax.MultiExpert(experts, K, fusion, temp, cfg["PRETRAINED"]).to(ax.DEVICE)
                base_model, _, _ = ax.train(base_model, src_tr, src_va, cfg, lam, log_prefix=f"   [{m} source]")
                base_state = copy.deepcopy(base_model.state_dict())
                for k in [0] + a.shots:
                    shots = np.concatenate([order[c][:k] for c in range(K)]) if k > 0 else np.array([], int)
                    if any(len(order[c]) <= k for c in range(K)):
                        print(f"   k={k}: some class has <= k pool images; skipped"); continue
                    calib = np.setdiff1d(pool, shots)
                    model = base_model
                    if k > 0:
                        model = ax.MultiExpert(experts, K, fusion, temp, False).to(ax.DEVICE); model.load_state_dict(base_state)
                        model = finetune(model, src_tr_half, L(tgt, shots, sp["remap_t"], True, min(half, len(shots))),
                                         a.steps, a.ft_lr, lam, cfg)
                    parts = {"calib": ax.collect(model, L(tgt, calib, sp["remap_t"])),
                             "test": ax.collect(model, t_test_ld), "unk": ax.collect(model, t_unk_ld)}
                    bank = ax.collect(model, L(src, src_bank_idx, sp["remap_s"]))["feats"]
                    if k > 0:
                        bank = np.concatenate([bank, ax.collect(model, L(tgt, shots, sp["remap_t"]))["feats"]])
                    parts["bank"] = bank
                    outs.setdefault(k, {})[m] = parts
                    if k > 0: del model
                del base_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            # oracle: train on the whole target pool (80 %), calibrate on the remaining 20 %
            if not a.no_oracle:
                tr_o, ca_o = ax._strat_split(pool, tt[pool], 0.2, seed)
                tr_o, va_o = ax._strat_split(tr_o, tt[tr_o], 0.1, seed)
                for m in a.models:
                    experts, fusion, temp, lam = ax.MODEL_SPECS[m]
                    ax.seed_everything(seed * 1000 + zlib.crc32((m + "oracle").encode()) % 997)
                    om = ax.MultiExpert(experts, K, fusion, temp, cfg["PRETRAINED"]).to(ax.DEVICE)
                    om, _, _ = ax.train(om, L(tgt, tr_o, sp["remap_t"], True), L(tgt, va_o, sp["remap_t"]), cfg, lam, log_prefix=f"   [{m} oracle]")
                    outs.setdefault("oracle", {})[m] = {"calib": ax.collect(om, L(tgt, ca_o, sp["remap_t"])),
                                                       "test": ax.collect(om, t_test_ld), "unk": ax.collect(om, t_unk_ld),
                                                       "bank": ax.collect(om, L(tgt, tr_o, sp["remap_t"]))["feats"]}
                    del om
            singles = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
            for k, mo in outs.items():
                if all(s in mo for s in singles):
                    de = deep_ensemble([{p: mo[s][p] for p in ("calib", "test", "unk")} for s in singles])
                    de["bank"] = np.concatenate([mo[s]["bank"] for s in singles], 1)
                    mo["deep_ensemble"] = de
                for m, o in mo.items():
                    res = evaluate(o, m == "deep_ensemble", o["bank"], seed, cfg)
                    for s, r in res.items():
                        rows.append(dict(direction=dr, seed=seed, model=m, k=str(k), score=s,
                                         unknown="+".join(sp["unknown"]), **r))
            pd.DataFrame(rows).to_csv(raw_csv, index=False)
            print(f"   seed {seed} done, elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
    print("Done ->", raw_csv)

if __name__ == "__main__":
    main()
