# =============================================================================
# AUDIT-X v2 - few-shot recovery baselines (notebook 14)
# Same splits and target pools as notebook 10. For single CNNs (deep ensemble built from them):
#   source        : unadapted source model (k = 0)
#   adabn0        : source model with BatchNorm statistics re-estimated on unlabelled target images (k = 0)
#   finetune      : fine-tuning with k labelled target images per family + source replay (as notebook 10)
#   adabn_finetune: AdaBN, then the same fine-tuning
#   shots_only    : ImageNet-pretrained model trained only on the k labelled target images (no source data)
# =============================================================================
import argparse, copy, json, os, time, zlib
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import auditx_eval_v2 as ax
from auditx_xrep import make_xsplit, deep_ensemble
from auditx_fewshot import finetune, evaluate, infinite

@torch.no_grad()
def adabn(model, loader):
    """Re-estimate BatchNorm running statistics on (unlabelled) target images."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns: m.reset_running_stats(); m.momentum = None
    model.train()
    for xb, _ in loader: model(xb.to(ax.DEVICE))
    for m in bns: m.momentum = 0.1
    model.eval(); return model

def shots_only(model, shot_ld, steps, lr, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["WD"])
    use_amp = cfg["AMP"] and ax.DEVICE.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    it = infinite(shot_ld); model.train()
    for _ in range(steps):
        xb, yb = next(it); xb, yb = xb.to(ax.DEVICE), yb.to(ax.DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=ax.DEVICE.type, enabled=use_amp):
            loss = F.cross_entropy(model(xb)["logits"], yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--malevis", required=True); ap.add_argument("--dumpware", required=True)
    ap.add_argument("--direction", default="MaleVis->Dumpware10")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--shots", nargs="*", type=int, default=[5, 10, 25])
    ap.add_argument("--methods", nargs="*", default=["finetune", "adabn_finetune", "shots_only"])
    ap.add_argument("--models", nargs="*", default=["single_mobilenet", "single_resnet18", "single_effnet_b0"])
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--shots-lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int); ap.add_argument("--img", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/auditx_v2_fewshot_baselines"); ap.add_argument("--max-hours", type=float)
    a = ap.parse_args()
    cfg = copy.deepcopy(ax.CFG); cfg.update(N_UNKNOWN=2)
    for k, v in dict(EPOCHS=a.epochs, IMG=a.img, BATCH=a.batch).items():
        if v: cfg[k] = v
    if a.workers is not None: cfg["WORKERS"] = a.workers
    if a.no_pretrained: cfg["PRETRAINED"] = False
    half = max(1, cfg["BATCH"] // 2); os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=1)
    s_name, t_name = a.direction.split("->"); paths = {"MaleVis": a.malevis, "Dumpware10": a.dumpware}
    src = datasets.ImageFolder(paths[s_name], transform=ax.make_transform(cfg["IMG"]))
    tgt = datasets.ImageFolder(paths[t_name], transform=ax.make_transform(cfg["IMG"]))
    tt = np.array(tgt.targets); never = {n.lower() for n in cfg["NEVER_UNKNOWN"]}
    raw_csv = f"{a.out}/fewshot_baselines_raw.csv"
    rows = pd.read_csv(raw_csv).to_dict("records") if os.path.exists(raw_csv) else []
    t0 = time.time()
    L = lambda base, idx, remap, sh=False, bs=None: DataLoader(ax.IndexedSubset(base, idx, remap), batch_size=bs or cfg["BATCH"],
                                                                 shuffle=sh, num_workers=cfg["WORKERS"], pin_memory=torch.cuda.is_available(), drop_last=sh)
    for seed in a.seeds:
        if any(r["seed"] == seed for r in rows): print(f"seed {seed} already done (resume)"); continue
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours: print("!! time budget reached"); break
        sp = make_xsplit(src, tgt, seed, cfg["N_UNKNOWN"], never); K = len(sp["known"])
        rng = np.random.RandomState(1000 + seed)            # identical shot draws to notebook 10
        pool = np.array(sp["t_cal"]); pool_lab = np.array([sp["remap_t"][x] for x in tt[pool]])
        order = {c: rng.permutation(pool[pool_lab == c]) for c in range(K)}
        print(f"\n[{a.direction} seed {seed}] unknown={sp['unknown']}", flush=True)
        src_tr = L(src, sp["s_train"], sp["remap_s"], True); src_tr_half = L(src, sp["s_train"], sp["remap_s"], True, half)
        src_va = L(src, sp["s_val"], sp["remap_s"])
        bank_idx = np.random.RandomState(seed).choice(sp["s_train"], min(len(sp["s_train"]), 3000), replace=False)
        t_test_ld, t_unk_ld, pool_ld = L(tgt, sp["t_test"], sp["remap_t"]), L(tgt, sp["t_unk"], sp["remap_t"]), L(tgt, pool, sp["remap_t"])
        outs = {}
        def record(key, m, model, calib_idx, bank_src=True, shots=None):
            parts = {"calib": ax.collect(model, L(tgt, calib_idx, sp["remap_t"])), "test": ax.collect(model, t_test_ld),
                     "unk": ax.collect(model, t_unk_ld)}
            bank = []
            if bank_src: bank.append(ax.collect(model, L(src, bank_idx, sp["remap_s"]))["feats"])
            if shots is not None and len(shots): bank.append(ax.collect(model, L(tgt, shots, sp["remap_t"]))["feats"])
            parts["bank"] = np.concatenate(bank); outs.setdefault(key, {})[m] = parts
        for m in a.models:
            experts, fusion, temp, lam = ax.MODEL_SPECS[m]
            ax.seed_everything(seed * 1000 + zlib.crc32(m.encode()) % 997)
            base_m = ax.MultiExpert(experts, K, fusion, temp, cfg["PRETRAINED"]).to(ax.DEVICE)
            base_m, _, _ = ax.train(base_m, src_tr, src_va, cfg, lam, log_prefix=f"   [{m} source]")
            st = copy.deepcopy(base_m.state_dict())
            record(("0", "source"), m, base_m, pool)
            mb = ax.MultiExpert(experts, K, fusion, temp, False).to(ax.DEVICE); mb.load_state_dict(st)
            record(("0", "adabn0"), m, adabn(mb, pool_ld), pool); del mb
            for k in a.shots:
                if any(len(order[c]) <= k for c in range(K)): print(f"   k={k} skipped (pool too small)"); continue
                shots = np.concatenate([order[c][:k] for c in range(K)]); calib = np.setdiff1d(pool, shots)
                shot_ld = L(tgt, shots, sp["remap_t"], True, min(half, len(shots)))
                for meth in a.methods:
                    if meth == "shots_only":
                        ax.seed_everything(seed * 1000 + zlib.crc32((m + "shots").encode()) % 997 + k)
                        mm = ax.MultiExpert(experts, K, fusion, temp, cfg["PRETRAINED"]).to(ax.DEVICE)
                        mm = shots_only(mm, L(tgt, shots, sp["remap_t"], True, min(cfg["BATCH"], len(shots))), a.steps, a.shots_lr, cfg)
                        record((str(k), meth), m, mm, calib, bank_src=False, shots=shots)
                    else:
                        mm = ax.MultiExpert(experts, K, fusion, temp, False).to(ax.DEVICE); mm.load_state_dict(st)
                        if meth == "adabn_finetune": mm = adabn(mm, pool_ld)
                        mm = finetune(mm, src_tr_half, shot_ld, a.steps, a.ft_lr, lam, cfg)
                        record((str(k), meth), m, mm, calib, shots=shots)
                    del mm
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            del base_m
        singles = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
        for (k, meth), mo in outs.items():
            if all(s in mo for s in singles):
                de = deep_ensemble([{p: mo[s][p] for p in ("calib", "test", "unk")} for s in singles])
                de["bank"] = np.concatenate([mo[s]["bank"] for s in singles], 1); mo["deep_ensemble"] = de
            for m, o in mo.items():
                for s, r in evaluate(o, m == "deep_ensemble", o["bank"], seed, cfg).items():
                    rows.append(dict(direction=a.direction, seed=seed, model=m, k=k, method=meth, score=s, **r))
        pd.DataFrame(rows).to_csv(raw_csv, index=False)
        print(f"   seed {seed} done, elapsed {(time.time()-t0)/3600:.2f} h", flush=True)
    print("Done ->", raw_csv)

if __name__ == "__main__":
    main()
