# =============================================================================
# AUDIT-X EXTRA - additional models trained on the EXISTING splits of the final runs (exploratory)
#   * single_{mobilenet,resnet18,effnet_b0}_aug : same recipe + per-sample GPU augmentation (random scale 0.85-1.0 and
#     translation up to 5 %) and label smoothing 0.1; epoch chosen on validation accuracy as before
#   * snapshot_{mobilenet,resnet18}             : one 12-epoch run with cosine warm restarts (3 cycles of 4 epochs);
#     the three end-of-cycle snapshots form the ensemble (no epoch selection)
# Splits are read from the attached final run (split{s}/split.json); nothing about the splits is recomputed.
# Output files have the same layout and keys as the final runs, so the analysis code can read them unchanged.
# =============================================================================
import argparse, copy, glob, hashlib, json, os, shutil, time, zlib
import numpy as np, torch, torch.nn.functional as F
from torchvision import datasets
import auditx_eval_v2 as ax
import auditx_final as af

DEV = af.DEV
AUG_MODELS = ["single_mobilenet_aug", "single_resnet18_aug", "single_effnet_b0_aug"]
SNAP_MODELS = ["snapshot_mobilenet", "snapshot_resnet18"]
BASE_OF = {"single_mobilenet_aug": "single_mobilenet", "single_resnet18_aug": "single_resnet18", "single_effnet_b0_aug": "single_effnet_b0",
           "snapshot_mobilenet": "single_mobilenet", "snapshot_resnet18": "single_resnet18"}

def augment(x):
    """per-sample random scale (0.85-1.0) and translation (<= 5 % of the image) with an affine grid, on the GPU"""
    n = x.size(0); s = torch.empty(n, device=x.device).uniform_(0.85, 1.0)
    t = torch.empty(n, 2, device=x.device).uniform_(-0.05, 0.05)
    theta = torch.zeros(n, 2, 3, device=x.device); theta[:, 0, 0] = s; theta[:, 1, 1] = s; theta[:, :, 2] = t
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

def train_aug(model, tr, va, epochs, lr, wd=1e-4, smoothing=0.1, tag=""):
    model.to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = DEV.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, hist = (-1, 1e9), None, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = n = 0
        for xb, yb in tr:
            xb = augment(af.norm_gpu(xb)); yb = yb.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type, enabled=amp):
                loss = F.cross_entropy(model(xb)["logits"], yb, label_smoothing=smoothing)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item() * len(yb); n += len(yb)
        sched.step(); acc, nll = af.evaluate(model, va)
        hist.append(dict(epoch=ep, train_loss=tl / n, val_acc=acc, val_nll=nll, sec=time.time() - t0))
        print(f"  {tag} ep {ep:2d} loss {tl / n:.4f} val acc {acc:.2f}% nll {nll:.4f} {time.time() - t0:.0f}s", flush=True)
        if (acc, -nll) > (best[0], -best[1]): best, best_state = (acc, nll), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); return model, hist

def train_snapshots(model, tr, va, epochs, lr, cycles=3, wd=1e-4, tag=""):
    """cosine annealing with warm restarts; a snapshot (state dict) is kept at the end of every cycle"""
    model.to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    per = epochs // cycles; sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=per)
    amp = DEV.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp); snaps, hist = [], []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = n = 0
        for xb, yb in tr:
            xb = af.norm_gpu(xb); yb = yb.to(DEV); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type, enabled=amp):
                loss = F.cross_entropy(model(xb)["logits"], yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item() * len(yb); n += len(yb)
        sched.step(); acc, nll = af.evaluate(model, va)
        hist.append(dict(epoch=ep, train_loss=tl / n, val_acc=acc, val_nll=nll, sec=time.time() - t0))
        print(f"  {tag} ep {ep:2d} loss {tl / n:.4f} val acc {acc:.2f}% nll {nll:.4f} {time.time() - t0:.0f}s", flush=True)
        if (ep + 1) % per == 0: snaps.append(copy.deepcopy(model.state_dict()))
    return snaps, hist

def stack_parts(parts_list):
    """combine several single-expert collections into one ensemble record (same keys as the final runs)"""
    out = {}
    for p in parts_list[0]:
        el = np.concatenate([o[p]["expert_logits"] for o in parts_list], 1)
        out[p] = dict(expert_logits=el, logits=el.mean(1), weights=np.full((len(el), el.shape[1]), 1.0 / el.shape[1]),
                      feats=np.concatenate([o[p]["feats"] for o in parts_list], 1), y=parts_list[0][p]["y"],
                      react_logits=np.mean([o[p]["react_logits"] for o in parts_list], 0))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--path", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--models", nargs="*", default=AUG_MODELS + SNAP_MODELS)
    ap.add_argument("--splits", nargs="*", type=int); ap.add_argument("--train-seeds", nargs="*", type=int)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--img", type=int, default=224); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--max-hours", type=float, default=10.5)
    a = ap.parse_args(); t_start = time.time(); os.makedirs(a.out, exist_ok=True); pre = not a.no_pretrained
    for prev in glob.glob(f"/kaggle/input/**/{os.path.basename(a.out)}", recursive=True):       # automatic resume
        if os.path.abspath(prev) == os.path.abspath(a.out): continue
        n = 0
        for dp, _, fns in os.walk(prev):
            for fn in fns:
                if fn.endswith((".npz", ".json")):
                    dst = os.path.join(a.out, os.path.relpath(os.path.join(dp, fn), prev))
                    if not os.path.exists(dst): os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(os.path.join(dp, fn), dst); n += 1
        print(f"resumed {n} files from {prev}", flush=True)
    man0 = json.load(open(os.path.join(a.run, "MANIFEST.json"))); assert man0["complete"], "the attached final run is incomplete"
    splits = a.splits if a.splits is not None else man0["args"]["splits"]; seeds = a.train_seeds if a.train_seeds is not None else man0["args"]["train_seeds"]
    base = datasets.ImageFolder(a.path); targets = np.array(base.targets)
    assert np.array_equal(targets, np.load(os.path.join(a.run, "targets.npy"))), "image order differs from the final run - wrong dataset folder?"
    print(f"{a.dataset}: using the splits of {a.run} (protocol {man0['protocol']})", flush=True)
    cache = af.build_cache(base, a.img); stopped = False
    for s in splits:
        sp = json.load(open(os.path.join(a.run, f"split{s}", "split.json"))); K = len(sp["known"]); remap = {c: i for i, c in enumerate(sp["known"])}
        rng = np.random.RandomState(s); tr_sub = np.sort(rng.choice(sp["train"], min(len(sp["train"]), 6000), replace=False))
        L = lambda idx, sh=False, bs=None: af.make_loader(cache, idx, remap, targets, bs or a.batch, sh, a.workers)
        tr_ld, va_ld = L(sp["train"], True), L(sp["val"], bs=64)
        ev = {"train": L(tr_sub, bs=128), "calib": L(sp["calib"], bs=128), "test": L(sp["test"], bs=128), "unk": L(sp["unk"], bs=128)}
        for t in seeds:
            d = os.path.join(a.out, f"split{s}", f"seed{t}"); os.makedirs(d, exist_ok=True)
            for m in a.models:
                npz = os.path.join(d, f"{m}.npz")
                if os.path.exists(npz): continue
                if (time.time() - t_start) / 3600 > a.max_hours: stopped = True; print("!! time budget reached; commit again with this version attached"); break
                ax.seed_everything(s * 1000 + zlib.crc32(m.encode()) % 997 + 7919 * t)
                model = af.build(BASE_OF[m], K, pre)
                def clips_of(mdl):
                    mdl.eval(); acts = [[] for _ in mdl.experts]
                    with torch.no_grad():
                        for bi, (xb, _) in enumerate(ev["train"]):
                            if bi >= 4: break
                            x = af.norm_gpu(xb)
                            for k_, e in enumerate(mdl.experts): acts[k_].append(e.backbone(x).float().flatten().cpu())
                    return [float(np.percentile(torch.cat(v).numpy(), 90)) for v in acts]
                if m in AUG_MODELS:
                    model, hist = train_aug(model, tr_ld, va_ld, a.epochs, a.lr, tag=f"[{a.dataset} s{s} t{t} {m}]")
                    cl = clips_of(model); o = {p: af.collect(model, ev[p], cl) for p in ev}
                else:
                    snaps, hist = train_snapshots(model, tr_ld, va_ld, a.epochs, a.lr, tag=f"[{a.dataset} s{s} t{t} {m}]")
                    per_snap = []
                    for st in snaps:
                        model.load_state_dict(st); cl = clips_of(model); per_snap.append({p: af.collect(model, ev[p], cl) for p in ev})
                    o = stack_parts(per_snap)
                np.savez_compressed(npz, meta=json.dumps(dict(history=hist, lr=a.lr, kind="augmented" if m in AUG_MODELS else "snapshot")),
                                    **{f"{p}__{k}": v for p, dd in o.items() for k, v in dd.items()})
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            print(f"  split {s} seed {t} done | elapsed {(time.time() - t_start) / 3600:.2f} h", flush=True)
            if stopped: break
        if stopped: break
    expected = [(s, t, m) for s in splits for t in seeds for m in a.models]
    missing = [x for x in expected if not os.path.exists(os.path.join(a.out, f"split{x[0]}", f"seed{x[1]}", f"{x[2]}.npz"))]
    json.dump(dict(dataset=a.dataset, base_run=a.run, base_protocol=man0["protocol"], n_expected=len(expected), n_done=len(expected) - len(missing),
                   missing=missing, complete=not missing, models=a.models, script_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1, default=str)
    print("=" * 90); print("COMPLETE" if not missing else f"INCOMPLETE: {len(missing)} of {len(expected)} runs missing - commit again with this version attached"); print("=" * 90)

if __name__ == "__main__":
    main()
