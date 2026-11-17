# =============================================================================
# AUDIT-X FINAL - one-dataset training run for the frozen protocol (PROTOCOL.md)
#   * protocol v4 splits: exact duplicates + capped within-family near-duplicates grouped (auditx_eval_v2.compute_groups)
#   * 5 unseen-family splits x 3 training seeds x 6 models:
#       agef_sharp, agef_tuned (feature-level gate + per-expert auxiliary losses, LR chosen on known-class validation data),
#       avg_joint, single_mobilenet, single_resnet18, single_effnet_b0   (deep ensemble is built from the singles at analysis time)
#   * saves everything the analysis needs: logits, expert logits, gate weights, L2-normalised features, raw single-model features
#     and head weights (ViM), ReAct logits, training histories; frozen DINOv2 features of every image; image hashes
#   * in-memory uint8 image cache (decode once), GPU-side normalisation, automatic resume, self-check manifest
# =============================================================================
import argparse, copy, glob, hashlib, json, os, shutil, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.special import softmax
from PIL import Image
from torchvision import datasets
import auditx_eval_v2 as ax

MODELS = ["agef_sharp", "agef_tuned", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEV = ax.DEVICE

# ----------------------------------------------------------------------------- image cache
def _load(path, img):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((img, img), Image.BILINEAR), dtype=np.uint8)

def build_cache(base, img, workers=8):
    t0 = time.time(); arr = np.empty((len(base.samples), img, img, 3), np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        for i, a in enumerate(ex.map(lambda p: _load(p[0], img), base.samples, chunksize=64)): arr[i] = a
    print(f"  image cache: {arr.shape} uint8 ({arr.nbytes / 1e9:.2f} GB) built in {(time.time() - t0) / 60:.1f} min", flush=True)
    return arr

class CacheDS(Dataset):
    def __init__(self, cache, idx, labels): self.c, self.idx, self.lab = cache, np.asarray(idx), labels
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): j = self.idx[i]; return torch.from_numpy(self.c[j]), self.lab[j]

def norm_gpu(xb):
    xb = xb.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return (xb - MEAN.to(DEV)) / STD.to(DEV)

def make_loader(cache, idx, remap, targets, bs, shuffle, workers=2):
    lab = np.array([remap.get(int(t), -1) for t in targets])
    return DataLoader(CacheDS(cache, idx, lab), batch_size=bs, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=False)

# ----------------------------------------------------------------------------- models
class TunedAGEF(nn.Module):
    """AGEF-Net variant: the gate reads the experts' L2-normalised penultimate features (not their logits), and every expert
    receives its own auxiliary cross-entropy loss so that no expert is starved of gradient."""
    def __init__(self, num_classes, pretrained=True, temp=1.0):
        super().__init__()
        self.experts = nn.ModuleList([ax.Expert(n, num_classes, pretrained) for n in ax.EXPERT_NAMES])
        d = sum(e.feat_dim for e in self.experts); self.E, self.fusion, self.temp = 3, "gate", temp
        self.gate = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 3))
    def forward(self, x):
        feats, logits = zip(*[e(x) for e in self.experts])
        fn = torch.cat([F.normalize(f.float(), dim=1) for f in feats], 1)
        st = torch.stack(logits, 1); w = F.softmax(self.gate(fn) / self.temp, 1)
        return dict(logits=(w.unsqueeze(-1) * st).sum(1), expert_logits=st, weights=w, feats=fn)

def build(model_name, K, pretrained):
    if model_name == "agef_tuned": return TunedAGEF(K, pretrained)
    experts, fusion, temp, _ = ax.MODEL_SPECS[model_name]
    return ax.MultiExpert(experts, K, fusion, temp, pretrained)

def lam_div(model_name):
    return 0.05 if model_name in ("agef_sharp", "agef_tuned") else 0.0

# ----------------------------------------------------------------------------- training
def train(model, model_name, tr, va, epochs, lr, wd=1e-4, aux=0.5, tag=""):
    model.to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    amp = DEV.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, hist = (-1, 1e9), None, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = n = 0
        for xb, yb in tr:
            xb = norm_gpu(xb); yb = yb.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEV.type, enabled=amp):
                out = model(xb); loss = F.cross_entropy(out["logits"], yb)
                if model_name == "agef_tuned":
                    loss = loss + aux * sum(F.cross_entropy(out["expert_logits"][:, i], yb) for i in range(3)) / 3
                lam = lam_div(model_name)
                if lam > 0:
                    wb = out["weights"].float().mean(0); loss = loss + lam * (1.0 - (-(wb * torch.log(wb + 1e-8)).sum()))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(yb); n += len(yb)
        sched.step()
        acc, nll = evaluate(model, va)
        hist.append(dict(epoch=ep, train_loss=tl / n, val_acc=acc, val_nll=nll, sec=time.time() - t0))
        print(f"  {tag} ep {ep:2d} loss {tl / n:.4f} val acc {acc:.2f}% nll {nll:.4f} {time.time() - t0:.0f}s", flush=True)
        if (acc, -nll) > (best[0], -best[1]): best, best_state = (acc, nll), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); return model, hist

@torch.no_grad()
def evaluate(model, ld):
    model.eval(); c = t = 0; nll = 0.0
    for xb, yb in ld:
        lg = model(norm_gpu(xb))["logits"].float(); yb = yb.to(DEV)
        c += (lg.argmax(1) == yb).sum().item(); t += len(yb); nll += F.cross_entropy(lg, yb, reduction="sum").item()
    return 100.0 * c / t, nll / t

@torch.no_grad()
def collect(model, ld, clips=None, keep_raw=False):
    """Fused / expert logits, gate weights, L2-normalised features (fp16); ReAct logits with per-expert clips; raw features."""
    model.eval(); acc = {k: [] for k in ("logits", "expert_logits", "weights", "feats", "y", "react_logits")}; raw = []
    for xb, yb in ld:
        x = norm_gpu(xb); fs = [e.backbone(x).float() for e in model.experts]
        def fuse(feats):
            st = torch.stack([e.head(f) for e, f in zip(model.experts, feats)], 1)
            if isinstance(model, TunedAGEF):
                w = F.softmax(model.gate(torch.cat([F.normalize(f, dim=1) for f in feats], 1)) / model.temp, 1)
            elif model.fusion == "gate":
                w = F.softmax(model.gate(st.flatten(1)) / model.temp, 1)
            else:
                w = torch.full((x.size(0), model.E), 1.0 / model.E, device=x.device)
            return (w.unsqueeze(-1) * st).sum(1), st, w
        fu, st, w = fuse(fs)
        acc["logits"].append(fu.cpu().numpy()); acc["expert_logits"].append(st.cpu().numpy()); acc["weights"].append(w.cpu().numpy())
        acc["feats"].append(torch.cat([F.normalize(f, dim=1) for f in fs], 1).cpu().numpy().astype(np.float16)); acc["y"].append(yb.numpy())
        if clips is not None: acc["react_logits"].append(fuse([f.clamp(max=c) for f, c in zip(fs, clips)])[0].cpu().numpy())
        if keep_raw: raw.append(fs[0].cpu().numpy().astype(np.float16))
    out = {k: np.concatenate(v) for k, v in acc.items() if v}
    if keep_raw: out["raw"] = np.concatenate(raw)
    return out

# ----------------------------------------------------------------------------- DINOv2 frozen features
@torch.no_grad()
def dinov2_features(cache, bs=128):
    try:
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False, trust_repo=True, skip_validation=True)
    except Exception as e:
        try:
            import timm; net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        except Exception as e2:
            print("  DINOv2 unavailable:", e, "|", e2); return None
    net = net.to(DEV).eval(); out = []
    for i in range(0, len(cache), bs):
        x = norm_gpu(torch.from_numpy(cache[i:i + bs]))
        if x.shape[-1] % 14: x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=DEV.type, enabled=DEV.type == "cuda"):
            out.append(net(x).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)

# ----------------------------------------------------------------------------- latency (one split, seed 0)
@torch.no_grad()
def latency(models, xb_uint8, reps=20):
    x = norm_gpu(xb_uint8); res = {}
    def timeit(fn, X, per_sample):
        def run():
            if per_sample:
                for i in range(X.size(0)): fn(X[i:i + 1])
            else: fn(X)
        for _ in range(2): run()
        if DEV.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps): run()
        if DEV.type == "cuda": torch.cuda.synchronize()
        return (time.time() - t0) / reps / X.size(0) * 1e3
    ag = models["agef_sharp"].eval()
    pm = F.softmax(ag.experts[0](x)[1], 1).max(1).values; theta = float(pm.median())
    def early(X):
        _, l0 = ag.experts[0](X); keep = F.softmax(l0, 1).max(1).values < theta
        if keep.any():
            st = torch.stack([l0[keep]] + [e(X[keep])[1] for e in ag.experts[1:]], 1)
            (F.softmax(ag.gate(st.flatten(1)) / ag.temp, 1).unsqueeze(-1) * st).sum(1)
    singles = ["single_mobilenet", "single_resnet18", "single_effnet_b0"]
    for bs, per in [("batch1", True), ("batch32", False)]:
        r = {m: timeit(lambda z, mm=models[m].eval(): mm(z), x, per) for m in models}
        r["deep_ensemble"] = timeit(lambda z: [models[s](z) for s in singles], x, per)
        r["agef_sharp_early_exit_50pct"] = timeit(early, x, per); res[bs] = r
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return res

# ----------------------------------------------------------------------------- main
def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--path", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", type=int, default=[0, 1, 2, 3, 4]); ap.add_argument("--train-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--models", nargs="*", default=MODELS); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--img", type=int, default=224); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--tuned-lrs", nargs="*", type=float, default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--latency", action="store_true"); ap.add_argument("--dinov2", action="store_true")
    ap.add_argument("--max-hours", type=float, default=11.3); ap.add_argument("--protocol-file")
    a = ap.parse_args(); t_start = time.time(); os.makedirs(a.out, exist_ok=True); pre = not a.no_pretrained
    # automatic resume: copy finished results from any attached previous version of this output folder
    for prev in glob.glob(f"/kaggle/input/**/{os.path.basename(a.out)}", recursive=True):
        if os.path.abspath(prev) == os.path.abspath(a.out): continue
        n = 0
        for dp, _, fns in os.walk(prev):
            for fn in fns:
                if fn.endswith((".npz", ".json", ".npy")):
                    dst = os.path.join(a.out, os.path.relpath(os.path.join(dp, fn), prev))
                    if not os.path.exists(dst): os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(os.path.join(dp, fn), dst); n += 1
        print(f"resumed {n} files from {prev}", flush=True)
    cfg = copy.deepcopy(ax.CFG)
    base = datasets.ImageFolder(a.path); targets = np.array(base.targets)
    np.save(os.path.join(a.out, "targets.npy"), targets); json.dump(base.classes, open(os.path.join(a.out, "classes.json"), "w"))
    near = a.dataset in cfg["NEAR_DUP_DATASETS"]
    G = ax.compute_groups(base, os.path.join(a.out, "image_keys.json"), cfg["GROUP_KEYS"], cfg["UNSPLITTABLE_SHARE"], cfg["MIN_CLASS_IMAGES"],
                          cfg["NEAR_DUP_BITS"] if near else None, cfg["NEAR_DUP_CAP"])
    json.dump(dict(G["stats"], protocol=G["protocol"]), open(os.path.join(a.out, "groups_stats.json"), "w"), indent=1)
    print("protocol:", G["protocol"], "| excluded:", G["stats"]["excluded_classes"], flush=True)
    cache = build_cache(base, a.img)
    if a.dinov2 and not os.path.exists(os.path.join(a.out, "dinov2_feats.npy")):
        f = dinov2_features(cache)
        if f is not None: np.save(os.path.join(a.out, "dinov2_feats.npy"), f); print("  DINOv2 features:", f.shape, flush=True)
    splits = {s: ax.make_split(base, s, cfg, G) for s in a.splits}
    for s, sp in splits.items():
        d = os.path.join(a.out, f"split{s}"); os.makedirs(d, exist_ok=True)
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sp.items() if k != "remap"}, open(os.path.join(d, "split.json"), "w"), indent=1)
    # --- tuned-AGEF learning rate, chosen on known-class validation data of split 0, training seed 0 only
    lr_file = os.path.join(a.out, "agef_tuned_lr.json")
    if "agef_tuned" in a.models and not os.path.exists(lr_file):
        sp = splits[a.splits[0]]; K = len(sp["known"]); res = []
        tr = make_loader(cache, sp["train"], sp["remap"], targets, a.batch, True, a.workers); va = make_loader(cache, sp["val"], sp["remap"], targets, 64, False, a.workers)
        for lr in a.tuned_lrs:
            ax.seed_everything(12345); m = build("agef_tuned", K, pre)
            m, h = train(m, "agef_tuned", tr, va, a.epochs, lr, tag=f"[lr sweep {lr:g}]")
            best = max(h, key=lambda r: (r["val_acc"], -r["val_nll"])); res.append(dict(lr=lr, val_acc=best["val_acc"], val_nll=best["val_nll"]))
            del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        chosen = max(res, key=lambda r: (r["val_acc"], -r["val_nll"]))["lr"]
        json.dump(dict(sweep=res, chosen=chosen, criterion="known-class validation accuracy, ties by validation NLL"), open(lr_file, "w"), indent=1)
        print("agef_tuned learning rate chosen on validation data:", chosen, flush=True)
    tuned_lr = json.load(open(lr_file))["chosen"] if os.path.exists(lr_file) else a.lr
    # --- main loop
    stopped = False
    for s, sp in splits.items():
        K = len(sp["known"]); rng = np.random.RandomState(s)
        tr_sub = np.sort(rng.choice(sp["train"], min(len(sp["train"]), 6000), replace=False))
        L = lambda idx, sh=False, bs=None: make_loader(cache, idx, sp["remap"], targets, bs or a.batch, sh, a.workers)
        tr_ld, va_ld = L(sp["train"], True), L(sp["val"], bs=64)
        ev = {"train": L(tr_sub, bs=128), "calib": L(sp["calib"], bs=128), "test": L(sp["test"], bs=128), "unk": L(sp["unk"], bs=128)}
        for t in a.train_seeds:
            d = os.path.join(a.out, f"split{s}", f"seed{t}"); os.makedirs(d, exist_ok=True); trained = {}
            for m in a.models:
                npz = os.path.join(d, f"{m}.npz")
                if os.path.exists(npz): continue
                if (time.time() - t_start) / 3600 > a.max_hours:
                    stopped = True; print(f"!! time budget reached before split {s} seed {t} {m}; commit again with this version attached"); break
                ax.seed_everything(s * 1000 + zlib.crc32(m.encode()) % 997 + 7919 * t)
                model = build(m, K, pre)
                model, hist = train(model, m, tr_ld, va_ld, a.epochs, tuned_lr if m == "agef_tuned" else a.lr, tag=f"[{a.dataset} s{s} t{t} {m}]")
                single = m.startswith("single_")
                # ReAct clip per expert = 90th percentile of its penultimate activations on (up to) 4 training batches
                clips = []
                with torch.no_grad():
                    model.eval(); acts = [[] for _ in model.experts]
                    for bi, (xb, _) in enumerate(ev["train"]):
                        if bi >= 4: break
                        x = norm_gpu(xb)
                        for k_, e in enumerate(model.experts): acts[k_].append(e.backbone(x).float().flatten().cpu())
                    clips = [float(np.percentile(torch.cat(v).numpy(), 90)) for v in acts]
                o = {"train": collect(model, ev["train"], clips, keep_raw=single)}
                for p in ("calib", "test", "unk"): o[p] = collect(model, ev[p], clips, keep_raw=single)
                extra = {}
                if single:
                    e = model.experts[0]; extra = dict(W=e.head.weight.detach().cpu().numpy(), b=e.head.bias.detach().cpu().numpy())
                np.savez_compressed(npz, meta=json.dumps(dict(history=hist, lr=tuned_lr if m == "agef_tuned" else a.lr, clips=clips)),
                                    **{f"{p}__{k}": v for p, dd in o.items() for k, v in dd.items()}, **extra)
                if a.latency and s == a.splits[0] and t == a.train_seeds[0]: trained[m] = model
                else: del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            if trained and len(trained) == len(a.models) and not os.path.exists(os.path.join(a.out, "latency.json")):
                lat = latency(trained, next(iter(ev["calib"]))[0][:32]); json.dump(lat, open(os.path.join(a.out, "latency.json"), "w"), indent=1)
                print("latency (ms/image):", json.dumps(lat), flush=True)
            trained.clear()
            print(f"  split {s} seed {t} done | elapsed {(time.time() - t_start) / 3600:.2f} h", flush=True)
            if stopped: break
        if stopped: break
    # --- self-check manifest
    expected = [(s, t, m) for s in a.splits for t in a.train_seeds for m in a.models]
    missing = [x for x in expected if not os.path.exists(os.path.join(a.out, f"split{x[0]}", f"seed{x[1]}", f"{x[2]}.npz"))]
    man = dict(dataset=a.dataset, protocol=G["protocol"], n_expected=len(expected), n_done=len(expected) - len(missing), missing=missing,
               complete=not missing, dinov2=os.path.exists(os.path.join(a.out, "dinov2_feats.npy")),
               script_sha256=sha(__file__), eval_sha256=sha(ax.__file__),
               protocol_sha256=sha(a.protocol_file) if a.protocol_file and os.path.exists(a.protocol_file) else None,
               args=vars(a), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(man, open(os.path.join(a.out, "MANIFEST.json"), "w"), indent=1, default=str)
    print("=" * 90); print("COMPLETE" if not missing else f"INCOMPLETE: {len(missing)} of {len(expected)} runs missing - commit again with this version attached")
    print("=" * 90)

if __name__ == "__main__":
    main()
