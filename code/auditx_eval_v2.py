# =============================================================================
# AUDIT-X  —  corrected evaluation pipeline (v2)
# =============================================================================
# One script that produces every number the paper needs, from saved artefacts.
#
# What it fixes relative to the original notebooks
#   1. Known data are split per run into  train / val / calib / test  (70/10/10/10).
#      - val   : epoch selection only (known classes only)
#      - calib : operating thresholds only (target FAR)
#      - test  : FAR and closed-set accuracy are REPORTED here, so FAR is a
#                measurement, not a tautology. Training samples are never scored.
#   2. Repeated over several random unknown-class splits (SPLIT_SEEDS); results are
#      mean ± std across splits, with bootstrap CIs and paired tests across splits.
#   3. Scores are computed in float64 from logits; non-saturating scores
#      (MaxLogit, Energy) are reported next to MSP / entropy, plus the fraction of
#      calibration scores tied at the threshold (saturation diagnostic).
#   4. Threshold-free metrics (AUROC, AUPR-out, FPR@95TPR) are primary;
#      ZDR at a calibrated 5% FAR is the operational metric.
#   5. Real baselines: single CNNs, a jointly-trained average ensemble, an
#      independently-trained deep ensemble, a genuine OpenMax (Weibull on distances
#      to class MAVs), Energy, MaxLogit, KNN on penultimate features.
#   6. Four-tag triage is reported with per-tag fractions on test-known and unknown.
#   7. Optional dynamic early exit that is actually conditional: the exit rule uses
#      MobileNet's own confidence only (the gate needs all three experts, so a
#      gate-entropy exit rule can never save compute). Cost is measured with
#      torch.utils.flop_counter and wall-clock timing per expert.
#   8. Classes named in NEVER_UNKNOWN (benign / "Other") are never held out as
#      zero-day, so flagging benign files cannot count as zero-day recall.
#   9. Everything is resumable: per-run scores are saved to .npz and skipped if present.
#
# Adversarial, federated, concept-drift and SOC-stream experiments are NOT here on
# purpose; see the accompanying notes for why they need redesign first.
#
# Usage (Kaggle cell):   !python auditx_eval_v2.py
# Usage (subset):        !python auditx_eval_v2.py --datasets Malimg --seeds 0 1
# Smoke test:            python auditx_eval_v2.py --smoke
# =============================================================================
import argparse, copy, json, math, os, time, warnings, zlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from scipy import stats
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 0. Configuration (edit paths here, or override from the command line)
# -----------------------------------------------------------------------------
CFG = dict(
    DATASETS=OrderedDict(
        Malimg="/kaggle/input/datasets/mohammadmehedimunna/malimg-dataset-25/Malimg_Dataset_25classes",
        MaleVis="/kaggle/input/datasets/mohammadmehedimunna/melevis/MaleVis",
        Dumpware10="/kaggle/working/dumpware10_merged",   # built from dumpware10_images/224/224/{TRAINING,TEST}
    ),
    # matched case-insensitively against class folder names; these are never held out as zero-day
    NEVER_UNKNOWN=["other", "others", "benign", "goodware", "clean", "legitimate"],
    UNKNOWN_RATIO=0.2,
    SPLIT_SEEDS=[0, 1, 2, 3, 4],
    FRACS=(0.70, 0.10, 0.10, 0.10),          # train / val / calib / test (known classes)
    EPOCHS=12,
    BATCH=32,
    LR=1e-3,
    WD=1e-4,
    IMG=224,
    WORKERS=4,
    PRETRAINED=True,
    AMP=True,
    TARGET_FAR=0.05,
    # Models to train per split. Keys are defined in MODEL_SPECS below.
    MODELS=["agef_sharp", "avg_joint", "single_mobilenet", "single_resnet18", "single_effnet_b0"],
    DEEP_ENSEMBLE=True,                        # built post hoc from the three single_* models
    TRIAGE_AXES=("msp", "entropy"),            # the paper's two axes
    KNN_K=50,
    KNN_MAX_TRAIN=6000,
    OPENMAX_TAIL=20,
    OPENMAX_ALPHA=10,
    RUN_DYNAMIC=True,
    DYNAMIC_EXIT_QUANTILES=[0.0, 0.25, 0.5, 0.75, 0.9],
    N_BOOT=10000,
    OUT_DIR="/kaggle/working/auditx_v2",
    # Duplicate-aware splitting (protocol v3): byte-identical or pixel-identical images form one group; a group is never
    # split across train/val/calib/test; groups containing more than one family (label conflicts) are removed; classes that
    # cannot be split (one group > 90 % of the class, or < 10 images left) are excluded from the experiment.
    # Near-identical images (perceptual hashes) are NOT grouped: low-information byte plots share identical dHashes and
    # transitive matching merges whole families; they are reported as a sensitivity analysis (notebook 12b).
    GROUPING=True,
    GROUP_KEYS=(0, 1),            # 0 = file MD5, 1 = pixel MD5, 2 = 256-bit dHash
    UNSPLITTABLE_SHARE=0.9,
    MIN_CLASS_IMAGES=10,
    # Near-duplicate grouping (protocol v4, applied to the datasets listed): within each family, images whose 256-bit dHash
    # differs by <= NEAR_DUP_BITS are merged, closest pairs first, and a merge is refused if the group would exceed
    # NEAR_DUP_CAP images, which prevents transitive chaining across whole families.
    # cap 250 = smallest cap that brings residual near-identical matches (256-bit dHash within 8 bits) between training and
    # evaluation images below 5 % on every dataset without making a class unsplittable through chaining; chosen from duplicate
    # statistics only, never from detection results.
    NEAR_DUP_DATASETS=("Malimg", "MaleVis", "Dumpware10"),
    # Amendment 1 (MaleVis only): identical 256-bit-hash clusters are merged exactly and never cut by the cap.
    IDENTICAL_WHOLE_DATASETS=("MaleVis",),
    NEAR_DUP_BITS=8,
    NEAR_DUP_CAP=250,
)
PROTOCOL = "grouped-L1-v3"

EXPERT_NAMES = ("mobilenet", "resnet18", "effnet_b0")
MODEL_SPECS = {
    # name                 experts            fusion  temp  lambda_div
    "agef_sharp":        (EXPERT_NAMES,       "gate", 1.0,  0.05),
    "agef_soft":         (EXPERT_NAMES,       "gate", 2.0,  0.05),
    "agef_nodiv":        (EXPERT_NAMES,       "gate", 1.0,  0.0),
    "avg_joint":         (EXPERT_NAMES,       "avg",  1.0,  0.0),
    "single_mobilenet":  (("mobilenet",),     "avg",  1.0,  0.0),
    "single_resnet18":   (("resnet18",),      "avg",  1.0,  0.0),
    "single_effnet_b0":  (("effnet_b0",),     "avg",  1.0,  0.0),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
def make_transform(img):
    return transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class IndexedSubset(Dataset):
    """Subset of an ImageFolder with labels remapped to 0..K-1 (unknown -> -1)."""
    def __init__(self, base, indices, remap):
        self.base, self.indices, self.remap = base, np.asarray(indices), remap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, t = self.base.samples[self.indices[i]]
        img = self.base.loader(path)
        if self.base.transform is not None:
            img = self.base.transform(img)
        return img, self.remap.get(t, -1)


def _strat_split(idx, y, test_size, seed):
    try:
        return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    except ValueError:  # a class too small to stratify
        return train_test_split(idx, test_size=test_size, random_state=seed)


def image_keys(path):
    """file MD5, decoded grey-level pixel MD5, and 256-bit dHash (17x16) as hex."""
    import hashlib
    from PIL import Image
    b = open(path, "rb").read()
    with Image.open(path) as im:
        g = im.convert("L")
        pmd5 = hashlib.md5(g.tobytes() + str(g.size).encode()).hexdigest()
        a = np.asarray(g.resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return hashlib.md5(b).hexdigest(), pmd5, np.packbits((a[:, 1:] > a[:, :-1]).flatten()).tobytes().hex()


def compute_groups(base, cache_path=None, key_types=(0, 1), unsplittable_share=0.9, min_class_images=10,
                   near_bits=None, near_cap=50):
    """Union of exact-key equalities (+ optional capped within-family near-duplicate merging) -> duplicate groups.
    Returns dict(groups, drop, exclude_classes, stats, protocol)."""
    keys = None
    if cache_path and os.path.exists(cache_path):
        c = json.load(open(cache_path))
        if c.get("n") == len(base.samples): keys = c["keys"]
    if keys is None:
        keys = []
        for i, (p, _) in enumerate(base.samples):
            keys.append(image_keys(p))
            if (i + 1) % 5000 == 0: print(f"    hashed {i + 1}/{len(base.samples)}", flush=True)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            json.dump(dict(n=len(base.samples), keys=keys), open(cache_path, "w"))
    n = len(keys); parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for t in key_types:
        first = {}
        for i, k in enumerate(keys):
            j = first.setdefault(k[t], i)
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj: parent[ri] = rj
    targets = np.array([t for _, t in base.samples])
    n_near_merges = 0
    if near_bits is not None:
        POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)
        hb = np.stack([np.frombuffer(bytes.fromhex(k[2]), dtype=np.uint8) for k in keys])
        size = np.zeros(n, int)
        for i in range(n): size[find(i)] += 1
        edges = []
        for c in np.unique(targets):
            idx = np.where(targets == c)[0]
            for a0 in range(0, len(idx), 256):
                q = idx[a0:a0 + 256]
                d = POP[hb[q][:, None, :] ^ hb[idx][None, :, :]].sum(-1)
                ii, jj = np.where(d <= near_bits)
                for u, v in zip(ii, jj):
                    if q[u] < idx[v]: edges.append((int(d[u, v]), int(q[u]), int(idx[v])))
        edges.sort()
        for _, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj and size[ri] + size[rj] <= near_cap:
                parent[ri] = rj; size[rj] += size[ri]; n_near_merges += 1
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    fam_per_group = pd.Series(targets).groupby(groups).nunique()
    conflict = fam_per_group[fam_per_group > 1].index.values
    drop = np.isin(groups, conflict)
    sizes = np.bincount(groups)
    exclude = []
    for c in range(len(base.classes)):
        m = (targets == c) & ~drop
        if m.sum() < min_class_images or (m.any() and np.bincount(groups[m]).max() > unsplittable_share * m.sum()):
            exclude.append(c)
    stats = dict(n_images=n, n_groups=int(len(sizes)), n_images_in_multi_groups=int((sizes[groups] > 1).sum()),
                 largest_group=int(sizes.max()), n_conflict_groups=int(len(conflict)), n_dropped=int(drop.sum()),
                 dropped_by_class={base.classes[c]: int(k) for c, k in zip(*np.unique(targets[drop], return_counts=True))},
                 excluded_classes=[base.classes[c] for c in exclude],
                 largest_group_share_by_class={base.classes[c]: round(float(np.bincount(groups[(targets == c) & ~drop]).max()
                                               / max(((targets == c) & ~drop).sum(), 1)), 3) if ((targets == c) & ~drop).any() else None
                                               for c in range(len(base.classes))},
                 near_dup=None if near_bits is None else dict(bits=near_bits, cap=near_cap, merges=n_near_merges))
    prot = PROTOCOL + ("+identical-hash-whole" if 2 in key_types else "") + \
           ("" if near_bits is None else f"+near-L3-{near_bits}bit-cap{near_cap}")
    return dict(groups=groups, drop=drop, exclude_classes=exclude, stats=stats, protocol=prot)


def make_split(base, seed, cfg, G=None):
    """Class-disjoint known/unknown split + 4-way split of known data (70/10/10/10).
    With G (from compute_groups): duplicate groups are kept within one part and label-conflict groups are removed."""
    classes = base.classes
    targets = np.array([t for _, t in base.samples])
    never = {n.lower() for n in cfg["NEVER_UNKNOWN"]}
    eligible = [c for c in range(len(classes)) if classes[c].lower() not in never]
    protected = [classes[c] for c in range(len(classes)) if classes[c].lower() in never]
    if protected:
        print(f"  classes always kept known (benign/other): {protected}")
    n_unk = max(1, int(round(len(classes) * cfg["UNKNOWN_RATIO"])))
    rng = np.random.RandomState(seed)
    unk = sorted(rng.choice(eligible, n_unk, replace=False).tolist())
    known = [c for c in range(len(classes)) if c not in unk]
    excluded = [] if G is None else list(G.get("exclude_classes", []))
    if excluded:
        unk = [c for c in unk if c not in excluded]; known = [c for c in known if c not in excluded]
    keep = np.ones(len(targets), bool) if G is None else ~G["drop"]
    known_idx = np.where(np.isin(targets, known) & keep)[0]
    unk_idx = np.where(np.isin(targets, unk) & keep)[0]
    if G is None:
        f_tr, f_va, f_ca, f_te = cfg["FRACS"]
        tr, rest = _strat_split(known_idx, targets[known_idx], 1 - f_tr, seed)
        va, rest2 = _strat_split(rest, targets[rest], (f_ca + f_te) / (f_va + f_ca + f_te), seed)
        ca, te = _strat_split(rest2, targets[rest2], f_te / (f_ca + f_te), seed)
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        fold = np.full(len(known_idx), -1)
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        for f, (_, te_) in enumerate(sgkf.split(known_idx, targets[known_idx], G["groups"][known_idx])):
            fold[te_] = f
        tr, va, ca, te = (known_idx[fold <= 6], known_idx[fold == 7], known_idx[fold == 8], known_idx[fold == 9])
        missing = [classes[c] for c in known if not np.isin(c, targets[tr])]
        assert not missing, f"known classes without training images after grouping: {missing}"
        for nm, part in [("val", va), ("calib", ca), ("test", te)]:
            lack = [classes[c] for c in known if not np.isin(c, targets[part])]
            if lack: print(f"  warning: seed {seed}: no {nm} images for {lack} (group sizes)")
    remap = {c: i for i, c in enumerate(known)}
    out = dict(known=known, unknown=unk, remap=remap,
               known_names=[classes[c] for c in known], unknown_names=[classes[c] for c in unk],
               train=tr, val=va, calib=ca, test=te, unk=unk_idx,
               protocol=G.get("protocol", PROTOCOL) if G is not None else "random-v2")
    if G is not None:
        out["n_dropped_conflict"] = int(G["drop"].sum())
        out["excluded_classes"] = [classes[c] for c in excluded]
    return out


def loader(base, idx, remap, cfg, shuffle):
    return DataLoader(IndexedSubset(base, idx, remap), batch_size=cfg["BATCH"], shuffle=shuffle,
                      num_workers=cfg["WORKERS"], pin_memory=torch.cuda.is_available(),
                      persistent_workers=False)


# -----------------------------------------------------------------------------
# 2. Models
# -----------------------------------------------------------------------------
class Expert(nn.Module):
    """Backbone returning penultimate features + its own linear head."""
    def __init__(self, name, num_classes, pretrained):
        super().__init__()
        w = "IMAGENET1K_V1" if pretrained else None
        if name == "mobilenet":
            m = models.mobilenet_v3_small(weights=w); d = m.classifier[3].in_features; m.classifier[3] = nn.Identity()
        elif name == "resnet18":
            m = models.resnet18(weights=w); d = m.fc.in_features; m.fc = nn.Identity()
        elif name == "effnet_b0":
            m = models.efficientnet_b0(weights=w); d = m.classifier[1].in_features; m.classifier[1] = nn.Identity()
        else:
            raise ValueError(name)
        self.name, self.backbone, self.head, self.feat_dim = name, m, nn.Linear(d, num_classes), d

    def forward(self, x):
        f = self.backbone(x)
        return f, self.head(f)


class MultiExpert(nn.Module):
    """AGEF-Net (fusion='gate') or uniform average (fusion='avg'). One expert = single CNN."""
    def __init__(self, expert_names, num_classes, fusion="gate", temp=1.0, pretrained=True):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n, num_classes, pretrained) for n in expert_names])
        self.E, self.fusion, self.temp = len(expert_names), fusion, temp
        if fusion == "gate":
            self.gate = nn.Sequential(nn.Linear(num_classes * self.E, 128), nn.ReLU(), nn.Linear(128, self.E))

    def forward(self, x):
        feats, logits = zip(*[e(x) for e in self.experts])
        stacked = torch.stack(logits, 1)                       # B,E,C
        if self.fusion == "gate":
            w = F.softmax(self.gate(stacked.flatten(1)) / self.temp, dim=1)
        else:
            w = torch.full((x.size(0), self.E), 1.0 / self.E, device=x.device)
        fused = (w.unsqueeze(-1) * stacked).sum(1)
        # per-expert L2 normalisation so no single backbone dominates KNN distances
        return dict(logits=fused, expert_logits=stacked, weights=w,
                    feats=torch.cat([F.normalize(f.float(), dim=1) for f in feats], 1))


# -----------------------------------------------------------------------------
# 3. Training (model selection on known-class val accuracy only)
# -----------------------------------------------------------------------------
def train(model, tr_ld, va_ld, cfg, lambda_div, log_prefix=""):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"], weight_decay=cfg["WD"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"])
    use_amp = cfg["AMP"] and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, hist = -1.0, None, []
    for ep in range(cfg["EPOCHS"]):
        model.train(); t0 = time.time(); tl, n = 0.0, 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
                out = model(xb)
                loss = F.cross_entropy(out["logits"], yb)
                if lambda_div > 0 and model.E > 1 and model.fusion == "gate":
                    wbar = out["weights"].float().mean(0)
                    loss = loss + lambda_div * (1.0 - (-(wbar * torch.log(wbar + 1e-8)).sum()))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * xb.size(0); n += xb.size(0)
        sched.step()
        acc = evaluate_acc(model, va_ld, use_amp)
        hist.append(dict(epoch=ep, train_loss=tl / n, val_acc=acc, sec=time.time() - t0))
        print(f"{log_prefix} ep {ep:2d} | loss {tl / n:.4f} | val acc {acc:.2f}% | {time.time() - t0:.0f}s", flush=True)
        if acc > best:
            best, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best, hist


@torch.no_grad()
def evaluate_acc(model, ld, use_amp=False):
    model.eval(); c = t = 0
    for xb, yb in ld:
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            p = model(xb.to(DEVICE))["logits"].argmax(1).cpu()
        c += (p == yb).sum().item(); t += yb.numel()
    return 100.0 * c / max(t, 1)


@torch.no_grad()
def collect(model, ld):
    """Return float64 numpy outputs for every sample in the loader (fp32 forward)."""
    model.eval(); acc = {k: [] for k in ("logits", "expert_logits", "weights", "feats", "y")}
    for xb, yb in ld:
        out = model(xb.to(DEVICE))
        for k in ("logits", "expert_logits", "weights", "feats"):
            acc[k].append(out[k].float().cpu().numpy().astype(np.float64))
        acc["y"].append(yb.numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


# -----------------------------------------------------------------------------
# 4. Scores (higher = more likely unknown)
# -----------------------------------------------------------------------------
def entropy(p):
    return -(p * np.log(np.clip(p, 1e-300, 1.0))).sum(-1)


def base_scores(o, is_deep_ensemble=False):
    """o: dict with logits [N,C], expert_logits [N,E,C], weights [N,E]."""
    if is_deep_ensemble:
        pe = softmax(o["expert_logits"], axis=-1)              # N,E,C
        p = pe.mean(1)
        logits = o["expert_logits"].mean(1)
    else:
        logits = o["logits"]; p = softmax(logits, axis=-1)
        pe = softmax(o["expert_logits"], axis=-1)
    s = OrderedDict()
    s["msp"] = -p.max(1)
    s["entropy"] = entropy(p)
    s["maxlogit"] = -logits.max(1)
    s["energy"] = -logsumexp(logits, axis=1)
    if pe.shape[1] > 1:
        s["mutual_info"] = entropy(pe.mean(1)) - entropy(pe).mean(1)
    if (not is_deep_ensemble) and o["weights"].shape[1] > 1 and np.ptp(o["weights"]) > 1e-9:
        s["gate_entropy"] = entropy(o["weights"])
    return s


class KNNScorer:
    """Sun et al. (2022): negative cosine similarity to k-th nearest training feature."""
    def __init__(self, train_feats, k, max_train, seed):
        rng = np.random.RandomState(seed)
        if len(train_feats) > max_train:
            train_feats = train_feats[rng.choice(len(train_feats), max_train, replace=False)]
        self.bank = torch.tensor(train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True), dtype=torch.float32)
        self.k = min(k, len(self.bank))

    def __call__(self, feats):
        q = torch.tensor(feats / np.linalg.norm(feats, axis=1, keepdims=True), dtype=torch.float32)
        out = []
        for i in range(0, len(q), 1024):
            sims = q[i:i + 1024] @ self.bank.T
            out.append(-sims.topk(self.k, dim=1).values[:, -1])
        return torch.cat(out).numpy().astype(np.float64)


class OpenMax:
    """Bendale & Boult (2016) on fused logits: per-class MAV, Weibull fit on the
    largest distances of correctly classified training samples, top-alpha revision."""
    def __init__(self, train_logits, train_y, tail, alpha):
        from scipy.stats import weibull_min
        self.C = train_logits.shape[1]; self.alpha = min(alpha, self.C)
        self.mav = np.zeros((self.C, self.C)); self.params = []
        pred = train_logits.argmax(1)
        for c in range(self.C):
            av = train_logits[(train_y == c) & (pred == c)]
            if len(av) == 0:
                av = train_logits[train_y == c]
            self.mav[c] = av.mean(0)
            d = np.linalg.norm(av - self.mav[c], axis=1)
            t = np.sort(d)[-min(tail, len(d)):]
            if len(t) < 3 or t.std() < 1e-9:
                self.params.append(None)
            else:
                self.params.append(weibull_min.fit(t, floc=0))
        self._cdf = weibull_min.cdf

    def __call__(self, logits):
        N = len(logits); out = np.zeros(N)
        ranks = np.argsort(-logits, axis=1)[:, :self.alpha]
        wts = np.array([(self.alpha - i) / self.alpha for i in range(self.alpha)])
        for n in range(N):
            v = logits[n].copy(); unk = 0.0
            for r, c in enumerate(ranks[n]):
                prm = self.params[c]
                cdf = 0.0 if prm is None else self._cdf(np.linalg.norm(logits[n] - self.mav[c]), *prm)
                om = wts[r] * cdf
                unk += v[c] * om
                v[c] = v[c] * (1 - om)
            z = np.concatenate([[unk], v])
            out[n] = softmax(z)[0]                               # P(unknown)
        return out


# -----------------------------------------------------------------------------
# 5. Metrics
# -----------------------------------------------------------------------------
def fpr_at_95tpr(s_known, s_unk):
    """Known = positive (accepted if score <= thr). FPR = unknowns accepted."""
    thr = np.quantile(s_known, 0.95)
    return float((s_unk <= thr).mean())


def operating_point(s_calib, s_test, s_unk, target_far):
    thr = np.quantile(s_calib, 1 - target_far)
    return dict(
        thr=float(thr),
        zdr=float((s_unk > thr).mean()),
        far_test=float((s_test > thr).mean()),
        far_calib=float((s_calib > thr).mean()),
        tie_frac_calib=float(np.isclose(s_calib, thr, rtol=0, atol=1e-12).mean()),
    )


def threshold_free(s_known_test, s_unk):
    y = np.r_[np.zeros(len(s_known_test)), np.ones(len(s_unk))]
    s = np.r_[s_known_test, s_unk]
    return dict(auroc=float(roc_auc_score(y, s)), aupr_out=float(average_precision_score(y, s)),
                fpr95=fpr_at_95tpr(s_known_test, s_unk))


def four_tag(conf_score, unc_score, tau_c, tau_u):
    """conf_score / unc_score are 'higher = more unknown'. Paper's Eq. 7 mapping:
    0 Auto-block (confident & certain), 1 Quarantine (unconfident & uncertain),
    2 Review (confident & uncertain), 3 Log (unconfident & certain)."""
    lowconf, highunc = conf_score > tau_c, unc_score > tau_u
    return np.where(~lowconf & ~highunc, 0, np.where(lowconf & highunc, 1, np.where(~lowconf & highunc, 2, 3)))


def triage_report(sc_calib, sc_test, sc_unk, axes, target_far):
    a, b = axes
    tau_a = np.quantile(sc_calib[a], 1 - target_far)
    tau_b = np.quantile(sc_calib[b], 1 - target_far)
    tk = four_tag(sc_test[a], sc_test[b], tau_a, tau_b)
    tu = four_tag(sc_unk[a], sc_unk[b], tau_a, tau_b)
    r = dict(tau_a=float(tau_a), tau_b=float(tau_b),
             zdr=float(np.isin(tu, [1, 2]).mean()), far_test=float(np.isin(tk, [1, 2]).mean()),
             wr_test=float(np.isin(tk, [0, 3]).mean()))
    for t, nm in enumerate(["autoblock", "quarantine", "review", "log"]):
        r[f"known_{nm}"] = float((tk == t).mean()); r[f"unk_{nm}"] = float((tu == t).mean())
    return r


# -----------------------------------------------------------------------------
# 6. Dynamic early exit (honest version)
# -----------------------------------------------------------------------------
def measure_expert_costs(model, cfg):
    """FLOPs per image (torch flop counter) and mean latency per 32-image batch, per expert + gate."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval(); x1 = torch.randn(1, 3, cfg["IMG"], cfg["IMG"], device=DEVICE)
    xb = torch.randn(cfg["BATCH"], 3, cfg["IMG"], cfg["IMG"], device=DEVICE)
    costs = {}
    with torch.no_grad():
        for e in model.experts:
            fc = FlopCounterMode(display=False)
            with fc:
                e(x1)
            for _ in range(3):
                e(xb)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                e(xb)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            costs[e.name] = dict(gflops=fc.get_total_flops() / 1e9, ms_per_batch=(time.time() - t0) / 10 * 1e3)
    return costs


def dynamic_eval(o_cal, o_test, o_unk, costs, cfg):
    """Exit after MobileNet (expert 0) when its own MSP >= theta; theta chosen as a
    quantile of calib MobileNet-MSP so that a target fraction of known traffic exits.
    The detection score is entropy of whichever output is used; its threshold is
    re-calibrated on calib under the same policy."""
    rows = []
    f_m, f_rest = costs["mobilenet"]["gflops"], sum(v["gflops"] for k, v in costs.items() if k != "mobilenet")
    t_m, t_rest = costs["mobilenet"]["ms_per_batch"], sum(v["ms_per_batch"] for k, v in costs.items() if k != "mobilenet")

    def policy(o, theta):
        pm = softmax(o["expert_logits"][:, 0], axis=-1)
        exit_ = pm.max(1) >= theta
        p = np.where(exit_[:, None], pm, softmax(o["logits"], axis=-1))
        return entropy(p), exit_

    mcal = softmax(o_cal["expert_logits"][:, 0], axis=-1).max(1)
    for q in cfg["DYNAMIC_EXIT_QUANTILES"]:
        theta = np.inf if q == 0 else np.quantile(mcal, 1 - q)
        s_cal, _ = policy(o_cal, theta)
        s_te, ex_te = policy(o_test, theta)
        s_un, ex_un = policy(o_unk, theta)
        op = operating_point(s_cal, s_te, s_un, cfg["TARGET_FAR"])
        exit_rate = np.r_[ex_te, ex_un].mean()
        rows.append(dict(target_exit=q, theta=float(theta), exit_rate_known=float(ex_te.mean()),
                         exit_rate_unknown=float(ex_un.mean()),
                         gflops_per_img=f_m + (1 - exit_rate) * f_rest,
                         gflops_static=f_m + f_rest,
                         est_ms_per_batch=t_m + (1 - exit_rate) * t_rest,
                         ms_static=t_m + t_rest, **op))
    return rows


# -----------------------------------------------------------------------------
# 7. One (dataset, seed) run
# -----------------------------------------------------------------------------
def run_split(ds_name, base, seed, cfg):
    out_dir = os.path.join(cfg["OUT_DIR"], ds_name, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    sp = make_split(base, seed, cfg, cfg.get("_GROUPS", {}).get(ds_name))
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sp.items() if k != "remap"},
              open(os.path.join(out_dir, "split.json"), "w"), indent=1)
    print(f"\n[{ds_name} seed {seed}] known={len(sp['known'])} unknown={sp['unknown_names']} "
          f"| train {len(sp['train'])} val {len(sp['val'])} calib {len(sp['calib'])} test {len(sp['test'])} unk {len(sp['unk'])}")

    lds = {k: loader(base, sp[k], sp["remap"], cfg, shuffle=(k == "train")) for k in ("train", "val")}
    eval_lds = {k: loader(base, sp[k], sp["remap"], cfg, shuffle=False) for k in ("train", "calib", "test", "unk")}
    K = len(sp["known"])
    outputs = {}

    for mname in cfg["MODELS"]:
        npz = os.path.join(out_dir, f"{mname}.npz")
        if os.path.exists(npz):
            d = np.load(npz, allow_pickle=True)
            outputs[mname] = {part: {k: d[f"{part}__{k}"] for k in ("logits", "expert_logits", "weights", "feats", "y")}
                              for part in ("train", "calib", "test", "unk")}
            outputs[mname]["_meta"] = json.loads(str(d["meta"]))
            print(f"  [{mname}] loaded cached outputs"); continue
        experts, fusion, temp, lam = MODEL_SPECS[mname]
        seed_everything(seed * 1000 + zlib.crc32(mname.encode()) % 997)
        model = MultiExpert(experts, K, fusion, temp, cfg["PRETRAINED"]).to(DEVICE)
        model, best_val, hist = train(model, lds["train"], lds["val"], cfg, lam, log_prefix=f"  [{mname}]")
        o = {part: collect(model, ld) for part, ld in eval_lds.items()}
        meta = dict(best_val_acc=best_val, history=hist,
                    test_acc=float(100 * (o["test"]["logits"].argmax(1) == o["test"]["y"]).mean()))
        if cfg["RUN_DYNAMIC"] and fusion == "gate":
            meta["costs"] = measure_expert_costs(model, cfg)
        np.savez_compressed(npz, meta=json.dumps(meta),
                            **{f"{p}__{k}": v for p, d in o.items() for k, v in d.items()})
        o["_meta"] = meta; outputs[mname] = o
        del model
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    singles = [f"single_{n}" for n in EXPERT_NAMES]
    if cfg["DEEP_ENSEMBLE"] and all(s in outputs for s in singles):
        de = {}
        for part in ("train", "calib", "test", "unk"):
            de[part] = dict(
                expert_logits=np.concatenate([outputs[s][part]["expert_logits"] for s in singles], 1),
                feats=np.concatenate([outputs[s][part]["feats"] for s in singles], 1),
                y=outputs[singles[0]][part]["y"])
            de[part]["logits"] = de[part]["expert_logits"].mean(1)
            de[part]["weights"] = np.full((len(de[part]["y"]), 3), 1 / 3)
        p = softmax(de["test"]["expert_logits"], -1).mean(1)
        de["_meta"] = dict(test_acc=float(100 * (p.argmax(1) == de["test"]["y"]).mean()), deep_ensemble=True)
        outputs["deep_ensemble"] = de

    # ---- score + metric every model -------------------------------------------------
    rows, tri_rows, dyn_rows, cls_rows = [], [], [], []
    targets = np.array([t for _, t in base.samples])
    unk_family = np.array([base.classes[t] for t in targets[sp["unk"]]])       # loader order == sp["unk"] order
    for mname, o in outputs.items():
        is_de = mname == "deep_ensemble"
        sc = {part: base_scores(o[part], is_de) for part in ("calib", "test", "unk")}
        knn = KNNScorer(o["train"]["feats"], cfg["KNN_K"], cfg["KNN_MAX_TRAIN"], seed)
        for part in ("calib", "test", "unk"):
            sc[part]["knn"] = knn(o[part]["feats"])
        om = OpenMax(o["train"]["logits"], o["train"]["y"], cfg["OPENMAX_TAIL"], cfg["OPENMAX_ALPHA"])
        for part in ("calib", "test", "unk"):
            sc[part]["openmax"] = om(o[part]["logits"])
        for sname in sc["test"]:
            r = dict(dataset=ds_name, seed=seed, model=mname, score=sname,
                     test_acc=o["_meta"]["test_acc"], n_unknown_classes=len(sp["unknown"]))
            r.update(threshold_free(sc["test"][sname], sc["unk"][sname]))
            r.update(operating_point(sc["calib"][sname], sc["test"][sname], sc["unk"][sname], cfg["TARGET_FAR"]))
            rows.append(r)
        # per-class alarm rate: recall for each held-out family, false-alarm rate for each known class
        test_cls = np.array([sp["known_names"][y] for y in o["test"]["y"]])
        for sname in ("msp", "entropy", "energy", "knn"):
            if sname not in sc["test"]:
                continue
            thr = np.quantile(sc["calib"][sname], 1 - cfg["TARGET_FAR"])
            for fam in np.unique(unk_family):
                m_ = unk_family == fam
                cls_rows.append(dict(dataset=ds_name, seed=seed, model=mname, score=sname, cls=fam, role="unknown",
                                     n=int(m_.sum()), alarm_rate=float((sc["unk"][sname][m_] > thr).mean())))
            for kc in np.unique(test_cls):
                m_ = test_cls == kc
                cls_rows.append(dict(dataset=ds_name, seed=seed, model=mname, score=sname, cls=kc, role="known",
                                     n=int(m_.sum()), alarm_rate=float((sc["test"][sname][m_] > thr).mean())))
        tri = triage_report(sc["calib"], sc["test"], sc["unk"], cfg["TRIAGE_AXES"], cfg["TARGET_FAR"])
        tri_rows.append(dict(dataset=ds_name, seed=seed, model=mname, axes="/".join(cfg["TRIAGE_AXES"]), **tri))
        if cfg["RUN_DYNAMIC"] and "costs" in o["_meta"]:
            for d in dynamic_eval(o["calib"], o["test"], o["unk"], o["_meta"]["costs"], cfg):
                dyn_rows.append(dict(dataset=ds_name, seed=seed, model=mname, **d))
    return rows, tri_rows, dyn_rows, cls_rows


# -----------------------------------------------------------------------------
# 8. Aggregation across splits
# -----------------------------------------------------------------------------
def boot_ci(x, n_boot, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    m = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarise(raw, cfg):
    metrics = ["auroc", "aupr_out", "fpr95", "zdr", "far_test", "tie_frac_calib", "test_acc"]
    out = []
    for (d, m, s), g in raw.groupby(["dataset", "model", "score"]):
        r = dict(dataset=d, model=m, score=s, n_splits=len(g))
        for k in metrics:
            r[f"{k}_mean"], r[f"{k}_std"] = g[k].mean(), g[k].std(ddof=1) if len(g) > 1 else np.nan
        r["auroc_ci_lo"], r["auroc_ci_hi"] = boot_ci(g["auroc"], cfg["N_BOOT"])
        r["zdr_ci_lo"], r["zdr_ci_hi"] = boot_ci(g["zdr"], cfg["N_BOOT"])
        out.append(r)
    return pd.DataFrame(out)


def paired_tests(raw, cfg, ref=("agef_sharp", "entropy")):
    """Reference (model, score) vs every other (model, score), paired over split seeds."""
    out = []
    for d, gd in raw.groupby("dataset"):
        a = gd[(gd.model == ref[0]) & (gd.score == ref[1])].set_index("seed")
        if a.empty:
            continue
        for (m, s), gb in gd.groupby(["model", "score"]):
            if (m, s) == ref:
                continue
            b = gb.set_index("seed")
            common = a.index.intersection(b.index)
            if len(common) < 2:
                continue
            for k in ("auroc", "zdr"):
                diff = a.loc[common, k].values - b.loc[common, k].values
                t = stats.ttest_rel(a.loc[common, k], b.loc[common, k])
                try:
                    w = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                except ValueError:
                    w = np.nan
                lo, hi = boot_ci(diff, cfg["N_BOOT"])
                out.append(dict(dataset=d, reference=f"{ref[0]}:{ref[1]}", other=f"{m}:{s}", metric=k,
                                n=len(common), mean_diff=diff.mean(), diff_ci_lo=lo, diff_ci_hi=hi,
                                t_p=t.pvalue, wilcoxon_p=w))
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# 9. Smoke test data (synthetic, CPU) — verifies the pipeline end to end
# -----------------------------------------------------------------------------
def build_synthetic(root, n_classes=6, n_per=40, img=32, seed=0):
    from PIL import Image
    rng = np.random.RandomState(seed)
    for c in range(n_classes):
        os.makedirs(os.path.join(root, f"fam{c}"), exist_ok=True)
        base = rng.rand(img, img)
        for i in range(n_per):
            a = np.clip(base + 0.25 * rng.randn(img, img), 0, 1)
            Image.fromarray((a * 255).astype(np.uint8), "L").save(os.path.join(root, f"fam{c}", f"{i}.png"))
    os.makedirs(os.path.join(root, "Other"), exist_ok=True)
    for i in range(n_per):
        Image.fromarray((rng.rand(img, img) * 255).astype(np.uint8), "L").save(os.path.join(root, "Other", f"{i}.png"))


# -----------------------------------------------------------------------------
# 10. Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", help="subset of CFG['DATASETS'] keys")
    ap.add_argument("--path", nargs="*", help="override paths as Name=/path")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--img", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-dynamic", action="store_true")
    ap.add_argument("--no-grouping", action="store_true", help="old random splits (protocol v2)")
    ap.add_argument("--smoke", action="store_true", help="synthetic CPU run to verify the pipeline")
    ap.add_argument("--resume-from", help="folder with a previous OUT_DIR (e.g. an attached Kaggle output); cached runs are copied and skipped")
    ap.add_argument("--max-hours", type=float, default=None, help="do not start a new split after this many hours")
    args, _ = ap.parse_known_args()

    cfg = copy.deepcopy(CFG)
    if args.smoke:
        root = "/tmp/auditx_smoke_data"
        if not os.path.exists(root):
            build_synthetic(root)
        cfg.update(DATASETS=OrderedDict(Synthetic=root), NEVER_UNKNOWN=["other"], SPLIT_SEEDS=[0, 1], EPOCHS=2, IMG=64, BATCH=16,
                   WORKERS=0, PRETRAINED=False, OUT_DIR="/tmp/auditx_smoke_out", N_BOOT=500,
                   KNN_K=5, OPENMAX_TAIL=5, UNKNOWN_RATIO=0.34)
    if args.path:
        for kv in args.path:
            k, v = kv.split("=", 1); cfg["DATASETS"][k] = v
    if args.datasets:
        cfg["DATASETS"] = OrderedDict((k, cfg["DATASETS"][k]) for k in args.datasets)
    if args.seeds: cfg["SPLIT_SEEDS"] = args.seeds
    if args.models: cfg["MODELS"] = args.models
    if args.epochs: cfg["EPOCHS"] = args.epochs
    if args.img: cfg["IMG"] = args.img
    if args.batch: cfg["BATCH"] = args.batch
    if args.workers is not None: cfg["WORKERS"] = args.workers
    if args.out: cfg["OUT_DIR"] = args.out
    if args.no_pretrained: cfg["PRETRAINED"] = False
    if args.no_dynamic: cfg["RUN_DYNAMIC"] = False
    if args.no_grouping: cfg["GROUPING"] = False

    os.makedirs(cfg["OUT_DIR"], exist_ok=True)
    if args.resume_from and os.path.isdir(args.resume_from):
        import shutil
        n_copied = 0
        for dp, _, fns in os.walk(args.resume_from):
            for fn in fns:
                if fn.endswith(".npz") or fn == "split.json":
                    rel = os.path.relpath(os.path.join(dp, fn), args.resume_from)
                    dst = os.path.join(cfg["OUT_DIR"], rel)
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(os.path.join(dp, fn), dst); n_copied += 1
        print(f"Resumed {n_copied} cached files from {args.resume_from}")
    t_start = time.time()
    json.dump({k: (list(v.items()) if isinstance(v, OrderedDict) else v) for k, v in cfg.items()},
              open(os.path.join(cfg["OUT_DIR"], "config.json"), "w"), indent=1)
    print(f"Device: {DEVICE} | torch {torch.__version__} | out: {cfg['OUT_DIR']}")

    all_rows, all_tri, all_dyn, all_cls = [], [], [], []
    for ds_name, path in cfg["DATASETS"].items():
        if not path or not os.path.isdir(path):
            print(f"!! skipping {ds_name}: path not found ({path})"); continue
        base = datasets.ImageFolder(path, transform=make_transform(cfg["IMG"]))
        print(f"\n===== {ds_name}: {len(base)} images, {len(base.classes)} classes | protocol {PROTOCOL if cfg['GROUPING'] else 'random-v2'} =====")
        if cfg["GROUPING"]:
            near = ds_name in cfg.get("NEAR_DUP_DATASETS", ())
            G = compute_groups(base, os.path.join(cfg["OUT_DIR"], ds_name, "image_keys.json"), cfg["GROUP_KEYS"],
                               cfg["UNSPLITTABLE_SHARE"], cfg["MIN_CLASS_IMAGES"],
                               cfg["NEAR_DUP_BITS"] if near else None, cfg["NEAR_DUP_CAP"])
            print("  split protocol:", G["protocol"])
            cfg.setdefault("_GROUPS", {})[ds_name] = G
            json.dump(G["stats"], open(os.path.join(cfg["OUT_DIR"], ds_name, "groups_stats.json"), "w"), indent=1)
            print("  duplicate groups:", json.dumps({k: v for k, v in G["stats"].items() if k not in ("dropped_by_class", "largest_group_share_by_class")}),
                  "| dropped (label conflicts):", G["stats"]["dropped_by_class"], "| excluded classes:", G["stats"]["excluded_classes"])
        for seed in cfg["SPLIT_SEEDS"]:
            if args.max_hours and (time.time() - t_start) / 3600 > args.max_hours:
                print(f"!! time budget of {args.max_hours} h reached; not starting seed {seed}. "
                      f"Re-run with --resume-from to continue."); break
            r, t, d, c = run_split(ds_name, base, seed, cfg)
            all_cls += c
            pd.DataFrame(all_cls).to_csv(os.path.join(cfg["OUT_DIR"], "per_class_raw.csv"), index=False)
            print(f"  elapsed {(time.time() - t_start) / 3600:.2f} h", flush=True)
            all_rows += r; all_tri += t; all_dyn += d
            # write after every split so a timed-out session still leaves results
            pd.DataFrame(all_rows).to_csv(os.path.join(cfg["OUT_DIR"], "results_raw.csv"), index=False)
            pd.DataFrame(all_tri).to_csv(os.path.join(cfg["OUT_DIR"], "triage_raw.csv"), index=False)
            if all_dyn:
                pd.DataFrame(all_dyn).to_csv(os.path.join(cfg["OUT_DIR"], "dynamic_raw.csv"), index=False)

    if not all_rows:
        print("No results."); return
    raw = pd.DataFrame(all_rows)
    summ = summarise(raw, cfg); summ.to_csv(os.path.join(cfg["OUT_DIR"], "summary.csv"), index=False)
    paired_tests(raw, cfg).to_csv(os.path.join(cfg["OUT_DIR"], "paired_tests.csv"), index=False)
    tri = pd.DataFrame(all_tri)
    tri.drop(columns=["axes", "seed"]).groupby(["dataset", "model"]).agg(["mean", "std"]).to_csv(os.path.join(cfg["OUT_DIR"], "triage_summary.csv"))
    if all_dyn:
        dyn = pd.DataFrame(all_dyn)
        dyn.groupby(["dataset", "model", "target_exit"]).mean(numeric_only=True).to_csv(
            os.path.join(cfg["OUT_DIR"], "dynamic_summary.csv"))

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    show = summ[summ.score.isin(["msp", "entropy", "energy", "knn", "openmax"])][
        ["dataset", "model", "score", "n_splits", "auroc_mean", "auroc_std", "fpr95_mean",
         "zdr_mean", "zdr_std", "far_test_mean", "tie_frac_calib_mean", "test_acc_mean"]]
    print("\n", show.round(4).to_string(index=False))
    print(f"\nAll outputs in {cfg['OUT_DIR']}")


if __name__ == "__main__":
    main()
