"""Exact duplicate exposure of run splits (no GPU, no images): share of known evaluation images (calibration + test)
that have a training image with an identical file or pixels, an identical 256-bit hash, or a hash within 8 bits
(any family, as in the residual-duplicate table of the paper).

Usage: python exposure_exact.py OUT.csv  DATASET RUN_DIR [RUN_DIR ...]  [-- DATASET RUN_DIR ...]
Each RUN_DIR is a training-run output folder with image_keys.json, MANIFEST.json and split<s>/split.json.
All run folders of one dataset must share the same image keys and unknown families.
"""
import json, os, sys
import numpy as np, pandas as pd

def splits(d):
    return {int(x[5:]): json.load(open(f"{d}/{x}/split.json")) for x in sorted(os.listdir(d))
            if x.startswith("split") and os.path.exists(f"{d}/{x}/split.json")}

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def popcount64(x):  # works on NumPy < 2.0, where np.bitwise_count is missing
    if hasattr(np, "bitwise_count"): return np.bitwise_count(x)
    return _POP[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)

def exposure(keys, sp):
    fm = np.array([k[0] for k in keys]); pm = np.array([k[1] for k in keys]); dh = np.array([k[2] for k in keys])
    w = np.frombuffer(bytes.fromhex("".join(dh)), dtype=">u8").reshape(len(dh), 4)
    tr = np.asarray(sp["train"], int); ev = np.concatenate([sp["calib"], sp["test"]]).astype(int)
    exact = np.isin(fm[ev], fm[tr]) | np.isin(pm[ev], pm[tr]); same = np.isin(dh[ev], dh[tr]); near = np.zeros(len(ev), bool)
    for i in range(0, len(ev), 300):
        near[i:i + 300] = (popcount64(w[ev[i:i + 300]][:, None, :] ^ w[tr][None, :, :]).sum(2) <= 8).any(1)
    return 100 * exact.mean(), 100 * same.mean(), 100 * near.mean()

rows = []; groups = []; cur = []
for a in sys.argv[2:]:
    if a == "--": groups.append(cur); cur = []
    else: cur.append(a)
groups.append(cur)
for g in groups:
    ds, dirs = g[0], g[1:]
    keys0 = json.load(open(f"{dirs[0]}/image_keys.json"))["keys"]; S0 = splits(dirs[0])
    for d in dirs:
        man = json.load(open(f"{d}/MANIFEST.json")); keys = json.load(open(f"{d}/image_keys.json"))["keys"]; S = splits(d)
        assert keys == keys0, f"{ds}: {d} has different image keys"
        for s in S:
            assert S[s]["unknown_names"] == S0[s]["unknown_names"] and sorted(S[s]["unk"]) == sorted(S0[s]["unk"]), f"{ds} split {s}: unknown families differ"
            e = exposure(keys, S[s])
            rows.append(dict(dataset=ds, protocol=man["protocol"], split=s, exact=e[0], identical_hash=e[1], within8=e[2]))
        print(f"{ds}: {man['protocol']} done", flush=True)
D = pd.DataFrame(rows); D.to_csv(sys.argv[1], index=False)
print(D.groupby(["dataset", "protocol"])[["exact", "identical_hash", "within8"]].mean().round(2).to_string())
