"""Check that the protocol documents and training scripts in this repository are the ones recorded by the runs.

Every training run recorded the SHA-256 of the protocol document and of the training script in its manifest; the
analysis logs (results/primary_analysis/VERDICTS.txt) print them. This script recomputes the hashes of the files here.
Usage: python tools/verify_hashes.py
"""
import hashlib, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EXPECTED = {  # file: (sha256 prefix recorded by the runs, which runs)
    "protocol/PROTOCOL.md": ("aaa563ecc3c7ee8bcc59d66a33bcfa60cf8f5e7c98eccff60a966b3e4849853e", "Malimg, Dumpware10, original MaleVis"),
    "protocol/PROTOCOL_AMENDMENT_1.md": ("9e25083cc4eae36aa1da15f9ca6ce0d0d8ed724e6e7c8cc9319e71f3303eaddf", "MaleVis (Amendment 1)"),
    "code/auditx_final_v4.py": ("ef76bd1e6704", "Malimg, Dumpware10, original MaleVis"),
    "code/auditx_final.py": ("611bb69d73be", "MaleVis (Amendment 1)"),
}
sha = lambda p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
ok = True
for f, (exp, runs) in EXPECTED.items():
    h = sha(f); good = h.startswith(exp); ok &= good
    print(f"{'OK ' if good else 'BAD'} {f:36s} {h[:12]}  recorded by: {runs}")
log = open(os.path.join(ROOT, "results", "primary_analysis", "VERDICTS.txt")).read()
for f, (exp, _) in EXPECTED.items():
    if exp[:12] not in log:
        print(f"note: {exp[:12]} ({f}) not printed in VERDICTS.txt")
# The run notebooks embed every file they use (%%writefile cells): check they are byte-identical to code/ and protocol/
import glob, json
files = {hashlib.sha256(open(p, "rb").read()).hexdigest(): os.path.relpath(p, ROOT)
         for p in glob.glob(os.path.join(ROOT, "code", "*.py")) + glob.glob(os.path.join(ROOT, "protocol", "*.md"))}
for nbp in sorted(glob.glob(os.path.join(ROOT, "notebooks", "final", "FINAL_[ABC]*.ipynb"))):
    used = []
    for c in json.load(open(nbp))["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and src.startswith("%%writefile"):
            name, body = src.split("\n", 1)[0].split("/")[-1], src.split("\n", 1)[1]
            hit = next((files[h] for h in (hashlib.sha256(body.encode()).hexdigest(),
                                           hashlib.sha256((body + "\n").encode()).hexdigest()) if h in files), None)
            ok &= hit is not None
            used.append(f"{name} = {hit or 'NOT FOUND IN REPOSITORY'}")
    print(f"{os.path.basename(nbp)}: " + "; ".join(used))
print("ALL HASHES MATCH" if ok else "HASH MISMATCH")
sys.exit(0 if ok else 1)
