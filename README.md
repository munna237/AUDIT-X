# AUDIT-X: leakage-controlled evaluation of unseen-family malware detection

Code, protocol, splits, results and literature audit for the paper

> *How Much Can Duplicates and Evaluation Shortcuts Inflate Image-Based Malware Detection? A Leakage-Controlled Evaluation of Unseen Families and an Audit of Reporting Practice* (IEEE Access, under review).

Archived release: DOI **[to be added after the first Zenodo release]**

The repository contains no malware images or binaries. The datasets must be obtained from their original
sources; this repository publishes the hashes and split assignments needed to reproduce every partition.

## Contents

| Folder | What it contains |
|---|---|
| `protocol/` | The evaluation protocol and Amendment 1, unchanged since the runs; `SHA256SUMS.txt` |
| `code/` | Training and analysis scripts, byte-identical to the versions recorded by the runs |
| `notebooks/final/` | The Kaggle notebooks that ran the final experiments (FINAL_A to FINAL_X) |
| `notebooks/development/` | Earlier notebooks; only the cross-representation and few-shot notebooks (08, 10, 14) feed the paper, as a secondary analysis |
| `splits/` | Per-image hashes, duplicate groups and the five split assignments per dataset |
| `results/` | Analysis outputs (CSV tables, summaries) and the run logs |
| `tools/` | Scripts to verify hashes, regenerate the splits and match a local dataset copy |
| `audit/` | The literature audit: protocol, coding manual, search log, sampling frame, agreement scripts |
| `docs/` | Run guides used for the Kaggle runs |

## Verify that the protocol came before the results

Every training run wrote the SHA-256 of the protocol document and of the training script into its manifest, and the
analysis printed them (`results/primary_analysis/VERDICTS.txt`, first lines). To check that the files here are the
ones the runs used:

```bash
python tools/verify_hashes.py
```

| File | SHA-256 | Used by |
|---|---|---|
| `protocol/PROTOCOL.md` | `aaa563ecc3c7ee8bcc59d66a33bcfa60cf8f5e7c98eccff60a966b3e4849853e` | Malimg, Dumpware10, original MaleVis run |
| `protocol/PROTOCOL_AMENDMENT_1.md` | `9e25083cc4eae36aa1da15f9ca6ce0d0d8ed724e6e7c8cc9319e71f3303eaddf` | MaleVis (Amendment 1, primary MaleVis result) |
| `code/auditx_final_v4.py` | `ef76bd1e6704…` | Malimg, Dumpware10, original MaleVis run |
| `code/auditx_final.py` | `611bb69d73be…` | MaleVis (Amendment 1) |

The two training scripts differ in one line: for MaleVis, Amendment 1 keeps clusters of images with an identical
256-bit hash in one partition. The manifests do not record the hash of the evaluation module that the training script
imports, but the run notebooks embed every file they use: FINAL_B (Malimg), FINAL_C (Dumpware10) and FINAL_A
(original MaleVis) contain exactly `auditx_final_v4.py` and `auditx_eval_v2_v4.py`, and FINAL_A2 (MaleVis,
Amendment 1) contains exactly `auditx_final.py`, `auditx_eval_v2.py` and both protocol documents.

## Reproduce the splits without any images

```bash
pip install scikit-learn==1.6.1 numpy pandas pillow
python tools/regenerate_splits.py MaleVis
python tools/regenerate_splits.py Dumpware10
```

The splits depend only on the image hashes, the family labels and the ImageFolder order, all published in
`splits/<dataset>/images.csv`. scikit-learn 1.6.1 is required, because `StratifiedGroupKFold` assigns groups to folds
differently in other versions. With 1.6.1, the regenerated splits of the original MaleVis run match the split files
that run saved, and the MaleVis and Dumpware10 splits match the calibration-set sizes and held-out families recorded
by the final runs. See `splits/README.md`; the Malimg split files are added separately (`splits/Malimg/README.md`).

## Datasets

| Dataset | Source | Images in the copy used | Images used |
|---|---|---|---|
| Malimg | Nataraj et al., VizSec 2011 | 9,338 | 8,290 (Yuner.A, Autorun.K, Obfuscator.AD excluded) |
| MaleVis | Bozkır et al., SIU 2019 (Hacettepe University) | 14,226 | 13,726 (VBA excluded, Amendment 1) |
| Dumpware10 | Bozkır et al., Computers & Security 2021 | 4,294 | 4,294 |

Excluded families consist of images that fall into one (near-)identical duplicate group and cannot be split. To map
the published rows to the files of your own copy:

```bash
python tools/match_local_copy.py MaleVis /path/to/MaleVis
```

## Re-running the experiments

The final experiments ran as Kaggle notebooks on a Tesla T4 GPU (`notebooks/final/`, guides in `docs/`). Each
notebook writes the protocol and scripts to disk with `%%writefile` cells (byte-identical to the files in `protocol/`
and `code/`), trains all models for one dataset under the protocol, and writes a manifest with the protocol and
script hashes. To re-run, attach the dataset in Kaggle and commit the notebook with a T4 GPU. Expected run times are in `docs/RUN_GUIDE.md`. Analysis notebooks
(FINAL_D to FINAL_H) run on CPU and produce the folders in `results/`.

```bash
pip install -r requirements.txt
```

## Results

`results/README.md` lists which notebook produced each folder. The primary analysis is `results/primary_analysis/`
(`VERDICTS.txt` gives the six confirmatory comparisons); the original MaleVis run, before Amendment 1, is kept as a
sensitivity analysis.

## Literature audit

`audit/` contains the audit of reporting practice in malware-image studies: the coding protocol
(`AUDIT_PROTOCOL.md`), the coding manual, the search log, the external sampling frame (`STRATUM_B_FRAME.csv`) and the
agreement and tally scripts. The coded sheets of both coders, with a quotation for every code, are added once
independent second coding is complete.

## Licence

Code: MIT (`LICENSE`). Protocol, splits, results and audit: CC BY 4.0 (`LICENSE-DATA.md`). The datasets themselves
remain under their original terms.

## Citation

See `CITATION.cff`. Please cite the paper once published.
