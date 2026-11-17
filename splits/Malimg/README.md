# Malimg splits: to be added from the final run

The final Malimg run (notebook `FINAL_B_Malimg`) saved its splits in its Kaggle output folder `auditx_final_malimg/`.
To add them:

1. Open the committed version of `FINAL_B_Malimg` on Kaggle → **Output** → folder `auditx_final_malimg`.
2. Download `image_keys.json`, `classes.json`, `targets.npy`, `groups_stats.json` and `split0/split.json` …
   `split4/split.json`, keeping the folder structure.
3. Run `python tools/import_run_splits.py Malimg /path/to/auditx_final_malimg`.

The tool writes `images.csv` and `splits_summary.json` here only if every split's calibration size and withheld
families agree with `results/primary_analysis/` and no exact duplicate crosses partitions. Then delete this README's
instructions or keep them as provenance.

Expected: 9,338 images, 8,290 used; Yuner.A, Autorun.K and Obfuscator.AD excluded; 4–5 withheld families per split
(mean 4.2).
