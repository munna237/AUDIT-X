# AUDIT-X final run — step by step

## 0. Before any GPU time (10 minutes)
1. Upload `PROTOCOL.md` to OSF (osf.io, "Registrations") or to a public GitHub repository. Keep the link: it is the
   timestamp that the analysis plan existed before the results. Do not edit PROTOCOL.md afterwards; every run records its SHA-256.
2. Import the notebooks into Kaggle. Each person uses only their own Kaggle account.

## 1. GPU runs (can start in parallel)
| Notebook | Who | Attach (only this) | Expected |
|---|---|---|---|
| FINAL_A_MaleVis | Acram | melevis | ~19 h, two sessions |
| FINAL_B_Malimg | Munna | malimg-dataset-25 | ~9–10 h, one session |
| FINAL_C_Dumpware10 | Munna | dumpware10 | ~7 h, one session |

For each: Settings → GPU T4, Internet On → Save Version → Save & Run All (Commit).
* The notebook runs a 5-minute smoke test first and stops by itself if anything is wrong.
* The log ends with `COMPLETE` or `INCOMPLETE`.
* If `INCOMPLETE` (time limit): open a new version, Add Input → *this notebook's previous version*, commit again.
  Finished runs are copied and skipped. Repeat until `COMPLETE`.
* Munna shares his two notebooks with Acram (Share → add collaborator) so their outputs can be attached.

## 2. Analysis (CPU)
FINAL_D_Analysis_CPU: attach the committed outputs of FINAL_A, FINAL_B, FINAL_C (all COMPLETE) and of the earlier notebooks
08, 10, 14 (cross-representation, few-shot). Accelerator None → commit. Expected 2–4 h.
Check the first lines of the log: "protocol document identical across all runs". Then send
`auditx_final_analysis/` (all CSV files and VERDICTS.txt) for the manuscript update.

## Expected exclusions (not errors)
Malimg: Yuner.A, Autorun.K and Obfuscator.AD are excluded (near-identical image families that cannot be split).
