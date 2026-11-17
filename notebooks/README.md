# Notebooks

`final/`: the notebooks that produced every reported result. FINAL_A2 (MaleVis, Amendment 1), FINAL_B (Malimg) and
FINAL_C (Dumpware10) are the primary training runs; FINAL_A is the original MaleVis run (sensitivity analysis);
FINAL_X_* train additional fusion baselines; FINAL_D to FINAL_H are CPU analyses (see `results/README.md`).
Each training notebook writes the protocol and code to disk with `%%writefile` cells; `tools/verify_hashes.py` checks
that these are byte-identical to `protocol/` and `code/`.

`development/`: notebooks from before the final protocol, kept for transparency; no reported result depends on them.

`rq1_controls/`: the RQ1 controls added after the main analysis (naive splits, MaleVis cap-none, exact exposure), with
the committed Kaggle notebooks of the three naive runs (`run_rq1-naive-vs-grouped-<dataset>.ipynb`).
