# Round 5: extra models (GPU) and fifth-round analyses (CPU)

Nothing existing is re-run; the new models use exactly the splits of the final runs.

## 1. GPU (three notebooks; independent, can run in parallel on two accounts)
| Notebook | Attach | Estimated time |
|---|---|---|
| FINAL_X_MaleVis_Extra_Models    | dataset `melevis` + output of FINAL_A2           | ~3-3.5 h |
| FINAL_X_Malimg_Extra_Models     | dataset `malimg-dataset-25` + output of FINAL_B  | ~2.5-3 h |
| FINAL_X_Dumpware10_Extra_Models | dataset `dumpware10` + output of FINAL_C         | ~1.5-2 h |
GPU T4, Internet On, Save & Run All (Commit). Import each file (File -> Import Notebook) instead of re-using an old notebook.
Check the log: `reusing the splits of: .../auditx_final_...`, `SMOKE TEST PASSED`, and at the end `COMPLETE`.
If `INCOMPLETE`: attach that version as input and commit again (it resumes).

## 2. CPU (after the three GPU runs)
FINAL_H_Review5_Analyses_CPU - attach outputs of FINAL_A2, FINAL_B, FINAL_C, the three FINAL_X notebooks, and the three
datasets. Accelerator None, ~3-4 h. The log must show three lines `... | extra models: .../auditx_extra_...`.
Send: auditx_review5/
