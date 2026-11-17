# Final MaleVis run (Amendment 1) and re-analysis — step by step

## 0. Before the GPU run (10 minutes)
Deposit BOTH documents (OSF or Zenodo) and keep the DOIs/timestamps for the paper:
* PROTOCOL.md               sha256 aaa563ecc3c7ee8bcc59d66a33bcfa60cf8f5e7c98eccff60a966b3e4849853e  (already used by FINAL_A/B/C)
* PROTOCOL_AMENDMENT_1.md   sha256 9e25083cc4eae36aa1da15f9ca6ce0d0d8ed724e6e7c8cc9319e71f3303eaddf
Do not edit either file afterwards; the run records the amendment's hash.

## 1. GPU (one notebook, one session, ~8.5–9 h estimated)
FINAL_A2_MaleVis_Amendment1 — attach only `melevis`; GPU T4; Internet On; Save & Run All (Commit).
Check in the log: `protocol: grouped-L1-v3+identical-hash-whole+near-L3-8bit-cap250 | excluded: ['VBA']`, `SMOKE TEST PASSED`,
and at the end `COMPLETE`. If `INCOMPLETE`: attach this version as input and commit again.
Nothing else is retrained (Malimg and Dumpware10 keep FINAL_B and FINAL_C).

## 2. CPU re-analysis (attach FINAL_A2, FINAL_B, FINAL_C, and notebooks 08, 10, 14 for D)
* FINAL_D_Analysis_CPU          (primary analysis)          ~2–4 h
* FINAL_E_Dedup_Analysis_CPU    (deduplicated evaluation)   ~2–4 h
* FINAL_F_Review3_Analyses_CPU  (third-round analyses)      ~3–4 h
Each notebook has `MALEVIS = "v5"` in its run cell (uses FINAL_A2). The earlier FINAL_D/E outputs on the original MaleVis run
are kept as the sensitivity analysis; no need to re-run them.
Send: auditx_final_analysis/, auditx_final_analysis_dedup/, auditx_review3/.
