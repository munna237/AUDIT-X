# Results

| Folder | Produced by | Contents |
|---|---|---|
| `primary_analysis/` | `FINAL_D_Analysis_CPU` | Primary results: confirmatory comparisons (`T_confirmatory.csv`, `VERDICTS.txt`), all models and scores, shortcuts, operating points, variance, families, per-run results (`runs_long.csv`) |
| `dedup_analysis/` | `FINAL_E_Dedup_Analysis_CPU` | The same analysis after removing evaluation images with a duplicate in training |
| `review3_analyses/` | `FINAL_F_Review3_Analyses_CPU` | Homogeneous seed ensembles, seed bootstrap, heterogeneity, inflation against a pre-specified configuration, training curves, duplicate-rule sensitivity |
| `review4_analyses/` | `FINAL_G_Review4_Analyses_CPU` | Further sensitivity analyses (operating points, near twins, nested shortcuts, family jackknife, KNN bank size, learning-rate sweep) |
| `review5_analyses/` | `FINAL_H_Review5_Analyses_CPU` (+ `FINAL_X_*` extra models) | Similarity-based duplicate rule, taxonomy-aware hold-out, group calibration, additional fusion baselines |
| `sensitivity_original_malevis_run/` | `FINAL_D_Analysis_CPU` on the original MaleVis run | Results before Amendment 1, reported as a sensitivity analysis |
| `rq1_controls/` | RQ1/RQ3 control notebooks | Paired naive and cap-none comparisons, exact duplicate exposure, permutation nulls for E3, false alarms by split, group and family (RQ3) |
| `logs/` | Kaggle | Text logs of the training and analysis notebooks, including the recorded protocol and script hashes |

Every analysis summary begins by listing the runs it read and the protocol and script hashes of each.
