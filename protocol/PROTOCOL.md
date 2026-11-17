# Frozen evaluation protocol — unseen-family malware detection on image benchmarks

Version: FINAL-1. This document is fixed **before** the final training runs. Each run records the SHA-256 hash of this
file in its `MANIFEST.json`; the public upload time (OSF / GitHub) is the timestamp. Any deviation is reported as such.

## 1. Data and splits
* Datasets: Malimg (25 families, grayscale byte plots), MaleVis (25 families + benign, RGB byte plots),
  Dumpware10 (10 families + benign, memory-dump images; official TRAINING/TEST folders merged). Images resized to 224×224.
* Unseen-family setting: per split, 20 % of eligible classes are held out as unknown; benign classes are never held out.
  Five splits per dataset (seeds 0–4). This is a spatial open-set setting, not a temporal one.
* Duplicate-aware grouping (protocol `grouped-L1-v3+near-L3-8bit-cap250`): byte- or pixel-identical images form one group;
  within each family, images whose 256-bit difference hash (17×16) differs by ≤ 8 bits are merged, closest pairs first, and a
  merge is refused if the group would exceed 250 images. Groups never cross train / validation / calibration / test.
  The cap (250) is the smallest that keeps residual near-identical matches below 5 % on every dataset without making a class
  unsplittable; it was chosen from duplicate statistics only. Groups containing several families are removed; classes whose
  largest group exceeds 90 % of their images are excluded (expected: Malimg Yuner.A, Autorun.K, Obfuscator.AD).
* Known data split 70 / 10 / 10 / 10 into train / validation (epoch selection) / calibration (thresholds) / test (FAR, accuracy)
  with a stratified group split. Unknown-family images are used for evaluation only.

## 2. Models and training (identical recipe unless stated)
* Experts: ImageNet-pretrained MobileNetV3-Small, ResNet-18, EfficientNet-B0.
* Models: AGEF-Net (logit gate, τ = 1, diversity weight 0.05); **AGEF-Net-tuned** (gate on L2-normalised expert features,
  per-expert auxiliary cross-entropy with weight 0.5, diversity weight 0.05, learning rate chosen from {1e-4, 3e-4, 1e-3} by
  known-class validation accuracy, ties by validation NLL, on split 0 / training seed 0 only); average ensemble (jointly trained,
  uniform weights); three single CNNs; deep ensemble = mean softmax of the three independently trained singles.
* AdamW (lr 1e-3 unless tuned, weight decay 1e-4), cosine schedule, 12 epochs, batch 32, mixed precision, no augmentation;
  checkpoint with the best validation accuracy (ties: lower validation NLL).
* **Three training seeds per split** for every model (5 splits × 3 seeds × 6 models per dataset).
* Frozen DINOv2 ViT-S/14 features of every image (no training) as a generic-feature baseline.

## 3. Scores
Primary (confirmatory): **energy** (logit-based) and **KNN, k = 50** (feature-based, cosine, L2-normalised penultimate features).
Exploratory: MSP, entropy, MaxLogit, mutual information, gate entropy, KNN k ∈ {1, 5, 10}, Mahalanobis (shared Ledoit–Wolf
covariance, 512-d PCA), ViM (single CNNs and deep ensemble), ReAct + energy (90th-percentile clip), OpenMax,
Transcend-style credibility (class-conditional conformal p-value of the KNN distance), DINOv2 + KNN / Mahalanobis.

## 4. Metrics
AUROC (primary), AUPR, FPR at 95 % TPR; ZDR and FAR at thresholds set on the calibration split for 5 % and 1 % FAR with the
finite-sample conformal quantile; Clopper–Pearson intervals for FAR; a 1 % operating point is called *reliable* only if the
calibration set implies ≥ 10 expected exceedances. Results are averaged over the three training seeds.
No configuration is ever selected using unknown-family data.

## 5. Confirmatory comparisons (two-sided; Holm correction across this family; α = 0.05)
Inference: family-level cluster bootstrap (resample unknown families and known test images within dataset; all splits and
training seeds; 1,000 draws); per-dataset estimates with 95 % intervals; pooled estimate = mean of dataset estimates.
* C1  deep ensemble + energy  vs  AGEF-Net + energy
* C2  deep ensemble + KNN50   vs  AGEF-Net + KNN50
* C3  average ensemble + energy vs AGEF-Net + energy      — also TOST equivalence, margin ±0.03 AUROC (90 % interval)
* C4  AGEF-Net-tuned + energy vs  AGEF-Net + energy
* C5  AGEF-Net-tuned + energy vs  deep ensemble + energy
* C6  deep ensemble + energy  vs  best single CNN + energy, where "best" = highest known-class validation accuracy per split
Claims of superiority require the pooled interval to exclude 0 after Holm correction; dataset-level claims require the
dataset interval to exclude 0. "No advantage" is stated only as such; equivalence only if TOST holds.

## 6. Descriptive analyses (fixed in advance, no hypothesis test)
* Evaluation shortcuts: S1 thresholds on training data (reported vs held-out FAR); S3 best configuration on the same split's
  unknowns and S4 best configuration on its best split, both measured against the mean of the configuration pool, with the
  dependence on pool size (1–all configurations); a known-data-only rule (model with the highest calibration accuracy, energy).
* Variance decomposition of AUROC into split and training-seed components.
* Residual duplicates across partitions at five levels (file, pixel, identical 256-bit hash, ≤ 8 bits, 64-bit ≤ 2 bits).
* Per-family recall; early-exit compute for known and unknown inputs separately and measured GPU latency;
  per-class (Mondrian) calibration conditional on the predicted label; a short note on MSP/entropy four-tag triage.
* Cross-representation transfer and label-efficient adaptation reuse the completed runs (notebooks 08, 10, 14), reported as
  label efficiency of pretrained CNNs with the calibration-set size stated.

## 7. Reporting rules
All comparisons in §5 are reported whatever their outcome. Exploratory results are labelled exploratory. Malimg is reported
with its exclusions and degeneracy stated; Dumpware10 results are flagged as fragile (two unknown families per split).
