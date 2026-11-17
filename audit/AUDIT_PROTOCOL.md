# Structured audit of evaluation practices in image-based malware detection

Fixed before any paper is coded. Changes after coding starts are logged at the end of this file.

## Question
How often does published image-based malware detection work use evaluation practices that the manuscript shows can inflate
results: no duplicate control, image-level random splits, selection or early stopping on the test set, best-run reporting,
and (for open-set or zero-day work) thresholds set on training data or on the test unknowns?

## Inclusion criteria
* Peer-reviewed article or arXiv preprint, 2011-2026, in English.
* Deep-learning or machine-learning classification or detection of malware represented as images
  (byte plots, grey-scale or RGB conversions, memory-dump images).
* Evaluates on at least one of Malimg, MaleVis, Dumpware10, or another public malware-image dataset
  (e.g., BIG 2015 converted to images), OR claims detection of unseen / zero-day / unknown malware from images.
* Full text accessible (open access, arXiv or author copy), so that every code can be traced to the text.

## Exclusions
* Surveys and reviews without their own experiments (used only to find primary studies).
* Papers whose experiments do not use image representations.
* Duplicate versions of the same study (the most complete version is coded).

## Search
Purposive, structured search with web search engines and publisher sites, using the queries listed in SEARCH_LOG.md,
plus references of included papers. This is a structured sample, not a systematic review; the paper must say so.
Target: 25-30 included papers, with no restriction by venue or result.

## Coding items (each with the supporting quote or location; "NR" = not reported)
| Item | Codes |
|---|---|
| D1 Datasets | names |
| D2 Duplicate handling | removed / discussed but not removed / NR |
| D3 Split unit | image-level random / family-level (open-set) / time-based / k-fold image-level / NR |
| D4 Separate validation set for model selection | yes / no (test used) / NR |
| D5 Selection or early stopping on test data | yes (stated) / no / unclear |
| D6 Repeated runs | mean +/- dispersion over seeds or folds / single run / best of several / NR |
| D7 Unseen-family evaluation | yes (family hold-out) / no |
| D8 Threshold source (only if D7 = yes or a rejection threshold is used) | training data / validation / test / NR |
| D9 Code or splits released | yes / no |

Coding rules: code only what the text states; anything ambiguous is "unclear" with the quote; no inference from reported
numbers. Each paper is coded independently of its results. Every row must be verified by a second coder (an author) before
the audit is reported; disagreements are resolved by discussion and logged.

## Reporting
Counts and percentages per item, with the coded table in the supplement and quotes in the audit sheet.

## Change log
- 2026-11-03 (after 13 papers coded, stratum A). Added stratum B: the 69-study PRISMA corpus of Lima & Gondim (2026, "Malware Classification from Binary Images Using Machine Learning: A Review", UnB), used as an externally selected sampling frame. Every member of that corpus whose full text can be opened is coded with the same items; members already coded in stratum A are counted once and flagged. Reason: removes selection by the auditing authors. Results are reported separately for stratum A (our structured search) and stratum B (external frame), and pooled.
- 2026-11-03. Access constraint: MDPI and PubMed Central pages block automated retrieval; such papers are listed for coding by an author rather than dropped.
- 2026-11-03 (after 25 papers coded by the first coder; before second coding). Code-list clarifications, no change to the items: D2 adds "partial check" for a duplicate or overlap check that does not cover the train/test boundary; D3 adds "grouped (duplicate-aware)" and "not stated"; D6 splits "mean +/- dispersion" into "repeated, mean with dispersion" and "repeated, mean only"; D8 adds "NA" when no rejection threshold is used. The second coder uses this code list (CODER_INSTRUCTIONS.md) without seeing the first coder's codes.
- 2026-11-05 (verification, before resolution). P13 (Ige et al. 2024, Preprints.org, not peer-reviewed) does not meet the
  inclusion criterion "peer-reviewed article or arXiv preprint" and is excluded; N = 24. Identifiers of the other studies unchanged.
