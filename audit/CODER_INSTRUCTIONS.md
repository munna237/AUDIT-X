# Coder instructions: audit of evaluation practice in image-based malware detection

For: Munna (second coder). From: Acram. Version 2026-11-03.

## 1. Why we are doing this

A reviewer asked whether the evaluation shortcuts our paper measures (no duplicate control, selection on test
data, best-run reporting, thresholds set on training data) actually occur in published malware-image papers. We
answer with an audit: we read published papers and record, item by item, what each one **reports** about its
evaluation. One coder (the first coder) has coded 25 papers. The protocol requires that a second person codes every
paper independently before the counts can be published. That is your job.

## 2. Your two tasks

| Task | Sheet | Papers | Time |
|---|---|---|---|
| **A. Second coding** (do this first) | `1_SECOND_CODING_SHEET.csv` | the 25 papers already coded by the first coder | about 6–8 hours |
| **B. New papers** | `2_NEW_PAPERS_SHEET.csv` | papers the first coder could not open (paywalled) | about 20–30 minutes each |

For task B, do the **priority 1** rows first (Gibert et al. 2019 and review references 17, 55, 59), then priority 2.
Ten to fifteen new papers is enough; you do not need to finish the list.

## 3. The most important rule: code blind

Do **not** look at the first coder's sheet (`AUDIT_SHEET.csv`), the draft manuscript section, or the audit summary
before you have finished task A. If you have already seen them, tell Acram, and still code from the paper alone.
Do not discuss individual papers with the first coder until both of you have finished. Independent coding is what
makes the agreement statistic meaningful.

## 4. General coding rules

1. **Code only what the paper states.** Never infer from the numbers, the figures, or what authors "probably" did.
2. **Every code needs a quote.** Copy the exact words from the paper into the `_quote` column, with the section,
   table or page, for example: `"We randomly split the dataset into 80% training and 20% testing" (Sec. 4.1)`.
   Keep quotes short (one sentence is usually enough). If the code is "NR" (not reported), write what you searched,
   for example `no mention of duplicates in Sec. 3-5 or tables`.
3. **If it is ambiguous, use the ambiguous code** (`unclear` or `NR`) and quote the ambiguous sentence. Do not guess.
4. **Use only the codes in Section 5**, spelled exactly as listed. The comparison script rejects anything else.
5. **Read the whole method and results sections**, plus tables, figure captions and any supplement or appendix.
   Search the PDF for these words: duplicate, identical, hash, dedup, split, random, stratif, fold, validation,
   test set, hyperparameter, tun, grid, best, run, seed, repeat, average, mean, std, unseen, unknown, zero-day,
   novel, open-set, threshold, github, code, available.
6. **Several experiments in one paper:** code the most explicit description that applies to any reported experiment,
   and mention the others in the quote. Example: a paper with a closed-set experiment and a separate family hold-out
   experiment gets D7 = `yes`.
7. **One paper = one row.** If the same study exists as a preprint and a journal version, code the most complete
   version and note the other in `notes`.
8. Record the minutes you spent in `minutes`, and anything odd (contradictions in the paper, missing pages) in `notes`.

## 5. The items and their codes

### D1 Datasets (free text)
List every dataset the paper evaluates on, with the number of images if stated, for example
`Malimg (9,339), MaleVis (14,226)`. Note if a dataset is the authors' own.

### D2 Duplicate handling
Does the paper deal with duplicate or near-duplicate images (identical files, identical pixels, near-identical
variants)?

| Code | Use when |
|---|---|
| `removed` | Duplicates are removed, or grouped so that copies cannot be on both sides of the train/test split |
| `partial check` | Some duplicate or overlap check is reported, but it does not cover the train/test boundary of the main experiments (e.g. only checks a separate test set, or groups images only to pick training examples) |
| `discussed only` | Duplicates are mentioned as an issue but nothing is done about them in the splits |
| `NR` | Nothing about duplicates |

Note: removing duplicate *files* when downloading or "cleaning" data counts only if the text says identical images
were removed. Data augmentation is not duplicate handling.

### D3 Split unit
What is the unit that is assigned to training or testing?

| Code | Use when |
|---|---|
| `image-level` | The text says images (samples, files) are split randomly, stratified by class, by k-fold, or in a fixed order within each class |
| `family-level` | At least one experiment holds out whole families (classes) from training |
| `time-based` | Training data come from an earlier period than test data |
| `NR` | Only ratios are given (e.g. "80/20") without saying what was split, or the paper uses a predefined dataset split without describing it, or no split is described |

Be strict: "we split the data 80/20" is `NR`, because it does not say whether images, files or families were split.
"We randomly selected 80% of the images of each class" is `image-level`.

### D4 Separate validation set for model selection
| Code | Use when |
|---|---|
| `yes` | A validation set (or cross-validation inside the training data) separate from the test data is described |
| `no` | Only training and test data are described, or the set called "validation" is the one whose results are reported as test results |
| `NR` | No split is described at all |

### D5 Selection or tuning on test data
Were hyperparameters, epochs, thresholds, architectures, loss weights or other choices selected using the data on
which results are reported?

| Code | Use when |
|---|---|
| `yes` | The text states, or a table shows, that a choice was made from results on the test (or reported) data, e.g. "we chose the learning rate that gave the best test accuracy", or a tuning table reports test accuracy per setting and the best setting is used |
| `no` | The text states that choices were made on validation or training data only |
| `unclear` | Choices were tuned but the data used for tuning are not stated, or tuning is not described at all |

Note: "we trained for 50 epochs" with no selection is not selection on test data; if the paper says nothing about how
anything was chosen, code `unclear` and quote what it says about training settings.

### D6 Repeated runs
How many training runs stand behind the reported numbers?

| Code | Use when |
|---|---|
| `single` | One run per configuration, or the number of runs is not stated |
| `best of several` | Several runs, splits or seeds were tried and the best one is reported |
| `repeated mean only` | A mean over runs, folds or seeds is reported without a standard deviation, range or interval |
| `repeated mean+dispersion` | A mean with standard deviation, range or confidence interval over runs, folds or seeds |
| `NR` | Use only if the paper reports no quantitative results at all |

Cross-validation counts as repeated runs only if results are averaged over folds. Bootstrap intervals over test
samples of one trained model do not count as repeated runs.

### D7 Unseen-family evaluation
| Code | Use when |
|---|---|
| `yes` | At least one experiment tests on families (classes) that were not in training |
| `no` | All test families also appear in training, even if the paper says "unseen samples" or "zero-day" |

If the paper **claims** unseen, zero-day or unknown detection (title, abstract or introduction) but D7 = `no`, write
`claims unseen detection without family hold-out` in `notes`, with the quote of the claim.

### D8 Threshold source (only when D7 = `yes` or a rejection threshold is used)
| Code | Use when |
|---|---|
| `training` | The threshold for flagging unknown or rejecting samples is set on training data |
| `validation` | Set on a held-out validation or calibration set of known families |
| `test` | Set on the test data or on the unknown families themselves |
| `NR` | A threshold is used but its source is not stated |
| `NA` | No threshold is used (e.g. only AUROC is reported, or a plain closed-set classifier) |

### D9 Code or splits released
| Code | Use when |
|---|---|
| `yes` | A working public link to code or split files is given (GitHub, Zenodo, etc.) |
| `no` | No link, "available on request", or a link to the dataset only |

## 6. Task B: extra rules for new papers

1. Find the full text with university access (IEEE Xplore, ScienceDirect, SpringerLink, ACM) or an author copy.
   Put the DOI or link in `url_or_doi`, and `yes` or `no` in `full_text_found`. If you cannot find it, leave the
   codes empty and write `not accessible` in `notes`.
2. Check inclusion before coding. Include only if all hold:
   - peer-reviewed paper or arXiv preprint, 2011–2026, in English;
   - deep-learning or machine-learning classification or detection of malware **represented as images**
     (byte plots, grey-scale or RGB conversions, memory-dump images, entropy images used as 2-D inputs);
   - evaluates on a public malware-image dataset (Malimg, MaleVis, Dumpware10, BIG 2015 converted to images, MalNet,
     etc.) **or** claims detection of unseen, zero-day or unknown malware from images.
   Exclude surveys without their own experiments, papers without image representations, and duplicate versions.
   Write `yes` or `no` in `included`, and the reason in `exclusion_reason` if `no`.
3. Give each included paper the next ID, starting at **P26**, in the `id` column.
4. The `frame_ref` column gives the reference number in the Lima & Gondim (2026) review, which lists the full
   reference; the `citation` column is only a short label. For stratum A rows marked "verify title", confirm the
   exact title from the publisher page.
5. Code exactly as in Sections 4–5.

## 7. When you are done

Send both CSV files to Acram (keep the file names). Do not edit any other file. Acram runs
`python compare_coders.py AUDIT_SHEET.csv 1_SECOND_CODING_SHEET.csv`, which reports agreement per item and writes
`DISAGREEMENTS.csv`. You and the first coder then go through every disagreement together, agree a final code with
the reason, and record it in `resolved_code` and `resolution_note`. Both coders' original codes are kept; nothing
is overwritten.

## 8. Using an AI assistant (allowed, with limits)

You may use an AI assistant to help you find passages in a paper, but **you are the coder**: the protocol requires
that every row is verified by an author. Rules:

- Give the assistant only this file and one paper at a time. Do **not** give it `AUDIT_SHEET.csv`, the manuscript, or
  any of the first coder's codes.
- Check every quote it gives you against the PDF, word for word. AI assistants sometimes produce quotes that are not
  in the paper. If you cannot find the quote in the PDF, do not use it.
- Make the final decision on every code yourself, using Section 5, and write `AI-assisted` in `notes` for that paper.

Prompt to paste, followed by the paper's full text or PDF:

```
You are helping a human researcher code one published paper for a literature audit. Use only the coding manual
below and the paper I provide. For each item D1 to D9:
1. give the code, chosen strictly from the allowed codes in the manual;
2. give the exact supporting sentence(s) copied word for word from the paper, with section, table or page;
3. if the paper does not state it, give the code for "not reported" or "unclear" and list the sections you checked.
Rules: code only what the paper states; never infer from reported numbers; when ambiguous, choose the ambiguous
code; never paraphrase inside a quote; if you are not sure a sentence is in the paper, say so instead of quoting.
Also report: whether the paper meets the inclusion criteria (Section 6), and whether it claims unseen or zero-day
detection without a family hold-out (D7 note).
Output a table with columns: item | code | quote | location | confidence (high/medium/low).

[paste Sections 4, 5 and 6 of CODER_INSTRUCTIONS.md here]
[paste or attach the paper here]
```

## 9. Files in this package

| File | Purpose |
|---|---|
| `CODER_INSTRUCTIONS.md` | This document |
| `AUDIT_PROTOCOL.md` | The audit protocol fixed before coding, with its change log |
| `1_SECOND_CODING_SHEET.csv` | Task A: the 25 papers, codes empty |
| `2_NEW_PAPERS_SHEET.csv` | Task B: papers to find and code |
| `audit_codes.py`, `compare_coders.py`, `tally.py` | Scripts Acram runs afterwards (you do not need them) |

Questions about a rule: ask Acram, and write the answer in `notes` so we can add it to the change log.
