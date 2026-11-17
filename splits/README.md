# Splits and image hashes

One folder per dataset. No images are included; rows follow torchvision `ImageFolder` order (family folders sorted
by name, files sorted by name within each family).

## `images.csv`

| Column | Meaning |
|---|---|
| `index` | Position of the image in ImageFolder order; the training code uses these indices |
| `family` | Family folder name (`Other` is the benign class of MaleVis and Dumpware10) |
| `file_md5` | MD5 of the image file |
| `pixel_md5` | MD5 of the decoded grey-level pixels and image size |
| `dhash256` | 256-bit difference hash (17×16), hexadecimal |
| `duplicate_group` | Duplicate group under the protocol rule; every group lies in one partition |
| `family_excluded` | 1 if the family cannot be split and is excluded |
| `split0` … `split4` | Partition in each of the five splits: `train`, `val`, `calib`, `test`, `unseen` or `excluded` |

`train` (70%) fits the models, `val` (10%) selects the epoch, `calib` (10%) sets the alarm thresholds and `test` (10%)
measures the false-alarm rate; `unseen` holds every image of the families withheld in that split.

`splits_summary.json` gives, per split, the withheld families and partition sizes.

## How these files were produced and checked

**MaleVis and Dumpware10** were regenerated from the image hashes with the exact code each run used
(`tools/regenerate_splits.py`, scikit-learn 1.6.1), then checked against the final runs. For every split the
calibration-set size and the withheld families agree with `results/primary_analysis/` (`runs_long.csv`,
`T_families.csv`), and no duplicate group is cut across partitions. As a test of the method, the same procedure
reproduced, index for index, all five split files saved by the original MaleVis run.

**Malimg** is added from the split files saved by the final Malimg run (see `Malimg/README.md`). Regenerating it from
the hash file available to us reproduced the withheld families but not the partition sizes, so we publish the run's own
files instead.
