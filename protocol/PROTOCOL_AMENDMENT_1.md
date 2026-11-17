# Amendment 1 to PROTOCOL.md (FINAL-1) — MaleVis duplicate rule

Fixed before the amended MaleVis run. The run records the SHA-256 hash of this file in its MANIFEST.json.

## Reason
Under PROTOCOL.md, four MaleVis clusters of images with identical 256-bit difference hashes (VBA 500, Vilsel 294, Hlux 264,
Amonetize 260) exceeded the 250-image cap and were cut into pieces; in four of five splits pieces fell on both sides of the
train/evaluation boundary (11.7 % of known evaluation images with an identical-hash training image). This amendment removes
that failure mode for MaleVis only.

## Change (MaleVis only)
1. Images with an identical 256-bit difference hash are merged exactly, like byte- and pixel-identical images, and such
   clusters are never cut by the cap.
2. Everything else is unchanged: within-family near-duplicate merging at <= 8 bits with the 250-image cap, removal of
   label-conflict groups, exclusion of classes whose largest group exceeds 90 % of their images (expected: VBA, which consists
   of 500 identical-hash images), unknown-family draw, 70/10/10/10 stratified group split, models, three training seeds,
   learning-rate selection for the tuned AGEF-Net on split 0 / seed 0, scores, metrics, confirmatory comparisons C1–C6 and
   the statistical procedure.
3. Malimg and Dumpware10 are not re-run: under PROTOCOL.md their identical-hash residual is already 0.0 %.

## Expected consequences (computed from the image hashes before the run)
Identical-hash matches between training and evaluation images: 0.0 % in every split by construction; within 8 bits:
0–3.9 %. Families with few large groups (Snarasite: 6 groups; Hlux: 11 groups) are unevenly represented across
validation, calibration and test, and may be absent from some of these parts in some splits.

## Reporting
The amended MaleVis run is the primary MaleVis result. The MaleVis run under PROTOCOL.md and its deduplicated evaluation are
reported as sensitivity analyses. All six confirmatory comparisons are reported whatever their outcome.
