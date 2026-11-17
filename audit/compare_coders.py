"""Compare the second coder's sheet with the first coder's sheet. Run by Acram after Munna returns his sheet.

Usage: python compare_coders.py AUDIT_SHEET.csv 1_SECOND_CODING_SHEET.csv
Prints per-item agreement and Cohen's kappa, and writes DISAGREEMENTS.csv (one row per disagreeing item, with both
codes and both quotes) for the discussion. Agreed codes stand; disagreements are resolved by discussion and logged.
"""
import csv, sys, collections
import audit_codes as A

first = {r["id"]: r for r in csv.DictReader(open(sys.argv[1]))}
second = {r["id"]: r for r in csv.DictReader(open(sys.argv[2]))}
ids = [i for i in second if i in first and any(second[i][f"{it}_code"].strip() for it in A.CODES)]
missing = [i for i in first if i not in ids]
print(f"papers coded by both: {len(ids)}" + (f" | not yet second-coded: {missing}" if missing else ""))

def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)

out, invalid = [], []
print(f"{'item':4} {'agree':>9} {'kappa':>6}")
for it, col in A.FIRST_CODER_COLUMNS.items():
    a = [A.from_first_coder(it, first[i][col]) for i in ids]
    b = [A.normalise(it, second[i][f"{it}_code"]) for i in ids]
    invalid += [(i, it, v) for i, v in zip(ids, b) if v.startswith("INVALID")]
    agree = sum(x == y for x, y in zip(a, b))
    print(f"{it:4} {agree:>4}/{len(ids):<4} {kappa(a, b):6.2f}")
    for i, x, y in zip(ids, a, b):
        if x != y:
            out.append(dict(id=i, item=it, first_code=x, second_code=y, first_evidence=first[i]["evidence"],
                            second_quote=second[i][f"{it}_quote"], resolved_code="", resolution_note=""))
with open("DISAGREEMENTS.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "item", "first_code", "second_code", "first_evidence", "second_quote",
                                      "resolved_code", "resolution_note"], lineterminator="\n")
    w.writeheader(); w.writerows(out)
print(f"{len(out)} disagreements written to DISAGREEMENTS.csv")
if invalid: print("codes not on the list (fix before comparing):", invalid)
