"""Final counts for the manuscript from the agreed codes.

Usage: python tally.py AGREED_SHEET.csv
AGREED_SHEET.csv has columns id, stratum (A, B or A+B) and D2_code ... D9_code with values from audit_codes.CODES.
"""
import csv, sys, collections
import audit_codes as A

rows = list(csv.DictReader(open(sys.argv[1])))
groups = {"pooled": rows, "A": [r for r in rows if "A" in r["stratum"]], "B": [r for r in rows if "B" in r["stratum"]]}
for g, sub in groups.items():
    print(f"\n== {g} (N = {len(sub)})")
    for it in A.CODES:
        c = collections.Counter(A.normalise(it, r[f"{it}_code"]) for r in sub)
        print(f"  {it}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
d7 = [r for r in rows if A.normalise("D7", r["D7_code"]) == "yes"]
print("\nD8 among unseen-family studies:", dict(collections.Counter(A.normalise("D8", r["D8_code"]) for r in d7)))
