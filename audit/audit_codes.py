"""Controlled code list for the literature audit, and the mapping of the first coder's free-text codes onto it."""
CODES = {
    "D2": ["removed", "partial check", "discussed only", "NR"],
    "D3": ["image-level", "family-level", "time-based", "NR"],
    "D4": ["yes", "no", "NR"],
    "D5": ["yes", "no", "unclear"],
    "D6": ["single", "best of several", "repeated mean only", "repeated mean+dispersion", "NR"],
    "D7": ["yes", "no"],
    "D8": ["training", "validation", "test", "NR", "NA"],
    "D9": ["yes", "no"],
}
FIRST_CODER_COLUMNS = {"D2": "D2_duplicates", "D3": "D3_split_unit", "D4": "D4_separate_validation",
                       "D5": "D5_selection_on_test", "D6": "D6_repeated_runs", "D7": "D7_unseen_family_eval",
                       "D8": "D8_threshold_source", "D9": "D9_code_released"}


def from_first_coder(item, text):
    """Map one free-text code of AUDIT_SHEET.csv to the controlled list."""
    v = text.strip().lower()
    if item == "D2":
        return "removed" if v.startswith("removed") else "partial check" if v.startswith("checked") else \
               "discussed only" if v.startswith("discussed") else "NR"
    if item == "D3":
        if v.startswith("nr"): return "NR"
        if "family-level" in v or "class-level" in v or "disjoint famil" in v: return "family-level"
        if "time" in v[:12]: return "time-based"
        return "image-level"
    if item == "D4":
        return "yes" if v.startswith("yes") else "no" if v.startswith("no") else "NR"
    if item == "D5":
        return "yes" if v.startswith("yes") else "no" if v.startswith("no") else "unclear"
    if item == "D6":
        if v.startswith("single"): return "single"
        if v.startswith("best"): return "best of several"
        if "range" in v or "sd" in v.split() or "dispersion" in v and "no dispersion" not in v: return "repeated mean+dispersion"
        return "repeated mean only"
    if item == "D7":
        return "yes" if v.startswith("yes") else "no"
    if item == "D8":
        return "training" if v.startswith("training") else "validation" if v.startswith("validation") else \
               "test" if v.startswith("test") else "NA" if v.startswith("na") else "NR"
    if item == "D9":
        return "yes" if v.startswith("yes") else "no"


def normalise(item, value):
    """Accept the second coder's value if it is on the list (case-insensitive); otherwise return it marked invalid."""
    for c in CODES[item]:
        if value.strip().lower() == c.lower(): return c
    return f"INVALID:{value.strip()}"
