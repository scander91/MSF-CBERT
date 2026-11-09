"""Structured-attribute verbalization appended to the event summary.

# TEMPLATES are the published phrasing. The label is NEVER part of the input:
# guarded by an assertion at build time.
"""
import re

TEMPLATES = {
    "region_txt":   "The attack occurred in the {} region.",
    "country_txt":  "The country was {}.",
    "provstate":    "The province or state was {}.",
    "city":         "The city was {}.",
    "gname":        "The perpetrator group was {}.",
    "targtype1_txt": "The target type was {}.",
    "weaptype1_txt": "The weapon type was {}.",
}


_ABBREV = re.compile(r"(?:\b[A-Z]|\b(?:St|Mr|Mrs|Ms|Dr|Gen|Lt|Col|Maj|Capt|Sgt|Rev|Sen|Rep|Gov|Pres|No|Mt|Ft|Jr|Sr|Prof|Inc|Co|Corp|Dept|approx|vs))\.$")


def first_sentence_text(text):
    """Keep the date stamp and the first sentence of a GTD summary (row-local, deterministic)."""
    m = re.match(r"^(\d{2}/\d{2}/\d{4}:\s*)?(.*)$", str(text), flags=re.S)
    stamp, body = m.group(1) or "", m.group(2)
    for e in re.finditer(r"[.!?](?=\s+[\"'(]?[A-Z0-9])", body):
        if not _ABBREV.search(body[:e.end()]):
            return (stamp + body[:e.end()]).strip()
    return (stamp + body).strip()


def verbalize_row(row, selected_features):
    parts = []
    for col in selected_features:
        val = row.get(col)
        if val is None or (isinstance(val, float)) or str(val).strip() in ("", "nan", "Unknown"):
            continue
        parts.append(TEMPLATES.get(col, col + ": {}.").format(val))
    return " ".join(parts)


def build_input_text(row, selected_features, use_structured=True):
    txt = str(row["text"])
    if use_structured:
        v = verbalize_row(row, selected_features)
        assert str(row["label"]) not in v, "label string leaked into verbalized input"
        txt = (txt + " " + v).strip()
    return txt


# Numeric attributes rendered as text (routing-isolation arm, 2026-09-22). Same twelve
# Tier-B-kept numerics as the dense branch, same train-fit medians for missing values, so
# the arm differs from `structdense` ONLY in how the numbers reach the model (text tokens
# vs. a standardised dense pathway). Values are row-local raw GTD codes: counts as
# integers, binary flags as yes/no, GTD's -9 ("unknown") as unknown.
NUM_TEMPLATES = {
    "nkill":    ("{} killed.", "count"),
    "nwound":   ("{} wounded.", "count"),
    "success":  ("Attack success: {}.", "flag"),
    "suicide":  ("Suicide attack: {}.", "flag"),
    "extended": ("Extended incident: {}.", "flag"),
    "multiple": ("Part of multiple incidents: {}.", "flag"),
    "crit1":    ("Criterion 1: {}.", "flag"),
    "crit2":    ("Criterion 2: {}.", "flag"),
    "crit3":    ("Criterion 3: {}.", "flag"),
    "vicinity": ("In the vicinity: {}.", "flag"),
    "property": ("Property damage: {}.", "flag"),
    "int_any":  ("International: {}.", "flag"),
    "casualties": ("{} casualties.", "count"),
    "ishostkid": ("Hostages or kidnapping: {}.", "flag"),
}


def verbalize_numerics(row, num_cols, medians):
    parts = []
    for c in num_cols:
        v = row.get(c)
        try:
            v = float(medians[c]) if v is None or v != v else float(v)
        except (TypeError, ValueError):
            v = float(medians[c])
        tpl, kind = NUM_TEMPLATES.get(c, (c + ": {}.", "count"))
        if kind == "flag":
            s = "unknown" if v < 0 else ("yes" if v >= 0.5 else "no")
        else:
            s = str(int(round(v)))
        parts.append(tpl.format(s))
    return " ".join(parts)
