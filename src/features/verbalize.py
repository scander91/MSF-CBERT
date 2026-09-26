"""Structured-attribute verbalization appended to the event summary.

# TEMPLATES are the published phrasing. The label is NEVER part of the input:
# guarded by an assertion at build time.
"""
TEMPLATES = {
    "region_txt":   "The attack occurred in the {} region.",
    "country_txt":  "The country was {}.",
    "provstate":    "The province or state was {}.",
    "city":         "The city was {}.",
    "gname":        "The perpetrator group was {}.",
    "targtype1_txt": "The target type was {}.",
    "weaptype1_txt": "The weapon type was {}.",
}


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
