"""Compact, column-width version of the paper's row-normalised MSF-CBERT confusion matrix (pooled
test predictions of the ten published MSF-CBERT runs, read-only). Same data as the original figure;
larger type (7 pt labels before scaling, abbreviated class names), no support counts in tick labels, no colour bar (cell values printed).
Writes exp/results/figures/{pdf,png,svg}/fig_confusion_matrix_compact.* and data/*.csv."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PUB = pathlib.Path.home() / "projects" / "mtl_cbert_v2" / "runs"
FIG = pathlib.Path(__file__).resolve().parent / "results" / "figures"
SEEDS = [13, 21, 42, 87, 100, 7, 33, 55, 76, 91]
SHORT = {"Armed Assault": "Armed", "Assassination": "Assassin.", "Bombing/Explosion": "Bombing",
         "Facility/Infrastructure Attack": "Facility", "Hijacking": "Hijacking",
         "Hostage Taking (Barricade Incident)": "Barricade", "Hostage Taking (Kidnapping)": "Kidnapping",
         "Unarmed Assault": "Unarmed", "Unknown": "Unknown"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def main():
    labels = json.loads((PUB / "seed_42_structdense_nograph" / "metrics.json").read_text())["labels"]
    p = pd.concat([pd.read_csv(PUB / f"seed_{s}_structdense_nograph" / "preds_test.csv") for s in SEEDS])
    k = len(labels)
    cm = np.zeros((k, k)); np.add.at(cm, (p["true"].values, p["pred"].values), 1)
    r = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.5, 1.02))
    ax.imshow(r, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(k):
        for j in range(k):
            v = 100 * r[i, j]
            if round(v) >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                        color="white" if r[i, j] > 0.55 else "#1a1a1a", fontweight="bold" if i == j else "normal")
    names = [SHORT[l] for l in labels]
    ax.set_yticks(range(k), names, fontsize=7); ax.set_xticks(range(k), names, fontsize=7, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel("predicted class", fontsize=7.5, labelpad=1); ax.set_ylabel("true class", fontsize=7.5, labelpad=1)
    ax.tick_params(length=1.5, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    for sub in ("pdf", "png", "svg", "data"):
        (FIG / sub).mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / ext / f"fig_confusion_matrix_compact.{ext}")
    pd.DataFrame(r, index=labels, columns=labels).to_csv(FIG / "data" / "fig_confusion_matrix_compact.csv")
    print("support", cm.sum(1).astype(int).tolist(), "diag %", np.round(100 * np.diag(r), 1).tolist())


if __name__ == "__main__":
    main()
