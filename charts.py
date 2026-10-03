"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_scatter.png, 02_models.png, 03_learning.png, 04_lda.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
COL = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
SPC = dict(zip(SPECIES, (BLUE, ORANGE, GREEN)))
SHORT = {s: s.replace("Iris-", "") for s in SPECIES}
NAMES = {"always setosa": "always setosa", "rule: petal width": "petal-width rule", "LDA": "LDA (Fisher's method)", "logistic regression": "logistic regression",
         "5 nearest neighbours": "5 nearest neighbours", "RBF SVM": "RBF SVM", "random forest": "random forest"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def scatter(ax):
    d = pd.read_csv(HERE / "data" / "bezdekIris.data", header=None, names=COL + ["species"]).dropna(how="all")
    hard = pd.read_csv(R / "hard_flowers.csv").query("wrong_of_50 >= 20")["line"].to_numpy() - 1
    for s in SPECIES:
        t = d[d["species"] == s]; ax.scatter(t["petal_length"], t["petal_width"], s=34, color=SPC[s], alpha=0.75, linewidths=0, label=SHORT[s])
    ax.scatter(d["petal_length"].iloc[hard], d["petal_width"].iloc[hard], s=150, facecolors="none", edgecolors=INK, linewidths=1.2, label="7 flowers most often misclassified")
    ax.set_xlabel("petal length (cm)"); ax.set_ylabel("petal width (cm)"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    ax.set_title("Petal length and width: setosa stands apart, the other two overlap")


def models(ax):
    m = pd.read_csv(R / "models.csv").iloc[::-1]; y = np.arange(len(m)); rule = float(m.loc[m["model"] == "rule: petal width", "accuracy_pct"].iloc[0])
    ax.barh(y, m["accuracy_pct"], color=[GRAY if mm == "always setosa" else ORANGE if mm.startswith("rule") else BLUE for mm in m["model"]], height=0.65)
    hi = np.minimum(m["ci_hi"], 100)                                     # an interval can run past 100%; it is cut there
    ax.errorbar(m["accuracy_pct"], y, xerr=[m["accuracy_pct"] - m["ci_lo"], hi - m["accuracy_pct"]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
    for yi, (_, r) in zip(y, m.iterrows()):
        ax.text(1.5, yi, f"{r['accuracy_pct']:.1f}%", va="center", color=BG, fontsize=10)
    ax.axvline(rule, color=ORANGE, linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[n] for n in m["model"]]); ax.set_xlim(0, 100); ax.set_xlabel("accuracy, 10 x 5-fold cross-validation, with 95% intervals (cut at 100%)"); ax.grid(axis="y", visible=False)
    ax.set_title("One measurement against five models")


def learning(ax):
    s = pd.read_csv(R / "learning_summary.csv"); colors = {"LDA": GREEN, "rule: petal width": ORANGE, "logistic regression": BLUE, "5 nearest neighbours": GRAY}
    for n, c in colors.items():
        t = s[s["model"] == n].sort_values("flowers"); ax.plot(t["flowers"], 100 * t["mean"], color=c, linewidth=2.2, marker="o", markersize=5, label=NAMES[n])
    flowers = sorted(s["flowers"].unique()); ax.set_xscale("log"); ax.set_xticks(flowers); ax.set_xticklabels([str(f) for f in flowers]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(84, 100); ax.set_xlabel("flowers used for training (a third of each species), log scale"); ax.set_ylabel("accuracy on held-out flowers (%)")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower right"); ax.set_title("How many flowers does it take?")


def lda(ax):
    z = pd.read_csv(R / "lda_axes.csv"); sh = pd.read_csv(R / "lda_summary.csv").iloc[0]; rng = np.random.default_rng(0)
    sign = -1 if z.loc[z["species"] == SPECIES[0], "ld1"].mean() > z.loc[z["species"] == SPECIES[2], "ld1"].mean() else 1      # the axis has no preferred direction: put setosa on the left
    z["x"] = sign * z["ld1"]; z["y"] = z["species"].map({s: i for i, s in enumerate(SPECIES)}) + rng.uniform(-0.2, 0.2, len(z))
    for s in SPECIES:
        t = z[z["species"] == s]; ax.scatter(t["x"], t["y"], s=30, color=SPC[s], alpha=0.8, linewidths=0)
    w = z[z["lda_wrong"] == 1]; ax.scatter(w["x"], w["y"], s=150, facecolors="none", edgecolors=INK, linewidths=1.2)
    ax.legend([Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK, markersize=9)], ["flowers the discriminant gets wrong, fitted on all 150"], frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    ax.set_yticks(range(3)); ax.set_yticklabels([SHORT[s] for s in SPECIES]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"first discriminant axis: {100 * sh['ld1_share']:.1f}% of the between-species variance"); ax.set_title("Fisher's discriminant axis: setosa apart, the other two nearly")


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.15, 1]})
    scatter(a); learning(b); a.set_title("Setosa stands apart, the other two overlap")
    for ax in (a, b):
        ax.title.set_fontsize(15)
    fig.suptitle("Can four measurements name an iris? 150 flowers, three species", x=0.01, ha="left", fontsize=20, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.4)); scatter(ax); save(fig, "01_scatter.png")
    fig, ax = plt.subplots(figsize=(10, 3.8)); models(ax); save(fig, "02_models.png")
    fig, ax = plt.subplots(figsize=(10, 3.8)); learning(ax); save(fig, "03_learning.png")
    fig, ax = plt.subplots(figsize=(10, 3.4)); lda(ax); save(fig, "04_lda.png")


if __name__ == "__main__":
    main()
