"""Steps 2-4: audit the table, compare one-measurement rules with five models, see how much data it takes, and find the flowers every model gets wrong.

    ./venv/bin/python analyze.py
        -> results/audit.csv  species_summary.csv  file_diff.csv  duplicates.csv  rule_full_fit.csv  folds.csv  cv_predictions.csv  fold_scores.csv  models.csv  single_measurement.csv
           file_effect.csv  learning_samples.csv  learning_curve.csv  learning_summary.csv  hard_flowers.csv  confusion.csv  lda_axes.csv  lda_summary.csv

The main table is bezdekIris.data (the two rows of iris.data that differ from Fisher's article are corrected); file_effect.csv repeats the comparison on iris.data.
Flowers are numbered by their line in the file (1 to 150); folds.csv and learning_samples.csv use row = line - 1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).parent
R = HERE / "results"
MEAS = ["sepal length", "sepal width", "petal length", "petal width"]
COLS = [m.replace(" ", "_") for m in MEAS]
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
RULES = [f"rule: {m}" for m in MEAS]
MODELS = ["LDA", "logistic regression", "5 nearest neighbours", "RBF SVM", "random forest"]
ALL = ["always setosa"] + RULES + MODELS
SPLITS, REPEATS = 5, 10
PER_CLASS = [5, 8, 12, 20, 30, 40]


def load(name):
    d = pd.read_csv(HERE / "data" / name, header=None, names=COLS + ["species"]).dropna(how="all")
    return d, d["species"].map({s: i for i, s in enumerate(SPECIES)}).to_numpy()


def fit_rule(x, y):
    """Two thresholds, three intervals, each interval predicts its majority species. Every pair of thresholds at the midpoints between neighbouring distinct values is tried; the fewest
    training errors wins, ties go to the smaller first threshold, then the smaller second, and a tied majority goes to the lower species number."""
    u = np.unique(x); mids = (u[:-1] + u[1:]) / 2
    assert len(mids) >= 2
    cum = np.array([[np.sum((x <= m) & (y == c)) for c in range(3)] for m in mids])        # flowers of each species at or below each midpoint
    tot = np.bincount(y, minlength=3); best = None
    for a in range(len(mids)):
        for b in range(a + 1, len(mids)):
            left, mid, right = cum[a], cum[b] - cum[a], tot - cum[b]
            correct = left.max() + mid.max() + right.max()
            if best is None or correct > best[0]:
                best = (int(correct), float(mids[a]), float(mids[b]), [int(left.argmax()), int(mid.argmax()), int(right.argmax())])
    return best[1], best[2], best[3], len(x) - best[0]


def predict_rule(x, t1, t2, labels):
    return np.where(x <= t1, labels[0], np.where(x <= t2, labels[1], labels[2]))


def build(name):
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    if name == "logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, tol=1e-10))
    if name == "5 nearest neighbours":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == "RBF SVM":
        return make_pipeline(StandardScaler(), SVC())
    return RandomForestClassifier(n_estimators=300, random_state=0)


def predict_all(X, y, tr, te, names):
    """predicted species (0 to 2) of the flowers `te` by each method named, fitted on the flowers `tr`"""
    out = {}
    for n in names:
        if n == "always setosa":
            out[n] = np.zeros(len(te), int)
        elif n.startswith("rule: "):
            c = COLS[MEAS.index(n[6:])]; t1, t2, lab, _ = fit_rule(X[c].to_numpy()[tr], y[tr]); out[n] = predict_rule(X[c].to_numpy()[te], t1, t2, lab)
        else:
            out[n] = build(n).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return out


def corrected(values, n_test_over_train):
    """Nadeau-Bengio: the 50 fold scores are not independent (the training sets overlap), so the variance of their mean is inflated by (1/J + n_test/n_train)."""
    v = np.asarray(values, float); J = len(v)
    se = np.sqrt((1 / J + n_test_over_train) * v.var(ddof=1)); t = stats.t.ppf(0.975, J - 1)
    return v.mean(), se, v.mean() - t * se, v.mean() + t * se, 2 * stats.t.sf(abs(v.mean() / se), J - 1) if se > 0 else float("nan")


def run_cv(X, y, splits):
    folds = pd.DataFrame([(i // SPLITS, i % SPLITS, int(r)) for i, (_, te) in enumerate(splits) for r in te], columns=["repeat", "fold", "row"])
    preds = folds.assign(y=y[folds["row"]]); cols = {n: np.zeros(len(folds), int) for n in ALL}
    for i, (tr, te) in enumerate(splits):
        idx = np.flatnonzero(((folds["repeat"] == i // SPLITS) & (folds["fold"] == i % SPLITS)).to_numpy())     # this fold's block of rows, in the order of `te`
        for n, p in predict_all(X, y, tr, te, ALL).items():
            cols[n][idx] = p
    for n in ALL:
        preds[n] = cols[n]
    return folds, preds


def fold_accuracy(preds):
    g = preds.assign(**{n: preds[n] == preds["y"] for n in ALL}).groupby(["repeat", "fold"])[ALL].mean().reset_index()
    return g.melt(id_vars=["repeat", "fold"], var_name="model", value_name="accuracy")


def main():
    R.mkdir(exist_ok=True)
    d, y = load("bezdekIris.data"); o, yo = load("iris.data"); X = d[COLS]; Xo = o[COLS]
    assert len(d) == len(o) == 150 and (y == yo).all()

    # Step 2: audit
    diff = [(i + 1, c, float(o[c].iloc[i]), float(d[c].iloc[i])) for i in range(150) for c in COLS if o[c].iloc[i] != d[c].iloc[i]]
    pd.DataFrame(diff, columns=["line", "measurement", "iris.data", "bezdekIris.data"]).to_csv(R / "file_diff.csv", index=False)
    g = d.groupby("species")
    summ = pd.concat({m: g[c].agg(["min", "mean", "max"]) for m, c in zip(MEAS, COLS)}, names=["measurement", "species"]).reset_index()
    summ.to_csv(R / "species_summary.csv", index=False)
    audit = [("flowers", len(d))] + [(s, int((d["species"] == s).sum())) for s in SPECIES] + [("missing cells", int(d.isna().sum().sum())), ("duplicate rows (bezdekIris.data)", int(d.duplicated().sum())),
             ("duplicate rows (iris.data)", int(o.duplicated().sum())), ("lines that differ between the two files", len({x[0] for x in diff})), ("measurements that differ", len(diff))]
    for c, m in (("petal_length", "petal length"), ("petal_width", "petal width")):
        se, ve, vi = (d.loc[d["species"] == s, c] for s in SPECIES)
        lo, hi = vi.min(), ve.max()                                           # the stretch where versicolor and virginica both occur
        audit += [(f"{m}: setosa maximum", se.max()), (f"{m}: versicolor minimum", ve.min()), (f"{m}: versicolor maximum", hi), (f"{m}: virginica minimum", lo), (f"{m}: virginica maximum", vi.max()),
                  (f"{m}: versicolor flowers in the overlap", int(((ve >= lo) & (ve <= hi)).sum())), (f"{m}: virginica flowers in the overlap", int(((vi >= lo) & (vi <= hi)).sum()))]
    pd.DataFrame(audit, columns=["check", "value"]).to_csv(R / "audit.csv", index=False)
    dup = []
    for name, t in (("iris.data", o), ("bezdekIris.data", d)):
        first = {}
        for i, row in enumerate(t.itertuples(index=False)):
            if row in first:
                dup.append({"file": name, "line": i + 1, "copy_of_line": first[row]})
            else:
                first[row] = i + 1
    pd.DataFrame(dup).to_csv(R / "duplicates.csv", index=False)
    full = []
    for m, c in zip(MEAS, COLS):
        t1, t2, lab, err = fit_rule(X[c].to_numpy(), y); full.append({"measurement": m, "t1": t1, "t2": t2, "low": SPECIES[lab[0]], "middle": SPECIES[lab[1]], "high": SPECIES[lab[2]], "errors_on_all_150": err})
    pd.DataFrame(full).to_csv(R / "rule_full_fit.csv", index=False)

    # Step 3: ten repeats of 5-fold cross-validation, the same folds for every method, on both copies of the table
    splits = list(RepeatedStratifiedKFold(n_splits=SPLITS, n_repeats=REPEATS, random_state=0).split(X, y))
    folds, preds = run_cv(X, y, splits); folds.to_csv(R / "folds.csv", index=False); preds.to_csv(R / "cv_predictions.csv", index=False)
    _, preds_o = run_cv(Xo, yo, splits)
    sc = fold_accuracy(preds); sc.to_csv(R / "fold_scores.csv", index=False); sc_o = fold_accuracy(preds_o)
    ratio = (150 // SPLITS) / (150 - 150 // SPLITS)
    mean = {n: 100 * sc.loc[sc["model"] == n, "accuracy"].mean() for n in ALL}; mean_o = {n: 100 * sc_o.loc[sc_o["model"] == n, "accuracy"].mean() for n in ALL}
    best = max(RULES, key=lambda n: mean[n])                                  # the best one-measurement rule, picked on the same folds (a little flattering to the rule)

    # Step 4: intervals, and each method against the best rule on the same folds
    def acc(n): return sc.loc[sc["model"] == n].sort_values(["repeat", "fold"])["accuracy"].to_numpy()
    rows = []
    for n in ["always setosa", best] + MODELS:
        m, _, lo, hi, _ = corrected(acc(n), ratio); dm, _, dlo, dhi, dp = corrected(acc(n) - acc(best), ratio)
        rows.append({"model": n, "accuracy_pct": 100 * m, "ci_lo": 100 * lo, "ci_hi": 100 * hi, "diff_vs_rule_pts": 100 * dm, "diff_lo": 100 * dlo, "diff_hi": 100 * dhi, "p_vs_rule": dp})
    pd.DataFrame(rows).round(4).to_csv(R / "models.csv", index=False)
    rows = []
    for n in RULES:
        m, _, lo, hi, _ = corrected(acc(n), ratio); rows.append({"model": n, "accuracy_pct": 100 * m, "ci_lo": 100 * lo, "ci_hi": 100 * hi})
    pd.DataFrame(rows).round(4).to_csv(R / "single_measurement.csv", index=False)
    pd.DataFrame([{"model": n, "corrected_file_pct": mean[n], "original_file_pct": mean_o[n], "change_pts": mean_o[n] - mean[n]} for n in ALL]).round(4).to_csv(R / "file_effect.csv", index=False)

    # how much data does it take? Train on a few flowers per species (drawn from each training fold), score on the test fold
    bc = COLS[MEAS.index(best[6:])]; samples, lc = [], []
    for i, (tr, te) in enumerate(splits):
        for k in PER_CLASS:
            rng = np.random.default_rng([i // SPLITS, i % SPLITS, k])
            sub = np.sort(np.concatenate([rng.choice(tr[y[tr] == c], k, replace=False) for c in range(3)]))
            samples.append({"repeat": i // SPLITS, "fold": i % SPLITS, "per_species": k, "rows": " ".join(map(str, sub))})
            for n, p in predict_all(X, y, sub, te, [best, "LDA", "logistic regression", "5 nearest neighbours"]).items():
                lc.append({"repeat": i // SPLITS, "fold": i % SPLITS, "per_species": k, "model": n, "accuracy": float((p == y[te]).mean())})
    pd.DataFrame(samples).to_csv(R / "learning_samples.csv", index=False); lc = pd.DataFrame(lc); lc.to_csv(R / "learning_curve.csv", index=False)
    ls = lc.groupby(["per_species", "model"])["accuracy"].agg(mean="mean", p10=lambda v: np.quantile(v, 0.1), p90=lambda v: np.quantile(v, 0.9)).reset_index()
    ls["flowers"] = 3 * ls["per_species"]; ls.round(4).to_csv(R / "learning_summary.csv", index=False)

    # the flowers the five models get wrong: 10 repeats x 5 models = 50 predictions per flower
    wrong = np.zeros(150, int); votes = np.zeros((150, 3), int)
    for n in MODELS:
        for r_, p_, y_ in zip(preds["row"], preds[n], preds["y"]):
            wrong[r_] += p_ != y_
            if p_ != y_:
                votes[r_, p_] += 1
    hard = pd.DataFrame({"line": np.arange(1, 151), "species": d["species"].to_numpy(), **{c: d[c].to_numpy() for c in COLS}, "wrong_of_50": wrong,
                         "usually_called": [SPECIES[int(votes[i].argmax())] if wrong[i] else "" for i in range(150)]})
    hard[hard["wrong_of_50"] > 0].sort_values(["wrong_of_50", "line"], ascending=[False, True]).to_csv(R / "hard_flowers.csv", index=False)

    conf = [{"model": n, "true": SPECIES[t], "predicted": SPECIES[q], "count": int(((preds["y"] == t) & (preds[n] == q)).sum())} for n in ALL for t in range(3) for q in range(3)]
    pd.DataFrame(conf).to_csv(R / "confusion.csv", index=False)       # pooled over the 10 repeats: 1,500 predictions per method

    # Fisher's linear discriminant on all 150 flowers
    lda = LinearDiscriminantAnalysis().fit(X, y); Z = lda.transform(X)
    pd.DataFrame({"line": np.arange(1, 151), "species": d["species"], "ld1": Z[:, 0], "ld2": Z[:, 1], "lda_wrong": (lda.predict(X) != y).astype(int)}).round(5).to_csv(R / "lda_axes.csv", index=False)
    pd.DataFrame([{"ld1_share": lda.explained_variance_ratio_[0], "ld2_share": lda.explained_variance_ratio_[1], "errors_on_all_150": int((lda.predict(X) != y).sum())}]).round(6).to_csv(R / "lda_summary.csv", index=False)

    for f in ("audit", "file_diff", "duplicates", "rule_full_fit", "models", "single_measurement", "file_effect", "learning_summary", "hard_flowers", "lda_summary"):
        print("==", f); print(pd.read_csv(R / f"{f}.csv").to_string(index=False))
    print("best one-measurement rule:", best)


if __name__ == "__main__":
    main()
