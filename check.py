"""Step 5: the project's check. Recompute with different code: the audit and the two files' differences from the raw text, the one-measurement rules by a plain-Python search on every
fold, every fold's accuracy, the corrected intervals from textbook formulas, Fisher's linear discriminant refitted by hand (class means, pooled covariance, a linear solve) on every
fold, on every learning-curve sample and on all 150 flowers, the share of the first discriminant axis from the eigenvalues of Sw^-1 Sb, and the hard-flower and confusion counts.
The other four models are not refitted: their saved predictions are the only thing recounted.

    ./venv/bin/python check.py
"""
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
R = HERE / "results"
MEAS = ["sepal length", "sepal width", "petal length", "petal width"]
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
RULES = [f"rule: {m}" for m in MEAS]
MODELS = ["LDA", "logistic regression", "5 nearest neighbours", "RBF SVM", "random forest"]
ALL = ["always setosa"] + RULES + MODELS
n_ok = 0


def same(label, got, want, tol=1e-6, abs_tol=None):
    global n_ok
    n_ok += 1
    limit = abs_tol if abs_tol is not None else tol * max(1.0, abs(float(want)))
    assert abs(float(got) - float(want)) <= limit, f"{label}: recount {got} != results {want}"


def table(name):
    return list(csv.DictReader(open(R / f"{name}.csv")))


def raw(name):
    rows = []
    for line in open(HERE / "data" / name):
        if line.strip():
            a = line.strip().split(","); assert len(a) == 5 and all(a), line
            rows.append(([float(v) for v in a[:4]], SPECIES.index(a[4])))
    return rows


def corrected(vals, ratio):
    J = len(vals); m = sum(vals) / J
    var = sum((v - m) ** 2 for v in vals) / (J - 1); se = math.sqrt((1 / J + ratio) * var); t = stats.t.ppf(0.975, J - 1)
    return m, m - t * se, m + t * se, (2 * stats.t.sf(abs(m / se), J - 1) if se > 0 else float("nan"))


def fit_rule(xs, ys):
    """the same rule as analyze.py, written as plain loops: every pair of midpoints, fewest errors, ties to the smaller thresholds, a tied majority to the lower species"""
    vals = sorted(set(xs)); mids = [(a + b) / 2 for a, b in zip(vals, vals[1:])]; best = None
    for a in range(len(mids)):
        for b in range(a + 1, len(mids)):
            counts = [[0, 0, 0] for _ in range(3)]
            for x, c in zip(xs, ys):
                counts[0 if x <= mids[a] else 1 if x <= mids[b] else 2][c] += 1
            correct = sum(max(row) for row in counts)
            if best is None or correct > best[0]:
                best = (correct, mids[a], mids[b], [row.index(max(row)) for row in counts])
    return best[1], best[2], best[3], len(xs) - best[0]


def rule_predict(xs_tr, ys_tr, xs_te):
    t1, t2, lab, _ = fit_rule(xs_tr, ys_tr)
    return [lab[0] if x <= t1 else lab[1] if x <= t2 else lab[2] for x in xs_te]


def lda_scores_fit(Xtr, ytr):
    """class means, pooled within-class covariance, and the linear discriminant of each class: x . S^-1 m - m . S^-1 m / 2 + log(prior)"""
    Xtr = np.asarray(Xtr, float); n = len(Xtr); K = 3
    mu = [Xtr[np.array(ytr) == c].mean(axis=0) for c in range(K)]
    S = sum(np.outer(x - mu[c], x - mu[c]) for x, c in zip(Xtr, ytr)) / (n - K)
    w = [np.linalg.solve(S, m) for m in mu]; prior = Counter(ytr)
    return [(w[c], -0.5 * mu[c] @ w[c] + math.log(prior[c] / n)) for c in range(K)]


def lda_predict(Xtr, ytr, Xte):
    f = lda_scores_fit(Xtr, ytr)
    return [max(range(3), key=lambda c: (np.asarray(x) @ f[c][0] + f[c][1], -c)) for x in Xte]


def quantile(v, q):
    v = sorted(v); pos = q * (len(v) - 1); lo = math.floor(pos)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (pos - lo)


def main():
    d = raw("bezdekIris.data"); o = raw("iris.data"); X = [r[0] for r in d]; y = [r[1] for r in d]; Xo = [r[0] for r in o]; yo = [r[1] for r in o]
    assert len(d) == len(o) == 150 and y == yo

    # the audit, from the raw text
    a = {r["check"]: float(r["value"]) for r in table("audit")}
    same("flowers", len(d), a["flowers"])
    for i, s in enumerate(SPECIES):
        same(f"count {s}", y.count(i), a[s])
    same("missing cells", sum(line.count(",,") + (line.strip().startswith(",")) for line in open(HERE / "data" / "bezdekIris.data")), a["missing cells"])
    dups = []
    for name, rows in (("iris.data", o), ("bezdekIris.data", d)):
        first = {}
        for i, (x, c) in enumerate(rows):
            k = (tuple(x), c)
            if k in first:
                dups.append((name, i + 1, first[k]))
            else:
                first[k] = i + 1
    same("duplicate rows (bezdekIris.data)", sum(1 for t in dups if t[0] == "bezdekIris.data"), a["duplicate rows (bezdekIris.data)"])
    same("duplicate rows (iris.data)", sum(1 for t in dups if t[0] == "iris.data"), a["duplicate rows (iris.data)"])
    got = {(r["file"], int(r["line"]), int(r["copy_of_line"])) for r in table("duplicates")}
    assert got == set(dups), (got, dups); same("duplicates listed", len(got), len(dups))
    diff = [(i + 1, MEAS[j].replace(" ", "_"), o[i][0][j], d[i][0][j]) for i in range(150) for j in range(4) if o[i][0][j] != d[i][0][j]]
    same("lines that differ", len({t[0] for t in diff}), a["lines that differ between the two files"]); same("measurements that differ", len(diff), a["measurements that differ"])
    got = {(int(r["line"]), r["measurement"], float(r["iris.data"]), float(r["bezdekIris.data"])) for r in table("file_diff")}
    assert got == set(diff), (got, diff); same("file_diff rows", len(got), len(diff))
    by = {s: [r[0] for r in d if r[1] == i] for i, s in enumerate(SPECIES)}
    for r in table("species_summary"):
        v = [x[MEAS.index(r["measurement"])] for x in by[r["species"]]]
        same(f"{r['measurement']} {r['species']} min", min(v), r["min"]); same("mean", sum(v) / len(v), r["mean"], tol=1e-9); same("max", max(v), r["max"])
    for j, m in ((2, "petal length"), (3, "petal width")):
        se, ve, vi = ([x[j] for x in by[s]] for s in SPECIES); lo, hi = min(vi), max(ve)
        for key, val in ((f"{m}: setosa maximum", max(se)), (f"{m}: versicolor minimum", min(ve)), (f"{m}: versicolor maximum", hi), (f"{m}: virginica minimum", lo), (f"{m}: virginica maximum", max(vi)),
                         (f"{m}: versicolor flowers in the overlap", sum(lo <= v <= hi for v in ve)), (f"{m}: virginica flowers in the overlap", sum(lo <= v <= hi for v in vi))):
            same(key, val, a[key])

    # the rules fitted on all 150 flowers
    for r in table("rule_full_fit"):
        j = MEAS.index(r["measurement"]); t1, t2, lab, err = fit_rule([x[j] for x in X], y)
        same(f"{r['measurement']} t1", t1, r["t1"], tol=1e-9); same("t2", t2, r["t2"], tol=1e-9); same("errors", err, r["errors_on_all_150"])
        assert [SPECIES[c] for c in lab] == [r["low"], r["middle"], r["high"]], r

    # folds and predictions
    folds = table("folds"); pred = table("cv_predictions"); assert [int(f["row"]) for f in folds] == [int(p["row"]) for p in pred]
    test = {}
    for p in pred:
        test.setdefault((int(p["repeat"]), int(p["fold"])), []).append(int(p["row"]))
        assert int(p["y"]) == y[int(p["row"])]
    keys = sorted(test); assert len(keys) == 50 and all(len(v) == 30 for v in test.values())
    col = {n: {(int(p["repeat"]), int(p["fold"]), int(p["row"])): int(p[n]) for p in pred} for n in ALL}
    mism_rule = {n: 0 for n in RULES}; mism_lda = 0
    for k in keys:
        te = test[k]; tr = [i for i in range(150) if i not in set(te)]
        for n in RULES:
            j = MEAS.index(n[6:]); p_ = rule_predict([X[i][j] for i in tr], [y[i] for i in tr], [X[i][j] for i in te])
            bad = sum(p_[t] != col[n][(k[0], k[1], te[t])] for t in range(30)); mism_rule[n] += bad; same(f"{n} fold {k}", bad, 0)
        p_ = lda_predict([X[i] for i in tr], [y[i] for i in tr], [X[i] for i in te])
        bad = sum(p_[t] != col["LDA"][(k[0], k[1], te[t])] for t in range(30)); mism_lda += bad; same(f"LDA fold {k}", bad, 0)
    same("always setosa is all zeros", sum(col["always setosa"].values()), 0)

    # fold scores, intervals, comparisons with the best rule
    acc = {n: [sum(col[n][(k[0], k[1], i)] == y[i] for i in test[k]) / 30 for k in keys] for n in ALL}
    for r in table("fold_scores"):
        same(f"fold score {r['model']} {r['repeat']}.{r['fold']}", acc[r["model"]][keys.index((int(r["repeat"]), int(r["fold"])))], r["accuracy"], tol=1e-9)
    ratio = 30 / 120; mean = {n: 100 * sum(v) / 50 for n, v in acc.items()}; best = max(RULES, key=lambda n: mean[n])
    ab = 6e-5                                                                    # the csv values are rounded to four decimals
    rows = table("models"); assert [r["model"] for r in rows] == ["always setosa", best] + MODELS, [r["model"] for r in rows]
    for r in rows:
        m, lo, hi, _ = corrected(acc[r["model"]], ratio); same(f"{r['model']} accuracy", 100 * m, r["accuracy_pct"], abs_tol=ab); same("ci_lo", 100 * lo, r["ci_lo"], abs_tol=ab); same("ci_hi", 100 * hi, r["ci_hi"], abs_tol=ab)
        dm, dlo, dhi, dp = corrected([u - v for u, v in zip(acc[r["model"]], acc[best])], ratio)
        same("diff", 100 * dm, r["diff_vs_rule_pts"], abs_tol=ab); same("diff_lo", 100 * dlo, r["diff_lo"], abs_tol=ab); same("diff_hi", 100 * dhi, r["diff_hi"], abs_tol=ab)
        if r["model"] != best:
            same("p", dp, r["p_vs_rule"], abs_tol=ab)
    for r in table("single_measurement"):
        m, lo, hi, _ = corrected(acc[r["model"]], ratio); same(f"{r['model']} accuracy", 100 * m, r["accuracy_pct"], abs_tol=ab); same("ci_lo", 100 * lo, r["ci_lo"], abs_tol=ab); same("ci_hi", 100 * hi, r["ci_hi"], abs_tol=ab)

    # the same comparison on iris.data, for the methods that can be refitted by hand
    eff = {r["model"]: r for r in table("file_effect")}
    for n in ALL:
        same(f"file_effect corrected {n}", mean[n], eff[n]["corrected_file_pct"], abs_tol=ab)
    for n in RULES + ["LDA"]:
        sc = []
        for k in keys:
            te = test[k]; tr = [i for i in range(150) if i not in set(te)]
            if n == "LDA":
                p_ = lda_predict([Xo[i] for i in tr], [yo[i] for i in tr], [Xo[i] for i in te])
            else:
                j = MEAS.index(n[6:]); p_ = rule_predict([Xo[i][j] for i in tr], [yo[i] for i in tr], [Xo[i][j] for i in te])
            sc.append(sum(p_[t] == yo[te[t]] for t in range(30)) / 30)
        same(f"file_effect original {n}", 100 * sum(sc) / 50, eff[n]["original_file_pct"], abs_tol=ab)
        same(f"file_effect change {n}", 100 * sum(sc) / 50 - mean[n], eff[n]["change_pts"], abs_tol=ab)

    # how much data it takes: refit LDA and the best rule by hand on every sample
    samples = {(int(r["repeat"]), int(r["fold"]), int(r["per_species"])): [int(v) for v in r["rows"].split()] for r in table("learning_samples")}
    lc = table("learning_curve"); lcv = {(int(r["repeat"]), int(r["fold"]), int(r["per_species"]), r["model"]): float(r["accuracy"]) for r in lc}
    assert len(samples) == 300 and len(lcv) == 1200
    jb = MEAS.index(best[6:])
    for (rep, fo, k), rows_ in samples.items():
        te = test[(rep, fo)]; assert len(rows_) == 3 * k and len(set(rows_)) == len(rows_) and not set(rows_) & set(te)
        assert sorted(Counter(y[i] for i in rows_).items()) == [(0, k), (1, k), (2, k)]
        p_ = lda_predict([X[i] for i in rows_], [y[i] for i in rows_], [X[i] for i in te]); same(f"LDA sample {rep}.{fo}.{k}", sum(p_[t] == y[te[t]] for t in range(30)) / 30, lcv[(rep, fo, k, "LDA")], tol=1e-9)
        p_ = rule_predict([X[i][jb] for i in rows_], [y[i] for i in rows_], [X[i][jb] for i in te]); same(f"rule sample {rep}.{fo}.{k}", sum(p_[t] == y[te[t]] for t in range(30)) / 30, lcv[(rep, fo, k, best)], tol=1e-9)
    for r in table("learning_summary"):
        v = [lcv[(rep, fo, int(r["per_species"]), r["model"])] for rep, fo in keys]
        same(f"learning mean {r['model']} {r['per_species']}", sum(v) / 50, r["mean"], abs_tol=ab); same("p10", quantile(v, 0.1), r["p10"], abs_tol=ab); same("p90", quantile(v, 0.9), r["p90"], abs_tol=ab)
        same("flowers", 3 * int(r["per_species"]), r["flowers"])
    for n in ("LDA", best):                                                         # 40 flowers per species is the whole training fold
        same(f"learning curve ends at the cross-validation score {n}", sum(lcv[(rep, fo, 40, n)] for rep, fo in keys) / 50, sum(acc[n]) / 50, tol=1e-9)

    # hard flowers and the confusion tables, recounted from the saved predictions
    wrong = [0] * 150; votes = [Counter() for _ in range(150)]
    for n in MODELS:
        for (rep, fo, i), p_ in col[n].items():
            if p_ != y[i]:
                wrong[i] += 1; votes[i][p_] += 1
    hard = table("hard_flowers"); assert {int(r["line"]) for r in hard} == {i + 1 for i in range(150) if wrong[i]}
    assert [int(r["wrong_of_50"]) for r in hard] == sorted((int(r["wrong_of_50"]) for r in hard), reverse=True)
    for r in hard:
        i = int(r["line"]) - 1
        same(f"wrong_of_50 flower {i + 1}", wrong[i], r["wrong_of_50"]); assert SPECIES[min(votes[i], key=lambda c: (-votes[i][c], c))] == r["usually_called"]
        assert r["species"] == SPECIES[y[i]] and [float(r[c]) for c in ("sepal_length", "sepal_width", "petal_length", "petal_width")] == X[i]
    for r in table("confusion"):
        same(f"confusion {r['model']} {r['true']}->{r['predicted']}", sum(1 for (rep, fo, i), p_ in col[r["model"]].items() if y[i] == SPECIES.index(r["true"]) and p_ == SPECIES.index(r["predicted"])), r["count"])

    # Fisher's linear discriminant on all 150 flowers: the share of the first axis from the eigenvalues of Sw^-1 Sb, and the errors on the flowers it was fitted on
    Xa = np.asarray(X); mu = Xa.mean(axis=0); mus = [Xa[np.array(y) == c].mean(axis=0) for c in range(3)]
    Sw = sum(np.outer(x - mus[c], x - mus[c]) for x, c in zip(Xa, y)); Sb = sum(50 * np.outer(mus[c] - mu, mus[c] - mu) for c in range(3))
    ev = sorted(np.linalg.eigvals(np.linalg.solve(Sw, Sb)).real, reverse=True)
    s = table("lda_summary")[0]; same("LD1 share", ev[0] / (ev[0] + ev[1]), s["ld1_share"], abs_tol=1e-5); same("LD2 share", ev[1] / (ev[0] + ev[1]), s["ld2_share"], abs_tol=1e-5)
    same("third eigenvalue is zero", ev[2] if len(ev) > 2 else 0, 0, abs_tol=1e-9)
    own = lda_predict(X, y, X); same("LDA errors on all 150", sum(p_ != t for p_, t in zip(own, y)), s["errors_on_all_150"])
    for r in table("lda_axes"):
        same(f"lda_wrong flower {r['line']}", own[int(r["line"]) - 1] != y[int(r["line"]) - 1], r["lda_wrong"])

    print(f"check.py: {n_ok} recomputed numbers agree with results/ (the hand-fitted LDA and the rules matched the saved predictions on all 50 folds: {mism_lda} and {sum(mism_rule.values())} differences)")


if __name__ == "__main__":
    main()
