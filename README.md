# Can four measurements name an iris?

150 iris flowers, each measured four ways (sepal and petal, length and width, in cm), three species. Four questions:
- which measurements separate the species, and how well does a single one do;
- what five classifiers add over that one measurement;
- how many flowers a model needs before it stops improving;
- which flowers every model gets wrong, and whether the two typos known to be in the table matter.

The questions follow GeeksforGeeks' "Iris Flower Dataset Analysis and Predictions". The data is the UCI copy of the dataset R. A. Fisher used in 1936, a real set of flower measurements, not a course file.

Build log: https://mrrishit909.github.io/projects/iris-fisher/

## Data

UCI Machine Learning Repository, "Iris" (dataset 53; creator R. A. Fisher, donor Michael Marshall), licence CC BY 4.0. 150 flowers, 50 per species (setosa, versicolor, virginica): sepal length, sepal width,
petal length and petal width in cm, to 0.1. The zip holds two copies of the table: `iris.data`, which differs from Fisher's article in lines 35 and 38 (UCI's own note says so), and `bezdekIris.data`, with those
two lines corrected. The analysis uses the corrected copy and repeats the comparison on `iris.data`, so the effect of the typos is measured. Each file is 4.5 KB, so both are committed in `data/`; `download.py` refreshes them.

```
./venv/bin/pip install pandas numpy scipy scikit-learn matplotlib certifi      # in a venv
./venv/bin/python download.py    # data/*
./venv/bin/python analyze.py     # results/*
./venv/bin/python check.py
./venv/bin/python charts.py
```

## Steps

1. **Download** both copies of the table.
2. **Audit:** counts, missing values, duplicates, which values differ between the two files, the range of each measurement by species, and where the species overlap.
3. **One measurement:** for each of the four measurements, a rule with two thresholds and three intervals, each interval predicting its majority species, found by trying every pair of thresholds.
4. **Models:** the same 10 x 5-fold cross-validation folds for always guessing setosa, the four rules, Fisher's linear discriminant (LDA), logistic regression, 5 nearest neighbours, an RBF support-vector machine and
   a random forest. The 50 fold scores are not independent, so intervals use the Nadeau-Bengio correction, and each method is compared with the best rule on the same folds. The whole comparison is repeated on `iris.data`.
5. **How much data it takes:** the best rule, LDA, logistic regression and 5 nearest neighbours trained on 5, 8, 12, 20, 30 and 40 flowers per species (drawn from each training fold) and scored on the held-out fold.
6. **Which flowers are hard:** each flower is predicted by five models in each of 10 repeats (50 predictions), by a model that never saw it; the confusion counts and the flowers most often wrong. Fisher's discriminant is also fitted
   on all 150 flowers to see how much of the between-species variance its first axis carries.
7. **Check.** `check.py` recomputes 1,862 numbers: the audit and the differences between the files from the raw text, the rules by a plain-Python search on every fold, every fold's accuracy, the corrected intervals, Fisher's
   discriminant refitted by hand (class means, pooled covariance, a linear solve) on every fold, every learning-curve sample and all 150 flowers, the axis share from the eigenvalues, and the hard-flower and confusion counts.
   The hand-fitted discriminant and the rules agree with the saved predictions on all 50 folds, with no differences. The other four models are not refitted.
8. **Charts.**

## Results

- **The data:** 150 flowers, 50 per species, nothing missing. `bezdekIris.data` has one duplicate row (line 143 repeats line 102); `iris.data` has three (lines 35 and 38 copy line 10, the two typos, and line 143).
- **The two typos:** the files differ in three values on two lines (line 35: petal width 0.1 against 0.2; line 38: sepal width 3.1 against 3.6 and petal length 1.5 against 1.4). Running everything on `iris.data` changes no method's
  accuracy by more than 0.2 points; LDA's does not change.
- **Setosa stands apart.** Its longest petal is 1.9 cm and the shortest versicolor petal is 3.0 cm; its widest petal is 0.6 cm against 1.0 cm for versicolor.
- **The other two overlap.** Versicolor petals run up to 5.1 cm and virginica from 4.5 cm: 37 of the 100 flowers (21 versicolor, 16 virginica) have a petal length in that stretch. For petal width, 38 flowers (22 and 16) lie between 1.4 and 1.8 cm.
- **One measurement nearly does it.** Cross-validated accuracy (95% interval): petal width 95.2% (91.3 to 99.1), petal length 94.5% (90.6 to 98.4), sepal length 73.0% (66.9 to 79.1), sepal width 53.0% (46.4 to 59.6); always
  guessing setosa 33.3%. Fitted on all 150 flowers, the petal-width rule is: setosa below 0.80 cm, versicolor up to 1.65 cm, virginica above; it gets 6 of the 150 wrong.
- **The models add little.** LDA 98.0% (95.8 to 100), logistic regression 95.5% (91.7 to 99.4), 5 nearest neighbours 95.9% (92.2 to 99.5), RBF SVM 96.0% (92.5 to 99.5), random forest 95.2% (90.9 to 99.5). Against the petal-width rule
  LDA gains 2.8 points (interval -0.5 to +6.1, p = 0.097); the others gain between 0.0 and 0.8 points, and their intervals include zero.
- **How much data it takes.** On 15 flowers (5 per species) LDA is right 95.1% of the time, the petal-width rule 94.3%, logistic regression 89.1% and 5 nearest neighbours 87.1%; on 60 flowers 97.3%, 95.1%, 95.1% and 94.3%; on 120
  98.0%, 95.2%, 95.5% and 95.9%. The rule stays between 93.8% and 95.2% at every size.
- **Which flowers are hard.** Of 7,500 predictions by the five models (150 flowers x 10 repeats x 5 models), 291 (3.9%) are wrong: 290 swap versicolor and virginica (141 and 149) and one calls a setosa a versicolor (flower 42, once).
  134 flowers are never wrong. Seven are wrong in at least 20 of their 50 predictions: lines 84 (49), 134 (47), 107 (40), 120 (39), 71 (35), 78 (34) and 73 (20).
- **Fisher's axis.** Fitted on all 150 flowers, the first discriminant axis carries 99.1% of the between-species variance, and the discriminant misclassifies 3 of the 150 (lines 71, 84 and 134).

## Not done, and caveats

- 150 flowers of three species, measured to 0.1 cm: the results say nothing about other irises. Intervals near 100% run past it; they are cut at 100% in the charts and tables.
- The best rule was picked on the same folds it is compared on (petal width 95.2% against petal length 94.5%), which flatters it slightly.
- LDA's lead over the rule is not distinguishable from chance here (p = 0.097); with 150 flowers, a gap of a few points cannot be resolved. The corrected test is an approximation.
- The learning curve is the mean of 50 splits, each with a random draw of the training flowers; it has no intervals, and the test folds are the same 30-flower folds as in step 4.
- "Hard" (wrong in at least 20 of 50 predictions) is a cut-off of mine. The predictions are out-of-fold, so each comes from a model that never saw that flower.
- Logistic regression, 5 nearest neighbours, the SVM and the forest use default settings (scaled inputs, 300 trees, a fixed seed) and are not refitted by the check; the hand-fitted pieces are LDA and the rules.
