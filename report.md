# Decision Tree From Scratch — Implementation and Testing Report

**Date:** 26 October 2026  
**Language:** Python  
**Libraries:** Pandas and Matplotlib (no scikit-learn)

## 1. Objective

Build a classification decision tree from a Pandas DataFrame, using Shannon entropy and Information Gain to choose splits. Visualize the trained tree and address the first major correctness blind spots: numerical features, missing values, and branch-label collisions.

This implementation handles **categorical and numeric input columns** and **categorical target labels**. It is not yet a production-ready classifier for arbitrary datasets.

## 2. Mathematical foundation

Entropy of a target column with class probabilities \(p_i\):

\[
H(S)=-\sum_{i=1}^{k}p_i\log_2(p_i)
\]

Information Gain for a candidate partition into subsets \(S_j\):

\[
IG(S,\text{split})=H(S)-\sum_j\frac{|S_j|}{|S|}H(S_j)
\]

The chosen split is the one with **maximum IG**. Recursion continues on each subset until the target is pure, no useful split remains, no features remain, or `max_depth` is reached.

## 3. What we implemented, in order

### Step 1 — Entropy

`entropy(y)` counts target labels, converts counts to probabilities, and calculates \(-\sum p\log_2p\). It rejects empty or missing target labels.

### Step 2 — Categorical Information Gain

For a categorical feature (such as `Weather`), partition rows by their actual values. Calculate the weighted child entropy and subtract it from the parent entropy. The implementation preserves the original category value type, rather than converting it into a string.

### Step 3 — Recursive tree construction

`build_tree(df, target)` chooses the feature/split with maximum gain, creates children for each partition, and calls itself on those children. A leaf stores the majority class and node metadata (`samples`, `entropy`, and `counts`).

### Step 4 — Numerical features

Previously, `Age` values were treated as separate categories. Now, for sorted distinct numeric values \(x_i\), the algorithm evaluates candidate midpoints:

\[
t_i=\frac{x_i+x_{i+1}}{2}
\]

For each threshold, the tree compares `Age <= t` and `Age > t`, including a missing-value partition if necessary. The best threshold maximizes IG. Numerical features can be split again at deeper levels.

**Worked example:** `Age = [18, 22, 25, 30, 35, 40]` with `Buy = [No, No, No, Yes, Yes, Yes]`. The optimal threshold is **27.5** with **IG = 1.0**.

### Step 5 — Missing feature values

Missing numeric feature values cannot be compared against a threshold. Missing categorical values are also distinct from any real category. We therefore create an explicit **missing branch**, when needed, and include its entropy in the IG calculation:

\[
IG=H(S)-\left(\frac{n_L}{n}H(L)+\frac{n_R}{n}H(R)+\frac{n_M}{n}H(M)\right)
\]

This prevents missing rows from silently disappearing during training. For categorical features, the same weighted-entropy principle applies across all categories plus an optional missing branch.

**Worked example:** `Age = [20, 25, NaN, 35, 40]`, `Play = [No, No, No, Yes, Yes]`. The optimal threshold is **30.0**, and IG is approximately **0.97095**. The missing row becomes a separate branch predicting `No`.

### Step 6 — Preventing branch collisions

The earlier code converted missing values to the literal string `"<missing>"` and all other categories to strings. That could overwrite branches (for example, a real category named `"<missing>"`, or integer `1` versus string `"1"`).

The revised tree uses a **list of branches with structured conditions**, such as `{"kind": "missing"}`, `{"kind": "category", "value": value}`, and numeric comparisons. Labels are converted to readable strings only when drawing the figure.

### Step 7 — Predictions and visualization

`predict_one(tree, row)` traverses the learned rules until it reaches a leaf. `predict(tree, features)` applies that operation across DataFrame rows. For an unseen category or absent feature, prediction falls back to the current node's majority class.

`visualize_tree(tree, save_path=...)` draws split conditions, Information Gain, sample counts, entropy, and predicted leaf labels using Matplotlib.

## 4. Code files and usage

- `decision_tree_visualizer.py` — full classifier, split logic, predictions, and visualization.
- `test_decision_tree.py` — automated regression tests.

Install dependencies:

```bash
python -m pip install pandas matplotlib
```

Run the included example and save a PNG:

```bash
python decision_tree_visualizer.py
```

Train on your own CSV:

```python
import pandas as pd
from decision_tree_visualizer import build_tree, predict, visualize_tree

df = pd.read_csv("dataset.csv")
tree = build_tree(df, target="Play", max_depth=5)
print(predict(tree, df.drop(columns="Play")))
visualize_tree(tree, save_path="tree.png")
```

Run tests:

```bash
python -m unittest -v test_decision_tree.py
```

## 5. Regression test results

**Result: 11 tests passed, 0 failed** (executed in the working environment).

| Test | Result |
|---|---|
| Entropy for two equally likely classes | PASS |
| Best numerical threshold and IG | PASS |
| Missing numerical branch and sample conservation | PASS |
| Missing categorical value versus literal `<missing>` | PASS |
| Mixed category values `1` versus `"1"` | PASS |
| Missing categorical branch | PASS |
| Constant feature and majority-class stopping | PASS |
| Unknown category fallback | PASS |
| Maximum-depth stopping | PASS |
| Missing target rejection | PASS |
| Training-set predictions | PASS |

## 6. Remaining limitations / blind spots

1. **Generalization and overfitting:** no train/test evaluation, pruning, `min_samples_split`, or `min_samples_leaf` yet; `max_depth` is available.
2. **Performance:** exhaustive numeric threshold search is not optimized for large datasets; repeated DataFrame filtering can be slow.
3. **Feature typing:** numeric columns imported as strings will be treated as categorical; preprocess or convert their dtypes first.
4. **Unseen or invalid inputs:** unknown categories fall back to the majority class; no probabilistic routing or learned missing direction.
5. **Data cleaning:** unhashable cells (lists/dicts/sets) are rejected; non-finite numeric values other than missing are rejected; missing targets are rejected.
6. **Mixed or unusual values:** unusual extension dtypes and complex Python objects are not exhaustively supported.
7. **Visualization scaling:** deep or wide trees may produce overlapping labels.
8. **Target scope:** classification only; regression trees are not implemented.
9. **Class counts:** labels are stringified for display, so different label types with the same printed representation could collide in *display metadata*; the split target itself still uses original labels.
10. **Evaluation:** passing unit tests does not establish model quality on unseen datasets.

## 7. Suggested next development steps

1. Add train/test splitting and evaluate accuracy, precision, recall, and confusion matrix.
2. Add `min_samples_split` and `min_samples_leaf` to control overfitting.
3. Add reproducible cross-validation and compare predictions against a reference implementation such as scikit-learn (comparison only, not required for tree training).
4. Stress-test on larger datasets and improve visualization layout.

## 8. Summary

We started with entropy and Information Gain, implemented recursive categorical splitting, introduced numerical thresholds, and then fixed missing-value handling and branch-label collisions. The implementation now has prediction and visualization functions, plus a passing 11-test regression suite. Further work is needed before claiming it supports arbitrary datasets robustly.

## Held-out evaluation: Iris classification

We evaluated the from-scratch decision tree on the Iris dataset (150 examples, four numerical features, three species). We used a stratified 80/20 train/test split (`random_state=42`), fitting on 120 rows and evaluating on 30 unseen rows. The maximum depth was 5. The evaluation uses scikit-learn **only for loading data, splitting, and computing metrics**, not for building the decision tree.

| Metric | Held-out score |
|---|---:|
| Accuracy | 0.9333 (28/30) |
| Precision (macro) | 0.9333 |
| Recall (macro) | 0.9333 |
| F1 (macro) | 0.9333 |

Per-class precision/recall/F1: setosa 1.00/1.00/1.00; versicolor 0.90/0.90/0.90; virginica 0.90/0.90/0.90. These scores apply only to this split and dataset; they do not establish performance on arbitrary datasets.

### Reproduce

```bash
pip install -r requirements.txt
python -m unittest -v test_decision_tree.py
python evaluate_decision_tree.py
python decision_tree_visualizer.py
```

### Known limitations

- Only classification targets are supported (no regression trees).
- Numeric split search is brute-force and may be slow on large datasets.
- Missing feature values use their own branch; this may overfit small missing groups.
- Unseen categorical values fall back to the current node's majority class.
- `max_depth` is supported, but minimum leaf size, pruning, class weighting, and cross-validation are not yet implemented.
- Evaluation is one stratified split of Iris; repeat with cross-validation and additional real-world datasets before drawing general conclusions.
