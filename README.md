# Decision Tree From Scratch

An educational decision tree classifier built with Pandas and Python (without scikit-learn for tree training). Supports categorical splits, numerical thresholds, explicit missing-value branches, prediction, and Matplotlib visualization.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python -m unittest -v test_decision_tree.py
python evaluate_decision_tree.py
python decision_tree_visualizer.py
```

The evaluation prints accuracy, macro precision, macro recall, macro F1 and per-class results for an 80/20 stratified Iris split. The visualizer saves `decision_tree.png`.

## Use your own CSV

```python
import pandas as pd
from decision_tree_visualizer import build_tree, predict, visualize_tree

df = pd.read_csv('my_data.csv')
target = 'label'  # replace with your target column
model = build_tree(df, target, max_depth=5)
predictions = predict(model, df.drop(columns=[target]))
visualize_tree(model, save_path='my_tree.png')
```

For an honest accuracy estimate, evaluate on held-out rows rather than training rows. Read `report.md` for mathematical formulas, engineering decisions, tests, evaluation, and limitations.
