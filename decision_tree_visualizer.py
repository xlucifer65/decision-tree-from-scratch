"""Decision tree classifier from scratch: categorical/numeric splits and missing branches.

Dependencies: pandas, matplotlib. No scikit-learn required.
"""
from __future__ import annotations

import math
from numbers import Number
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def entropy(y: pd.Series) -> float:
    """Shannon entropy of non-missing class labels."""
    if y.empty or y.isna().any():
        raise ValueError("Target labels must be present and nonempty")
    p = y.value_counts(normalize=True, dropna=False)
    return float(-sum(float(v) * math.log2(float(v)) for v in p if v > 0))


def _validate(df: pd.DataFrame, target: str) -> None:
    if df.empty or target not in df or len(df.columns) < 2:
        raise ValueError("Provide a nonempty DataFrame with a target and at least one feature")
    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique")
    if df[target].isna().any():
        raise ValueError("Missing target labels are not supported")
    for feature in df.columns.drop(target):
        nonmissing = df[feature].dropna()
        if nonmissing.map(lambda x: isinstance(x, (list, dict, set))).any():
            raise TypeError(f"Feature {feature!r} contains unhashable/list-like cells")
        if is_numeric_dtype(df[feature]) and not is_bool_dtype(df[feature]):
            if not nonmissing.map(lambda x: isinstance(x, Number) and math.isfinite(float(x))).all():
                raise ValueError(f"Feature {feature!r} contains non-finite numeric values")


def _numeric(df: pd.DataFrame, feature: str) -> bool:
    return is_numeric_dtype(df[feature]) and not is_bool_dtype(df[feature])


def _groups(df: pd.DataFrame, feature: str, threshold: float | None = None) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Structured branch conditions prevent label collisions (including NaN)."""
    series = df[feature]
    result: list[tuple[dict[str, Any], pd.DataFrame]] = []
    if threshold is not None:
        for op, mask in (("<=", series.notna() & (series <= threshold)),
                         (">", series.notna() & (series > threshold))):
            subset = df.loc[mask]
            if not subset.empty:
                result.append(({"kind": "numeric", "op": op, "threshold": threshold}, subset))
    else:
        for value in series.dropna().unique():
            # Comparing a column against an original typed value, never stringifying keys.
            subset = df.loc[series.notna() & series.map(lambda v: False if pd.isna(v) else type(v) is type(value) and v == value)]
            if not subset.empty:
                result.append(({"kind": "category", "value": value}, subset))
    missing = df.loc[series.isna()]
    if not missing.empty:
        result.append(({"kind": "missing"}, missing))
    return result


def _gain(df: pd.DataFrame, target: str, groups: list[tuple[dict, pd.DataFrame]]) -> float:
    if sum(len(part) for _, part in groups) != len(df):
        raise AssertionError("A split must account for every training row")
    return max(0.0, entropy(df[target]) - sum(len(part) / len(df) * entropy(part[target]) for _, part in groups))


def best_split(df: pd.DataFrame, feature: str, target: str) -> dict[str, Any]:
    """Find the best split of one feature, including missing-value entropy."""
    _validate(df, target)
    if feature == target or feature not in df:
        raise ValueError("Feature must be a non-target column")
    if _numeric(df, feature):
        values = sorted(df[feature].dropna().unique())
        best = {"feature": feature, "type": "numeric", "threshold": None, "gain": 0.0, "groups": []}
        for a, b in zip(values, values[1:]):
            threshold = float(a / 2 + b / 2)  # reduces overflow risk vs (a+b)/2
            groups = _groups(df, feature, threshold)
            gain = _gain(df, target, groups)
            if gain > best["gain"] + 1e-12:
                best.update(threshold=threshold, gain=gain, groups=groups)
        return best
    groups = _groups(df, feature)
    return {"feature": feature, "type": "categorical", "threshold": None,
            "gain": _gain(df, target, groups), "groups": groups}


def information_gain(df: pd.DataFrame, feature: str, target: str) -> float:
    return best_split(df, feature, target)["gain"]


def build_tree(df: pd.DataFrame, target: str, max_depth: int | None = None) -> dict[str, Any]:
    """Build a classifier. Missing values are explicit branches; numeric features may be reused."""
    _validate(df, target)
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be nonnegative")

    def grow(data: pd.DataFrame, features: list[str], depth: int) -> dict[str, Any]:
        counts = {str(k): int(v) for k, v in data[target].value_counts().items()}
        majority = data[target].value_counts().index[0]
        node: dict[str, Any] = {"samples": len(data), "entropy": entropy(data[target]),
                                "counts": counts, "prediction": majority, "leaf": True}
        if data[target].nunique() == 1 or not features or (max_depth is not None and depth >= max_depth):
            return node
        options = [best_split(data, f, target) for f in features]
        best = max(options, key=lambda s: s["gain"])
        if best["gain"] <= 1e-12:
            return node
        node.update(leaf=False, feature=best["feature"], split_type=best["type"],
                    threshold=best["threshold"], ig=best["gain"], branches=[])
        next_features = features if best["type"] == "numeric" else [f for f in features if f != best["feature"]]
        for condition, subset in best["groups"]:
            node["branches"].append({"condition": condition, "child": grow(subset, next_features, depth + 1)})
        return node

    return grow(df, [c for c in df.columns if c != target], 0)


def _matches(value: Any, condition: dict[str, Any]) -> bool:
    if condition["kind"] == "missing":
        return bool(pd.isna(value))
    if pd.isna(value):
        return False
    if condition["kind"] == "category":
        expected = condition["value"]
        return type(value) is type(expected) and bool(value == expected)
    try:
        return bool(value <= condition["threshold"]) if condition["op"] == "<=" else bool(value > condition["threshold"])
    except (TypeError, ValueError):
        return False


def predict_one(tree: dict[str, Any], row: dict[str, Any] | pd.Series) -> Any:
    """Unknown values fall back to the majority class of the current node."""
    node = tree
    while not node["leaf"]:
        value = row.get(node["feature"], None)
        match = next((b for b in node["branches"] if _matches(value, b["condition"])), None)
        if match is None:
            return node["prediction"]
        node = match["child"]
    return node["prediction"]


def predict(tree: dict[str, Any], features: pd.DataFrame) -> pd.Series:
    return features.apply(lambda row: predict_one(tree, row), axis=1)


def _edge_label(condition: dict[str, Any]) -> str:
    if condition["kind"] == "missing":
        return "Missing"
    if condition["kind"] == "category":
        return repr(condition["value"])
    return f"{condition['op']} {condition['threshold']:g}"


def visualize_tree(tree: dict[str, Any], figsize: tuple[int, int] = (14, 8), save_path: str | None = None) -> None:
    """Draw tree; optionally save a PNG (also displays the figure)."""
    positions: dict[int, tuple[float, float]] = {}
    edges: list[tuple[dict, dict, str]] = []
    leaf_index = 0

    def layout(node: dict, depth: int) -> float:
        nonlocal leaf_index
        branches = node.get("branches", [])
        if not branches:
            x = float(leaf_index)
            leaf_index += 1
        else:
            xs = []
            for branch in branches:
                child = branch["child"]
                xs.append(layout(child, depth + 1))
                edges.append((node, child, _edge_label(branch["condition"])))
            x = (min(xs) + max(xs)) / 2
        positions[id(node)] = (x, -float(depth))
        return x

    layout(tree, 0)
    fig, ax = plt.subplots(figsize=figsize)
    for parent, child, label in edges:
        x1, y1 = positions[id(parent)]
        x2, y2 = positions[id(child)]
        ax.plot([x1, x2], [y1, y2], color="#7d8796", linewidth=1.5, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none"), zorder=3)

    def draw(node: dict) -> None:
        x, y = positions[id(node)]
        if node["leaf"]:
            label = f"Predict: {node['prediction']}\nn={node['samples']} | H={node['entropy']:.3f}\n{node['counts']}"
            color = "#d7f2de"
        else:
            rule = node["feature"] + (f" ≤ {node['threshold']:g}?" if node["split_type"] == "numeric" else "")
            label = f"{rule}\nIG={node['ig']:.3f}\nn={node['samples']} | H={node['entropy']:.3f}"
            color = "#dceafd"
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=.65", fc=color, ec="#52616b", lw=1.2), zorder=5)
        for branch in node.get("branches", []):
            draw(branch["child"])

    draw(tree)
    ax.set_xlim(-0.8, max(leaf_index - 0.2, 0.8))
    ax.set_ylim(min(y for _, y in positions.values()) - 0.65, 0.65)
    ax.axis("off")
    ax.set_title("Decision Tree — Entropy and Information Gain", pad=24, fontsize=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    data = pd.DataFrame({
        "Age": [20, 25, None, 35, 40, 28, 22],
        "Weather": ["Sunny", "Rain", "Sunny", "Rain", None, "Sunny", "Rain"],
        "Play": ["No", "No", "No", "Yes", "Yes", "Yes", "No"],
    })
    fitted = build_tree(data, "Play")
    print("Predictions:", predict(fitted, data.drop(columns="Play")).tolist())
    visualize_tree(fitted, save_path="decision_tree.png")
