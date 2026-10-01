"""Prompt builders. Pure functions so they are easy to test and audit.

Only aggregate statistics and a handful of sample rows are ever sent to the
model, never the full dataset.
"""

from __future__ import annotations

import pandas as pd

from autods.core.dtypes import categorical_columns, numeric_columns

ANALYST_SYSTEM = (
    "You are a careful senior data analyst. Explain findings in plain language a "
    "non-specialist can follow. Be concise and concrete, and never invent numbers "
    "that are not in the provided context. If the context is insufficient, say so."
)

MAX_CONTEXT_CHARS = 6000


def dataset_context(df: pd.DataFrame, sample_rows: int = 3) -> str:
    """Compact description of a dataset: shape, types, gaps, statistics and a sample."""
    nums = numeric_columns(df)
    cats = categorical_columns(df)
    missing = df.isna().sum()
    missing_info = ", ".join(f"{c}: {int(n)}" for c, n in missing[missing > 0].items()) or "none"

    lines = [
        f"Shape: {len(df):,} rows x {df.shape[1]} columns.",
        f"Numeric columns ({len(nums)}): {', '.join(nums[:40]) or 'none'}",
        f"Text/categorical columns ({len(cats)}): {', '.join(cats[:40]) or 'none'}",
        f"Missing values per column: {missing_info}",
    ]
    if nums:
        lines.append("Numeric summary:\n" + df[nums[:15]].describe().T.round(3).to_string())
    for col in cats[:8]:
        top = df[col].value_counts().head(5)
        lines.append(f"Top values of {col}: " + ", ".join(f"{k} ({v})" for k, v in top.items()))
    lines.append(f"First {sample_rows} rows:\n" + df.head(sample_rows).to_string(max_cols=20))

    text = "\n".join(lines)
    if len(text) > MAX_CONTEXT_CHARS:
        text = text[:MAX_CONTEXT_CHARS] + "\n… (context truncated)"
    return text


def column_insight_prompt(df: pd.DataFrame, column: str) -> str:
    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        values = s.dropna()
        skew = float(values.skew()) if values.nunique() > 2 else 0.0  # type: ignore[arg-type]
        stats = values.describe().round(4).to_string()
        detail = f"Numeric column '{column}'.\n{stats}\nSkewness: {skew:.3f}"
    else:
        counts = s.value_counts()
        top = ", ".join(f"{k} ({v})" for k, v in counts.head(8).items())
        detail = (
            f"Categorical column '{column}' with {s.nunique()} distinct values "
            f"and {int(s.isna().sum())} missing. Most common: {top}."
        )
    return (
        f"{detail}\n\nIn 2–3 sentences, describe what this distribution tells us and "
        "recommend any preparation step (e.g. transform, grouping rare values)."
    )


def relationship_prompt(df: pd.DataFrame, a: str, b: str) -> str:
    sa, sb = df[a], df[b]
    if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
        detail = f"Pearson correlation between '{a}' and '{b}': {sa.corr(sb):.3f}."
    elif pd.api.types.is_numeric_dtype(sa) or pd.api.types.is_numeric_dtype(sb):
        num, cat = (a, b) if pd.api.types.is_numeric_dtype(sa) else (b, a)
        means = df.groupby(cat, observed=True)[num].agg(["mean", "count"]).round(3)
        means = means.sort_values("mean", ascending=False).head(10)
        detail = f"Mean of '{num}' by '{cat}' (top 10):\n{means.to_string()}"
    else:
        table = pd.crosstab(df[a], df[b]).iloc[:10, :10]
        detail = f"Cross-tabulation of '{a}' by '{b}' (top 10 x 10):\n{table.to_string()}"
    return (
        f"{detail}\n\nIn 2–3 sentences, explain this relationship in plain terms and whether "
        "it looks strong enough to matter. Mention that correlation is not causation if relevant."
    )


def correlation_prompt(pairs: pd.DataFrame) -> str:
    return (
        "Strongest pairwise correlations in a dataset:\n"
        f"{pairs.to_string(index=False)}\n\n"
        "Pick the 2–3 most meaningful ones and explain what they might imply. "
        "Flag any pair so strong that one column is probably redundant."
    )


def feature_importance_prompt(target: str, task: str, importance: pd.DataFrame) -> str:
    return (
        f"A {task.lower()} model predicts '{target}'. Its most important input features are:\n"
        f"{importance.head(10).to_string(index=False)}\n\n"
        "In 2 short paragraphs for a business audience, explain why these features plausibly "
        "matter and what to be careful about (importance is not causation)."
    )


def assistant_system_prompt(context: str) -> str:
    return (
        f"{ANALYST_SYSTEM}\n\nYou are answering questions about the user's dataset. "
        "You cannot run code; answer from the dataset description below. When a question "
        "needs row-level computation that the description cannot answer, say so and explain "
        "how the user could find out in the app (for example the Explore or Analyze pages).\n\n"
        f"Dataset description:\n{context}"
    )
