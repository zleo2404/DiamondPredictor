"""Exploratory data analysis for the diamonds dataset.

Produces:
- Histograms of price and log(price)
- Boxplots of price grouped by cut, color, clarity
- Correlation heatmap of numeric features
- Scatter plot of carat vs price, coloured by clarity

Figures are written to reports/figures/eda_*.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diamond_price_prediction.data import (
    CLARITY_ORDER,
    COLOR_ORDER,
    CUT_ORDER,
    clean,
    load_raw,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def main() -> None:
    df_raw = load_raw()
    df = clean(df_raw)

    print(f"raw rows:     {len(df_raw)}")
    print(f"cleaned rows: {len(df)}")
    print(f"columns:      {list(df.columns)}")
    print()
    print(df.describe(include="all").T)

    #Price histograms
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["price"], bins=60, ax=axes[0], color="#4477AA")
    axes[0].set_title("Distribution of price (USD)")
    axes[0].set_xlabel("price (USD)")
    sns.histplot(np.log(df["price"]), bins=60, ax=axes[1], color="#4477AA")
    axes[1].set_title("Distribution of log(price)")
    axes[1].set_xlabel("log(price)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_price_histograms.png", dpi=130)
    plt.close(fig)

    #Boxplots by categorical attribute
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, order in zip(
        axes, ["cut", "color", "clarity"], [CUT_ORDER, COLOR_ORDER, CLARITY_ORDER]
    ):
        sns.boxplot(data=df, x=col, y="price", order=order, ax=ax, color="#88CCEE")
        ax.set_title(f"Price by {col}")
        ax.set_ylabel("price (USD)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_price_boxplots.png", dpi=130)
    plt.close(fig)

    #Correlation heatmap of numeric features
    numeric_cols = ["carat", "depth", "table", "x", "y", "z", "price"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Pearson correlation, numeric features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_correlation_heatmap.png", dpi=130)
    plt.close(fig)

    #Carat vs price, coloured by clarity
    sample = df.sample(n=8000, random_state=42)  # avoid overplot
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.scatterplot(
        data=sample,
        x="carat",
        y="price",
        hue="clarity",
        hue_order=CLARITY_ORDER,
        palette="viridis",
        s=12,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title("Carat vs price, coloured by clarity (8 000 random rows)")
    ax.set_xlabel("carat")
    ax.set_ylabel("price (USD)")
    ax.legend(title="clarity", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_carat_vs_price.png", dpi=130)
    plt.close(fig)

    print(f"\nWrote 4 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
