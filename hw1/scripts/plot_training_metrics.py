"""Generate report-ready MSE/Flow plots from a W&B-export CSV.

Expected input format matches W&B table export, for example:
    Step
    <run_name_mse> - train/loss
    <run_name_mse> - eval/mean_reward
    <run_name_flow> - train/loss
    <run_name_flow> - eval/mean_reward
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        "None of the expected columns were found. Tried: "
        + ", ".join(repr(c) for c in candidates)
    )


def _load_xy(df: pd.DataFrame, step_col: str, y_col: str) -> pd.DataFrame:
    data = df[[step_col, y_col]].copy()
    data[step_col] = pd.to_numeric(data[step_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[step_col, y_col]).sort_values(step_col)
    data = data.rename(columns={step_col: "step"})
    return data


def _smooth_weighted(series: pd.Series, alpha: float) -> pd.Series:
    if alpha <= 0.0 or alpha >= 1.0:
        return series
    return series.ewm(alpha=alpha, adjust=False).mean()


def _find_model_metric_column(
    df: pd.DataFrame, model_name: str, metric_suffixes: list[str]
) -> str:
    # W&B export pattern: "<run_name> - <metric>"
    cols = [c for c in df.columns if f"_{model_name} - " in c]
    for suffix in metric_suffixes:
        for col in cols:
            if col.endswith(suffix):
                return col
    raise ValueError(
        f"Could not find {model_name} column ending in one of: {metric_suffixes}"
    )


def _plot_loss(
    data: pd.DataFrame,
    y_col: str,
    smooth_alpha: float,
    title: str,
    out_path: Path,
) -> None:
    plot_df = data.copy()
    plot_df["loss_smooth"] = _smooth_weighted(plot_df[y_col], smooth_alpha)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    line_color = "#F58518"

    sns.lineplot(
        data=plot_df,
        x="step",
        y=y_col,
        ax=ax,
        color=line_color,
        linewidth=1.0,
        alpha=1,
    )

    # sns.lineplot(
    #     data=plot_df,
    #     x="step",
    #     y="loss_smooth",
    #     ax=ax,
    #     color=line_color,
    #     linewidth=1,
    #     label=f"Smoothed",
    # )

    # Mark the final timestep and call out final smoothed value.
    final_step = float(plot_df["step"].iloc[-1])
    final_raw = float(plot_df[y_col].iloc[-1])
    final_smooth = float(plot_df["loss_smooth"].iloc[-1])
    # ax.scatter([final_step], [final_raw], color=line_color, alpha=0.25, s=24, zorder=5)
    ax.scatter([final_step], [final_raw], color=line_color, s=28, zorder=6)
    ax.annotate(
        f"{final_raw:.4f}",
        xy=(final_step, final_raw),
        xytext=(0, 15),
        textcoords="offset points",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.5",
            "fc": "white",
            "ec": line_color,
            "alpha": 0.9,
        },
        # arrowprops={"arrowstyle": "->", "color": line_color, "lw": 0.9},
    )

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    # ax.legend(frameon=True, fontsize=10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_reward(
    data: pd.DataFrame,
    y_col: str,
    title: str,
    out_path: Path,
    *,
    add_final_callout: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    line_color = "#54A24B"
    sns.lineplot(
        data=data,
        x="step",
        y=y_col,
        marker="o",
        linewidth=2.2,
        color=line_color,
        ax=ax,
    )

    if add_final_callout:
        final_step = float(data["step"].iloc[-1])
        final_reward = float(data[y_col].iloc[-1])
        ax.scatter([final_step], [final_reward], color=line_color, s=36, zorder=6)
        ax.annotate(
            f"{final_reward:.4f}",
            xy=(final_step, final_reward),
            xytext=(0, 15),
            textcoords="offset points",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.5",
                "fc": "white",
                "ec": line_color,
                "alpha": 0.9,
            },
        )

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_reward_comparison(
    mse_data: pd.DataFrame,
    mse_col: str,
    flow_data: pd.DataFrame,
    flow_col: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.lineplot(
        data=mse_data,
        x="step",
        y=mse_col,
        marker="o",
        linewidth=2.2,
        color="#4C78A8",
        label="MSE",
        ax=ax,
    )
    sns.lineplot(
        data=flow_data,
        x="step",
        y=flow_col,
        marker="o",
        linewidth=2.2,
        color="#F58518",
        label="Flow",
        ax=ax,
    )
    ax.set_title("Mean Reward vs Step (MSE vs Flow)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.legend(frameon=True, fontsize=10)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(input_csv: Path, output_dir: Path, smooth_alpha: float) -> list[Path]:
    df = pd.read_csv(input_csv)
    step_col = _pick_column(df, ["Step", "step"])

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    mse_loss_col = _find_model_metric_column(df, "mse-long", ["train/loss", "loss"])
    mse_reward_col = _find_model_metric_column(
        df, "mse-long", ["eval/mean_reward", "eval/reward"]
    )
    flow_loss_col = _find_model_metric_column(df, "flow-long", ["train/loss", "loss"])
    flow_reward_col = _find_model_metric_column(
        df, "flow-long", ["eval/mean_reward", "eval/reward"]
    )

    mse_loss_df = _load_xy(df, step_col, mse_loss_col)
    mse_reward_df = _load_xy(df, step_col, mse_reward_col)
    flow_loss_df = _load_xy(df, step_col, flow_loss_col)
    flow_reward_df = _load_xy(df, step_col, flow_reward_col)

    out_paths = [
        output_dir / "mse_loss_vs_step.png",
        output_dir / "mse_mean_reward_vs_step.png",
        output_dir / "flow_loss_vs_step.png",
        output_dir / "flow_mean_reward_vs_step.png",
        output_dir / "mse_vs_flow_mean_reward_vs_step.png",
    ]

    _plot_loss(
        mse_loss_df,
        y_col=mse_loss_col,
        smooth_alpha=smooth_alpha,
        title="MSE Train Loss vs Step",
        out_path=out_paths[0],
    )
    _plot_reward(
        mse_reward_df,
        y_col=mse_reward_col,
        title="MSE Mean Reward vs Step",
        out_path=out_paths[1],
        add_final_callout=True,
    )
    _plot_loss(
        flow_loss_df,
        y_col=flow_loss_col,
        smooth_alpha=smooth_alpha,
        title="Flow Train Loss vs Step",
        out_path=out_paths[2],
    )
    _plot_reward(
        flow_reward_df,
        y_col=flow_reward_col,
        title="Flow Mean Reward vs Step",
        out_path=out_paths[3],
        add_final_callout=True,
    )
    _plot_reward_comparison(
        mse_data=mse_reward_df,
        mse_col=mse_reward_col,
        flow_data=flow_reward_df,
        flow_col=flow_reward_col,
        out_path=out_paths[4],
    )

    return out_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create polished seaborn training plots from a metrics CSV.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to W&B export CSV (e.g., csv/test.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to write PNGs.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.8,
        help="EWMA alpha for smoothing loss (0 < alpha < 1). Default: 0.8.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_paths = make_plots(
        input_csv=args.input,
        output_dir=args.output_dir,
        smooth_alpha=args.smooth_alpha,
    )
    for path in out_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
