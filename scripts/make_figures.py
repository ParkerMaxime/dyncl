#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
AGG_DIR = REPO_ROOT / "evaluation" / "aggregated"
OUT_DIR = REPO_ROOT / "figures" / "generated"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=200)
    plt.close(fig)


def plot_pareto(input_name: str, stem: str, title: str) -> None:
    data = load_json(AGG_DIR / input_name)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for point in data["points"]:
        x = point["x"]
        y = point["y"]
        yerr = point.get("yerr")
        if yerr is not None:
            ax.errorbar([x], [y], yerr=[yerr], fmt="o", capsize=3)
        else:
            ax.plot([x], [y], "o")
        ax.annotate(point["label"], (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel(data["x"].replace("_", " "))
    ax.set_ylabel(data["y"].replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    save(fig, stem)


def plot_trajectory() -> None:
    data = load_json(AGG_DIR / "figure_data_trajectory.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for series in data["series"]:
        xs = list(range(len(series["points"])))
        ys = [point["y"] for point in series["points"]]
        labels = [point["level"] for point in series["points"]]
        yerrs = [point.get("yerr") for point in series["points"]]
        if any(err is not None for err in yerrs):
            ax.errorbar(
                xs,
                ys,
                yerr=[err if err is not None else 0.0 for err in yerrs],
                marker="o",
                capsize=3,
                label=series["label"],
            )
        else:
            ax.plot(xs, ys, marker="o", label=series["label"])
        ax.set_xticks(xs, labels)
    ax.set_xlabel(data["x"])
    ax.set_ylabel(data["y"])
    ax.set_title("Cross-level autoregressive trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "cl_trajectory_rebuilt")


def main() -> None:
    ensure_dir(OUT_DIR)
    plot_pareto("figure_data_primary.json", "pareto_primary_rebuilt", "Pareto frontier on the primary level")
    plot_pareto("figure_data_middle.json", "pareto_middle_rebuilt", "Pareto frontier on the middle-school level")
    plot_trajectory()
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
