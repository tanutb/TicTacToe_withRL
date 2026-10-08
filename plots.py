"""Draw the README charts from web/data/evaluation.json into img/."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {"QLearning": "Q-Learning", "SARSA": "SARSA", "DoubleQLearning": "Double Q-Learning"}
COLORS = {"QLearning": "#3b6ef5", "SARSA": "#12a594", "DoubleQLearning": "#e5484d"}
OPPONENTS = {"random": "random player", "imperfect": "imperfect player (25% random)", "minimax": "perfect player"}
OUTCOME_COLORS = {"win_rate": "#12a594", "draw_rate": "#c9d1de", "loss_rate": "#e5484d"}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


def find(data, algorithm, seat, opponent):
    for row in data["summary"]:
        if (row["algorithm"], row["seat"], row["opponent"]) == (algorithm, seat, opponent):
            return row


def results_chart(data, path):
    """Win rate vs a random player, for X and O."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = list(NAMES.values())
    for ax, seat in zip(axes, "XO"):
        rows = [find(data, a, seat, "random") for a in NAMES]
        bars = ax.bar(labels, [r["win_rate"] for r in rows], yerr=[r["win_rate_std"] for r in rows],
                      color=list(COLORS.values()), width=0.6, capsize=4)
        ax.bar_label(bars, labels=[f"{r['win_rate']:.3f}" for r in rows], padding=4)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"Agent plays {seat} ({'first' if seat == 'X' else 'second'})")
    axes[0].set_ylabel("win rate")
    fig.suptitle(f"Win rate vs a random player  |  {data['episodes']:,} training episodes, "
                 f"{len(data['seeds'])} seeds, {data['games']:,} games per seed (no losses)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def outcomes_chart(data, path):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
    labels = list(NAMES.values())
    for row, opponent in enumerate(OPPONENTS):
        for col, seat in enumerate("XO"):
            ax = axes[row, col]
            records = [find(data, a, seat, opponent) for a in NAMES]
            left = [0.0] * 3
            for key, color in OUTCOME_COLORS.items():
                values = [r[key] * 100 for r in records]
                ax.barh(labels, values, left=left, color=color, height=0.6, label=key.split("_")[0].title())
                for i, v in enumerate(values):
                    if v >= 6:
                        ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center",
                                color="#1d2433" if key == "draw_rate" else "white", fontsize=9)
                left = [a + b for a, b in zip(left, values)]
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_title(f"Agent as {seat} vs {OPPONENTS[opponent]}")
    axes[-1, 0].set_xlabel("% of games")
    axes[-1, 1].set_xlabel("% of games")
    axes[0, 0].legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def learning_curves_chart(data, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    points = sorted({c["episodes"] for c in data["curves"]})
    for ax, seat in zip(axes, "XO"):
        for algorithm, name in NAMES.items():
            per_point = [[c["win_rate"] * 100 for c in data["curves"]
                          if c["algorithm"] == algorithm and c["seat"] == seat and c["episodes"] == p]
                         for p in points]
            ax.plot(points, [statistics.mean(v) for v in per_point], color=COLORS[algorithm], lw=2, label=name)
            ax.fill_between(points, [min(v) for v in per_point], [max(v) for v in per_point],
                            color=COLORS[algorithm], alpha=0.12)
        ax.set_title(f"Agent as {seat} vs random")
        ax.set_xlabel("training episodes")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k" if x else "0")
    axes[0].set_ylabel("win rate (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_charts(data, folder):
    folder.mkdir(parents=True, exist_ok=True)
    results_chart(data, folder / "results.png")
    outcomes_chart(data, folder / "outcomes.png")
    learning_curves_chart(data, folder / "learning_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="web/data/evaluation.json")
    parser.add_argument("--output", default="img")
    args = parser.parse_args()
    draw_charts(json.loads(Path(args.input).read_text(encoding="utf-8")), Path(args.output))
