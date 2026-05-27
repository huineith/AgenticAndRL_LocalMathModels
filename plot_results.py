"""
Snabb visuell sanity-check för H2 (agent vs SOTA-baseline).
H0 och H1 analyseras inline i notebooken med pandas — se cell-snippet
i README-kommentaren nedan.

Användning:
    python plot_results.py --save-dir /path/to/h2_results
"""
import argparse
import os
import csv

import matplotlib.pyplot as plt


def _read_summary(save_dir: str) -> list[dict]:
    """Returnera lista av rader från benchmark_summary.csv, en per run.
    Konverterar numeriska kolumner. Tomma celler blir None."""
    path = os.path.join(save_dir, "benchmark_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                if v == "" or v is None:
                    converted[k] = None
                elif k in ("run_id", "mode", "checkpoint"):
                    converted[k] = v
                elif k in ("correct", "total"):
                    converted[k] = int(v)
                else:
                    try:
                        converted[k] = float(v)
                    except ValueError:
                        converted[k] = v
            rows.append(converted)
    return rows


def plot_h2(save_dir: str) -> str:
    """Två paneler: accuracy-vs-tokens scatter + agent status-breakdown."""
    rows = _read_summary(save_dir)
    n_questions = max(r["total"] for r in rows if r["total"]) if rows else 0

    mode_color = {"no_agent": "steelblue", "agent": "indianred", "math_baseline": "darkorange"}
    mode_marker = {"no_agent": "o", "agent": "D", "math_baseline": "s"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Vänster: accuracy vs tokens
    for row in rows:
        mode = row["mode"]
        x = row["avg_tokens"]
        y = row["accuracy"] * 100
        color = mode_color.get(mode, "grey")
        marker = mode_marker.get(mode, "o")
        ax1.scatter(x, y, color=color, marker=marker, s=140, zorder=3,
                    edgecolors="black", linewidth=0.5)
        ax1.annotate(
            f"{row['run_id']}\n{y:.1f}%  |  {row['tokens_per_correct']:.0f} tok/ans",
            xy=(x, y), xytext=(8, 4), textcoords="offset points", fontsize=8,
        )

    legend_handles = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="indianred",
                   markersize=10, label="Agent", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="darkorange",
                   markersize=10, label="Math-7B baseline", markeredgecolor="black"),
    ]
    ax1.legend(handles=legend_handles, fontsize=9, loc="lower right")
    ax1.set_xlabel("Average tokens per question (input + output)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title(f"H2: Agentic compute vs SOTA baseline (n={n_questions})", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # --- Höger: agent status + PRM-kalibrering
    agent_rows = [r for r in rows if r["mode"] == "agent"]
    if agent_rows:
        statuses = ["confident", "unsure", "failed"]
        status_colors = {"confident": "seagreen", "unsure": "goldenrod", "failed": "indianred"}
        x_positions = list(range(len(agent_rows)))
        bottoms = [0.0] * len(agent_rows)
        for status in statuses:
            heights = [r[f"status_{status}_rate"] * r["total"] for r in agent_rows]
            ax2.bar(x_positions, heights, 0.6, bottom=bottoms,
                    color=status_colors[status], label=status)
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        for i, r in enumerate(agent_rows):
            acc_conf = r.get("accuracy_when_confident") or 0.0
            acc_unsure = r.get("accuracy_when_unsure") or 0.0
            ax2.text(i, r["total"] + 1,
                     f"acc|conf={acc_conf*100:.0f}%\nacc|unsure={acc_unsure*100:.0f}%",
                     ha="center", fontsize=8)

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([r["run_id"] for r in agent_rows])
        ax2.set_ylabel("Number of questions")
        ax2.set_title("Agent status breakdown + PRM calibration")
        ymax = max(r["total"] for r in agent_rows)
        ax2.set_ylim(0, ymax * 1.2)
        ax2.legend(loc="upper right")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "No agent runs in this save_dir",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_xticks([])
        ax2.set_yticks([])

    out_path = os.path.join(save_dir, "h2_agent_vs_baseline.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True,
                        help="Directory containing benchmark_summary.csv from run_h2_benchmark")
    args = parser.parse_args()
    plot_h2(args.save_dir)


if __name__ == "__main__":
    main()
