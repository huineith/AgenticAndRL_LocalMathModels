"""
Plotta GRPO-träningskurvor för de 6 tränade modellerna.

Läser {drive_save_dir}/grpo_{size}_{pct}pct_metrics.json (sparad av
save_training_metrics i grpo_trainer.py). Filerna innehåller log_history
(reward, loss per step) och val_accuracy_history (per epoch).

Användning:
    python plot_training.py --drive-dir /content/drive/MyDrive/results

Filnamnsmönstret matchar grpo_trainer.py:
    grpo_1_5b_2pct_metrics.json    (2.5% data → int(0.025*100)=2)
    grpo_1_5b_5pct_metrics.json
    grpo_1_5b_10pct_metrics.json
    grpo_3b_2pct_metrics.json
    grpo_3b_5pct_metrics.json
    grpo_3b_10pct_metrics.json

Producerar:
    {drive_save_dir}/training_curves.png   — 2×2 grid: reward + val_acc per storlek
"""
import argparse
import json
import os

import matplotlib.pyplot as plt


# Exakta filnamn vi förväntar oss. Saknade filer hoppas över med en utskrift.
EXPECTED_RUNS = [
    ("1_5b", 2),
    ("1_5b", 5),
    ("1_5b", 10),
    ("3b", 2),
    ("3b", 5),
    ("3b", 10),
]

# Färgskala per data-pct för konsekvent läsning över paneler.
PCT_COLOR = {2: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c"}
PCT_LABEL = {2: "2.5%", 5: "5%", 10: "10%"}
SIZE_DISPLAY = {"1_5b": "1.5B", "3b": "3B"}


def load_metrics(drive_dir: str, size: str, pct: int) -> dict | None:
    path = os.path.join(drive_dir, f"grpo_{size}_{pct}pct_metrics.json")
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-dir", required=True,
                        help="Directory with grpo_*_metrics.json files (DRIVE_SAVE_DIR)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: {drive_dir}/training_curves.png)")
    args = parser.parse_args()

    out_path = args.output or os.path.join(args.drive_dir, "training_curves.png")

    # Ladda alla 6
    all_metrics: dict[tuple[str, int], dict] = {}
    for size, pct in EXPECTED_RUNS:
        m = load_metrics(args.drive_dir, size, pct)
        if m is not None:
            all_metrics[(size, pct)] = m

    if not all_metrics:
        print("No metrics files found. Nothing to plot.")
        return

    # 2×2 grid: rader = modellstorlek, kolumner = reward / val_accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for row_idx, size in enumerate(["1_5b", "3b"]):
        ax_reward = axes[row_idx, 0]
        ax_acc = axes[row_idx, 1]

        for pct in [2, 5, 10]:
            key = (size, pct)
            if key not in all_metrics:
                continue
            m = all_metrics[key]

            # Reward — log_history items där "reward" finns
            log = m.get("log_history", [])
            reward_steps = [e["step"] for e in log if "reward" in e]
            rewards = [e["reward"] for e in log if "reward" in e]
            if reward_steps:
                ax_reward.plot(reward_steps, rewards, "-",
                               color=PCT_COLOR[pct], linewidth=1.5,
                               label=f"{PCT_LABEL[pct]} data")

            # Val accuracy — separat lista i metrics
            acc_hist = m.get("val_accuracy_history", [])
            if acc_hist:
                ep = [h["epoch"] for h in acc_hist]
                acc = [h["val_accuracy"] * 100 for h in acc_hist]
                ax_acc.plot(ep, acc, "-o",
                            color=PCT_COLOR[pct], markersize=8, linewidth=2,
                            label=f"{PCT_LABEL[pct]} data")

        ax_reward.set_title(f"{SIZE_DISPLAY[size]} — mean reward per step", fontsize=12)
        ax_reward.set_xlabel("Step")
        ax_reward.set_ylabel("Mean reward")
        ax_reward.legend(loc="lower right", fontsize=9)
        ax_reward.grid(True, linestyle="--", alpha=0.4)

        ax_acc.set_title(f"{SIZE_DISPLAY[size]} — validation accuracy per epoch", fontsize=12)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(loc="lower right", fontsize=9)
        ax_acc.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("GRPO training curves — 1.5B vs 3B at 2.5/5/10% data", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
