"""
visualizer.py
Gemensam modul för interaktiv visualisering i Jupyter Notebook.
Helt befriad från CLI-kod och anpassad för Pandas DataFrame-input.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. GRAF: GRPO TRÄNINGSKURVOR (2x2 Grid)
# =====================================================================
def plot_grpo_training(drive_dir: str, save_path: str = None):
    """
    Läser metrics.json-filer och ritar upp Reward och Validerings-Accuracy.
    """
    EXPECTED_RUNS = [
        ("1_5b", 2), ("1_5b", 5), ("1_5b", 10),
        ("3b", 2), ("3b", 5), ("3b", 10)
    ]
    PCT_COLOR = {2: "#e74c3c", 5: "#3498db", 10: "#2ecc71"}
    PCT_LABEL = {2: "2.5% Data", 5: "5% Data", 10: "10% Data"}
    SIZE_DISPLAY = {"1_5b": "Qwen2.5-Math-1.5B", "3b": "Qwen2.5-Math-3B"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_mapping = {
        "1_5b": {"reward": axes[0, 0], "acc": axes[0, 1]},
        "3b":   {"reward": axes[1, 0], "acc": axes[1, 1]}
    }

    file_found = False
    for size, pct in EXPECTED_RUNS:
        filename = f"grpo_{size}_{pct}pct_metrics.json"
        full_path = os.path.join(drive_dir, filename)
        if not os.path.exists(full_path):
            continue
            
        file_found = True
        with open(full_path, "r") as f:
            metrics = json.load(f)
            
        ax_reward = plot_mapping[size]["reward"]
        ax_acc = plot_mapping[size]["acc"]
        
        # Reward per steg
        log_history = metrics.get("log_history", [])
        reward_steps = [e["step"] for e in log_history if "reward" in e]
        rewards = [e["reward"] for e in log_history if "reward" in e]
        if reward_steps:
            ax_reward.plot(reward_steps, rewards, "-", color=PCT_COLOR[pct], linewidth=1.5, label=PCT_LABEL[pct])
            
        # Accuracy per epok
        acc_hist = metrics.get("val_accuracy_history", [])
        if acc_hist:
            epochs = [h["epoch"] for h in acc_hist]
            accs = [h["val_accuracy"] * 100 for h in acc_hist]
            ax_acc.plot(epochs, accs, "-o", color=PCT_COLOR[pct], markersize=6, linewidth=2, label=PCT_LABEL[pct])

    if not file_found:
        print(f"[ERROR] Inga träningsfiler hittades i {drive_dir}")
        plt.close()
        return

    for size in ["1_5b", "3b"]:
        plot_mapping[size]["reward"].set_title(f"{SIZE_DISPLAY[size]} — Mean Reward per Step", fontsize=11, fontweight="bold")
        plot_mapping[size]["reward"].set_xlabel("Training Step")
        plot_mapping[size]["reward"].set_ylabel("Reward Score")
        plot_mapping[size]["reward"].legend(loc="lower right")
        plot_mapping[size]["reward"].grid(True, linestyle="--", alpha=0.5)
        
        plot_mapping[size]["acc"].set_title(f"{SIZE_DISPLAY[size]} — Val Accuracy per Epoch", fontsize=11, fontweight="bold")
        plot_mapping[size]["acc"].set_xlabel("Epoch")
        plot_mapping[size]["acc"].set_ylabel("Accuracy (%)")
        plot_mapping[size]["acc"].legend(loc="lower right")
        plot_mapping[size]["acc"].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("GRPO Alignment Performance (Model Scale vs. Data Efficiency)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =====================================================================
# 2. GRAF: SKALNINGSLAGAR & DATAEFFEKTIVITET (Hypotes 1)
# =====================================================================
def plot_scaling_laws(df: pd.DataFrame, save_path: str = None):
    """
    Ritar linjer från 0% (H0) till 10% data för att visa hur GRPO skalar över storlekar.
    """
    # Mappa om run_id till dataprocent för linjerna
    # H0 sätter vi till 0% data
    data_mapping = {
        "1_5b_untrained": (1.5, 0), "3b_untrained": (3.0, 0),
        "1_5b_2pct": (1.5, 2.5),    "3b_2pct": (3.0, 2.5),
        "1_5b_5pct": (1.5, 5.0),    "3b_5pct": (3.0, 5.0),
        "1_5b_10pct": (1.5, 10.0),  "3b_10pct": (3.0, 10.0)
    }

    plot_data = []
    for _, r in df.items() if isinstance(df, dict) else df.iterrows():
        run_id = r["run_id"]
        if run_id in data_mapping:
            size, pct = data_mapping[run_id]
            plot_data.append({
                "Size": size,
                "Data_Pct": pct,
                "Accuracy": r["accuracy"] * 100
            })

    if not plot_data:
        print("[INFO] Kunde inte extrahera H0/H1-data för skalningsgrafen.")
        return

    plot_df = pd.DataFrame(plot_data).sort_values("Data_Pct")

    plt.figure(figsize=(8, 5))
    
    # Rita linje för 1.5B
    df_15 = plot_df[plot_df["Size"] == 1.5]
    plt.plot(df_15["Data_Pct"], df_15["Accuracy"], "-o", color="#e74c3c", linewidth=2.5, label="Qwen 1.5B + GRPO", markersize=8)
    
    # Rita linje för 3B
    df_3 = plot_df[plot_df["Size"] == 3.0]
    plt.plot(df_3["Data_Pct"], df_3["Accuracy"], "-o", color="#3498db", linewidth=2.5, label="Qwen 3B + GRPO", markersize=8)

    plt.xlabel("Mängd GRPO-träningsdata (%)", fontsize=11)
    plt.ylabel("GSM8K Accuracy (%)", fontsize=11)
    plt.title("Hypotes 1: Skalningslagar & Dataeffektivitet under GRPO", fontsize=12, fontweight="bold")
    plt.xticks([0, 2.5, 5.0, 10.0], ["0% (Otränad)", "2.5%", "5.0%", "10.0%"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =====================================================================
# 3. GRAF: AGENTIC COMPUTE VS BASELINES (Hypotes 2 med din förbättring!)
# =====================================================================
def plot_agent_compute(df: pd.DataFrame, save_path: str = None):
    """
    Jämför Agenten, Rå tränad 3B (utan agent) och SOTA Math-7B Baseline.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- PANEL 1: Scatter-plot med din nya kontroll ---
    for _, r in df.iterrows():
        mode = r["mode"]
        run_id = r["run_id"]
        acc = r["accuracy"] * 100
        tokens = r.get("avg_tokens", 0)

        if mode == "agent":
            # 1. Agent-arkitekturen (H2)
            ax1.scatter(tokens, acc, label=f"Agent Loop ({run_id})", color="#2ecc71", marker="o", s=150, edgecolors="black", zorder=3)
        elif mode == "math_baseline":
            # 2. SOTA Baslinjen 7B
            ax1.scatter(tokens, acc, label=f"SOTA Baseline (Math-7B)", color="#9b59b6", marker="X", s=180, edgecolors="black", zorder=3)
        elif run_id == "3b_10pct" and mode != "agent":
            # 3. DIN FÖRBÄTTRING: Rå tränad modell UTAN agent-loop!
            ax1.scatter(tokens, acc, label=f"Rå Tränad Bas (3B 10%)", color="#3498db", marker="s", s=130, edgecolors="black", zorder=3)
        elif "1_5b" in run_id or "3b" in run_id:
            # Övriga H1-punkter i bakgrunden för kontext
            ax1.scatter(tokens, acc, color="gray", alpha=0.3, marker=".", s=80)

    ax1.set_xlabel("Genomsnittligt antal tokens per fråga (Compute)", fontsize=11)
    ax1.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax1.set_title("H2: Test-Time Compute (Agent vs. Rå vs. 7B Baseline)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Snygga till legend så att dubbletter rensas bort
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="lower right", frameon=True)

    # --- PANEL 2: PRM Beslutsfördelning ---
    agent_rows = df[df["mode"] == "agent"].to_dict(orient="records")
    if agent_rows:
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        x_positions = range(len(agent_rows))
        bottoms = [0] * len(agent_rows)

        for idx, status in enumerate(["confident", "unsure", "failed"]):
            heights = [r.get(f"share_{status}", 0.0) * r.get("total", 50) for r in agent_rows]
            ax2.bar(x_positions, heights, bottom=bottoms, label=status.capitalize(), color=colors[idx], alpha=0.85, width=0.4)
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        for i, r in enumerate(agent_rows):
            ax2.text(i, r.get("total", 50) * 1.02,
                     f"Acc|Conf: {r.get('accuracy_when_confident',0)*100:.0f}%\nAcc|Unsure: {r.get('accuracy_when_unsure',0)*100:.0f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([r["run_id"] for r in agent_rows])
        ax2.set_ylabel("Antal frågor", fontsize=11)
        ax2.set_title("PRM Kalibrering & Beslutsfördelning", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax2.set_ylim(0, max(r.get("total", 50) for r in agent_rows) * 1.3)
    else:
        ax2.text(0.5, 0.5, "Kör bänkmärkningen i 'agent'-läge\nför att se PRM-statistik.", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()