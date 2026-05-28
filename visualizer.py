"""
visualizer.py
Gemensam modul för interaktiv visualisering i Jupyter Notebook.
Helt befriad från CLI-kod. Normaliserad för 'Samples Seen' (Dataeffektivitet).
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. GRAF: GRPO TRÄNINGSKURVOR (Normaliserade på Samples Seen)
# =====================================================================
def plot_grpo_training(drive_dir: str, save_path: str = None):
    """
    Läser metrics.json-filer och normaliserar X-axeln till 'Samples Seen' 
    för att ge en vetenskapligt korrekt jämförelse av dataeffektivitet.
    """
    # Vi kollar både 5pct och 4pct på grund av flyttalsavrundningar i Python
    EXPECTED_RUNS = [
        ("1_5b", 2), ("1_5b", 5), ("1_5b", 4), ("1_5b", 10),
        ("3b", 2), ("3b", 5), ("3b", 4), ("3b", 10)
    ]
    
    PCT_COLOR = {2: "#e74c3c", 5: "#3498db", 4: "#3498db", 10: "#2ecc71"}
    PCT_LABEL = {2: "2.5% Data", 5: "5% Data", 4: "5% Data", 10: "10% Data"}
    SIZE_DISPLAY = {"1_5b": "Qwen2.5-Math-1.5B (GRPO)", "3b": "Qwen2.5-Math-3B (GRPO)"}

    # Uppskattat antal unika frågor per träningssteg (Global Batch Size)
    # Justera dessa om du vet din exakta num_generations * batch_size, 
    # men standard i din pipeline sätter en stabil proportionell skalfaktor:
    BATCH_SAMPLES_PER_STEP = 16 

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_mapping = {
        "1_5b": {"reward": axes[0, 0], "acc": axes[0, 1]},
        "3b":   {"reward": axes[1, 0], "acc": axes[1, 1]}
    }

    print("Söker efter träningsfiler i:", drive_dir)
    file_found = False
    
    for size, pct in EXPECTED_RUNS:
        filename = f"grpo_{size}_{pct}pct_metrics.json"
        full_path = os.path.join(drive_dir, filename)
        
        if not os.path.exists(full_path):
            print(f"  [X] Saknas: {filename}")
            continue
            
        print(f"  [✓] Hittad: {filename} -> Normaliserar axlar...")
        file_found = True
        
        with open(full_path, "r") as f:
            metrics = json.load(f)
            
        ax_reward = plot_mapping[size]["reward"]
        ax_acc = plot_mapping[size]["acc"]
        
        log_history = metrics.get("log_history", [])
        
        # --- 1. Normalisera Reward-kurvan ---
        reward_steps = [e["step"] for e in log_history if "reward" in e]
        rewards = [e["reward"] for e in log_history if "reward" in e]
        
        if reward_steps:
            # Transformera träningssteg till antal sedda träningsexempel
            samples_seen_reward = [step * BATCH_SAMPLES_PER_STEP for step in reward_steps]
            ax_reward.plot(samples_seen_reward, rewards, "-", color=PCT_COLOR[pct], 
                           linewidth=1.5, label=PCT_LABEL[pct])
            
        # --- 2. Normalisera Validerings-Accuracy ---
        acc_hist = metrics.get("val_accuracy_history", [])
        if acc_hist:
            epochs = [h["epoch"] for h in acc_hist]
            accs = [h["val_accuracy"] * 100 for h in acc_hist]
            
            # Översätt stela epok-nummer till exakta globala träningssteg
            samples_seen_acc = []
            for ep in epochs:
                closest_step = None
                min_diff = float("inf")
                for e in log_history:
                    if "epoch" in e:
                        diff = abs(e["epoch"] - ep)
                        if diff < min_diff:
                            min_diff = diff
                            closest_step = e["step"]
                
                # Om logg saknas, gör en linjär uppskattning baserat på reward-stegen
                if closest_step is None and reward_steps:
                    max_step = max(reward_steps)
                    max_epoch = max(epochs) if epochs else 1
                    closest_step = int((ep / max_epoch) * max_step)
                
                step_val = closest_step if closest_step is not None else ep
                # Konvertera steget till Samples Seen
                samples_seen_acc.append(step_val * BATCH_SAMPLES_PER_STEP)

            ax_acc.plot(samples_seen_acc, accs, "-o", color=PCT_COLOR[pct], 
                        markersize=6, linewidth=2, label=PCT_LABEL[pct])

    if not file_found:
        print(f"\n[FEL] Inga metrics-filer hittades i katalogen: {drive_dir}")
        plt.close()
        return

    # Snygga till designen
    for size in ["1_5b", "3b"]:
        ax_reward = plot_mapping[size]["reward"]
        ax_acc = plot_mapping[size]["acc"]
        
        ax_reward.set_title(f"{SIZE_DISPLAY[size]} — Mean Reward", fontsize=11, fontweight="bold")
        ax_reward.set_xlabel("Exponeringar (Samples Seen)")
        ax_reward.set_ylabel("Reward Score")
        ax_reward.legend(loc="lower right")
        ax_reward.grid(True, linestyle="--", alpha=0.5)
        
        ax_acc.set_title(f"{SIZE_DISPLAY[size]} — Validation Accuracy", fontsize=11, fontweight="bold")
        ax_acc.set_xlabel("Exponeringar (Samples Seen)")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.legend(loc="lower right")
        ax_acc.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("GRPO Alignment Performance (Sample-Efficient Compute Axis)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n[INFO] Träningsgraf sparad till: {save_path}")
        
    plt.show()


# =====================================================================
# 2. GRAF: SKALNINGSLAGAR & DATAEFFEKTIVITET (Hypotes 1 — FIXAD)
# =====================================================================
def plot_scaling_laws(df: pd.DataFrame, save_path: str = None):
    """
    Ritar linjer från 0% (H0) till 10% data genom att mönstermatcha run_id.
    Säkerställer att textsträngar från CSV tolkas korrekt oavsett gemener/versaler.
    """
    plot_data = []
    
    for _, r in df.iterrows():
        run_id = str(r["run_id"]).lower()
        mode = str(r.get("mode", "")).lower()
        
        # Vi exkluderar agent-körningar från den rena skalningsgrafen
        if mode == "agent" or "agent" in run_id:
            continue
            
        # Identifiera modellstorlek (1.5B vs 3B)
        if "1_5b" in run_id or "1.5b" in run_id:
            size = 1.5
        elif "3b" in run_id:
            size = 3.0
        else:
            continue # Hoppa över Math-7B baslinjen här
            
        # Identifiera dataprocent (X-axeln)
        if "untrained" in run_id or "h0" in run_id or ("baseline" in run_id and "math" not in run_id):
            pct = 0.0
        elif "2pct" in run_id or "2.5" in run_id:
            pct = 2.5
        elif "5pct" in run_id or "4pct" in run_id:
            pct = 5.0
        elif "10pct" in run_id:
            pct = 10.0
        else:
            continue
            
        plot_data.append({
            "Size": size,
            "Data_Pct": pct,
            "Accuracy": r["accuracy"] * 100
        })

    if not plot_data:
        print("\n[VARNING] Kunde inte matcha kolumnerna i din CSV. Dina registrerade run_ids är:")
        print(df["run_id"].unique())
        return

    plot_df = pd.DataFrame(plot_data).sort_values("Data_Pct")

    plt.figure(figsize=(8, 5))
    
    # Linje för 1.5B
    df_15 = plot_df[plot_df["Size"] == 1.5]
    if not df_15.empty:
        plt.plot(df_15["Data_Pct"], df_15["Accuracy"], "-o", color="#e74c3c", linewidth=2.5, label="Qwen 1.5B + GRPO", markersize=8)
    
    # Linje för 3B
    df_3 = plot_df[plot_df["Size"] == 3.0]
    if not df_3.empty:
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
# 3. GRAF: AGENTIC COMPUTE VS BASELINES (Hypotes 2)
# =====================================================================
def plot_agent_compute(df: pd.DataFrame, save_path: str = None):
    """
    Jämför Agenten, Rå tränad 3B (utan agent) och SOTA Math-7B Baseline.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for _, r in df.iterrows():
        mode = str(r["mode"]).lower()
        run_id = str(r["run_id"]).lower()
        acc = r["accuracy"] * 100
        tokens = r.get("avg_tokens", 0)

        if mode == "agent" or "agent" in run_id:
            ax1.scatter(tokens, acc, label="Agent Loop (3B 10% + PRM)", color="#2ecc71", marker="o", s=150, edgecolors="black", zorder=3)
        elif mode == "math_baseline" or "math_7b" in run_id:
            ax1.scatter(tokens, acc, label="SOTA Baseline (Math-7B)", color="#9b59b6", marker="X", s=180, edgecolors="black", zorder=3)
        elif "3b_10pct" in run_id and mode != "agent":
            ax1.scatter(tokens, acc, label="Rå Tränad Bas (3B 10%)", color="#3498db", marker="s", s=130, edgecolors="black", zorder=3)
        elif "1_5b" in run_id or "3b" in run_id:
            ax1.scatter(tokens, acc, color="gray", alpha=0.3, marker=".", s=80)

    ax1.set_xlabel("Genomsnittligt antal tokens per fråga (Compute)", fontsize=11)
    ax1.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax1.set_title("H2: Test-Time Compute (Agent vs. Rå vs. 7B Baseline)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="lower right", frameon=True)

    # PRM Beslutsfördelning Panel 2
    agent_rows = df[df["mode"].str.lower() == "agent"].to_dict(orient="records")
    if not agent_rows:
        # Fallback om strängmatchningen i mode missar
        agent_rows = df[df["run_id"].str.contains("agent", case=False)].to_dict(orient="records")

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