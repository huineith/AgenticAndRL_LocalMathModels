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
    Läser metrics/trainer_state-filer och normaliserar X-axeln till 'Samples Seen' 
    för att ge en vetenskapligt korrekt jämförelse av dataeffektivitet.
    Söker både i rotkatalogen och i undermappar (t.ex. grpo_1_5b_5pct_best).
    """
    EXPECTED_RUNS = [
        ("1_5b", 2), ("1_5b", 4), ("1_5b", 5), ("1_5b", 10),
        ("3b", 2), ("3b", 4), ("3b", 5), ("3b", 10)
    ]
    
    PCT_COLOR = {2: "#e74c3c", 4: "#3498db", 5: "#3498db", 10: "#2ecc71"}
    PCT_LABEL = {2: "2.5% Data", 4: "5% Data", 5: "5% Data", 10: "10% Data"}
    SIZE_DISPLAY = {"1_5b": "Qwen2.5-Math-1.5B (GRPO)", "3b": "Qwen2.5-Math-3B (GRPO)"}

    BATCH_SAMPLES_PER_STEP = 16 

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_mapping = {
        "1_5b": {"reward": axes[0, 0], "acc": axes[0, 1]},
        "3b":   {"reward": axes[1, 0], "acc": axes[1, 1]}
    }

    print("Söker efter träningsfiler i:", drive_dir)
    file_found = False
    
    for size, pct in EXPECTED_RUNS:
        # 1. Kolla standardfilen i roten först
        root_filename = f"grpo_{size}_{pct}pct_metrics.json"
        full_path = os.path.join(drive_dir, root_filename)
        
        # 2. Om den inte finns, leta i undermapparna som du listade
        if not os.path.exists(full_path):
            # Prioriterade mappar att leta i
            subdirs_to_check = [
                f"grpo_{size}_{pct}pct_best",
                f"grpo_{size}_{pct}pct_epoch2",
                f"grpo_{size}_{pct}pct_epoch1"
            ]
            for subdir in subdirs_to_check:
                subdir_path = os.path.join(drive_dir, subdir)
                if os.path.exists(subdir_path):
                    # Hugging Face sparar oftast historiken i 'trainer_state.json'
                    for possible_json in ["trainer_state.json", "metrics.json"]:
                        check_file = os.path.join(subdir_path, possible_json)
                        if os.path.exists(check_file):
                            full_path = check_file
                            break
                if full_path != os.path.join(drive_dir, root_filename): # Hittade en match
                    break

        # Om vi fortfarande inte hittade något, hoppa över
        if not os.path.exists(full_path):
            print(f"  [X] Saknas: grpo_{size}_{pct}pct (Hittade ingen json i roten eller undermappar)")
            continue
            
        print(f"  [✓] Hittad data: {os.path.basename(os.path.dirname(full_path)) or 'Roten'}/{os.path.basename(full_path)}")
        file_found = True
        
        with open(full_path, "r") as f:
            metrics = json.load(f)
            
        ax_reward = plot_mapping[size]["reward"]
        ax_acc = plot_mapping[size]["acc"]
        
        # Säkra upp ifall det är en Hugging Face 'trainer_state.json' (som använder log_history direkt i roten)
        log_history = metrics.get("log_history", metrics) if isinstance(metrics, dict) else metrics
        if not isinstance(log_history, list) and isinstance(metrics, dict):
            log_history = metrics.get("log_history", [])

        # --- 1. Normalisera Reward-kurvan ---
        # Olika tränare loggar under olika nycklar (t.ex. 'reward', 'train/reward' eller 'rewards/chosen')
        reward_steps = []
        rewards = []
        for e in log_history:
            step = e.get("step")
            # Leta efter belöningsnycklar robust
            r_val = e.get("reward", e.get("train/reward", e.get("rewards/chosen", None)))
            if step is not None and r_val is not None:
                reward_steps.append(step)
                rewards.append(r_val)
        
        if reward_steps:
            samples_seen_reward = [step * BATCH_SAMPLES_PER_STEP for step in reward_steps]
            ax_reward.plot(samples_seen_reward, rewards, "-", color=PCT_COLOR[pct], 
                           linewidth=1.5, label=PCT_LABEL[pct])
            
        # --- 2. Normalisera Validerings-Accuracy ---
        # Hämtar antingen din anpassade historik eller kollar i standard log_history
        acc_hist = metrics.get("val_accuracy_history", [])
        if not acc_hist:
            # Fallback: kolla om validering/evaluering loggats direkt i historiken
            for e in log_history:
                if "eval_accuracy" in e or "val_accuracy" in e:
                    acc_val = e.get("eval_accuracy", e.get("val_accuracy"))
                    step_val = e.get("step", 0)
                    acc_hist.append({"epoch": step_val / 100, "val_accuracy": acc_val}) # grov uppskattning av epok ifall det saknas

        if acc_hist:
            epochs = [h["epoch"] for h in acc_hist]
            accs = [h["val_accuracy"] * (100 if h["val_accuracy"] <= 1.0 else 1) for h in acc_hist]
            
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
                
                if closest_step is None and reward_steps:
                    max_step = max(reward_steps)
                    max_epoch = max(epochs) if epochs else 1
                    closest_step = int((ep / max_epoch) * max_step)
                
                step_val = closest_step if closest_step is not None else ep
                samples_seen_acc.append(step_val * BATCH_SAMPLES_PER_STEP)

            ax_acc.plot(samples_seen_acc, accs, "-o", color=PCT_COLOR[pct], 
                        markersize=6, linewidth=2, label=PCT_LABEL[pct])

    if not file_found:
        print(f"\n[FEL] Inga metrics- eller trainer_state-filer hittades i katalogen: {drive_dir}")
        plt.close()
        return

    # Snygga till designen och rensa dubbletter i legends
    for size in ["1_5b", "3b"]:
        for ax_type in ["reward", "acc"]:
            ax = plot_mapping[size][ax_type]
            if ax_type == "reward":
                ax.set_title(f"{SIZE_DISPLAY[size]} — Mean Reward", fontsize=11, fontweight="bold")
                ax.set_ylabel("Reward Score")
            else:
                ax.set_title(f"{SIZE_DISPLAY[size]} — Validation Accuracy", fontsize=11, fontweight="bold")
                ax.set_ylabel("Accuracy (%)")
                
            ax.set_xlabel("Exponeringar (Samples Seen)")
            ax.grid(True, linestyle="--", alpha=0.5)
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="lower right")

    plt.suptitle("GRPO Alignment Performance (Sample-Efficient Compute Axis)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n[INFO] Träningsgraf sparad till: {save_path}")
        
    plt.show()


# =====================================================================
# 2. GRAF: SKALNINGSLAGAR & DATAEFFEKTIVITET (Hypotes 1 — SÄKRAD)
# =====================================================================
def plot_scaling_laws(df: pd.DataFrame, save_path: str = None):
    """
    Ritar linjer från 0% (H0) till 10% data genom att mönstermatcha run_id.
    Fungerar robust med ID-strukturer från benchmarker.py (t.ex. '1_5b_5pct').
    """
    plot_data = []
    
    for _, r in df.iterrows():
        run_id = str(r["run_id"]).lower()
        mode = str(r.get("mode", "")).lower()
        
        # Exkludera agent-körningar och den externa math-7b baslinjen
        if "agent" in mode or "agent" in run_id or "7b" in run_id:
            continue
            
        # Identifiera modellstorlek robust
        if "1_5b" in run_id or "1.5b" in run_id:
            size = 1.5
        elif "3b" in run_id:
            size = 3.0
        else:
            continue
            
        # Identifiera dataprocent (X-axeln) utifrån benchmarker.py:s namngivning
        if "untrained" in run_id or "h0" in run_id or "0pct" in run_id:
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
            "Accuracy": float(r["accuracy"]) * 100
        })

    if not plot_data:
        print("\n[VARNING] Kunde inte matcha rader i din DataFrame till H1-skalningen. Dina tillgängliga run_ids är:")
        print(df["run_id"].unique())
        return

    plot_df = pd.DataFrame(plot_data).sort_values("Data_Pct")

    plt.figure(figsize=(8, 5))
    
    # Linje för 1.5B
    df_15 = plot_df[plot_df["Size"] == 1.5].drop_duplicates(subset=["Data_Pct"], keep="last")
    if not df_15.empty:
        plt.plot(df_15["Data_Pct"], df_15["Accuracy"], "-o", color="#e74c3c", linewidth=2.5, label="Qwen 1.5B + GRPO", markersize=8)
    
    # Linje för 3B
    df_3 = plot_df[plot_df["Size"] == 3.0].drop_duplicates(subset=["Data_Pct"], keep="last")
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
        mode = str(r.get("mode", "")).lower()
        run_id = str(r["run_id"]).lower()
        acc = float(r["accuracy"]) * 100
        tokens = r.get("avg_tokens", 0)

        if "agent" in mode or "agent" in run_id:
            ax1.scatter(tokens, acc, label="Agent Loop (3B 10% + PRM)", color="#2ecc71", marker="o", s=150, edgecolors="black", zorder=3)
        elif "math_baseline" in mode or "math_7b" in run_id:
            ax1.scatter(tokens, acc, label="SOTA Baseline (Math-7B)", color="#9b59b6", marker="X", s=180, edgecolors="black", zorder=3)
        elif "3b_10pct" in run_id and "agent" not in mode:
            ax1.scatter(tokens, acc, label="Rå Tränad Bas (3B 10%)", color="#3498db", marker="s", s=130, edgecolors="black", zorder=3)
        elif "1_5b" in run_id or "3b" in run_id:
            ax1.scatter(tokens, acc, color="gray", alpha=0.3, marker=".", s=80)

    ax1.set_xlabel("Genomsnittligt antal tokens per fråga (Compute)", fontsize=11)
    ax1.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax1.set_title("H2: Test-Time Compute (Agent vs. Rå vs. 7B Baseline)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax1.legend(by_label.values(), by_label.keys(), loc="lower right", frameon=True)

    # PRM Beslutsfördelning Panel 2
    agent_rows = df[df["mode"].str.lower().str.contains("agent", na=False)].to_dict(orient="records")
    if not agent_rows:
        agent_rows = df[df["run_id"].str.contains("agent", case=False)].to_dict(orient="records")

    if agent_rows:
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        x_positions = range(len(agent_rows))
        bottoms = [0] * len(agent_rows)

        # Letar efter antingen 'share_X' eller 'status_X_rate' från benchmarker.py-strukturen
        for idx, status in enumerate(["confident", "unsure", "failed"]):
            rate_key = f"status_{status}_rate"
            share_key = f"share_{status}"
            
            heights = []
            for r in agent_rows:
                rate = r.get(rate_key, r.get(share_key, 0.0))
                total = r.get("total", 50)
                heights.append(float(rate) * float(total))

            ax2.bar(x_positions, heights, bottom=bottoms, label=status.capitalize(), color=colors[idx], alpha=0.85, width=0.4)
            bottoms = [b + h for b, h in zip(bottoms, heights)]

        for i, r in enumerate(agent_rows):
            total_q = r.get("total", 50)
            ax2.text(i, total_q * 1.02,
                     f"Acc|Conf: {float(r.get('accuracy_when_confident', 0))*100:.0f}%\nAcc|Unsure: {float(r.get('accuracy_when_unsure', 0))*100:.0f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([r["run_id"] for r in agent_rows])
        ax2.set_ylabel("Antal frågor", fontsize=11)
        ax2.set_title("PRM Kalibrering & Beslutsfördelning", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax2.set_ylim(0, max(int(r.get("total", 50)) for r in agent_rows) * 1.3)
    else:
        ax2.text(0.5, 0.5, "Kör bänkmärkningen i 'agent'-läge\nför att se PRM-statistik.", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()