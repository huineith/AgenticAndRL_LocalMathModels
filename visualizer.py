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
    # FIX: Tog bort duplikaten (4) — bara 2, 5, 10 är verkliga körningar
    EXPECTED_RUNS = [
        ("1_5b", 2), ("1_5b", 5), ("1_5b", 10),
        ("3b",   2), ("3b",   5), ("3b",  10)
    ]

    PCT_COLOR = {2: "#e74c3c", 5: "#3498db", 10: "#2ecc71"}
    PCT_LABEL = {2: "2.5% Data", 5: "5% Data", 10: "10% Data"}
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
        root_path = os.path.join(drive_dir, root_filename)
        found_path = None

        if os.path.exists(root_path):
            found_path = root_path
        else:
            # FIX: Lade till _warmup och _grpo_tmp som saknades, plus renare logik med found_path
            subdirs_to_check = [
                f"grpo_{size}_{pct}pct_best",
                f"grpo_{size}_{pct}pct_epoch2",
                f"grpo_{size}_{pct}pct_epoch1",
                f"grpo_{size}_{pct}pct_warmup",
                f"grpo_{size}_{pct}pct_grpo_tmp",
            ]
            for subdir in subdirs_to_check:
                subdir_path = os.path.join(drive_dir, subdir)
                if os.path.exists(subdir_path):
                    for possible_json in ["trainer_state.json", "metrics.json"]:
                        check_file = os.path.join(subdir_path, possible_json)
                        if os.path.exists(check_file):
                            found_path = check_file
                            break
                if found_path:
                    break

        if not found_path:
            print(f"  [X] Saknas: grpo_{size}_{pct}pct (Hittade ingen json i roten eller undermappar)")
            continue

        print(f"  [✓] Hittad data: {os.path.basename(os.path.dirname(found_path)) or 'Roten'}/{os.path.basename(found_path)}")
        file_found = True

        with open(found_path, "r") as f:
            metrics = json.load(f)

        ax_reward = plot_mapping[size]["reward"]
        ax_acc = plot_mapping[size]["acc"]

        # FIX: Renare log_history-extraktion som inte riskerar att returnera hela metrics-dicten
        if isinstance(metrics, dict):
            log_history = metrics.get("log_history", [])
        elif isinstance(metrics, list):
            log_history = metrics
        else:
            log_history = []
        if not isinstance(log_history, list):
            log_history = []

        # --- 1. Normalisera Reward-kurvan ---
        reward_steps = []
        rewards = []
        for e in log_history:
            step = e.get("step")
            r_val = e.get("reward", e.get("train/reward", e.get("rewards/chosen", None)))
            if step is not None and r_val is not None:
                reward_steps.append(step)
                rewards.append(r_val)

        if reward_steps:
            samples_seen_reward = [step * BATCH_SAMPLES_PER_STEP for step in reward_steps]
            ax_reward.plot(samples_seen_reward, rewards, "-", color=PCT_COLOR[pct],
                           linewidth=1.5, label=PCT_LABEL[pct])

        # --- 2. Normalisera Validerings-Accuracy ---
        acc_hist = metrics.get("val_accuracy_history", []) if isinstance(metrics, dict) else []
        if not acc_hist:
            for e in log_history:
                if "eval_accuracy" in e or "val_accuracy" in e:
                    acc_val = e.get("eval_accuracy", e.get("val_accuracy"))
                    step_val = e.get("step", 0)
                    acc_hist.append({"epoch": step_val / 100, "val_accuracy": acc_val})

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

                # FIX: Använd 0 som sista fallback istället för ep (epoch-float ger nonsens-samples)
                step_val = closest_step if closest_step is not None else 0
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

        # FIX: Tog bort "4pct" — det är en hallucination
        if "untrained" in run_id or "h0" in run_id or "0pct" in run_id:
            pct = 0.0
        elif "2pct" in run_id or "2.5" in run_id:
            pct = 2.5
        elif "5pct" in run_id:
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
    Jämför Agenten, alla tränade modeller (utan agent) och SOTA Math-7B Baseline.
    X-axel: genomsnittliga tokens per fråga (compute).
    Y-axel: GSM8K Accuracy (%).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Färg och markör per kategori
    STYLE = {
        "agent":    {"color": "#2ecc71", "marker": "o", "s": 150, "label": "Agent Loop (3B 10% + PRM)"},
        "math_7b":  {"color": "#9b59b6", "marker": "X", "s": 180, "label": "SOTA Baseline (Math-7B)"},
        "3b_10pct": {"color": "#3498db", "marker": "s", "s": 130, "label": "3B 10% (ingen agent)"},
        "other":    {"color": "gray",    "marker": ".", "s": 80,  "label": None},
    }

    labels_added = set()

    for _, r in df.iterrows():
        mode   = str(r.get("mode", "")).lower()
        run_id = str(r["run_id"]).lower()
        acc    = float(r["accuracy"]) * 100
        tokens = float(r.get("avg_tokens", 0))

        # Kategorisera raden
        if "agent" in mode or "agent" in run_id:
            key = "agent"
        elif "math_baseline" in mode or "math_7b" in run_id:
            key = "math_7b"
        elif "3b_10pct" in run_id and "agent" not in mode:
            key = "3b_10pct"
        elif "1_5b" in run_id or ("3b" in run_id and "agent" not in mode):
            key = "other"
        else:
            continue

        style = STYLE[key]
        label = style["label"] if (style["label"] and key not in labels_added) else "_nolegend_"
        labels_added.add(key)

        ax.scatter(tokens, acc,
                   label=label,
                   color=style["color"],
                   marker=style["marker"],
                   s=style["s"],
                   edgecolors="black" if key != "other" else "none",
                   alpha=0.4 if key == "other" else 1.0,
                   zorder=3 if key != "other" else 2)

        # Namnge de viktigaste punkterna
        if key in ("agent", "math_7b", "3b_10pct"):
            ax.annotate(
                f"{run_id}\n{acc:.0f}%",
                xy=(tokens, acc),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
            )

    ax.set_xlabel("Genomsnittligt antal tokens per fråga (Compute)", fontsize=11)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax.set_title("H2: Test-Time Compute — Agent vs. Tränad Bas vs. 7B Baseline",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()