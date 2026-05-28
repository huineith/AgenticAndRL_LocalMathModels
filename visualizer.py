"""
visualizer.py
Gemensam modul för interaktiv visualisering i Jupyter Notebook.
Helt befriat från CLI-kod. Baserad på 'Epoch' för träningskurvor.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. GRAF: GRPO TRÄNINGSKURVOR (Baserat på Epoch)
# =====================================================================
def plot_grpo_training(drive_dir: str, save_path: str = None):
    """
    Läser metrics/trainer_state-filer och visar X-axeln per 'Epoch'
    för att ge en tydlig jämförelse av träningsförloppet.
    Söker både i rotkatalogen och i undermappar (t.ex. grpo_1_5b_5pct_best).
    """
    EXPECTED_RUNS = [
        ("1_5b", 2), ("1_5b", 5), ("1_5b", 10),
        ("3b",   2), ("3b",   5), ("3b",  10)
    ]

    PCT_COLOR = {2: "#e74c3c", 5: "#3498db", 10: "#2ecc71"}
    PCT_LABEL = {2: "2.5% Data", 5: "5% Data", 10: "10% Data"}
    SIZE_DISPLAY = {"1_5b": "Qwen2.5-Math-1.5B (GRPO)", "3b": "Qwen2.5-Math-3B (GRPO)"}

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

        if isinstance(metrics, dict):
            log_history = metrics.get("log_history", [])
        elif isinstance(metrics, list):
            log_history = metrics
        else:
            log_history = []
        if not isinstance(log_history, list):
            log_history = []

        # --- 1. Reward-kurvan (x = epoch) ---
        reward_epochs = []
        rewards = []
        for e in log_history:
            ep  = e.get("epoch")
            r_val = e.get("reward", e.get("train/reward", e.get("rewards/chosen", None)))
            if ep is not None and r_val is not None:
                reward_epochs.append(ep)
                rewards.append(r_val)

        if reward_epochs:
            ax_reward.plot(reward_epochs, rewards, "-", color=PCT_COLOR[pct],
                           linewidth=1.5, label=PCT_LABEL[pct])

        # --- 2. Validerings-Accuracy (x = epoch) ---
        acc_hist = metrics.get("val_accuracy_history", []) if isinstance(metrics, dict) else []
        if not acc_hist:
            for e in log_history:
                if "eval_accuracy" in e or "val_accuracy" in e:
                    acc_val  = e.get("eval_accuracy", e.get("val_accuracy"))
                    acc_hist.append({"epoch": e.get("epoch", 0), "val_accuracy": acc_val})

        if acc_hist:
            epochs = [h["epoch"] for h in acc_hist]
            accs   = [h["val_accuracy"] * (100 if h["val_accuracy"] <= 1.0 else 1) for h in acc_hist]
            ax_acc.plot(epochs, accs, "-o", color=PCT_COLOR[pct],
                        markersize=6, linewidth=2, label=PCT_LABEL[pct])

    if not file_found:
        print(f"\n[FEL] Inga metrics- eller trainer_state-filer hittades i katalogen: {drive_dir}")
        plt.close()
        return

    for size in ["1_5b", "3b"]:
        for ax_type in ["reward", "acc"]:
            ax = plot_mapping[size][ax_type]
            if ax_type == "reward":
                ax.set_title(f"{SIZE_DISPLAY[size]} — Mean Reward", fontsize=11, fontweight="bold")
                ax.set_ylabel("Reward Score")
            else:
                ax.set_title(f"{SIZE_DISPLAY[size]} — Validation Accuracy", fontsize=11, fontweight="bold")
                ax.set_ylabel("Accuracy (%)")

            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.5)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc="lower right")

    plt.suptitle("GRPO Alignment Performance", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n[INFO] Träningsgraf sparad till: {save_path}")

    plt.show()


# =====================================================================
# 2. GRAF: SKALNINGSLAGAR & DATAEFFEKTIVITET (Hypotes 1)
# =====================================================================
def plot_scaling_laws(df: pd.DataFrame, save_path: str = None):
    """
    Ritar linjer från No Training (0%) till 10% data genom att mönstermatcha run_id.
    Visar snygga och korrekta procentsatser (No Training, 2.5%, 5%, 10%).
    """
    plot_data = []

    for _, r in df.iterrows():
        run_id = str(r["run_id"]).lower()
        mode = str(r.get("mode", "")).lower()

        # Exkludera agent-körningar och den externa math-7b baslinjen
        is_agent = (mode == "agent") or (run_id.endswith("_agent")) or ("agent" in run_id)
        if is_agent or "7b" in run_id:
            continue

        # Identifiera modellstorlek robust
        if "1_5b" in run_id or "1.5b" in run_id:
            size = 1.5
        elif "3b" in run_id:
            size = 3.0
        else:
            continue

        # OBS: Viktigt att matcha "10pct" FÖRE "0pct" eftersom "10pct" innehåller strängen "0pct"
        if "10pct" in run_id or "10%" in run_id:
            pct = 10.0
        elif "5pct" in run_id or "5%" in run_id:
            pct = 5.0
        elif "2pct" in run_id or "2.5" in run_id:
            pct = 2.5
        elif "untrained" in run_id or "h0" in run_id or "0pct" in run_id or "no training" in run_id or "no_training" in run_id or "0%" in run_id:
            pct = 0.0
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

    fig, ax = plt.subplots(figsize=(8, 5))

    # Linje för 1.5B
    df_15 = plot_df[plot_df["Size"] == 1.5].drop_duplicates(subset=["Data_Pct"], keep="last")
    if not df_15.empty:
        ax.plot(df_15["Data_Pct"], df_15["Accuracy"], "-o", color="#e74c3c", linewidth=2.5, label="Qwen 1.5B + GRPO", markersize=8)

    # Linje för 3B
    df_3 = plot_df[plot_df["Size"] == 3.0].drop_duplicates(subset=["Data_Pct"], keep="last")
    if not df_3.empty:
        ax.plot(df_3["Data_Pct"], df_3["Accuracy"], "-o", color="#3498db", linewidth=2.5, label="Qwen 3B + GRPO", markersize=8)

    ax.set_xlabel("Mängd GRPO-träningsdata (%)", fontsize=11)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax.set_title("Hypotes 1: Skalningslagar & Dataeffektivitet under GRPO", fontsize=12, fontweight="bold")
    
    # Snygga till x-axeln enligt önskemål
    ax.set_xticks([0, 2.5, 5.0, 10.0])
    ax.set_xticklabels(["No Training", "2.5%", "5%", "10%"])
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =====================================================================
# 3. GRAF: AGENTIC COMPUTE VS BASELINES (Hypotes 2)
# =====================================================================
def plot_agent_compute(df: pd.DataFrame, save_path: str = None):
    """
    Jämför Agenten, den rena 3B 10% Baslinjen och SOTA Math-7B Baseline i förgrunden.
    Alla andra (t.ex. grpo-träningssteg) blir skuggade gråa punkter i bakgrunden.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Färg och stil för de tre fokusmodellerna (Förgrund)
    STYLE = {
        "agent":    {"color": "#e67e22", "marker": "o", "s": 160, "label": "Agent Loop (3B 10% + PRM)"}, # Orange
        "3b_10pct": {"color": "#2ecc71", "marker": "s", "s": 140, "label": "3B 10% Baseline (utan agent)"},# Grön
        "math_7b":  {"color": "#9b59b6", "marker": "X", "s": 200, "label": "SOTA Baseline (Math-7B)"},  # Lila
    }

    labels_added = set()
    foreground_rows = []
    background_rows = []

    # Max- och minvärden för att kunna räkna ut smart textplacering i efterhand
    max_tokens = df["avg_tokens"].max() if "avg_tokens" in df.columns else 1000
    max_acc = (df["accuracy"].max() * 100) if "accuracy" in df.columns else 100

    # Förbättrad och strikt sorteringslogik
    for _, r in df.iterrows():
        mode   = str(r.get("mode", "")).lower()
        run_id = str(r["run_id"]).lower()

        if "agent" in mode or "agent" in run_id:
            foreground_rows.append((r, "agent"))
        elif "math_baseline" in mode or "math_7b" in run_id or "math7b" in run_id:
            foreground_rows.append((r, "math_7b"))
        # Exakt matchning för den rena 3b_10pct baslinjen (får inte vara en aktiv grpo-träningskörning)
        elif run_id == "3b_10pct" or run_id == "3b_10pct_baseline" or ( "3b_10pct" in run_id and "grpo" not in run_id and "epoch" not in run_id and "best" not in run_id ):
            foreground_rows.append((r, "3b_10pct"))
        else:
            background_rows.append(r)

    # --- 1. Rita BAKGRUNDEN först (Skuggade modeller) ---
    bg_label_added = False
    for r in background_rows:
        acc    = float(r["accuracy"]) * 100
        tokens = float(r.get("avg_tokens", 0))
        
        ax.scatter(tokens, acc,
                   label="Övriga modeller (Bakgrund)" if not bg_label_added else "_nolegend_",
                   color="#bdc3c7", # Ljusgrå
                   marker="o",
                   s=50,
                   alpha=0.3,       # Ökad transparens för bättre skuggning
                   edgecolors="none",
                   zorder=2)        # Hamnar i bakgrunden
        bg_label_added = True

    # --- 2. Rita FÖRGRUNDEN (De tre fokusmodellerna) ---
    for r, key in foreground_rows:
        acc    = float(r["accuracy"]) * 100
        tokens = float(r.get("avg_tokens", 0))
        run_id = str(r["run_id"])

        style = STYLE[key]
        label = style["label"] if key not in labels_added else "_nolegend_"
        labels_added.add(key)

        ax.scatter(tokens, acc,
                   label=label,
                   color=style["color"],
                   marker=style["marker"],
                   s=style["s"],
                   edgecolors="black",
                   linewidths=1.2,
                   zorder=4)        # Hamnar framför allt annat

        # --- Smart, dynamisk textplacering för att undvika krockar med ramen ---
        # Standardförskjutning (höger och lite upp)
        x_offset = 12
        y_offset = 2
        ha_align = "left"
        va_align = "bottom"

        # Om punkten ligger väldigt nära högerkanten, flytta texten till vänster om punkten
        if tokens > max_tokens * 0.82:
            x_offset = -12
            ha_align = "right"
            
        # Om punkten ligger väldigt högt upp, skjut ner texten något
        if acc > max_acc * 0.92:
            y_offset = -12
            va_align = "top"

        ax.annotate(
            f"{run_id}\n{acc:.1f}%",
            xy=(tokens, acc),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=style["color"],
            horizontalalignment=ha_align,
            verticalalignment=va_align,
            zorder=5
        )

    ax.set_xlabel("Genomsnittligt antal tokens per fråga (Compute)", fontsize=11)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax.set_title("H2: Test-Time Compute — Agent vs. Tränad Bas vs. 7B Baseline",
                 fontsize=12, fontweight="bold")
    
    # Lägg till lite extra marginaler runt grafen så att ingen text klipps av mot kanterna
    ax.set_margins(0.12)
    
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#eaeded")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --- Summaryprint (Visar endast fokusmodellerna i tabellen för renhet) ---
    print("\n" + "=" * 60)
    print(f"  {'Fokusmodell (Förgrund)':<25} {'Acc':>6} {'Tokens/q':>10} {'Tok/correct':>13}")
    print("-" * 60)
    if foreground_rows:
        just_rows = [item[0] for item in foreground_rows]
        summary_df = pd.DataFrame(just_rows)
        for _, r in summary_df.sort_values("accuracy", ascending=False).iterrows():
            run_id = str(r["run_id"])
            acc    = float(r["accuracy"])
            tokens = float(r.get("avg_tokens", 0))
            tok_per_correct = (tokens / acc) if acc > 0 else float("inf")
            print(f"  {run_id:<25} {acc*100:>5.1f}%  {tokens:>9.0f}  {tok_per_correct:>12.0f}")
    print("=" * 60)