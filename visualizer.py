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
        ax_acc    = plot_mapping[size]["acc"]

        if isinstance(metrics, dict):
            log_history = metrics.get("log_history", [])
        elif isinstance(metrics, list):
            log_history = metrics
        else:
            log_history = []
        if not isinstance(log_history, list):
            log_history = []

        # --- Reward-kurvan (x = epoch) ---
        reward_epochs = []
        rewards = []
        for e in log_history:
            ep    = e.get("epoch")
            r_val = e.get("reward", e.get("train/reward", e.get("rewards/chosen", None)))
            if ep is not None and r_val is not None:
                reward_epochs.append(ep)
                rewards.append(r_val)

        if reward_epochs:
            ax_reward.plot(reward_epochs, rewards, "-", color=PCT_COLOR[pct],
                           linewidth=1.5, label=PCT_LABEL[pct])

        # --- Validerings-Accuracy (x = epoch) ---
        acc_hist = metrics.get("val_accuracy_history", []) if isinstance(metrics, dict) else []
        if not acc_hist:
            for e in log_history:
                if "eval_accuracy" in e or "val_accuracy" in e:
                    acc_val = e.get("eval_accuracy", e.get("val_accuracy"))
                    acc_hist.append({"epoch": e.get("epoch", 0), "val_accuracy": acc_val})

        if acc_hist:
            epochs = [h["epoch"] for h in acc_hist]
            # FIX: normalisera konsekvent — konvertera till % endast om värdet är i [0, 1]
            accs = [
                h["val_accuracy"] * 100 if h["val_accuracy"] <= 1.0 else h["val_accuracy"]
                for h in acc_hist
            ]
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
        mode   = str(r.get("mode", "")).lower()

        is_agent = (mode == "agent") or (run_id.endswith("_agent")) or ("agent" in run_id)
        if is_agent or "7b" in run_id:
            continue

        if "1_5b" in run_id or "1.5b" in run_id:
            size = 1.5
        elif "3b" in run_id:
            size = 3.0
        else:
            continue

        # OBS: matcha "10pct" FÖRE "5pct"/"2pct" för att undvika delsträngskollisioner
        if "10pct" in run_id or "10%" in run_id:
            pct = 10.0
        elif "5pct" in run_id or "5%" in run_id:
            pct = 5.0
        elif "2pct" in run_id or "2.5" in run_id:
            pct = 2.5
        elif ("untrained" in run_id or "h0" in run_id or "0pct" in run_id
              or "no training" in run_id or "no_training" in run_id or "0%" in run_id):
            pct = 0.0
        else:
            continue

        raw_acc = float(r["accuracy"])
        # FIX: normalisera till % oavsett om källdata är i [0,1] eller redan i %
        acc_pct = raw_acc * 100 if raw_acc <= 1.0 else raw_acc

        plot_data.append({"Size": size, "Data_Pct": pct, "Accuracy": acc_pct})

    if not plot_data:
        print("\n[VARNING] Kunde inte matcha rader i din DataFrame till H1-skalningen. Dina tillgängliga run_ids är:")
        print(df["run_id"].unique())
        return

    plot_df = pd.DataFrame(plot_data).sort_values("Data_Pct")

    fig, ax = plt.subplots(figsize=(8, 5))

    # FIX: sortera på Data_Pct innan drop_duplicates så att "last" alltid är högst pct
    df_15 = (plot_df[plot_df["Size"] == 1.5]
             .sort_values("Data_Pct")
             .drop_duplicates(subset=["Data_Pct"], keep="last"))
    if not df_15.empty:
        ax.plot(df_15["Data_Pct"], df_15["Accuracy"], "-o",
                color="#e74c3c", linewidth=2.5, label="Qwen 1.5B + GRPO", markersize=8)

    df_3 = (plot_df[plot_df["Size"] == 3.0]
            .sort_values("Data_Pct")
            .drop_duplicates(subset=["Data_Pct"], keep="last"))
    if not df_3.empty:
        ax.plot(df_3["Data_Pct"], df_3["Accuracy"], "-o",
                color="#3498db", linewidth=2.5, label="Qwen 3B + GRPO", markersize=8)

    ax.set_xlabel("Mängd GRPO-träningsdata (%)", fontsize=11)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax.set_title("Hypotes 1: Skalningslagar & Dataeffektivitet under GRPO",
                 fontsize=12, fontweight="bold")
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
    Plottar 3b_10pct, 3b_10pct_agent och math_7b_baseline i förgrunden med labels.
    Alla andra modeller plottas som skuggade grå punkter utan labels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    STYLE = {
        "3b_10pct":       {"color": "#2ecc71", "marker": "s", "s": 140, "label": "3B 10% Baseline"},
        "3b_10pct_agent": {"color": "#e67e22", "marker": "o", "s": 160, "label": "Agent Loop (3B 10% + PRM)"},
        "math_7b":        {"color": "#9b59b6", "marker": "X", "s": 200, "label": "SOTA Baseline (Math-7B)"},
    }

    FOREGROUND_IDS = {"3b_10pct", "3b_10pct_agent", "math_7b_baseline"}

    foreground_rows = []
    background_rows = []

    for _, r in df.iterrows():
        run_id = str(r["run_id"]).lower().strip()
        if run_id in FOREGROUND_IDS:
            style_key = "math_7b" if run_id == "math_7b_baseline" else run_id
            foreground_rows.append((r, style_key))
        else:
            background_rows.append(r)

    # --- 1. Bakgrund (skuggade, utan label) ---
    for r in background_rows:
        raw_acc = float(r["accuracy"])
        acc     = raw_acc * 100 if raw_acc <= 1.0 else raw_acc
        tokens  = float(r.get("avg_tokens", 0))
        ax.scatter(tokens, acc,
                   color="#bdc3c7", marker="o", s=50,
                   alpha=1, edgecolors="none", zorder=2)

    # --- 2. Förgrund: samla punkter ---
    fg_points = []
    for r, key in foreground_rows:
        raw_acc = float(r["accuracy"])
        acc     = raw_acc * 100 if raw_acc <= 1.0 else raw_acc
        tokens  = float(r.get("avg_tokens", 0))
        fg_points.append((tokens, acc, key, str(r["run_id"])))

    # Rita scatter-punkterna
    labels_added = set()
    for tokens, acc, key, run_id in fg_points:
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
                   zorder=4)

    ax.margins(0.15)
    ax.autoscale_view()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Annotationer med kollisions- och kantdetektering
    placed_boxes = []

    def overlaps(bx0, by0, bx1, by1):
        for (ox0, oy0, ox1, oy1) in placed_boxes:
            if bx0 < ox1 and bx1 > ox0 and by0 < oy1 and by1 > oy0:
                return True
        return False

    OFFSETS = [
        ( 12,  6, "left",   "bottom"),
        (-12,  6, "right",  "bottom"),
        ( 12, -6, "left",   "top"),
        (-12, -6, "right",  "top"),
        (  0, 14, "center", "bottom"),
        (  0,-14, "center", "top"),
        ( 18,  0, "left",   "center"),
        (-18,  0, "right",  "center"),
    ]

    fig_w, fig_h = fig.get_size_inches()
    char_w_data  = x_range / (fig_w * 8.5)
    line_h_data  = y_range / (fig_h * 6.5)

    for tokens, acc, key, run_id in fg_points:
        style = STYLE[key]
        text  = f"{run_id}\n{acc:.1f}%"
        lines = text.split("\n")
        tw    = max(len(l) for l in lines) * char_w_data
        th    = len(lines) * line_h_data

        # FIX: starta med None för att spåra om vi faktiskt hittar en giltig position
        chosen = None

        for xoff_pt, yoff_pt, ha, va in OFFSETS:
            xoff_data = xoff_pt / 72 * (x_range / fig_w)
            yoff_data = yoff_pt / 72 * (y_range / fig_h)

            if ha == "left":
                bx0 = tokens + xoff_data
                bx1 = bx0 + tw
            elif ha == "right":
                bx1 = tokens + xoff_data
                bx0 = bx1 - tw
            else:
                bx0 = tokens + xoff_data - tw / 2
                bx1 = tokens + xoff_data + tw / 2

            if va == "bottom":
                by0 = acc + yoff_data
                by1 = by0 + th
            elif va == "top":
                by1 = acc + yoff_data
                by0 = by1 - th
            else:
                by0 = acc + yoff_data - th / 2
                by1 = acc + yoff_data + th / 2

            margin_x = x_range * 0.01
            margin_y = y_range * 0.01
            in_bounds = (bx0 >= x_min + margin_x and bx1 <= x_max - margin_x and
                         by0 >= y_min + margin_y and by1 <= y_max - margin_y)

            if in_bounds and not overlaps(bx0, by0, bx1, by1):
                chosen = (xoff_pt, yoff_pt, ha, va, bx0, by0, bx1, by1)
                break

        # FIX: om ingen position hittades, räkna ut fallback-boxen explicit (OFFSETS[0])
        if chosen is None:
            xoff_pt, yoff_pt, ha, va = OFFSETS[0]
            xoff_data = xoff_pt / 72 * (x_range / fig_w)
            yoff_data = yoff_pt / 72 * (y_range / fig_h)
            bx0 = tokens + xoff_data
            bx1 = bx0 + tw
            by0 = acc + yoff_data
            by1 = by0 + th
            chosen = (xoff_pt, yoff_pt, ha, va, bx0, by0, bx1, by1)

        xoff_pt, yoff_pt, ha, va, bx0, by0, bx1, by1 = chosen
        placed_boxes.append((bx0, by0, bx1, by1))

        ax.annotate(
            text,
            xy=(tokens, acc),
            xytext=(xoff_pt, yoff_pt),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color=style["color"],
            horizontalalignment=ha,
            verticalalignment=va,
            zorder=5
        )


    for r in background_rows:
        raw_acc = float(r["accuracy"])
        acc     = raw_acc * 100 if raw_acc <= 1.0 else raw_acc
        tokens  = float(r.get("avg_tokens", 0))
        run_id  = str(r["run_id"])

        text  = f"{run_id}\n{acc:.1f}%"
        lines = text.split("\n")
        tw    = max(len(l) for l in lines) * char_w_data
        th    = len(lines) * line_h_data

        chosen = None
        for xoff_pt, yoff_pt, ha, va in OFFSETS:
            xoff_data = xoff_pt / 72 * (x_range / fig_w)
            yoff_data = yoff_pt / 72 * (y_range / fig_h)

            if ha == "left":
                bx0, bx1 = tokens + xoff_data, tokens + xoff_data + tw
            elif ha == "right":
                bx1 = tokens + xoff_data
                bx0 = bx1 - tw
            else:
                bx0 = tokens + xoff_data - tw / 2
                bx1 = tokens + xoff_data + tw / 2

            if va == "bottom":
                by0, by1 = acc + yoff_data, acc + yoff_data + th
            elif va == "top":
                by1, by0 = acc + yoff_data, acc + yoff_data - th
            else:
                by0 = acc + yoff_data - th / 2
                by1 = acc + yoff_data + th / 2

            margin_x = x_range * 0.01
            margin_y = y_range * 0.01
            in_bounds = (bx0 >= x_min + margin_x and bx1 <= x_max - margin_x and
                        by0 >= y_min + margin_y and by1 <= y_max - margin_y)

            if in_bounds and not overlaps(bx0, by0, bx1, by1):
                chosen = (xoff_pt, yoff_pt, ha, va, bx0, by0, bx1, by1)
                break

        if chosen is None:
            # Hoppa över — ingen plats hittades, bättre än att stapla labels
            continue

        xoff_pt, yoff_pt, ha, va, bx0, by0, bx1, by1 = chosen
        placed_boxes.append((bx0, by0, bx1, by1))

        ax.annotate(
            text,
            xy=(tokens, acc),
            xytext=(xoff_pt, yoff_pt),
            textcoords="offset points",
            fontsize=7,                      # mindre än förgrundslabels (9pt)
            fontweight="normal",             # inte bold
            color="#888888",                 # grå, matchar punkterna
            horizontalalignment=ha,
            verticalalignment=va,
            zorder=3,                        # under förgrundslabels (5) men över bakgrund (2)
            arrowprops=dict(
                arrowstyle="-",              # enkel linje, ingen pil
                color="#bbbbbb",
                lw=0.6,
            ),
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.6,
            )
        )    

    ax.set_xlabel("Genomsnittligt antal tokens per fråga (Compute)", fontsize=11)
    ax.set_ylabel("GSM8K Accuracy (%)", fontsize=11)
    ax.set_title("H2: Test-Time Compute — Agent vs. Tränad Bas vs. 7B Baseline",
                 fontsize=12, fontweight="bold")

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#eaeded")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  {'Fokusmodell (Förgrund)':<25} {'Acc':>6} {'Tokens/q':>10} {'Tok/correct':>13}")
    print("-" * 60)
    if foreground_rows:
        summary_df = pd.DataFrame([item[0] for item in foreground_rows])
        for _, r in summary_df.sort_values("accuracy", ascending=False).iterrows():
            run_id  = str(r["run_id"])
            raw_acc = float(r["accuracy"])
            acc     = raw_acc if raw_acc > 1.0 else raw_acc * 100
            tokens  = float(r.get("avg_tokens", 0))
            tok_per_correct = (tokens / (acc / 100)) if acc > 0 else float("inf")
            print(f"  {run_id:<25} {acc:>5.1f}%  {tokens:>9.0f}  {tok_per_correct:>12.0f}")
    print("=" * 60)