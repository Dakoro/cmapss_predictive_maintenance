"""
CMAPSS FD001 — Reliability & RUL dashboard.

Run:
    python app.py

Expects the three FD001 files in DATA_DIR (default ./data).
On first run, trains CatBoost and caches it as catboost_fd001.cbm.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln
from catboost import CatBoostRegressor
import gradio as gr

DATA_DIR = Path(__file__).parent / "data/CMAPSSData"
MODEL_PATH = Path(__file__).parent / "models/catboost_fd001.cbm"

COLS = ["unit", "cycle"] + [f"op_{i}" for i in range(1, 4)] + [f"s_{i}" for i in range(1, 22)]
DROP = ["op_1", "op_2", "op_3", "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
SENSORS = [c for c in COLS if c.startswith("s_") and c not in DROP]
WINDOW = 15
RUL_CAP = 125


def load_data():
    train = pd.read_csv(DATA_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COLS, engine="python")
    test = pd.read_csv(DATA_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COLS, engine="python")
    rul = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["RUL"]).squeeze("columns").values
    return train, test, rul


def build_features(df):
    out = [df[["unit", "cycle"] + SENSORS].copy()]
    g = df.groupby("unit")[SENSORS]
    roll = g.rolling(WINDOW, min_periods=1)
    out.append(roll.mean().reset_index(level=0, drop=True).add_suffix(f"_m{WINDOW}"))
    out.append(roll.std().reset_index(level=0, drop=True).fillna(0).add_suffix(f"_sd{WINDOW}"))

    def slope(x):
        if len(x) < 2:
            return 0.0
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    sl = g.rolling(WINDOW, min_periods=2).apply(slope, raw=True).reset_index(level=0, drop=True)
    out.append(sl.fillna(0).add_suffix(f"_sl{WINDOW}"))
    return pd.concat(out, axis=1)


def train_or_load(train_df, feature_cols):
    if MODEL_PATH.exists():
        m = CatBoostRegressor()
        m.load_model(str(MODEL_PATH))
        return m
    print("Training CatBoost (first run, ~30s)...")
    t = train_df.copy()
    t["RUL"] = t.groupby("unit").cycle.transform("max") - t.cycle
    t["RUL_capped"] = t["RUL"].clip(upper=RUL_CAP)
    X = build_features(t)
    y = t["RUL_capped"].values
    rng = np.random.default_rng(0)
    val_u = rng.choice(np.arange(1, 101), size=20, replace=False)
    is_val = X["unit"].isin(val_u).values
    m = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6,
                          loss_function="RMSE", early_stopping_rounds=50,
                          random_seed=42, verbose=False)
    m.fit(X.loc[~is_val, feature_cols].values, y[~is_val],
          eval_set=(X.loc[is_val, feature_cols].values, y[is_val]))
    m.save_model(str(MODEL_PATH))
    return m


# --- Load everything once at import time ---
train_df, test_df, rul_true = load_data()
life_train = train_df.groupby("unit").cycle.max().values
life_test = test_df.groupby("unit").cycle.max().values + rul_true
life_all = np.concatenate([life_train, life_test]).astype(float)

beta, _, eta = stats.weibull_min.fit(life_all, floc=0)
mttf_weibull = eta * np.exp(gammaln(1 + 1 / beta))
B10_weibull = eta * (-np.log(0.9)) ** (1 / beta)

feat_template = build_features(train_df.head(100))
FEATURE_COLS = [c for c in feat_template.columns if c not in ("unit", "cycle")]
model = train_or_load(train_df, FEATURE_COLS)

test_features = build_features(test_df)
TEST_UNITS = sorted(test_df["unit"].unique().tolist())


def fleet_overview():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    t_grid = np.linspace(1, life_all.max() + 20, 400)
    S = stats.weibull_min.sf(t_grid, beta, 0, eta)
    sorted_t = np.sort(life_all)
    S_emp = 1 - np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    ax1.step(sorted_t, S_emp, where="post", color="tab:blue", label="empirical", lw=1.5)
    ax1.plot(t_grid, S, "r-", lw=2, label=f"Weibull (β={beta:.2f}, η={eta:.0f})")
    ax1.axhline(0.1, color="gray", ls=":", lw=0.7)
    ax1.axvline(B10_weibull, color="gray", ls=":", lw=0.7)
    ax1.set_xlabel("Cycles")
    ax1.set_ylabel("S(t)")
    ax1.set_title("Fleet survival — FD001 (n=200)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    last = test_features.groupby("unit").tail(1).sort_values("unit")
    preds = np.clip(model.predict(last[FEATURE_COLS].values), 0, None)
    rmse = np.sqrt(np.mean((preds - rul_true) ** 2))
    ax2.scatter(rul_true, preds, alpha=0.7, edgecolor="black", lw=0.5)
    lim = max(rul_true.max(), preds.max()) + 5
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax2.set_xlabel("True RUL")
    ax2.set_ylabel("Predicted RUL (CatBoost)")
    ax2.set_title(f"Test set — RMSE = {rmse:.1f} cycles")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def inspect_unit(unit_id):
    sub = test_features[test_features.unit == unit_id].sort_values("cycle").reset_index(drop=True)
    preds = np.clip(model.predict(sub[FEATURE_COLS].values), 0, None)
    last_cycle = int(sub["cycle"].max())
    pred_now = float(preds[-1])
    true_now = float(rul_true[unit_id - 1])
    true_total_life = last_cycle + true_now

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))

    ax = axes[0, 0]
    true_rul = true_now + (last_cycle - sub["cycle"].values)
    ax.plot(sub["cycle"], true_rul, "k-", lw=1.6, label="true RUL")
    ax.plot(sub["cycle"], preds, "r-", lw=1.4, label="CatBoost prediction")
    ax.axhline(RUL_CAP, color="gray", ls=":", lw=0.8, label=f"training cap = {RUL_CAP}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Unit #{unit_id} — last observation at cycle {last_cycle}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    t_grid = np.linspace(1, life_all.max() + 20, 400)
    S = stats.weibull_min.sf(t_grid, beta, 0, eta)
    ax.plot(t_grid, S, "r-", lw=2, label=f"Weibull (β={beta:.2f})")
    ax.axvline(last_cycle, color="tab:blue", lw=2,
               label=f"current age = {last_cycle}")
    ax.axvline(true_total_life, color="black", ls="--", lw=1.2,
               label=f"actual failure = {true_total_life:.0f}")
    S_now = stats.weibull_min.sf(last_cycle, beta, 0, eta)
    ax.plot(last_cycle, S_now, "o", color="tab:blue", ms=9)
    ax.annotate(f"S={S_now:.1%}", (last_cycle, S_now),
                xytext=(10, 10), textcoords="offset points")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("S(t)")
    ax.set_title("Survival curve — position of this unit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for s in ["s_4", "s_11", "s_12", "s_15"]:
        ax.plot(sub["cycle"], sub[s], lw=1.2, label=s)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Sensor value (raw)")
    ax.set_title("Top-correlated sensors — raw trajectories")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for s in ["s_4", "s_11", "s_12", "s_15"]:
        ax.plot(sub["cycle"], sub[f"{s}_sl{WINDOW}"], lw=1.2, label=f"slope({s})")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"Rolling slope (window={WINDOW})")
    ax.set_title("Degradation signal — rolling slopes")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    verdict = ("CRITICAL — act now" if pred_now < 20
               else "WARNING — plan maintenance" if pred_now < 50
               else "HEALTHY")
    summary = f"""### Unit #{unit_id} — diagnosis

| Metric | Value |
|---|---|
| Current age | **{last_cycle} cycles** |
| Predicted RUL | **{pred_now:.1f} cycles** |
| True RUL | {true_now:.0f} cycles |
| Absolute error | {abs(pred_now - true_now):.1f} cycles |
| Weibull survival S(age) | {S_now:.1%} |
| Weibull MTTF | {mttf_weibull:.0f} cycles |
| Status | **{verdict}** |
"""
    return fig, summary


with gr.Blocks(title="CMAPSS FD001 — Reliability Dashboard") as demo:
    gr.Markdown("# CMAPSS FD001 — Reliability & Remaining Useful Life")
    gr.Markdown(
        f"**Fleet model**  Weibull(β={beta:.2f}, η={eta:.0f}) · "
        f"MTTF={mttf_weibull:.0f} · B10={B10_weibull:.0f}  \n"
        "**RUL model**  CatBoost on rolling sensor features (window=15, cap=125)"
    )

    with gr.Tab("Fleet overview"):
        gr.Markdown("Survival curve of the fleet (empirical vs Weibull) "
                    "and RUL predictions on the 100 test units.")
        btn = gr.Button("Generate", variant="primary")
        plot_fleet = gr.Plot()
        btn.click(fleet_overview, outputs=plot_fleet)
        demo.load(fleet_overview, outputs=plot_fleet)

    with gr.Tab("Unit inspection"):
        gr.Markdown("Pick a test unit to see its sensor trajectory, "
                    "RUL prediction, and position on the fleet survival curve.")
        unit_dd = gr.Dropdown(choices=TEST_UNITS, value=TEST_UNITS[0],
                              label="Test unit")
        with gr.Row():
            with gr.Column(scale=3):
                plot_unit = gr.Plot()
            with gr.Column(scale=1):
                md_unit = gr.Markdown()
        unit_dd.change(inspect_unit, inputs=unit_dd, outputs=[plot_unit, md_unit])
        demo.load(inspect_unit, inputs=unit_dd, outputs=[plot_unit, md_unit])


if __name__ == "__main__":
    demo.launch()