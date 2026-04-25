"""
CMAPSS FD001/FD002/FD004 — Reliability & RUL dashboard.

Run:
    python app_multi.py

Expects the 9 CMAPSS files in DATA_DIR (default ./data):
    train_FDxxx.txt, test_FDxxx.txt, RUL_FDxxx.txt  for xxx in {001, 002, 004}

On first run per dataset, trains CatBoost and caches it to disk.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sstats
from scipy.special import gammaln
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
import gradio as gr

from utils import (load as load_raw, build_features, select_informative_sensors, fit_regime_stats, em_weibull_mixture)

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
 
CAP = 150          # Pareto-optimum from the cap sweep
N_REGIMES = {"FD001": 1, "FD002": 6, "FD004": 6}
 
 
@dataclass
class DatasetModel:
    ds: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul_true: np.ndarray
    km: Optional[KMeans]
    sensors: list
    regime_stats: dict
    model: CatBoostRegressor
    feature_cols: list
    test_feats: pd.DataFrame
    # Reliability
    life_all: np.ndarray
    beta: float
    eta: float
    mixture: Optional[dict]   # Only for FD004
 
 
def prepare_dataset(ds: str) -> DatasetModel:
    cache = CACHE_DIR / f"{ds}.cbm"
 
    tr, te, rul_true = load_raw(ds)
    K = N_REGIMES[ds]
    if K == 1:
        tr = tr.assign(regime=0)
        te = te.assign(regime=0)
        km = None
    else:
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(
            tr[["op_1", "op_2", "op_3"]].values)
        tr = tr.assign(regime=km.labels_)
        te = te.assign(regime=km.predict(te[["op_1", "op_2", "op_3"]].values))
 
    sensors = select_informative_sensors(tr, "regime")
    regime_stats = fit_regime_stats(tr, sensors, "regime")
    feats_tr = build_features(tr, sensors, normalized=True, stats=regime_stats)
    feats_tr["regime"] = feats_tr["regime"].astype(int)
    y = (tr.groupby("unit").cycle.transform("max") - tr.cycle).clip(upper=CAP).values
    feature_cols = [c for c in feats_tr.columns if c not in ("unit", "cycle")]
 
    if cache.exists():
        model = CatBoostRegressor()
        model.load_model(str(cache))
    else:
        print(f"Training CatBoost for {ds} (first run)...")
        rng = np.random.default_rng(0)
        all_u = feats_tr["unit"].unique()
        val_u = rng.choice(all_u, size=max(20, len(all_u) // 5), replace=False)
        is_val = feats_tr["unit"].isin(val_u).values
        model = CatBoostRegressor(
            iterations=1500, learning_rate=0.05, depth=6,
            loss_function="RMSE", eval_metric="RMSE",
            early_stopping_rounds=50, random_seed=42, verbose=False,
            cat_features=["regime"],
        )
        model.fit(feats_tr.loc[~is_val, feature_cols], y[~is_val],
                  eval_set=(feats_tr.loc[is_val, feature_cols], y[is_val]))
        model.save_model(str(cache))
 
    feats_te = build_features(te, sensors, normalized=True, stats=regime_stats)
    feats_te["regime"] = feats_te["regime"].astype(int)
 
    life_tr = tr.groupby("unit").cycle.max().values.astype(float)
    life_te = (te.groupby("unit").cycle.max().values + rul_true).astype(float)
    life_all = np.concatenate([life_tr, life_te])
    beta, _, eta = sstats.weibull_min.fit(life_all, floc=0)
 
    mixture = None
    if ds == "FD004":
        mixture = em_weibull_mixture(life_all, K=2, n_iter=200)
 
    return DatasetModel(ds=ds, train=tr, test=te, rul_true=rul_true, km=km,
                        sensors=sensors, regime_stats=regime_stats, model=model,
                        feature_cols=feature_cols, test_feats=feats_te,
                        life_all=life_all, beta=beta, eta=eta, mixture=mixture)
 
 
MODELS = {ds: prepare_dataset(ds) for ds in ["FD001", "FD002", "FD004"]}
print("All datasets ready:", list(MODELS.keys()))
 
 
# --------------------------- Callbacks ---------------------------
def fleet_overview(ds: str):
    m = MODELS[ds]
 
    last = m.test_feats.groupby("unit").tail(1).sort_values("unit")
    preds = np.clip(m.model.predict(last[m.feature_cols]), 0, None)
    rmse = float(np.sqrt(((preds - m.rul_true) ** 2).mean()))
    mae = float(np.mean(np.abs(preds - m.rul_true)))
 
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{ds} — fleet survival (n={len(m.life_all)})",
                        f"{ds} — test set, RMSE = {rmse:.1f}, MAE = {mae:.1f}"),
        horizontal_spacing=0.12,
    )
 
    life = np.sort(m.life_all)
    S_emp = 1 - np.arange(1, len(life) + 1) / len(life)
    fig.add_trace(go.Scatter(x=life, y=S_emp, mode="lines", line_shape="hv",
                             name="empirical", line=dict(color="#1f77b4", width=2)),
                  row=1, col=1)
    t = np.linspace(1, life.max() + 30, 500)
    fig.add_trace(go.Scatter(x=t, y=sstats.weibull_min.sf(t, m.beta, 0, m.eta),
                             mode="lines",
                             name=f"Weibull β={m.beta:.2f} η={m.eta:.0f}",
                             line=dict(color="crimson", width=2)),
                  row=1, col=1)
    if m.mixture is not None:
        mx = m.mixture
        S_mix = (mx["pi"][0] * sstats.weibull_min.sf(t, mx["beta"][0], 0, mx["eta"][0])
                 + mx["pi"][1] * sstats.weibull_min.sf(t, mx["beta"][1], 0, mx["eta"][1]))
        fig.add_trace(go.Scatter(x=t, y=S_mix, mode="lines", name="2-comp mixture",
                                 line=dict(color="green", width=2, dash="dash")),
                      row=1, col=1)
 
    fig.add_trace(go.Scatter(
        x=m.rul_true, y=preds, mode="markers",
        marker=dict(size=7, color="#1f77b4", opacity=0.65,
                    line=dict(color="black", width=0.5)),
        name="test units",
        hovertemplate="true=%{x:.0f}<br>pred=%{y:.1f}<extra></extra>",
    ), row=1, col=2)
    lim = max(m.rul_true.max(), preds.max()) + 5
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                             line=dict(color="black", width=1, dash="dash"),
                             showlegend=False, hoverinfo="skip"),
                  row=1, col=2)
 
    fig.update_xaxes(title_text="Cycles", row=1, col=1)
    fig.update_yaxes(title_text="S(t)", row=1, col=1)
    fig.update_xaxes(title_text="True RUL", row=1, col=2)
    fig.update_yaxes(title_text="Predicted RUL", row=1, col=2)
    fig.update_layout(height=440, margin=dict(t=60, b=50, l=50, r=20),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="left", x=0))
 
    summary = [
        f"### {ds} model card",
         "| Metric | Value |",
         "|---|---|",
        f"| Units (train / test) | {m.train.unit.nunique()} / {m.test.unit.nunique()} |",
        f"| Operational regimes | {N_REGIMES[ds]} |",
        f"| Informative sensors | {len(m.sensors)} |",
        f"| Weibull β / η | {m.beta:.2f} / {m.eta:.0f} |",
        f"| Weibull MTTF | {m.eta * np.exp(gammaln(1 + 1/m.beta)):.0f} cycles |",
        f"| CatBoost test RMSE | **{rmse:.1f}** |",
        f"| CatBoost test MAE | {mae:.1f} |",
        f"| Training RUL cap | {CAP} |",
    ]
    if m.mixture is not None:
        mx = m.mixture
        summary += [
            "",
            "**FD004 failure-mode mixture:**",
            f"- Component 1 (short-life): π={mx['pi'][0]:.2f}, β={mx['beta'][0]:.2f}, η={mx['eta'][0]:.0f}",
            f"- Component 2 (long-life): π={mx['pi'][1]:.2f}, β={mx['beta'][1]:.2f}, η={mx['eta'][1]:.0f}",
        ]
    return fig, "\n".join(summary)
 
 
def unit_choices(ds: str):
    m = MODELS[ds]
    units = sorted(m.test.unit.unique().tolist())
    return gr.update(choices=units, value=units[0])
 
 
def inspect_unit(ds: str, unit_id: int):
    if unit_id is None:
        return None, ""
    m = MODELS[ds]
    unit_id = int(unit_id)
    sub = m.test_feats[m.test_feats.unit == unit_id].sort_values("cycle").reset_index(drop=True)
    if len(sub) == 0:
        return None, f"Unit #{unit_id} not in {ds} test set."
    preds = np.clip(m.model.predict(sub[m.feature_cols]), 0, None)
    last_cycle = int(sub["cycle"].max())
    pred_now = float(preds[-1])
    true_now = float(m.rul_true[unit_id - 1])
    true_total_life = last_cycle + true_now
 
    # Top 4 sensors most correlated with capped RUL on train
    sub_train = m.train.copy()
    sub_train["RUL"] = (sub_train.groupby("unit").cycle.transform("max")
                        - sub_train.cycle).clip(upper=CAP)
    corrs = (sub_train[m.sensors + ["RUL"]].corr()["RUL"]
             .drop("RUL").abs().sort_values(ascending=False))
    top_sensors = corrs.head(4).index.tolist()
 
    raw = m.test[m.test.unit == unit_id].sort_values("cycle")
    S_now = float(sstats.weibull_min.sf(last_cycle, m.beta, 0, m.eta))
    has_regimes = m.km is not None
 
    titles = [
        f"Unit #{unit_id} — RUL trajectory",
        f"Position on fleet survival (S={S_now:.1%})",
        "Top-correlated sensors — raw",
        "Degradation signal (rolling slope, regime-normalized)",
    ]
    if has_regimes:
        titles.append(
            f"Operational regime timeline (visited: "
            f"{sorted(raw.regime.unique().tolist())})"
        )
        specs = [[{}, {}], [{}, {}], [{"colspan": 2}, None]]
        rows, height = 3, 900
    else:
        specs = [[{}, {}], [{}, {}]]
        rows, height = 2, 700
 
    fig = make_subplots(
        rows=rows, cols=2, specs=specs, subplot_titles=titles,
        vertical_spacing=0.10, horizontal_spacing=0.10,
    )
 
    # 1. RUL trajectory
    true_rul_traj = true_now + (last_cycle - sub["cycle"].values)
    fig.add_trace(go.Scatter(x=sub["cycle"], y=true_rul_traj, mode="lines",
                             name="true RUL", line=dict(color="black", width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["cycle"], y=preds, mode="lines",
                             name="CatBoost prediction",
                             line=dict(color="crimson", width=2)),
                  row=1, col=1)
    fig.add_hline(y=CAP, line=dict(color="gray", width=1, dash="dot"),
                  annotation_text=f"training cap={CAP}",
                  annotation_position="top right",
                  row=1, col=1)
 
    # 2. Fleet Weibull + position
    t = np.linspace(1, m.life_all.max() + 20, 400)
    fig.add_trace(go.Scatter(x=t, y=sstats.weibull_min.sf(t, m.beta, 0, m.eta),
                             mode="lines", name=f"Weibull β={m.beta:.2f}",
                             line=dict(color="crimson", width=2)),
                  row=1, col=2)
    if m.mixture is not None:
        mx = m.mixture
        S_mix = (mx["pi"][0] * sstats.weibull_min.sf(t, mx["beta"][0], 0, mx["eta"][0])
                 + mx["pi"][1] * sstats.weibull_min.sf(t, mx["beta"][1], 0, mx["eta"][1]))
        fig.add_trace(go.Scatter(x=t, y=S_mix, mode="lines", name="mixture",
                                 line=dict(color="green", width=2, dash="dash")),
                      row=1, col=2)
    fig.add_vline(x=last_cycle, line=dict(color="#1f77b4", width=2),
                  annotation_text=f"age = {last_cycle}",
                  annotation_position="top right",
                  row=1, col=2)
    fig.add_vline(x=true_total_life,
                  line=dict(color="black", width=1.5, dash="dash"),
                  annotation_text=f"failure @ {true_total_life:.0f}",
                  annotation_position="bottom right",
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=[last_cycle], y=[S_now], mode="markers",
                             marker=dict(color="#1f77b4", size=11,
                                         line=dict(color="black", width=1)),
                             name="current state", showlegend=False,
                             hovertemplate=f"S={S_now:.1%}<extra></extra>"),
                  row=1, col=2)
 
    # 3 & 4. Top sensors — raw + rolling slope
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for s, color in zip(top_sensors, palette):
        fig.add_trace(go.Scatter(x=raw["cycle"], y=raw[s], mode="lines",
                                 name=s, line=dict(color=color, width=1.5),
                                 legendgroup=s),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=sub["cycle"], y=sub[f"{s}_n_sl"], mode="lines",
                                 name=f"slope({s})",
                                 line=dict(color=color, width=1.5, dash="dot"),
                                 legendgroup=s, showlegend=False),
                      row=2, col=2)
    fig.add_hline(y=0, line=dict(color="black", width=0.8), row=2, col=2)
 
    # 5. Regime timeline (full-width on row 3)
    if has_regimes:
        fig.add_trace(go.Scatter(x=raw["cycle"], y=raw["regime"], mode="lines",
                                 line_shape="hv",
                                 line=dict(color="#9467bd", width=2),
                                 name="regime", showlegend=False),
                      row=3, col=1)
        fig.update_yaxes(tickmode="array",
                         tickvals=list(range(N_REGIMES[ds])),
                         row=3, col=1)
 
    fig.update_xaxes(title_text="Cycle", row=1, col=1)
    fig.update_yaxes(title_text="RUL (cycles)", row=1, col=1)
    fig.update_xaxes(title_text="Cycles", row=1, col=2)
    fig.update_yaxes(title_text="S(t)", row=1, col=2)
    fig.update_xaxes(title_text="Cycle", row=2, col=1)
    fig.update_yaxes(title_text="Sensor value", row=2, col=1)
    fig.update_xaxes(title_text="Cycle", row=2, col=2)
    fig.update_yaxes(title_text="Rolling slope", row=2, col=2)
    if has_regimes:
        fig.update_xaxes(title_text="Cycle", row=3, col=1)
        fig.update_yaxes(title_text="Regime", row=3, col=1)
 
    fig.update_layout(height=height, margin=dict(t=50, b=40, l=50, r=20),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.08,
                                  xanchor="left", x=0))
 
    verdict = ("CRITICAL — act now" if pred_now < 20
               else "WARNING — plan maintenance" if pred_now < 50
               else "HEALTHY")
    md = f"""### Unit #{unit_id} — {ds}
 
| Metric | Value |
|---|---|
| Dataset | {ds} ({N_REGIMES[ds]} regime{'s' if N_REGIMES[ds] > 1 else ''}) |
| Current age | **{last_cycle}** cycles |
| Predicted RUL | **{pred_now:.1f}** cycles |
| True RUL | {true_now:.0f} cycles |
| |Pred − True| | {abs(pred_now - true_now):.1f} cycles |
| Weibull S(age) | {S_now:.1%} |
| Status | **{verdict}** |
"""
    return fig, md
 
 
# --------------------------- UI ---------------------------
with gr.Blocks(title="CMAPSS Multi-Dataset Reliability Dashboard") as demo:
    gr.Markdown("# CMAPSS Reliability & RUL Dashboard — FD001 / FD002 / FD004")
    gr.Markdown(
        "Each dataset is handled end-to-end: regime clustering · per-regime "
        f"sensor normalization · Weibull fleet model · CatBoost RUL prediction (cap={CAP})."
    )
 
    ds_selector = gr.Radio(choices=["FD001", "FD002", "FD004"],
                           value="FD001", label="Dataset")
 
    with gr.Tab("Fleet overview"):
        gr.Markdown(
            "Survival of the fleet (empirical vs Weibull, and 2-component "
            "mixture on FD004) + test-set RUL predictions."
        )
        with gr.Row():
            with gr.Column(scale=3):
                plot_fleet = gr.Plot()
            with gr.Column(scale=1):
                md_fleet = gr.Markdown()
 
    with gr.Tab("Unit inspection"):
        gr.Markdown(
            "Pick a test unit to see RUL prediction, position on the fleet "
            "survival curve, sensor trajectories, and operational regime timeline."
        )
        unit_dd = gr.Dropdown(choices=sorted(MODELS["FD001"].test.unit.unique().tolist()),
                              value=1, label="Test unit", allow_custom_value=False)
        with gr.Row():
            with gr.Column(scale=3):
                plot_unit = gr.Plot()
            with gr.Column(scale=1):
                md_unit = gr.Markdown()
 
    # Wire events
    ds_selector.change(fleet_overview, inputs=ds_selector, outputs=[plot_fleet, md_fleet])
    ds_selector.change(unit_choices, inputs=ds_selector, outputs=unit_dd)
    ds_selector.change(inspect_unit, inputs=[ds_selector, unit_dd], outputs=[plot_unit, md_unit])
    unit_dd.change(inspect_unit, inputs=[ds_selector, unit_dd], outputs=[plot_unit, md_unit])
 
    demo.load(fleet_overview, inputs=ds_selector, outputs=[plot_fleet, md_fleet])
    demo.load(inspect_unit, inputs=[ds_selector, unit_dd], outputs=[plot_unit, md_unit])
 
 
if __name__ == "__main__":
    demo.launch()
 