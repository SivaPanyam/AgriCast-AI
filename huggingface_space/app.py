import gradio as gr
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load trained bundle
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sarima_model.pkl"

bundle = None
if MODEL_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

def run_forecast(horizon, model_choice, monsoon_shock, fuel_shock, tariff_shock):
    if not bundle:
        return "Model not found. Please train sarima_model.pkl.", None, None

    steps = int(horizon)
    sarima_model = bundle.get("sarima_model") or bundle.get("fitted_model")
    hw_model = bundle.get("hw_model")
    xgb_artifact = bundle.get("xgb_artifact")
    ts = bundle.get("historical_series")

    # 1. SARIMA
    f_res = sarima_model.get_forecast(steps=steps)
    f_df = f_res.summary_frame()
    se = f_res.se_mean
    dates = [d.strftime("%Y-%m-%d") for d in f_df.index]
    sarima_vals = [float(v) for v in f_df["mean"].values]

    # 2. Holt-Winters
    if hw_model:
        hw_preds = hw_model.forecast(steps=steps)
        hw_vals = [float(v) for v in hw_preds.values]
    else:
        hw_vals = sarima_vals

    # 3. XGBoost
    if xgb_artifact:
        xgb_reg = xgb_artifact["model"]
        window = list(xgb_artifact["last_window"])
        xgb_vals = []
        for _ in range(steps):
            feat = np.array(window[-12:][::-1]).reshape(1, -1)
            p = float(xgb_reg.predict(feat)[0])
            window.append(p)
            xgb_vals.append(p)
    else:
        xgb_vals = sarima_vals

    # 4. Smart Ensemble (55% SARIMA, 25% HW, 20% XGB)
    ensemble_vals = [
        round(0.55 * sarima_vals[i] + 0.25 * hw_vals[i] + 0.20 * xgb_vals[i], 2)
        for i in range(steps)
    ]

    # Select values
    if model_choice == "Smart Ensemble":
        chosen_vals = ensemble_vals
    elif model_choice == "SARIMA":
        chosen_vals = sarima_vals
    elif model_choice == "Holt-Winters":
        chosen_vals = hw_vals
    else:
        chosen_vals = xgb_vals

    # Apply macro stress-testing shocks
    total_shock_pct = monsoon_shock + fuel_shock + tariff_shock
    multiplier = 1 + (total_shock_pct / 100.0)
    simulated_vals = [round(v * multiplier, 2) for v in chosen_vals]

    # Historical tail (last 24 months)
    hist_tail = ts.tail(24)
    hist_dates = [d.strftime("%Y-%m-%d") for d in hist_tail.index]
    hist_prices = [float(v) for v in hist_tail.values]

    # Plotly visualization with Fan Intervals
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        mode="lines+markers",
        name="Historical Actual",
        line=dict(color="#38bdf8", width=2.5)
    ))

    # Upper and Lower 95% CI for fan chart
    upper_95 = [round((v + 1.96 * float(se.iloc[i])) * multiplier, 2) for i, v in enumerate(chosen_vals)]
    lower_95 = [round(max(0, (v - 1.96 * float(se.iloc[i])) * multiplier), 2) for i, v in enumerate(chosen_vals)]

    # 95% Confidence Band
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper_95 + lower_95[::-1],
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Band",
        hoverinfo="skip"
    ))

    # Baseline vs Simulated
    fig.add_trace(go.Scatter(
        x=dates, y=chosen_vals,
        mode="lines+markers",
        name="Baseline Forecast",
        line=dict(color="#10b981", width=2.5, dash="dash")
    ))

    if total_shock_pct != 0:
        fig.add_trace(go.Scatter(
            x=dates, y=simulated_vals,
            mode="lines+markers",
            name=f"Stress-Tested ({'+' if total_shock_pct > 0 else ''}{total_shock_pct}%)",
            line=dict(color="#f59e0b", width=3)
        ))

    fig.update_layout(
        title=f"AgriCast-AI: {model_choice} Trajectory ({horizon} Months)",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (₹/quintal)",
        paper_bgcolor="#020617",
        plot_bgcolor="#090d16",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Narrative output
    last_price = float(ts.iloc[-1])
    first_pred = simulated_vals[0]
    pct_chg = ((first_pred - last_price) / last_price) * 100
    
    status = "Bullish (Upward)" if pct_chg > 2 else ("Bearish (Downward)" if pct_chg < -2 else "Consolidating (Sideways)")
    recommendation = (
        f"**Market Stance: {status}**\n\n"
        f"• **Benchmark Price:** ₹{last_price:,.2f}/quintal\n"
        f"• **Next Month Projected Price:** ₹{first_pred:,.2f}/quintal ({'+' if pct_chg > 0 else ''}{pct_chg:.2f}%)\n"
        f"• **Net Macro Shock Impact:** {total_shock_pct}%\n"
        f"• **Executive Guidance:** {'Forward-hedge procurement volume to mitigate upside risk.' if pct_chg > 2 else 'Adopt staggered just-in-time purchasing to minimize carrying costs.'}"
    )

    # Table df
    table_df = pd.DataFrame({
        "Month": dates,
        "Baseline (₹)": chosen_vals,
        "Stress-Tested (₹)": simulated_vals,
        "Lower 95% CI": lower_95,
        "Upper 95% CI": upper_95
    })

    return recommendation, fig, table_df


with gr.Blocks(theme=gr.themes.Base(primary_hue="emerald")) as demo:
    gr.Markdown("""
    # 🌾 AgriCast-AI: Commodity Price Intelligence & Forecasting
    ### Enterprise Multi-Model Stacking (SARIMA • Holt-Winters • XGBoost) with Climate & Macro Stress-Testing
    """)

    with gr.Row():
        with gr.Column(scale=1):
            horizon_input = gr.Radio([6, 12, 24], value=12, label="Forecast Horizon (Months)")
            model_input = gr.Dropdown(
                ["Smart Ensemble", "SARIMA", "Holt-Winters", "XGBoost"],
                value="Smart Ensemble",
                label="Forecasting Model"
            )

            gr.Markdown("### ⚡ Macro & Climate Stress Testing")
            monsoon_slider = gr.Slider(-15, 25, value=0, step=5, label="Monsoon / Drought Shock (%)")
            fuel_slider = gr.Slider(0, 20, value=0, step=5, label="Fuel & Logistics Inflation (%)")
            tariff_slider = gr.Slider(-20, 20, value=0, step=5, label="Tariff / Export Restriction (%)")

            run_btn = gr.Button("Generate Forecast & Scenarios", variant="primary")

        with gr.Column(scale=2):
            narrative_output = gr.Markdown()
            chart_output = gr.Plot()
            table_output = gr.Dataframe(label="Forecast Horizon Projections")

    run_btn.click(
        fn=run_forecast,
        inputs=[horizon_input, model_input, monsoon_slider, fuel_slider, tariff_slider],
        outputs=[narrative_output, chart_output, table_output]
    )

if __name__ == "__main__":
    demo.launch()
