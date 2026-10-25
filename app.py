from __future__ import annotations

from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

st.set_page_config(page_title="UAC Care Load Forecasting", page_icon="📈", layout="wide")

METRIC_COLUMNS = {
    "Children apprehended and placed in CBP custody*": "cbp_intake",
    "Children in CBP custody": "cbp_care",
    "Children transferred out of CBP custody": "transfers_to_hhs",
    "Children in HHS Care": "hhs_care",
    "Children discharged from HHS Care": "discharges",
}
MODEL_NAMES = [
    "Naive persistence", "7-day moving average", "Exponential smoothing",
    "SARIMA (weekly)", "Random Forest", "Gradient Boosting",
]
LEAD_BANDS = {"1–7 days": (1, 7), "8–14 days": (8, 14), "15–30 days": (15, 30)}


def load_csv(source: bytes) -> pd.DataFrame:
    raw = pd.read_csv(BytesIO(source), thousands=",")
    raw = raw.dropna(how="all").copy()
    if "Date" not in raw:
        raise ValueError("The CSV must include a Date column.")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    if raw["Date"].isna().any():
        raise ValueError("Some nonblank rows have dates that could not be parsed.")
    raw = raw.rename(columns=METRIC_COLUMNS)
    required = list(METRIC_COLUMNS.values())
    missing = [column for column in required if column not in raw]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    for column in required:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw[required].isna().any().any():
        raise ValueError("Metric values are missing or non-numeric on a dated row.")
    if raw["Date"].duplicated().any():
        raise ValueError("Duplicate dates found. Resolve them before forecasting.")
    return raw.sort_values("Date").set_index("Date")


def feature_frame(series: pd.Series) -> pd.DataFrame:
    x = pd.DataFrame(index=series.index)
    for lag in (1, 7, 14):
        x[f"lag_{lag}"] = series.shift(lag)
    prior = series.shift(1)
    for window in (7, 14):
        x[f"mean_{window}"] = prior.rolling(window).mean()
        x[f"std_{window}"] = prior.rolling(window).std()
    x["day_of_week"] = x.index.dayofweek
    x["month"] = x.index.month
    x["day_of_year"] = x.index.dayofyear
    return x


def ml_forecast(daily_history: pd.Series, observed_dates: pd.DatetimeIndex, horizon: int, model_name: str) -> pd.Series:
    x = feature_frame(daily_history)
    mask = x.notna().all(axis=1) & x.index.isin(observed_dates)
    if mask.sum() < 30:
        raise ValueError("Not enough observed care-load records for the ML model.")
    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, max_features=0.9,
            random_state=42, n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=2,
            loss="huber", random_state=42,
        )
    model.fit(x.loc[mask], daily_history.loc[mask])
    extended = daily_history.copy()
    predictions = []
    future = pd.date_range(extended.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    for date in future:
        with_future = pd.concat([extended, pd.Series([np.nan], index=[date])])
        row = feature_frame(with_future).loc[[date]]
        prediction = float(model.predict(row)[0])
        predictions.append(prediction)
        extended.loc[date] = prediction
    return pd.Series(predictions, index=future)


def forecast_models(history_observed: pd.Series, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    full_index = pd.date_range(history_observed.index.min(), history_observed.index.max(), freq="D")
    daily = history_observed.reindex(full_index).interpolate(method="time")
    observed_dates = history_observed.index
    future = pd.date_range(history_observed.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecasts: dict[str, pd.Series] = {
        "Naive persistence": pd.Series(float(daily.iloc[-1]), index=future),
        "7-day moving average": pd.Series(float(daily.tail(7).mean()), index=future),
    }
    errors = []
    candidates = [
        ("Exponential smoothing", lambda: ExponentialSmoothing(
            daily, trend="add", damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True).forecast(horizon)),
        ("SARIMA (weekly)", lambda: SARIMAX(
            daily, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=150).forecast(horizon)),
        ("Random Forest", lambda: ml_forecast(daily, observed_dates, horizon, "Random Forest")),
        ("Gradient Boosting", lambda: ml_forecast(daily, observed_dates, horizon, "Gradient Boosting")),
    ]
    for name, fit in candidates:
        try:
            values = np.asarray(fit(), dtype=float).reshape(-1)
            forecasts[name] = pd.Series(values[:horizon], index=future)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    result = pd.DataFrame(forecasts).clip(lower=0)
    return result, errors


@st.cache_data(show_spinner="Evaluating models at rolling forecast origins…")
def walk_forward_evaluation(data_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(data_bytes)
    target = df["hhs_care"]
    start_pos = int(np.floor(0.80 * len(target)))
    validation = target.iloc[start_pos:]
    if validation.empty:
        return pd.DataFrame(), pd.DataFrame()
    end_origin = target.index.max() - pd.Timedelta(days=30)
    candidates = validation.index[validation.index <= end_origin]
    origins = []
    for date in candidates:
        if not origins or (date - origins[-1]).days >= 28:
            origins.append(date)
    if len(origins) > 12:
        stride = int(np.ceil(len(origins) / 12))
        origins = origins[::stride]
    records = []
    failures = []
    for origin in origins:
        history = target.loc[target.index <= origin]
        if len(history) < 60:
            continue
        preds, errors = forecast_models(history, 30)
        failures.extend([f"{origin.date()} — {message}" for message in errors])
        actual = target.reindex(preds.index)
        for day, date in enumerate(preds.index, start=1):
            if pd.isna(actual.loc[date]):
                continue
            for model_name in preds.columns:
                pred = float(preds.loc[date, model_name])
                observed = float(actual.loc[date])
                records.append({
                    "origin": origin, "date": date, "lead_days": day,
                    "lead_band": next(name for name, (lo, hi) in LEAD_BANDS.items() if lo <= day <= hi),
                    "model": model_name, "actual": observed, "forecast": pred,
                    "error": observed - pred, "abs_error": abs(observed - pred),
                    "sq_error": (observed - pred) ** 2,
                    "ape": abs(observed - pred) / abs(observed) if observed else np.nan,
                })
    return pd.DataFrame(records), pd.DataFrame({"issue": failures})


def metric_table(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    result = (records.groupby(["model", "lead_band"], sort=False)
        .agg(n=("abs_error", "size"), MAE=("abs_error", "mean"),
             MSE=("sq_error", "mean"), MAPE=("ape", "mean"))
        .reset_index())
    result["RMSE"] = np.sqrt(result.pop("MSE"))
    result["MAPE (%)"] = 100 * result.pop("MAPE")
    result["Forecast accuracy (%)"] = 100 - result["MAPE (%)"]
    return result[["model", "lead_band", "n", "MAE", "RMSE", "MAPE (%)", "Forecast accuracy (%)"]]


def band_bounds(evaluation: pd.DataFrame, model: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    bounds_lo, bounds_hi = [], []
    for lead in range(1, horizon + 1):
        band = next(name for name, (lo, hi) in LEAD_BANDS.items() if lo <= lead <= hi)
        errors = evaluation.loc[
            (evaluation["model"] == model) & (evaluation["lead_band"] == band), "error"
        ]
        if len(errors) >= 8:
            qlo, qhi = errors.quantile([0.10, 0.90]).to_numpy()
        else:
            qlo, qhi = -np.nan, np.nan
        bounds_lo.append(qlo)
        bounds_hi.append(qhi)
    return np.asarray(bounds_lo), np.asarray(bounds_hi)


def plot_load(history: pd.Series, forecasts: pd.DataFrame, selected: str,
              horizon: int, low: np.ndarray | None = None,
              high: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    recent = history.tail(120)
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, name="Observed HHS care load", line={"color": "#27364b"}))
    future = forecasts.index
    if low is not None and high is not None:
        fig.add_trace(go.Scatter(x=future, y=high, mode="lines", line={"width": 0},
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future, y=low, mode="lines", line={"width": 0},
                                 fill="tonexty", fillcolor="rgba(36,130,180,0.18)",
                                 name="Empirical 10–90% error band"))
    fig.add_trace(go.Scatter(x=future, y=forecasts[selected], name=f"Selected: {selected}",
                             line={"color": "#147d92", "width": 3}))
    fig.update_layout(height=470, yaxis_title="Children in HHS care", xaxis_title="Date",
                      margin={"l": 10, "r": 10, "t": 25, "b": 10}, legend={"orientation": "h"})
    return fig


def main():
    st.title("UAC Care Load & Placement Demand Forecasts")
    st.caption("Planning prototype • forecasts describe the supplied reporting data and are not operational capacity decisions.")
    bundled = Path(__file__).with_name("HHS_Unaccompanied_Alien_Children_Program.csv")
    st.sidebar.header("Data and forecast settings")
    upload = st.sidebar.file_uploader("Upload refreshed CSV (optional)", type=["csv"])
    if upload is not None:
        data_bytes = upload.getvalue()
        data_label = upload.name
    elif bundled.exists():
        data_bytes = bundled.read_bytes()
        data_label = bundled.name
    else:
        st.info("Upload the source CSV to begin.")
        st.stop()

    try:
        data = load_csv(data_bytes)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    max_horizon = st.sidebar.selectbox("Forecast horizon", [7, 14, 30], index=2, format_func=lambda x: f"{x} days")
    evaluation, eval_issues = walk_forward_evaluation(data_bytes)
    metrics = metric_table(evaluation)
    if metrics.empty:
        st.error("Not enough history to evaluate forecasts. Provide at least several months of dated records.")
        st.stop()
    available_models = [name for name in MODEL_NAMES if name in metrics["model"].unique()]
    default_model = (metrics[metrics["lead_band"] == "1–7 days"].sort_values("MAE")["model"].iloc[0]
                     if (metrics["lead_band"] == "1–7 days").any() else available_models[0])
    selected = st.sidebar.selectbox("Primary model", available_models, index=available_models.index(default_model))
    compare_models = st.sidebar.multiselect("Models to show", available_models, default=available_models)
    capacity = st.sidebar.number_input("Approved capacity threshold (children)", min_value=0, value=0, step=100,
                                       help="Enter an approved operational capacity. Leave at 0 to disable breach KPIs.")
    risk_cutoff = st.sidebar.slider("Breach alert probability (%)", 5, 90, 25, 5)
    st.sidebar.caption(f"Data file: {data_label}")

    horizon = max_horizon
    predictions, fit_issues = forecast_models(data["hhs_care"], horizon)
    for name in compare_models:
        if name not in predictions:
            st.sidebar.warning(f"{name} did not fit on the current data and was omitted.")
    if selected not in predictions:
        selected = predictions.columns[0]
    primary = predictions[selected].to_numpy()
    qlo, qhi = band_bounds(evaluation, selected, horizon)
    forecast_low = np.maximum(0, primary + qlo)
    forecast_high = np.maximum(0, primary + qhi)
    last_observed = float(data["hhs_care"].iloc[-1])
    current_date = data.index.max()
    selected_metrics = metrics[(metrics["model"] == selected) & (metrics["lead_band"] == "1–7 days")]
    short_mae = float(selected_metrics["MAE"].iloc[0]) if not selected_metrics.empty else np.nan
    model_spread = predictions.sub(predictions.median(axis=1), axis=0).abs().median(axis=1).mean()
    stability = max(0.0, 100 * (1 - float(model_spread) / max(last_observed, 1)))

    st.info(f"Data: {data.index.min():%d %b %Y} to {current_date:%d %b %Y} • {len(data):,} observations • "
            f"{(pd.date_range(data.index.min(), data.index.max(), freq='D').size-len(data)):,} missing calendar dates. "
            "The forecast starts the day after the last report.")
    if current_date < pd.Timestamp.today().normalize() - pd.Timedelta(days=14):
        st.warning("The source extract is stale. Forecast dates are historical; upload current data before using this dashboard for planning.")
    if not (data.index.dayofweek == 5).any():
        st.warning("No Saturday observations are present. Weekly seasonality and Saturday estimates may reflect missing reporting rather than real activity.")

    end_value = float(predictions[selected].iloc[-1])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest reported HHS care load", f"{last_observed:,.0f}", delta=current_date.strftime("%d %b %Y"), delta_color="off")
    k2.metric(f"{horizon}-day forecast", f"{end_value:,.0f}", delta=f"{end_value-last_observed:+,.0f} vs. latest")
    k3.metric("Selected model 1–7 day MAE", f"{short_mae:,.1f}" if np.isfinite(short_mae) else "—", help="Historical rolling-origin MAE, in children.")
    k4.metric("Forecast stability index", f"{stability:.1f}/100", help="Heuristic cross-model agreement score; not forecast accuracy.")

    tab_load, tab_eval, tab_discharge, tab_scenario = st.tabs(["Care load forecast", "Model evaluation", "Discharge demand", "Scenarios & KPIs"])
    with tab_load:
        st.subheader("Future HHS care load")
        st.plotly_chart(plot_load(data["hhs_care"], predictions, selected, horizon, forecast_low, forecast_high), width="stretch")
        st.caption("Shaded region uses historical signed forecast errors from the matching lead-time band (10th–90th percentiles). This is an empirical prediction band, not a calibrated confidence interval.")
        table = pd.DataFrame({"Forecast": predictions[selected], "Lower empirical band": forecast_low,
                              "Upper empirical band": forecast_high})
        st.dataframe(table.round(0), width="stretch")
        st.download_button("Download care-load forecast CSV", table.to_csv().encode("utf-8"),
                           file_name="care_load_forecast.csv", mime="text/csv")
    with tab_eval:
        st.subheader("Rolling-origin model evaluation")
        st.caption("Each origin uses only earlier records. Scores are calculated only on dates with an observed HHS care-load value.")
        shown = metrics[metrics["model"].isin(compare_models or available_models)].copy()
        order = ["1–7 days", "8–14 days", "15–30 days"]
        shown["lead_band"] = pd.Categorical(shown["lead_band"], order, ordered=True)
        st.dataframe(shown.sort_values(["lead_band", "MAE"]).round(2), width="stretch", hide_index=True)
        chart = go.Figure()
        for model_name in (compare_models or available_models):
            rows = shown[shown["model"] == model_name].sort_values("lead_band")
            if not rows.empty:
                chart.add_trace(go.Scatter(x=rows["lead_band"].astype(str), y=rows["MAE"],
                                           mode="lines+markers", name=model_name))
        chart.update_layout(height=350, xaxis_title="Forecast lead time", yaxis_title="MAE (children)",
                            margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(chart, width="stretch")
        st.caption("MAPE complement (100 − MAPE) is shown as a descriptive percentage; it is not the probability a forecast is correct. Prefer MAE/RMSE in children.")
        if not eval_issues.empty:
            with st.expander("Models/origins that did not fit"):
                st.dataframe(eval_issues, width="stretch", hide_index=True)
    with tab_discharge:
        st.subheader("Discharge demand baseline")
        discharge = data["discharges"].dropna()
        recent_rate = float(discharge.tail(14).mean())
        discharge_errors = []
        for origin in evaluation["origin"].drop_duplicates() if not evaluation.empty else []:
            past = discharge.loc[discharge.index <= origin]
            future_actual = discharge.loc[(discharge.index > origin) & (discharge.index <= origin + pd.Timedelta(days=30))]
            if len(past) >= 7:
                estimate = float(past.tail(7).mean())
                discharge_errors.extend((future_actual.to_numpy() - estimate).tolist())
        if discharge_errors:
            lo, hi = np.quantile(discharge_errors, [0.10, 0.90])
            dlow, dhigh = max(0, recent_rate + lo), max(0, recent_rate + hi)
            d_mae = float(np.mean(np.abs(discharge_errors)))
        else:
            dlow, dhigh, d_mae = np.nan, np.nan, np.nan
        d1, d2, d3 = st.columns(3)
        d1.metric("Mean of last 14 reported discharge values", f"{recent_rate:.1f}", help="Children per reported observation; not a verified daily total.")
        d2.metric("Walk-forward baseline MAE", f"{d_mae:.1f}" if np.isfinite(d_mae) else "—", help="Trailing mean of seven reported values.")
        d3.metric("Empirical band", f"{dlow:.1f}–{dhigh:.1f}" if np.isfinite(dlow) else "—", help="Based on reported-day validation errors.")
        d_dates = pd.date_range(current_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        fig = go.Figure()
        recent = discharge.tail(120)
        fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines+markers", name="Observed discharges"))
        fig.add_trace(go.Scatter(x=d_dates, y=np.full(horizon, recent_rate), name="Flat reported-day baseline", line={"dash": "dash"}))
        if np.isfinite(dlow) and np.isfinite(dhigh):
            fig.add_trace(go.Scatter(x=d_dates, y=np.full(horizon, dhigh), line={"width": 0}, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=d_dates, y=np.full(horizon, dlow), line={"width": 0}, fill="tonexty", fillcolor="rgba(240,150,40,0.18)", name="Empirical error band"))
        fig.update_layout(height=400, yaxis_title="Children discharged (reported dates)", margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(fig, width="stretch")
        st.warning("Missing discharge dates are not zero. This panel forecasts a reported-day rate; do not multiply it by calendar days to plan placement totals unless reporting completeness is verified.")
    with tab_scenario:
        st.subheader("Capacity and early-warning KPIs")
        if capacity <= 0:
            st.info("Enter an approved capacity threshold in the sidebar to calculate a capacity-risk proxy and surge lead time.")
        else:
            probabilities, crossings = [], []
            for lead, (date, point) in enumerate(predictions[selected].items(), start=1):
                band = next(name for name, (lo, hi) in LEAD_BANDS.items() if lo <= lead <= hi)
                residuals = evaluation.loc[(evaluation["model"] == selected) & (evaluation["lead_band"] == band), "error"]
                p = float((point + residuals >= capacity).mean()) if len(residuals) else np.nan
                probabilities.append(p)
                _, upper_q = residuals.quantile([0.10, 0.90]).to_numpy() if len(residuals) >= 8 else (np.nan, np.nan)
                if np.isfinite(upper_q) and point + upper_q >= capacity:
                    crossings.append((lead, date, p))
            max_prob = float(np.nanmax(probabilities)) if np.isfinite(probabilities).any() else np.nan
            threshold = risk_cutoff / 100
            risk_dates = [i + 1 for i, p in enumerate(probabilities) if np.isfinite(p) and p >= threshold]
            surge_days = risk_dates[0] if risk_dates else None
            c1, c2 = st.columns(2)
            c1.metric("Highest backtest-based breach-risk proxy", f"{max_prob:.0%}" if np.isfinite(max_prob) else "Insufficient validation")
            c2.metric("Surge lead time", f"{surge_days} days" if surge_days else "No threshold crossing in horizon")
            st.caption(f"Risk proxy uses empirical validation residuals for the selected model and a user-entered capacity threshold of {capacity:,.0f}. It is unstable when the number of validation cases is small and should not be treated as a calibrated probability.")
            pfig = go.Figure(go.Scatter(x=predictions.index, y=probabilities, mode="lines+markers", name="Breach-risk proxy"))
            pfig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Alert threshold {risk_cutoff}%")
            pfig.update_layout(height=300, yaxis={"title": "Empirical risk proxy", "tickformat": ".0%", "range": [0, 1]}, margin={"l": 10, "r": 10, "t": 25, "b": 10})
            st.plotly_chart(pfig, width="stretch")
        st.divider()
        st.subheader("Scenario comparison")
        st.caption("Scenario paths add a constant user-entered daily net-flow change to the selected baseline. These are sensitivity illustrations, not separately trained or validated forecasts.")
        daily_shock = st.slider("Assumed additional net flow per day (children)", -50, 100, 20, 5)
        scenario_days = np.arange(1, horizon + 1)
        scenario = primary + daily_shock * scenario_days
        scenario_df = pd.DataFrame({"Baseline model": primary, "User scenario": scenario}, index=predictions.index)
        sfig = go.Figure()
        sfig.add_trace(go.Scatter(x=scenario_df.index, y=scenario_df["Baseline model"], name=f"Baseline: {selected}"))
        sfig.add_trace(go.Scatter(x=scenario_df.index, y=scenario_df["User scenario"], name=f"Scenario ({daily_shock:+} net/day)", line={"dash": "dash"}))
        sfig.update_layout(height=390, yaxis_title="Children in HHS care", margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(sfig, width="stretch")
        st.metric("Forecast stability index", f"{stability:.1f}/100",
                  help="Heuristic model agreement: higher values mean models cluster closer together relative to the latest care load. This does not measure historical accuracy.")
        st.markdown("**KPI definitions used here**\n- Forecast accuracy: 100 − MAPE, descriptive only.\n- Surge lead time: first forecast day the empirical breach-risk proxy crosses the selected alert threshold.\n- Capacity breach probability: backtest-residual risk proxy, not calibrated probability.\n- Forecast stability index: heuristic cross-model agreement score.")

    with st.expander("Method and important limitations"):
        st.markdown("""
        - HHS care load is a stock. Missing stock days are linearly interpolated for model input; metrics are scored only on observed dates.
        - Intake, transfers, and discharges are flows. Missing flow dates remain unknown and are not filled with zero.
        - The source has no Saturday records in the supplied extract. Weekly seasonality may partly reflect reporting gaps.
        - Rolling-origin forecasts are evaluated over the final 20% of records at origins at least 28 days apart, with a maximum 30-day evaluation horizon.
        - The discharge panel is a trailing reported-observation baseline, not a complete daily placement forecast.
        - No capacity, facility, staffing, or acuity data are supplied. Add an approved capacity threshold before interpreting risk indicators.
        - Model fit and backtest results do not establish future accuracy. Refresh the data and review forecasts with operational stakeholders.
        """)
    if fit_issues:
        with st.expander("Forecast fit notes"):
            st.write("\n".join(fit_issues))
    st.caption("For planning exploration only. Use current, validated data and approved thresholds before operational decisions.")


if __name__ == "__main__":
    main()
