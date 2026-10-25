# Executive Summary
## Predictive Forecasting of Care Load & Placement Demand

**Prepared:** October 25, 2026  
**Audience:** HHS program and operational planning stakeholders  
**Scope:** Historical proof of concept using the supplied UAC reporting extract.

### Decision brief

The analysis demonstrates a reproducible way to compare short-horizon forecasts of the number of children in HHS care and to summarize reported discharge activity. In the provided historical backtest, SARIMA produced the lowest mean absolute error (MAE) among the compared models at all evaluated lead times. The result is promising for further validation, but it is not evidence of operational readiness.

**The supplied extract ends December 21, 2025.** The forecast produced from it begins December 22, 2025 and is historical as of this brief. Refresh the data before using any dashboard forecast for current planning. The application supports uploading a refreshed CSV and recalculates its forecasts and metrics from that file.

### What the historical evaluation showed

| Lead time | SARIMA MAE | Persistence baseline MAE |
|---|---:|---:|
| 1–7 days | 18.47 children | 31.70 children |
| 8–14 days | 42.95 children | 62.59 children |
| 15–30 days | 103.73 children | 119.85 children |

The notebook selected SARIMA for its historical 30-day forecast: 2,485 children on December 22, 2025, rising gradually to about 2,557 by January 20, 2026. This forecast should be interpreted only as an example from the old data extract. Errors rose with longer horizons, as expected. The evaluation included seven rolling origins, which is a small validation sample.

The displayed “Forecast accuracy (%)” is calculated as 100 minus MAPE. It is not the probability that a forecast is correct. Stakeholder reporting should emphasize MAE and RMSE in children, identify the lead time, and show the number of observations used.

### Data and interpretation limits

- The CSV had 1,170 raw rows; 450 fully blank rows were removed, leaving 720 unique dated observations. Across the 1,075-day span, 355 calendar dates are absent.
- No Saturday records appear in the supplied extract. Weekly seasonal patterns and Saturday forecasts may partly reflect reporting coverage and interpolation.
- Missing flow records are unknown, not zero. The discharge panel estimates a recent reported-day rate; it does not establish a complete calendar-week placement total.
- The empirical forecast bands and capacity-risk display use historical residuals. They are exploratory proxies, not calibrated probability or confidence intervals.
- Capacity-breach indicators require an approved capacity threshold. The dataset does not include facility capacity, staffed beds, medical acuity, or caseworker workload.

### Dashboard capabilities

The Streamlit app provides an interactive care-load chart, forecast horizon and model selectors, rolling-origin MAE/RMSE/MAPE comparison, a reported-day discharge baseline, empirical error bands, an optional threshold-based capacity-risk proxy, and user-adjustable net-flow scenarios. The dashboard recalculates from the uploaded CSV; it is not connected to a live HHS data feed.

### Recommended next steps

1. Obtain and validate current daily data; confirm what missing dates mean and investigate the Saturday gap.
2. Extend backtesting with more rolling origins and a final untouched holdout. Check interval coverage and compare against persistence and moving-average baselines.
3. Confirm flow definitions and reporting completeness before estimating total placement demand or intake-exit imbalance.
4. Add approved capacity, staffing, and facility-level inputs before using breach risk or surge lead time in planning.
5. Have program, child-welfare, clinical, and data-governance staff review assumptions and outputs before operational use.

**Conclusion:** The prototype can help structure a forecasting workflow and compare models. It should support further analysis, not substitute for current operational data or decision-maker review. The research paper remains a separate deliverable.
