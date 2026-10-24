# Predictive Forecasting of Care Load & Placement Demand

## Contents
- `predictive_forecasting_care_load_placement.ipynb` — Jupyter notebook with executed outputs, figures, model comparison, and interpretation.
- `HHS_Unaccompanied_Alien_Children_Program.csv` — source data used for the included run.

## Open and rerun
1. Keep the notebook and CSV in the same folder.
2. Open the notebook in JupyterLab, Jupyter Notebook, or Google Colab (upload both files).
3. Install dependencies in the notebook environment if needed: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `statsmodels`.
4. Run all cells from top to bottom.

## Scope and data date
The included results use records from 2023-01-12 through 2025-12-21. The forecast in the included output starts 2025-12-22 and is historical as of October 2026. Refresh the CSV before using the forecast for current planning.

The source has 355 missing calendar dates and no Saturday records. Flow metrics are not imputed; absent dates are not treated as zero. The stock series is linearly interpolated for daily model inputs, and forecast evaluation scores only observed care-load dates. Capacity-breach probability is not estimated because no approved capacity threshold is included.

This is an analytical prototype. The empirical forecast bands are not calibrated probability intervals, and the backtest contains seven rolling origins. Validate with refreshed data and program stakeholders before operational use.
