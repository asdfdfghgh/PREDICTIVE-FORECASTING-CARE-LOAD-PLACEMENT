# Predictive Forecasting of Care Load & Placement Demand

## Package contents
- `predictive_forecasting_care_load_placement.ipynb` — notebook with the completed EDA, historical model evaluation, and saved outputs.
- `app.py` — interactive Streamlit forecasting dashboard.
- `HHS_Unaccompanied_Alien_Children_Program.csv` — source dataset for the included notebook run and dashboard default.
- `executive_summary.md` — stakeholder summary of results, limits, and recommendations.
- `requirements.txt` — Python dependencies.
- `run_dashboard.bat` — Windows launcher after dependencies are installed.

## Run the dashboard
1. Extract the package and open a terminal in this folder.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Start the app: `python -m streamlit run app.py` (or double-click `run_dashboard.bat`).
4. Upload a refreshed CSV in the sidebar to update forecasts and evaluation. The app expects the same columns as the included source.

The dashboard is interactive but is not connected to a live HHS feed.

## Data status and limitations
The included run covers January 12, 2023 through December 21, 2025. Its forecast dates are historical. The CSV contains 355 absent calendar dates and no Saturday records. Missing flow observations are not treated as zero; the HHS care-load stock is interpolated for model inputs and evaluation uses observed care-load dates. Capacity indicators are only rough residual-based proxies and require an approved threshold.

This is a planning prototype, not an operationally validated system. Refresh the data and review model performance with program stakeholders before using outputs for resource decisions.
