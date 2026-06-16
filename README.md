# Steel Procurement Model — Dilip Buildcon Ltd.

Predicts total steel requirement for infrastructure projects based on project type and package length.

## How to deploy (Streamlit Cloud — free)

1. Create a free account at https://streamlit.io/cloud
2. Push this folder to a GitHub repository (public or private)
3. On Streamlit Cloud click **New app** → select your repo → set main file as `app.py`
4. Click **Deploy** — your app will be live at `https://yourname-appname.streamlit.app`

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the app does

- **Predictor tab**: Enter project type, length, and steel procured → get predicted total, progress %, and remaining demand
- **Model Comparison tab**: Training vs CV error for Linear Regression, Random Forest, XGBoost with charts
- **Benchmarks tab**: MT/km ranges by project type with interpretation notes
- **Training Data tab**: All 22 training projects with scatter plot

## Model details

- Algorithm: Random Forest (200 trees, selected by lowest LOOCV CV MAPE)
- Training data: 22 full-scope DBL projects, FY 2023–26
- Features: Project Type (one-hot encoded) + Package Length (km)
- CV method: Leave-One-Out Cross Validation (LOOCV)
- CV MAPE: 210.7%
