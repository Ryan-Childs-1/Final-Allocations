# Allocation Iterative FLM Streamlit App

Flat Streamlit app for the Model 3 single-review neural iterative FLM allocator.

Put these files in the same GitHub repo folder as your `.npz` model artifacts. The app includes the model registry, feature configuration, optimizer parameters, feature dashboard, and test-result diagnostics. The `.npz` model files are expected to already exist in the repo root.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Required model artifacts beside `app.py`:
- shared_demand_model.npz
- shared_final_supply_model.npz
- allocate_classifier_model.npz
- allocate_ranker_model.npz
- allocate_auxiliary_model.npz
- allocate_regressor_model.npz
- review_classifier_model.npz
- review_ranker_model.npz
- review_auxiliary_model.npz
- review_regressor_model.npz
- iterative_flm_step_scorer_model.npz


Notes:
- `prediction_detail.csv` is intentionally excluded from the flat package because it is large. The app still shows test results from the lighter summary, business-rule, component, group audit, cycle trace, and largest-error files.
- Every Plotly chart is rendered with a unique Streamlit key to avoid duplicate element ID errors on Streamlit Cloud.
