# Bitcoin Volatility Forecasting | Research & Data Science

[![Python quality checks](https://github.com/kanizfatemar0xy/Bitcoin_dissertation/actions/workflows/python-tests.yml/badge.svg)](https://github.com/kanizfatemar0xy/Bitcoin_dissertation/actions/workflows/python-tests.yml)

**Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow · time-series research**

A research-oriented exploration of next-day Bitcoin volatility using historical market changes, news and social-media sentiment. Developed by **Kaniz Fatema Roxy**.

**[👩‍💻 Developer portfolio](https://kanizfatemar0xy.github.io/travel-website-project/portfolio.html)** · **[Read the training code](src/model_training.py)** · **[Tests](tests/test_split_utils.py)** · **[CI checks](https://github.com/kanizfatemar0xy/Bitcoin_dissertation/actions/workflows/python-tests.yml)**

## Research workflow at a glance

| Stage | Details |
| --- | --- |
| Inputs | Historical Bitcoin prices and text-sentiment datasets |
| Preprocessing | Daily returns, rolling volatility and daily sentiment aggregates |
| Modelling | Random Forest, Gradient Boosting, XGBoost, SVR, LSTM |
| Validation | Chronological holdout; scaler fitted on training data only |
| Reproducibility | Python unit tests, documented requirements and GitHub Actions checks |

> **Evaluation caveat:** Previously committed trained model files and comparison plots predate the corrected leakage controls. They **must be retrained and re-evaluated** before making reliable comparative-performance claims. Existing CI checks validate source and split helpers, **not** full training or benchmark metrics.


## Contents
- `src/data_prep.py`: process historical prices, news and tweets into a daily feature table
- `src/model_training.py`: train Random Forest, XGBoost, Gradient Boosting, SVR and an LSTM
- `src/split_utils.py`: deterministic chronological holdout with **training-only scaler fitting**
- `src/evaluate_models.py`: evaluate saved models and render a comparison chart
- `tests/test_split_utils.py`: regression tests for the time split and leakage prevention

## Environment and reproducibility
Use Python 3.11 in a virtual environment. Install dependencies with `pip install -r requirements.txt` (ML and TensorFlow installs may take time and have platform-specific requirements). Run from the repository root:

```bash
python src/data_prep.py
python src/model_training.py
python src/evaluate_models.py
python -m unittest discover -s tests -v
```

Historical input data and previously generated saved models are retained as originally committed. **Important:** Earlier saved models/figures may have been trained using a scaler fitted on the entire dataset and LSTM early stopping on the test set. They must be **retrained** before their metrics can be treated as leakage-controlled. The new code fixes these specific issues; no new benchmark result is claimed until retraining is completed.

## Research limitations
- Observational forecasting and text sentiment are not investment advice or a trading system.
- Reused news/tweet datasets can have licensing or privacy constraints. Review before redistribution or public commercial use.
- Time-based splits avoid random shuffling but do not themselves establish generalisation across different market regimes.
- Binary pickle models must only be loaded from trusted sources.
- Model metrics depend on dataset version, preprocessing, time period and environment. The full training pipeline has not been rerun on GitHub in CI because TensorFlow and the datasets are comparatively heavy.

## Automated checks
The lightweight GitHub Actions workflow compiles Python source and tests the split helper without training the heavy neural model. Run the full training commands locally or on a capable runner to validate end-to-end behaviour.

**Developer:** [Kaniz Fatema Roxy](https://github.com/kanizfatemar0xy)
