# Bitcoin Volatility Forecasting — Research & Data Science

[![Python quality checks](https://github.com/kanizfatemar0xy/Bitcoin_dissertation/actions/workflows/python-tests.yml/badge.svg)](https://github.com/kanizfatemar0xy/Bitcoin_dissertation/actions/workflows/python-tests.yml)

**Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow · Time-Series Analysis**

A research-oriented project by **Kaniz Fatema Roxy**, examining next-day Bitcoin volatility using historical market prices and daily news/social-media sentiment signals. The code demonstrates data preparation, supervised model training and the importance of evaluating time-ordered data without look-ahead leakage.

**[Research code](src)** · **[Validation tests](tests/test_split_utils.py)** · **[CI results](https://github.com/kanizfatemar0xy/Bitcoin_dissertation/actions/workflows/python-tests.yml)** · **[Developer portfolio](https://kanizfatemar0xy.github.io/travel-website-project/portfolio.html)**

## Research workflow

| Stage | Description |
| --- | --- |
| Data sources | Historical Bitcoin prices, news titles and tweet text |
| Preprocessing | Daily returns, 7-day rolling volatility and sentiment aggregation |
| Predictors | Recent volatility, daily returns, news sentiment, tweet sentiment |
| Candidate models | Random Forest, XGBoost, Gradient Boosting, SVR and LSTM |
| Validation | Chronological holdout, train-only scaling and separate evaluation |
| Outputs | Model files and comparison-chart generation |

## Project structure

- [`src/data_prep.py`](src/data_prep.py) — input cleaning, volatility and sentiment calculations
- [`src/model_training.py`](src/model_training.py) — training and saving candidate models
- [`src/split_utils.py`](src/split_utils.py) — chronological split and leakage-aware scaling
- [`src/evaluate_models.py`](src/evaluate_models.py) — load models, calculate errors and plot results
- [`tests/test_split_utils.py`](tests/test_split_utils.py) — split/scale regression tests
- [`requirements.txt`](requirements.txt) — runtime dependencies
- `data/`, `models/saved_models/` — original research inputs and saved artifacts

## Reproduce locally

Use **Python 3.11**, a virtual environment and sufficient memory/storage for TensorFlow and the datasets. Run the following from the repository root:

```bash
python -m pip install -r requirements.txt
python src/data_prep.py
python src/model_training.py
python src/evaluate_models.py
```

To run the lightweight source-quality tests:

```bash
python -m unittest discover -s tests -v
```

**Automated CI checks do not run the full training pipeline**; they cover Python checks and split-helper tests only. Reproducing the entire experiment requires a suitable local or cloud environment.

## Research integrity and known limitations

> **Important:** Previously committed saved models and comparison figures were produced before the evaluation improvements and may reflect training leakage (including whole-dataset scaling and/or early stopping using test data). **Retrain and regenerate evaluations before citing any metrics from those artifacts.** No new model-performance claim is made in this repository until that work is completed.

- Chronological splitting improves evaluation discipline but cannot guarantee generalisation to new market regimes.
- Input data and social/news text may have licensing and privacy restrictions. Review them before redistribution or commercial reuse.
- Serialized pickle models must only be opened when their sources are trusted.
- This is an academic/research demonstration, **not financial or investment advice**, a live trading tool, or evidence of future market performance.

## For reviewers

The clearest starting points are [the chronological split helper](src/split_utils.py), [its tests](tests/test_split_utils.py), and [the training workflow](src/model_training.py). They illustrate the data science process rather than promising validated financial predictions.

**Developer:** [Kaniz Fatema Roxy](https://github.com/kanizfatemar0xy) · [Portfolio](https://kanizfatemar0xy.github.io/travel-website-project/portfolio.html) · [LinkedIn](https://www.linkedin.com/in/kaniz-fatema-roxy)
