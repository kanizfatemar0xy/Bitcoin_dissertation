"""Train volatility regression models with a strict chronological holdout.

Saved historical models from earlier commits must be retrained after this fix.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from split_utils import split_and_scale

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/merged_data.csv'
MODEL_DIR = ROOT / 'models/saved_models'
FEATURES = ('Volatility_7d', 'Daily_Return', 'News_Sentiment', 'Tweet_Sentiment')


def load_dataset(path=DATA):
    if not Path(path).is_file():
        raise FileNotFoundError(f'{path} not found. Run src/data_prep.py first.')
    df = pd.read_csv(path)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f'Missing expected columns: {sorted(missing)}')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date', kind='stable').dropna(subset=['Date'])
    df = df.dropna(subset=list(FEATURES)).copy()
    # Target for a row at time t is the next row's 7-day realised volatility.
    df['Target_Volatility'] = df['Volatility_7d'].shift(-1)
    df = df.dropna(subset=['Target_Volatility'])
    return df[list(FEATURES)], df['Target_Volatility']


def train_and_save_all_models():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / 'scaler.pkl').open('wb') as handle:
        pickle.dump(scaler, handle)

    from xgboost import XGBRegressor
    models = {
        'Random Forest': (RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42), 'random_forest_model.pkl'),
        'XGBoost': (XGBRegressor(n_estimators=150, learning_rate=.05, max_depth=5, random_state=42), 'xgboost_model.pkl'),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=150, learning_rate=.05, random_state=42), 'gradient_boosting_model.pkl'),
        'SVR': (SVR(kernel='rbf', C=1.5, epsilon=0.05), 'svr_model.pkl'),
    }
    metrics = []
    for name, (model, filename) in models.items():
        print(f'Training {name}...')
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics.append({'Model': name, 'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
                        'MAE': float(mean_absolute_error(y_test, pred))})
        with (MODEL_DIR / filename).open('wb') as handle:
            pickle.dump(model, handle)

    # Never use the held-out test set as Keras' early stopping validation data.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    if len(y_train) < 20:
        raise ValueError('LSTM requires at least 20 training rows for inner validation')
    X_train_3d = X_train.reshape(len(X_train), 1, X_train.shape[1])
    X_test_3d = X_test.reshape(len(X_test), 1, X_test.shape[1])
    lstm = Sequential([Input(shape=(1, X_train.shape[1])), LSTM(64), Dense(32, activation='relu'), Dense(1)])
    lstm.compile(optimizer='adam', loss='mse')
    lstm.fit(X_train_3d, y_train, epochs=50, batch_size=16, validation_split=.1,
             shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', patience=5,
             restore_best_weights=True)], verbose=1)
    pred = lstm.predict(X_test_3d, verbose=0).reshape(-1)
    metrics.append({'Model': 'LSTM', 'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
                    'MAE': float(mean_absolute_error(y_test, pred))})
    lstm.save(MODEL_DIR / 'lstm_model.keras')
    output = pd.DataFrame(metrics)
    output.to_csv(ROOT / 'model_metrics_latest.csv', index=False)
    print(output.to_string(index=False))
    print('Metrics generated from the holdout after training-only scaler fitting.')


if __name__ == '__main__':
    train_and_save_all_models()
