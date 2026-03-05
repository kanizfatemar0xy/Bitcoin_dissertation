import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')

PROCESSED_DATA_PATH = 'data/processed/merged_data.csv'
MODEL_DIR = 'models/saved_models'

def train_and_save_all_models():
    print("1. Loading and Preprocessing Data...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] {PROCESSED_DATA_PATH} not found. Please run data prep first.")
        return

    # Clean the data and set the next day's volatility as the target
    df = df.dropna(subset=['Volatility_7d', 'Daily_Return'])
    df['Target_Volatility'] = df['Volatility_7d'].shift(-1)
    df = df.dropna(subset=['Target_Volatility'])
    
    features = ['Volatility_7d', 'Daily_Return', 'News_Sentiment', 'Tweet_Sentiment']
    X = df[features]
    y = df['Target_Volatility']
    
    # Feature Scaling (important for better training of SVR and LSTM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler as well so it can be used later for live predictions
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    
    results = {}
    
    print("\n Starting High-Quality Training for 5 Models...\n")
    
    # ---------------------------------------------------------
    # 1. Random Forest
    # ---------------------------------------------------------
    print(" Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    results['Random Forest'] = {'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)), 'MAE': mean_absolute_error(y_test, rf_pred)}
    
    with open(os.path.join(MODEL_DIR, 'random_forest_model.pkl'), 'wb') as f:
        pickle.dump(rf_model, f)

    # ---------------------------------------------------------
    # 2. XGBoost
    # ---------------------------------------------------------
    print(" Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results['XGBoost'] = {'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)), 'MAE': mean_absolute_error(y_test, xgb_pred)}
    
    with open(os.path.join(MODEL_DIR, 'xgboost_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)

    # ---------------------------------------------------------
    # 3. Gradient Boosting
    # ---------------------------------------------------------
    print(" Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    results['Gradient Boosting'] = {'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)), 'MAE': mean_absolute_error(y_test, gb_pred)}
    
    with open(os.path.join(MODEL_DIR, 'gradient_boosting_model.pkl'), 'wb') as f:
        pickle.dump(gb_model, f)

    # ---------------------------------------------------------
    # 4. SVR (Support Vector Regressor)
    # ---------------------------------------------------------
    print(" Training SVR...")
    svr_model = SVR(kernel='rbf', C=1.5, epsilon=0.05)
    svr_model.fit(X_train, y_train)
    svr_pred = svr_model.predict(X_test)
    results['SVR'] = {'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)), 'MAE': mean_absolute_error(y_test, svr_pred)}
    
    with open(os.path.join(MODEL_DIR, 'svr_model.pkl'), 'wb') as f:
        pickle.dump(svr_model, f)

    # ---------------------------------------------------------
    # 5. LSTM (Deep Learning)
    # ---------------------------------------------------------
    print(" Training LSTM (Deep Learning)...")
    
    # LSTM requires 3D input data
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, activation='relu', return_sequences=False, input_shape=(1, X_train.shape[1])))
    lstm_model.add(Dense(32, activation='relu'))
    lstm_model.add(Dense(1))
    
    lstm_model.compile(optimizer='adam', loss='mse')
    
    # EarlyStopping: Stop training if validation loss does not improve for 5 epochs
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # verbose=1 shows the training progress (epochs) in the terminal
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, 
                   validation_data=(X_test_lstm, y_test), 
                   callbacks=[early_stop], verbose=1)
    
    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    results['LSTM'] = {'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)), 'MAE': mean_absolute_error(y_test, lstm_pred)}
    
    # Deep Learning models are usually saved in .keras or .h5 format
    lstm_model.save(os.path.join(MODEL_DIR, 'lstm_model.keras'))

    # ---------------------------------------------------------
    # Final Output & Comparison
    # ---------------------------------------------------------
    print("\n Success! All 5 Models Trained and Saved in 'models/saved_models/' directory.\n")
    
    print(" FINAL MODEL EVALUATION RESULTS:")
    print("-" * 55)
    print(f"{'Model Name':<20} | {'RMSE':<12} | {'MAE':<12}")
    print("-" * 55)
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics['RMSE']:.5f}      | {metrics['MAE']:.5f}")
    print("-" * 55)

if __name__ == "__main__":
    train_and_save_all_models()