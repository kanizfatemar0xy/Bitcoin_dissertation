import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings('ignore')

# Folder Paths
DATA_PATH = 'data/processed/merged_data.csv'
MODEL_DIR = 'models/saved_models'
OUTPUT_IMG = 'model_comparison_chart.png'

def evaluate_and_plot():
    print("1. Loading Test Data and Scaler...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['Volatility_7d', 'Daily_Return'])
    df['Target_Volatility'] = df['Volatility_7d'].shift(-1)
    df = df.dropna(subset=['Target_Volatility'])
    
    features = ['Volatility_7d', 'Daily_Return', 'News_Sentiment', 'Tweet_Sentiment']
    X = df[features]
    y = df['Target_Volatility']
    
    # Load the same scaler that was used during training
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    X_scaled = scaler.transform(X)
    
    # Data Split (same as training: 80% train, 20% test)
    split_idx = int(len(X_scaled) * 0.8)
    X_test = X_scaled[split_idx:]
    y_test = y.iloc[split_idx:]
    
    results = []
    
    models = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'Gradient Boosting': 'gradient_boosting_model.pkl',
        'SVR': 'svr_model.pkl',
        'LSTM': 'lstm_model.keras'
    }
    
    print("\n2. Evaluating all 5 Models on Test Set...")
    for name, filename in models.items():
        try:
            if name == 'LSTM':
                model = load_model(os.path.join(MODEL_DIR, filename))
                
                # LSTM requires 3D input data
                X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
                pred = model.predict(X_test_lstm, verbose=0).flatten()
            else:
                with open(os.path.join(MODEL_DIR, filename), 'rb') as f:
                    model = pickle.load(f)
                pred = model.predict(X_test)
            
            # Calculate evaluation errors
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            
            results.append({'Model': name, 'Metric': 'RMSE', 'Score': rmse})
            results.append({'Model': name, 'Metric': 'MAE', 'Score': mae})
            print(f" Successfully evaluated {name}")
            
        except Exception as e:
            print(f" Error loading {name}: {e}")
            
    # Convert results into a DataFrame for plotting
    results_df = pd.DataFrame(results)
    
    print("\n3. Generating Professional Bar Chart...")
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    # Create grouped bar chart
    ax = sns.barplot(data=results_df, x='Model', y='Score', hue='Metric', palette=['#ff4b4b', '#4b4bff'])
    
    plt.title('Comparison of 5 Machine Learning Models (RMSE & MAE)', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Error Score (Lower is Better)', fontsize=14, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    
    # Add exact values on top of each bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.5f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=11, fontweight='bold')
                    
    plt.legend(title='Evaluation Metric', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    
    # Save the chart image
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"\n Success! High-Quality Chart saved as '{OUTPUT_IMG}' in your project folder.")

if __name__ == "__main__":
    evaluate_and_plot()