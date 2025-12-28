# train.py
# CLI script to train stock prediction model for a single stock

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import get_nse_symbols, fetch_stock_data
from utils.features import create_features, get_feature_columns
from models.predictor import train_model


def main():
    print("=" * 50)
    print("  Stock Price Prediction - Model Training")
    print("=" * 50)
    
    # Load available symbols
    print("\nLoading NSE symbols...")
    symbols = get_nse_symbols(yahoo_format=False)
    print(f"Found {len(symbols)} symbols")
    
    # Get user input
    print("\nEnter stock symbol (e.g., RELIANCE, TCS, INFY):")
    symbol_input = input("> ").strip().upper()
    
    # Validate symbol
    if symbol_input not in symbols:
        print(f"Warning: '{symbol_input}' not found in NSE list. Proceeding anyway...")
    
    yahoo_symbol = f"{symbol_input}.NS"
    
    # Get date range
    print("\nEnter start date (YYYY-MM-DD) [default: 2022-01-01]:")
    start_date = input("> ").strip() or "2022-01-01"
    
    print("Enter end date (YYYY-MM-DD) [default: 2024-12-31]:")
    end_date = input("> ").strip() or "2024-12-31"
    
    # Fetch data
    print(f"\nFetching data for {yahoo_symbol}...")
    df = fetch_stock_data(yahoo_symbol, start_date, end_date)
    
    if df.empty:
        print("Error: No data fetched. Check symbol or date range.")
        return
    
    print(f"Fetched {len(df)} rows")
    
    # Generate features
    print("\nGenerating features...")
    df_features = create_features(df)
    print(f"Features created: {len(df_features)} rows after dropping NaNs")
    
    if len(df_features) < 50:
        print("Error: Not enough data for training (need at least 50 rows)")
        return
    
    # Train model
    print("\nTraining XGBoost model...")
    model_filename = f"{symbol_input.lower()}_model.joblib"
    
    model, metrics = train_model(
        df_features,
        feature_cols=get_feature_columns(),
        save_path=model_filename
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("  Training Complete!")
    print("=" * 50)
    print(f"\nModel saved: models/{model_filename}")
    print(f"\nMetrics:")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()