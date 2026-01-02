# train.py
# CLI script to train stock prediction model for a single stock

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import get_nse_symbols, fetch_stock_data
from utils.features import create_features, get_feature_columns
from models.predictor import train_model, train_multi_horizon_models, train_quantile_models, MODELS_DIR
import time
import datetime
import joblib
from typing import Callable, Optional

# Metadata filename helper
def _meta_path(symbol: str) -> Path:
    return MODELS_DIR / f"{symbol.lower()}_model_meta.joblib"


def get_last_trained_timestamp(symbol: str) -> Optional[datetime.datetime]:
    path = _meta_path(symbol)
    if not path.exists():
        return None
    try:
        meta = joblib.load(path)
        return datetime.datetime.fromisoformat(meta.get('last_trained'))
    except Exception:
        return None


def _save_last_trained(symbol: str):
    path = _meta_path(symbol)
    meta = {'last_trained': datetime.datetime.utcnow().isoformat()}
    joblib.dump(meta, path)


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


def train_with_progress(
    symbol: str,
    start_date: str,
    end_date: str,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    retrain_threshold_hours: int = 24
):
    """
    Train models with progress updates emitted via `progress_callback(stage, percent)`.

    progress stages (approximate percentages):
      - data_loading: 10
      - feature_engineering: 30
      - training_model_1: 60
      - training_model_2: 85
      - saving_models: 100

    Returns (success: bool, message: str, metrics: dict)
    """
    def _emit(stage: str, pct: int):
        try:
            if progress_callback:
                progress_callback(stage, max(0, min(100, int(pct))))
        except Exception:
            pass

    symbol_clean = symbol.replace('.NS', '')

    # Check recent model
    last_trained = get_last_trained_timestamp(symbol_clean)
    if last_trained is not None:
        elapsed = datetime.datetime.utcnow() - last_trained
        if elapsed.total_seconds() < retrain_threshold_hours * 3600:
            return False, f"Recent model exists (last trained {last_trained.isoformat()} UTC). Skipping retrain.", {}

    try:
        _emit('data_loading', 5)
        df = fetch_stock_data(symbol, start_date, end_date)
        _emit('data_loading', 10)

        if df is None or df.empty:
            return False, "No data fetched for symbol/date range", {}

        _emit('feature_engineering', 25)
        df_features = create_features(df)
        _emit('feature_engineering', 35)

        if len(df_features) < 50:
            return False, "Not enough data after feature engineering (need >=50 rows)", {}

        base_filename = f"{symbol_clean.lower()}_model.joblib"

        # Train single model (fast)
        _emit('training_model_1', 45)
        model, metrics = train_model(
            df_features,
            feature_cols=get_feature_columns(),
            save_path=base_filename
        )
        _emit('training_model_1', 60)

        # Train ensemble / multi-horizon (GBM + Ridge)
        _emit('training_model_2', 65)
        multi_models, multi_metrics = train_multi_horizon_models(
            df_features,
            feature_cols=get_feature_columns(),
            save_path=base_filename
        )
        _emit('training_model_2', 85)

        # Train quantile models (optional - kept lightweight)
        _emit('training_model_2', 88)
        q_models, q_metrics = train_quantile_models(
            df_features,
            feature_cols=get_feature_columns(),
            save_path=base_filename
        )

        # Save metadata
        _emit('saving_models', 95)
        _save_last_trained(symbol_clean)
        _emit('saving_models', 100)

        return True, f"Models trained and saved for {symbol_clean}", metrics

    except Exception as e:
        _emit('error', 100)
        return False, f"Training failed: {str(e)}", {}