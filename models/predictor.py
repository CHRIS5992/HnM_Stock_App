# predictor.py
# XGBoost model for stock price prediction with multi-horizon forecasting

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Union

# Path for saving models
MODELS_DIR = Path(__file__).parent
MODELS_DIR.mkdir(exist_ok=True)

# Default horizons for multi-horizon forecasting
HORIZONS = {'1d': 1, '5d': 5, '20d': 20}
DEFAULT_FEATURE_COLS = ['returns', 'ma_5', 'ma_10', 'ma_20', 'volatility', 'rsi']

# Quantiles for confidence bands (low, mid, high)
QUANTILES = {'low': 0.1, 'mid': 0.5, 'high': 0.9}

# Regime-based confidence adjustment multipliers
# These widen/narrow confidence bands based on market regime
# Values > 1.0 widen bands (less confidence), < 1.0 narrow bands (more confidence)
REGIME_ADJUSTMENTS = {
    'Trending': 0.9,           # Narrow bands - trends are more predictable
    'Ranging': 1.3,            # Widen bands - sideways markets are harder to predict
    'High Volatility': 1.6,    # Significantly widen - high uncertainty
    'Transitional': 1.1,       # Slightly widen - regime changing
    'Insufficient Data': 1.5,  # Widen due to lack of information
}


def prepare_data(df: pd.DataFrame, feature_cols: list[str], target_col: str = 'Close'):
    """Prepare X and y for training."""
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


def create_horizon_targets(df: pd.DataFrame, horizons: dict[str, int] = None) -> pd.DataFrame:
    """
    Create target columns for each prediction horizon.
    
    Args:
        df: DataFrame with 'Close' column
        horizons: Dict mapping horizon name to days ahead (e.g., {'1d': 1, '5d': 5})
        
    Returns:
        DataFrame with added target columns (target_1d, target_5d, etc.)
    """
    if horizons is None:
        horizons = HORIZONS
    
    df = df.copy()
    for name, days in horizons.items():
        df[f'target_{name}'] = df['Close'].shift(-days)
    
    # Drop rows with NaN targets (last N rows)
    max_horizon = max(horizons.values())
    df = df.iloc[:-max_horizon].reset_index(drop=True)
    
    return df


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    target_col: str = 'Close',
    horizon_days: int = 1,
    test_size: float = 0.2,
    save_path: str = None
) -> tuple[XGBRegressor, dict]:
    """
    Train XGBoost model on stock data with features (single horizon).
    
    Args:
        df: DataFrame with features (from create_features)
        feature_cols: List of feature column names
        target_col: Column to predict (default: 'Close')
        test_size: Fraction for test split
        save_path: Path to save model (optional)
        
    Returns:
        Trained model and metrics dict
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    # Build target as log return over horizon_days and normalize by rolling vol
    df_local = df.copy()
    # log return for horizon
    df_local['target_return'] = np.log(df_local['Close'].shift(-horizon_days) / df_local['Close'])
    # rolling vol based on daily log returns
    daily_logret = np.log(df_local['Close'] / df_local['Close'].shift(1))
    df_local['vol'] = daily_logret.rolling(window=20, min_periods=5).std().fillna(method='bfill').replace(0, 1e-6)

    # Drop rows with NaN targets (last horizon days)
    df_local = df_local.iloc[:-horizon_days].reset_index(drop=True)

    # Prepare features and normalized target
    X = df_local[feature_cols].values
    y = (df_local['target_return'].values) / df_local['vol'].values
    
    # Time-series split (don't shuffle - preserve order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    # Save model if path provided
    if save_path:
        # Save the model that predicts normalized returns
        save_model(model, save_path)
    
    return model, metrics


def train_multi_horizon_models(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    horizons: dict[str, int] = None,
    test_size: float = 0.2,
    save_path: str = None,
    weights: dict = None
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Train a fast ensemble (GradientBoostingRegressor + Ridge) per horizon.

    Walk-forward split: last `test_size` fraction is validation (no shuffling).
    Models are trained once per horizon and final predictions are a weighted
    average of GBM and Ridge (default weights: 0.6, 0.4).

    Returns a dict of per-horizon ensembles and a dict of metrics (MAE reported).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    if horizons is None:
        horizons = HORIZONS
    if weights is None:
        weights = {'gbm': 0.6, 'ridge': 0.4}

    # We'll compute log-return targets and normalize by rolling volatility
    df_local = df.copy()
    # daily log return and rolling vol
    daily_logret = np.log(df_local['Close'] / df_local['Close'].shift(1))
    df_local['vol'] = daily_logret.rolling(window=20, min_periods=5).std().fillna(method='bfill').replace(0, 1e-6)

    # Build horizon targets (log returns) and drop last rows
    max_horizon = max(horizons.values())
    for name, days in horizons.items():
        df_local[f'target_{name}'] = np.log(df_local['Close'].shift(-days) / df_local['Close'])
    df_targets = df_local.iloc[:-max_horizon].reset_index(drop=True)

    # Pre-compute features once for speed
    X = df_targets[feature_cols].values
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_val = X[:split_idx], X[split_idx:]

    ensembles = {}
    all_metrics = {}

    for horizon_name in horizons.keys():
        target_col = f'target_{horizon_name}'
        # raw log-return target
        y_raw = df_targets[target_col].values
        vol = df_targets['vol'].values
        # normalized target (safe divide)
        y_norm = y_raw / (vol + 1e-12)
        y_train, y_val = y_norm[:split_idx], y_norm[split_idx:]

        # Fast Gradient Boosting (shallow, small trees) predicting normalized returns
        gbm = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        ridge = Ridge(alpha=1.0)

        # Train both models (fast)
        gbm.fit(X_train, y_train)
        ridge.fit(X_train, y_train)

        # Validation predictions (normalized) and weighted average
        pred_gbm = gbm.predict(X_val)
        pred_ridge = ridge.predict(X_val)
        pred_ensemble = (weights['gbm'] * pred_gbm) + (weights['ridge'] * pred_ridge)

        mae = mean_absolute_error(y_val, pred_ensemble)

        ensembles[horizon_name] = {
            'gbm': gbm,
            'ridge': ridge,
            'weights': weights.copy()
        }

        all_metrics[horizon_name] = {
            'mae': mae
        }

    # Save models and weights if path provided
    if save_path:
        # Save ensembles (contain models and weights)
        save_multi_horizon_models(ensembles, save_path)

    return ensembles, all_metrics


def predict_multi_horizon(
    df: pd.DataFrame,
    models: dict,
    feature_cols: list[str] = None
) -> dict[str, float]:
    """
    Predict closing prices for multiple horizons (1d, 5d, 20d).
    Optimized for speed with single prediction per model.
    
    Args:
        df: DataFrame with features (most recent data)
        models: Dict of trained models keyed by horizon name
        feature_cols: Feature column names
        
    Returns:
        Dictionary with predictions: {"1d": value, "5d": value, "20d": value}
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    # Get latest features once
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    # Latest features and last close
    X = df[feature_cols].iloc[-1:].values
    last_close = float(df['Close'].iloc[-1])

    # Recent volatility (rolling of daily log returns)
    daily_logret = np.log(df['Close'] / df['Close'].shift(1))
    recent_vol = daily_logret.tail(20).std()
    recent_vol = float(recent_vol) if not np.isnan(recent_vol) and recent_vol > 0 else 1e-6

    # Trend anchoring: use MA slope (ma_5 vs ma_20)
    ma5 = df.get('ma_5') if 'ma_5' in df.columns else None
    ma20 = df.get('ma_20') if 'ma_20' in df.columns else None
    bullish = False
    if ma5 is not None and ma20 is not None and len(df) >= 6:
        slope = ma5.iloc[-1] - ma5.iloc[-5] if len(ma5) >= 5 else ma5.iloc[-1] - ma5.iloc[0]
        bullish = (ma5.iloc[-1] > ma20.iloc[-1]) and (slope > 0)

    predictions = {}
    for horizon_name, mdl in models.items():
        # Predict normalized log-return
        if isinstance(mdl, dict) and 'gbm' in mdl and 'ridge' in mdl:
            pred_gbm_norm = mdl['gbm'].predict(X)[0]
            pred_ridge_norm = mdl['ridge'].predict(X)[0]
            w = mdl.get('weights', {'gbm': 0.6, 'ridge': 0.4})
            pred_norm = (w.get('gbm', 0.6) * pred_gbm_norm) + (w.get('ridge', 0.4) * pred_ridge_norm)
        else:
            pred_norm = mdl.predict(X)[0]

        # De-normalize to log-return
        pred_logret = float(pred_norm) * recent_vol

        # Trend anchoring: in bullish trend reduce negative bias
        if bullish and pred_logret < 0:
            pred_logret = pred_logret * 0.5 + 0.25 * recent_vol

        # Reconstruct price from last close (log-return)
        pred_price = last_close * float(np.exp(pred_logret))
        predictions[horizon_name] = round(float(pred_price), 2)

    return predictions


def train_quantile_models(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    horizons: dict[str, int] = None,
    quantiles: dict[str, float] = None,
    test_size: float = 0.2,
    save_path: str = None
) -> tuple[dict[str, dict[str, XGBRegressor]], dict[str, dict]]:
    """
    Train quantile regression models for confidence bands.
    Trains 3 models per horizon (low, mid, high quantiles).
    Optimized for speed - lightweight models suitable for laptops.
    
    Args:
        df: DataFrame with features (from create_features)
        feature_cols: List of feature column names
        horizons: Dict mapping horizon name to days ahead
        quantiles: Dict mapping quantile name to value (e.g., {'low': 0.1})
        test_size: Fraction for test split
        save_path: Base path to save models (optional)
        
    Returns:
        Tuple of (models dict, metrics dict)
        models: {horizon: {quantile_name: model}}
        metrics: {horizon: {quantile_name: metrics_dict}}
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    if horizons is None:
        horizons = HORIZONS
    if quantiles is None:
        quantiles = QUANTILES
    
    # Create target columns for all horizons
    df_targets = create_horizon_targets(df, horizons)
    
    # Pre-compute features once for speed
    X = df_targets[feature_cols].values
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    
    models = {}
    all_metrics = {}
    
    for horizon_name in horizons.keys():
        target_col = f'target_{horizon_name}'
        y = df_targets[target_col].values
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        models[horizon_name] = {}
        all_metrics[horizon_name] = {}
        
        for q_name, q_value in quantiles.items():
            # Train lightweight quantile XGBoost
            model = XGBRegressor(
                n_estimators=60,  # Reduced for speed
                max_depth=3,      # Shallower for speed
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist',
                objective='reg:quantileerror',
                quantile_alpha=q_value
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'quantile': q_value
            }
            
            models[horizon_name][q_name] = model
            all_metrics[horizon_name][q_name] = metrics
    
    # Save models if path provided
    if save_path:
        save_quantile_models(models, save_path)
    
    return models, all_metrics


def predict_with_confidence(
    df: pd.DataFrame,
    models: dict[str, dict[str, XGBRegressor]],
    feature_cols: list[str] = None
) -> dict[str, dict[str, float]]:
    """
    Predict closing prices with confidence bands for all horizons.
    Fast single-pass prediction for each quantile model.
    
    Args:
        df: DataFrame with features (most recent data)
        models: Nested dict {horizon: {quantile_name: model}}
        feature_cols: Feature column names
        
    Returns:
        Dictionary with predictions and confidence bands:
        {
            "1d": {"low": x, "mid": y, "high": z},
            "5d": {...},
            "20d": {...}
        }
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    # Get latest features once
    X = df[feature_cols].iloc[-1:].values
    
    predictions = {}
    for horizon_name, quantile_models in models.items():
        predictions[horizon_name] = {}
        for q_name, model in quantile_models.items():
            pred = model.predict(X)[0]
            predictions[horizon_name][q_name] = round(float(pred), 2)
    
    return predictions


def predict_with_regime_adjustment(
    df: pd.DataFrame,
    models: dict[str, dict[str, XGBRegressor]],
    regime: str = None,
    feature_cols: list[str] = None
) -> dict:
    """
    Predict with confidence bands adjusted for market regime.
    Widens/narrows bands based on regime without retraining.
    
    Adjustment Logic (deterministic & explainable):
    - Trending: Narrow bands by 10% (regime is predictable)
    - Ranging: Widen bands by 30% (sideways = harder to predict)
    - High Volatility: Widen bands by 60% (high uncertainty)
    - Transitional: Widen bands by 10% (regime changing)
    
    Args:
        df: DataFrame with OHLC and features
        models: Nested dict {horizon: {quantile_name: model}}
        regime: Market regime string (if None, will detect from df)
        feature_cols: Feature column names
        
    Returns:
        Dictionary with adjusted predictions and metadata:
        {
            "predictions": {
                "1d": {"low": x, "mid": y, "high": z},
                ...
            },
            "regime": "Trending",
            "adjustment_factor": 0.9,
            "confidence_level": "High"
        }
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    # Detect regime if not provided
    if regime is None:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.regime import detect_regime
            regime = detect_regime(df)
        except ImportError:
            regime = 'Transitional'  # Default if regime module unavailable
    
    # Get base predictions from quantile models
    base_preds = predict_with_confidence(df, models, feature_cols)
    
    # Get adjustment factor for regime
    adjustment = REGIME_ADJUSTMENTS.get(regime, 1.0)
    
    # Adjust confidence bands
    adjusted_preds = {}
    for horizon, bands in base_preds.items():
        mid = bands['mid']
        low = bands['low']
        high = bands['high']
        
        # Calculate original band width
        upper_width = high - mid
        lower_width = mid - low
        
        # Apply regime adjustment to band widths
        adjusted_upper = mid + (upper_width * adjustment)
        adjusted_lower = mid - (lower_width * adjustment)
        
        adjusted_preds[horizon] = {
            'low': round(float(adjusted_lower), 2),
            'mid': round(float(mid), 2),
            'high': round(float(adjusted_upper), 2)
        }
    
    # Determine confidence level based on adjustment
    if adjustment <= 0.95:
        confidence_level = "High"
    elif adjustment <= 1.15:
        confidence_level = "Medium"
    elif adjustment <= 1.4:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    return {
        'predictions': adjusted_preds,
        'regime': regime,
        'adjustment_factor': adjustment,
        'confidence_level': confidence_level
    }


def get_regime_explanation(regime: str) -> str:
    """
    Get human-readable explanation for regime-based adjustment.
    
    Args:
        regime: Market regime string
        
    Returns:
        Explanation string
    """
    explanations = {
        'Trending': "Market is trending. Predictions are more reliable, confidence bands narrowed by 10%.",
        'Ranging': "Market is ranging/sideways. Predictions less reliable, confidence bands widened by 30%.",
        'High Volatility': "High volatility detected. Significant uncertainty, confidence bands widened by 60%.",
        'Transitional': "Market regime is changing. Slight uncertainty, confidence bands widened by 10%.",
        'Insufficient Data': "Not enough data for regime detection. Confidence bands widened by 50%."
    }
    return explanations.get(regime, "Unknown regime. Using default confidence bands.")


def save_quantile_models(models: dict[str, dict[str, XGBRegressor]], base_filename: str):
    """
    Save quantile models to models/ folder.
    
    Args:
        models: Nested dict {horizon: {quantile_name: model}}
        base_filename: Base filename (e.g., 'reliance_model.joblib')
    """
    base = Path(base_filename)
    stem = base.stem
    suffix = base.suffix or '.joblib'
    
    path = MODELS_DIR / f"{stem}_quantile{suffix}"
    joblib.dump(models, path)
    print(f"Quantile models saved to {path}")


def load_quantile_models(base_filename: str) -> dict[str, dict[str, XGBRegressor]]:
    """
    Load quantile models from models/ folder.
    
    Args:
        base_filename: Base filename (e.g., 'reliance_model.joblib')
        
    Returns:
        Nested dict {horizon: {quantile_name: model}}
    """
    base = Path(base_filename)
    stem = base.stem
    suffix = base.suffix or '.joblib'
    
    path = MODELS_DIR / f"{stem}_quantile{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Quantile models not found: {path}")
    return joblib.load(path)


def predict_future(
    df: pd.DataFrame,
    model: XGBRegressor,
    days: int = 5,
    feature_cols: list[str] = None
) -> pd.DataFrame:
    """
    Predict next N days closing prices (iterative forecast).
    
    Args:
        df: DataFrame with features (most recent data)
        model: Trained XGBoost model
        days: Number of days to predict
        feature_cols: Feature column names
        
    Returns:
        DataFrame with predicted dates and prices
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    
    df = df.copy()
    predictions = []
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    # recent volatility and last close for de-normalization
    daily_logret = np.log(df['Close'] / df['Close'].shift(1))
    recent_vol = daily_logret.tail(20).std()
    recent_vol = float(recent_vol) if not np.isnan(recent_vol) and recent_vol > 0 else 1e-6

    for i in range(days):
        # Get latest features
        X = df[feature_cols].iloc[-1:].values

        # Support ensemble dicts or single models predicting normalized log-return
        if isinstance(model, dict) and 'gbm' in model and 'ridge' in model:
            pg = model['gbm'].predict(X)[0]
            pr = model['ridge'].predict(X)[0]
            w = model.get('weights', {'gbm': 0.6, 'ridge': 0.4})
            pred_norm = w.get('gbm', 0.6) * pg + w.get('ridge', 0.4) * pr
        else:
            pred_norm = model.predict(X)[0]

        # De-normalize to log-return and compute next close
        pred_logret = float(pred_norm) * recent_vol
        prev_close = float(df['Close'].iloc[-1])
        next_close = prev_close * float(np.exp(pred_logret))

        next_date = last_date + pd.Timedelta(days=1)

        # Skip weekends
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        predictions.append({
            'Date': next_date,
            'Predicted_Close': round(next_close, 2)
        })

        # Update features for next prediction (rolling update)
        new_row = df.iloc[-1:].copy()
        new_row['Date'] = next_date
        new_row['Close'] = next_close
        # recompute simple derived features
        new_row['returns'] = np.log(next_close / prev_close)
        # Rolling means: take recent closes
        closes = df['Close'].tolist()
        closes.append(next_close)
        new_row['ma_5'] = np.mean(closes[-5:]) if len(closes) >= 5 else np.mean(closes)
        new_row['ma_10'] = np.mean(closes[-10:]) if len(closes) >= 10 else np.mean(closes)
        new_row['ma_20'] = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)

        # new_row is already a one-row DataFrame slice; concat directly
        df = pd.concat([df, new_row], ignore_index=True)
        last_date = next_date
    
    return pd.DataFrame(predictions)


def save_model(model: XGBRegressor, filename: str):
    """Save model to models/ folder."""
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def save_multi_horizon_models(models: dict[str, XGBRegressor], base_filename: str):
    """
    Save multi-horizon models to models/ folder.
    
    Args:
        models: Dict of models keyed by horizon name
        base_filename: Base filename (e.g., 'reliance_model.joblib')
    """
    base = Path(base_filename)
    stem = base.stem
    suffix = base.suffix or '.joblib'
    
    # Save all models in a single file for efficiency
    path = MODELS_DIR / f"{stem}_multi{suffix}"
    joblib.dump(models, path)
    print(f"Multi-horizon models saved to {path}")


def load_model(filename: str) -> XGBRegressor:
    """Load model from models/ folder."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_multi_horizon_models(base_filename: str) -> dict[str, XGBRegressor]:
    """
    Load multi-horizon models from models/ folder.
    
    Args:
        base_filename: Base filename (e.g., 'reliance_model.joblib')
        
    Returns:
        Dict of models keyed by horizon name
    """
    base = Path(base_filename)
    stem = base.stem
    suffix = base.suffix or '.joblib'
    
    path = MODELS_DIR / f"{stem}_multi{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Multi-horizon models not found: {path}")
    return joblib.load(path)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import fetch_stock_data
    from utils.features import create_features, get_feature_columns
    
    # Fetch and prepare data
    df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2024-12-31")
    if not df.empty:
        df = create_features(df)
        
        # Train single model (original flow)
        print("=== Single Model Training ===")
        model, metrics = train_model(df, save_path="reliance_model.joblib")
        print(f"Metrics: {metrics}")
        
        # Predict next 5 days (iterative)
        preds = predict_future(df, model, days=5)
        print(f"\nIterative Predictions:\n{preds}")
        
        # Train multi-horizon models
        print("\n=== Multi-Horizon Model Training ===")
        models, all_metrics = train_multi_horizon_models(
            df, 
            save_path="reliance_model.joblib"
        )
        for horizon, m in all_metrics.items():
            print(f"{horizon}: MAE={m['mae']:.4f}")
        
        # Predict multi-horizon (fast, direct prediction)
        print("\n=== Multi-Horizon Predictions ===")
        multi_preds = predict_multi_horizon(df, models)
        print(f"Predictions: {multi_preds}")
        # Output: {"1d": value, "5d": value, "20d": value}
        
        # Train quantile models for confidence bands
        print("\n=== Quantile Model Training (Confidence Bands) ===")
        q_models, q_metrics = train_quantile_models(
            df,
            save_path="reliance_model.joblib"
        )
        for horizon, qm in q_metrics.items():
            print(f"{horizon}: low_mae={qm['low']['mae']:.2f}, mid_mae={qm['mid']['mae']:.2f}, high_mae={qm['high']['mae']:.2f}")
        
        # Predict with confidence bands
        print("\n=== Predictions with Confidence Bands ===")
        conf_preds = predict_with_confidence(df, q_models)
        print(f"Predictions: {conf_preds}")
        # Output: {"1d": {"low": x, "mid": y, "high": z}, "5d": {...}, "20d": {...}}