# features.py
# Feature engineering for stock price prediction

import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from OHLCV data.
    
    Args:
        df: DataFrame with Date, Open, High, Low, Close, Volume
        
    Returns:
        DataFrame with features, NaNs dropped
    """
    df = df.copy()
    
    # Ensure data is sorted by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    close = df['Close']
    
    # Returns
    df['returns'] = close.pct_change()
    
    # Moving averages
    df['ma_5'] = close.rolling(window=5).mean()
    df['ma_10'] = close.rolling(window=10).mean()
    df['ma_20'] = close.rolling(window=20).mean()
    
    # Volatility (rolling std of returns)
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # RSI (14-day)
    df['rsi'] = compute_rsi(close, period=14)
    
    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    
    return df


def get_feature_columns() -> list[str]:
    """Return list of feature column names."""
    return ['returns', 'ma_5', 'ma_10', 'ma_20', 'volatility', 'rsi']


if __name__ == "__main__":
    # Quick test with sample data
    from data_loader import fetch_stock_data
    
    df = fetch_stock_data("RELIANCE.NS", "2024-01-01", "2024-12-31")
    if not df.empty:
        df_features = create_features(df)
        print(f"Original shape: {df.shape}")
        print(f"With features: {df_features.shape}")
        print(f"\nFeature columns: {get_feature_columns()}")
        print(f"\nSample:\n{df_features[['Date', 'Close'] + get_feature_columns()].tail()}")
