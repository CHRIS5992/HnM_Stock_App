# regime.py
# Market regime detection using rule-based indicators

import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: Lookback period
        
    Returns:
        ATR series
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) for trend strength.
    ADX > 25 indicates trending, ADX < 20 indicates ranging.
    
    Args:
        df: DataFrame with High, Low, Close columns
        period: Lookback period
        
    Returns:
        ADX series
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values (Wilder's smoothing)
    atr = true_range.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Directional Index
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX is smoothed DX
    adx = dx.rolling(window=period).mean()
    
    return adx


def compute_volatility_regime(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate rolling volatility as percentage of price.
    
    Args:
        df: DataFrame with Close column
        period: Lookback period
        
    Returns:
        Volatility ratio series (current vol / average vol)
    """
    returns = df['Close'].pct_change()
    rolling_vol = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    avg_vol = rolling_vol.rolling(window=period * 3).mean()  # Longer-term average
    
    # Volatility ratio: current vs average
    vol_ratio = rolling_vol / (avg_vol + 1e-10)
    
    return vol_ratio


def detect_regime(
    df: pd.DataFrame,
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
    vol_high_threshold: float = 1.5,
    period: int = 14
) -> str:
    """
    Detect current market regime using ADX and volatility.
    
    Rules (evaluated in order):
    1. High Volatility: vol_ratio > 1.5 (50% above average)
    2. Trending: ADX > 25
    3. Ranging: ADX < 20
    4. Transitional: ADX between 20-25
    
    Args:
        df: DataFrame with OHLC data
        adx_trend_threshold: ADX above this = trending (default 25)
        adx_range_threshold: ADX below this = ranging (default 20)
        vol_high_threshold: Vol ratio above this = high volatility (default 1.5)
        period: Indicator period
        
    Returns:
        Regime label: "Trending", "Ranging", "High Volatility", or "Transitional"
    """
    if len(df) < period * 4:
        return "Insufficient Data"
    
    # Compute indicators
    adx = compute_adx(df, period)
    vol_ratio = compute_volatility_regime(df, period)
    
    # Get latest values
    current_adx = adx.iloc[-1]
    current_vol_ratio = vol_ratio.iloc[-1]
    
    # Handle NaN
    if pd.isna(current_adx) or pd.isna(current_vol_ratio):
        return "Insufficient Data"
    
    # Rule-based classification (order matters)
    if current_vol_ratio > vol_high_threshold:
        return "High Volatility"
    elif current_adx > adx_trend_threshold:
        return "Trending"
    elif current_adx < adx_range_threshold:
        return "Ranging"
    else:
        return "Transitional"


def get_regime_details(
    df: pd.DataFrame,
    period: int = 14
) -> dict:
    """
    Get detailed regime analysis with indicator values.
    
    Args:
        df: DataFrame with OHLC data
        period: Indicator period
        
    Returns:
        Dictionary with regime label and indicator values
    """
    if len(df) < period * 4:
        return {
            "regime": "Insufficient Data",
            "adx": None,
            "volatility_ratio": None,
            "trend_direction": None
        }
    
    # Compute indicators
    adx = compute_adx(df, period)
    vol_ratio = compute_volatility_regime(df, period)
    
    # Trend direction using simple MA comparison
    ma_short = df['Close'].rolling(window=10).mean()
    ma_long = df['Close'].rolling(window=30).mean()
    
    current_adx = adx.iloc[-1]
    current_vol_ratio = vol_ratio.iloc[-1]
    
    # Determine trend direction
    if pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]):
        if ma_short.iloc[-1] > ma_long.iloc[-1]:
            trend_direction = "Bullish"
        else:
            trend_direction = "Bearish"
    else:
        trend_direction = "Unknown"
    
    # Get regime label
    regime = detect_regime(df, period=period)
    
    return {
        "regime": regime,
        "adx": round(float(current_adx), 2) if pd.notna(current_adx) else None,
        "volatility_ratio": round(float(current_vol_ratio), 2) if pd.notna(current_vol_ratio) else None,
        "trend_direction": trend_direction
    }


def get_regime_history(
    df: pd.DataFrame,
    period: int = 14,
    adx_trend_threshold: float = 25.0,
    adx_range_threshold: float = 20.0,
    vol_high_threshold: float = 1.5
) -> pd.DataFrame:
    """
    Get regime classification for entire history.
    Useful for backtesting and visualization.
    
    Args:
        df: DataFrame with OHLC data
        period: Indicator period
        adx_trend_threshold: ADX above this = trending
        adx_range_threshold: ADX below this = ranging
        vol_high_threshold: Vol ratio above this = high volatility
        
    Returns:
        DataFrame with Date, ADX, Vol_Ratio, Regime columns
    """
    df = df.copy()
    
    # Compute indicators
    df['ADX'] = compute_adx(df, period)
    df['Vol_Ratio'] = compute_volatility_regime(df, period)
    
    # Classify each row
    def classify_row(row):
        if pd.isna(row['ADX']) or pd.isna(row['Vol_Ratio']):
            return "Insufficient Data"
        if row['Vol_Ratio'] > vol_high_threshold:
            return "High Volatility"
        elif row['ADX'] > adx_trend_threshold:
            return "Trending"
        elif row['ADX'] < adx_range_threshold:
            return "Ranging"
        else:
            return "Transitional"
    
    df['Regime'] = df.apply(classify_row, axis=1)
    
    return df[['Date', 'Close', 'ADX', 'Vol_Ratio', 'Regime']].copy()


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import fetch_stock_data
    
    # Fetch sample data
    df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2024-12-31")
    
    if not df.empty:
        # Detect current regime
        regime = detect_regime(df)
        print(f"Current Market Regime: {regime}")
        
        # Get detailed analysis
        details = get_regime_details(df)
        print(f"\nDetailed Analysis:")
        print(f"  Regime: {details['regime']}")
        print(f"  ADX: {details['adx']}")
        print(f"  Volatility Ratio: {details['volatility_ratio']}")
        print(f"  Trend Direction: {details['trend_direction']}")
        
        # Get regime history
        history = get_regime_history(df)
        print(f"\nRegime Distribution (last 100 days):")
        print(history['Regime'].tail(100).value_counts())
