# data_loader.py
# Handles loading NSE stock symbols and fetching historical price data

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal, Optional

# Paths
ROOT_DIR = Path(__file__).parent.parent
EQUITY_CSV = ROOT_DIR / "EQUITY_L.csv"
CACHE_DIR = ROOT_DIR / "data" / "cache"

# Valid intervals for Yahoo Finance
VALID_INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo']


def _get_cache_path(symbol: str, start: str, end: str, interval: str = '1d') -> Path:
    """Generate cache file path for a symbol, date range, and interval."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace(".", "_")
    return CACHE_DIR / f"{safe_symbol}_{start}_{end}_{interval}.csv"


def fetch_stock_data(
    symbol: str,
    start: str,
    end: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch historical stock price data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        use_cache: If True, use cached data if available
        
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """
    cache_path = _get_cache_path(symbol, start, end)
    
    # Return cached data if available
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=['Date'])
        return df
    
    # Fetch from Yahoo Finance
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Keep only required columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Clean date column (remove timezone)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Handle missing data
        df = df.dropna(subset=['Close'])
        df = df.fillna(method='ffill')
        
        # Cache the result
        if use_cache:
            df.to_csv(cache_path, index=False)
        
        return df
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])


def load_equity_csv() -> pd.DataFrame:
    """Load and clean the EQUITY_L.csv file."""
    df = pd.read_csv(EQUITY_CSV)
    # Clean column names: strip whitespace and lowercase
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def get_nse_symbols(yahoo_format: bool = True) -> list[str]:
    """
    Get list of valid NSE stock symbols.
    
    Args:
        yahoo_format: If True, append '.NS' for Yahoo Finance compatibility
        
    Returns:
        List of stock symbols
    """
    df = load_equity_csv()
    
    # Get symbols from the 'symbol' column
    symbols = df['symbol'].dropna().astype(str).str.strip().tolist()
    
    # Filter out empty strings
    symbols = [s for s in symbols if s]
    
    # Convert to Yahoo Finance format if requested
    if yahoo_format:
        symbols = [f"{s}.NS" for s in symbols]
    
    return symbols


def fetch_intraday_data(
    symbol: str,
    interval: str = '1h',
    period: str = '1mo',
    use_cache: bool = False
) -> pd.DataFrame:
    """
    Fetch intraday stock data from Yahoo Finance.
    
    Note: Yahoo Finance limits intraday data:
    - 1m: Last 7 days only
    - 2m-60m: Last 60 days
    - 1h: Last 730 days
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        interval: Data interval ('1m', '5m', '15m', '30m', '1h')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
        use_cache: If True, use cached data if available
        
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """
    if interval not in VALID_INTERVALS:
        print(f"Invalid interval: {interval}. Valid: {VALID_INTERVALS}")
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Generate cache path
    today = datetime.now().strftime('%Y-%m-%d')
    cache_path = CACHE_DIR / f"{symbol.replace('.', '_')}_{interval}_{period}_{today}.csv"
    
    # Return cached data if available
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=['Date'])
        return df
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Handle different column names for intraday data
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={date_col: 'Date'})
        
        # Keep only required columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Clean date column (remove timezone)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Handle missing data
        df = df.dropna(subset=['Close'])
        df = df.ffill()
        
        # Cache the result
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
        
        return df
        
    except Exception as e:
        print(f"Error fetching intraday data for {symbol}: {e}")
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])


def fetch_multi_timeframe(
    symbol: str,
    start: str,
    end: str
) -> dict[str, pd.DataFrame]:
    """
    Fetch data for multiple timeframes for comprehensive analysis.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary with timeframe keys and DataFrame values
    """
    data = {}
    
    # Daily data (full history)
    data['1d'] = fetch_stock_data(symbol, start, end)
    
    # Hourly data (limited to ~730 days max)
    data['1h'] = fetch_intraday_data(symbol, interval='1h', period='1mo')
    
    # 15-minute data (limited to ~60 days max)
    data['15m'] = fetch_intraday_data(symbol, interval='15m', period='5d')
    
    return data


if __name__ == "__main__":
    # Quick test
    symbols = get_nse_symbols()
    print(f"Total symbols: {len(symbols)}")
    print(f"First 5: {symbols[:5]}")
    
    # Test fetching daily data
    df = fetch_stock_data("RELIANCE.NS", "2024-01-01", "2024-12-31")
    print(f"\nRELIANCE.NS daily data shape: {df.shape}")
    if not df.empty:
        print(df.head())
    
    # Test fetching intraday data
    df_intra = fetch_intraday_data("RELIANCE.NS", interval='1h', period='5d')
    print(f"\nRELIANCE.NS hourly data shape: {df_intra.shape}")
    if not df_intra.empty:
        print(df_intra.head())

