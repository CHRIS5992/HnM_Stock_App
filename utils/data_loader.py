# data_loader.py
# Handles loading NSE stock symbols and fetching historical price data

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

# Paths
ROOT_DIR = Path(__file__).parent.parent
EQUITY_CSV = ROOT_DIR / "EQUITY_L.csv"
CACHE_DIR = ROOT_DIR / "data" / "cache"


def _get_cache_path(symbol: str, start: str, end: str) -> Path:
    """Generate cache file path for a symbol and date range."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace(".", "_")
    return CACHE_DIR / f"{safe_symbol}_{start}_{end}.csv"


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


if __name__ == "__main__":
    # Quick test
    symbols = get_nse_symbols()
    print(f"Total symbols: {len(symbols)}")
    print(f"First 5: {symbols[:5]}")
    
    # Test fetching data
    df = fetch_stock_data("RELIANCE.NS", "2024-01-01", "2024-12-31")
    print(f"\nRELIANCE.NS data shape: {df.shape}")
    if not df.empty:
        print(df.head())
