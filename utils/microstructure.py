# microstructure.py
# Microstructure Discovery: Volume Profile, Order Book Simulation, Market Depth
# For institutional-grade order flow analysis

import pandas as pd
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass


# ============== DATA CLASSES ==============

@dataclass
class VolumeNode:
    """Represents a volume-at-price node."""
    price_level: float
    volume: float
    buy_volume: float
    sell_volume: float
    is_poc: bool = False  # Point of Control
    is_hva: bool = False  # High Volume Area
    is_lva: bool = False  # Low Volume Area


@dataclass
class MarketDepthLevel:
    """Simulated order book level."""
    price: float
    bid_size: float
    ask_size: float
    cumulative_bid: float
    cumulative_ask: float


@dataclass
class OrderImbalance:
    """Order flow imbalance data."""
    timestamp: pd.Timestamp
    price: float
    imbalance_ratio: float  # -1 to 1 (negative = selling pressure)
    delta: float  # Buy volume - Sell volume
    cumulative_delta: float


# ============== VOLUME PROFILE ==============

def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70
) -> dict:
    """
    Calculate Volume Profile (Volume at Price / VPVR).
    
    Args:
        df: DataFrame with OHLC and Volume data
        num_bins: Number of price levels for the profile
        value_area_pct: Percentage of volume to include in Value Area
        
    Returns:
        Dictionary with volume profile data
    """
    if df.empty or 'Volume' not in df.columns:
        return {'nodes': [], 'poc': None, 'vah': None, 'val': None}
    
    # Get price range
    price_high = df['High'].max()
    price_low = df['Low'].min()
    price_range = price_high - price_low
    
    if price_range == 0:
        return {'nodes': [], 'poc': None, 'vah': None, 'val': None}
    
    # Create price bins
    bin_size = price_range / num_bins
    bins = np.linspace(price_low, price_high, num_bins + 1)
    
    # Initialize volume at each level
    volume_at_price = np.zeros(num_bins)
    buy_volume_at_price = np.zeros(num_bins)
    sell_volume_at_price = np.zeros(num_bins)
    
    # Distribute volume across price levels
    for idx, row in df.iterrows():
        candle_low = row['Low']
        candle_high = row['High']
        candle_open = row['Open']
        candle_close = row['Close']
        volume = row['Volume']
        
        if volume == 0 or pd.isna(volume):
            continue
        
        # Determine if bullish or bearish candle
        is_bullish = candle_close >= candle_open
        
        # Find which bins this candle touches
        for i in range(num_bins):
            bin_low = bins[i]
            bin_high = bins[i + 1]
            
            # Check overlap
            overlap_low = max(candle_low, bin_low)
            overlap_high = min(candle_high, bin_high)
            
            if overlap_high > overlap_low:
                # Calculate proportion of candle in this bin
                candle_range = candle_high - candle_low
                if candle_range > 0:
                    proportion = (overlap_high - overlap_low) / candle_range
                    bin_volume = volume * proportion
                else:
                    bin_volume = volume if candle_low == candle_high else 0
                
                volume_at_price[i] += bin_volume
                
                if is_bullish:
                    buy_volume_at_price[i] += bin_volume
                else:
                    sell_volume_at_price[i] += bin_volume
    
    # Find Point of Control (POC) - highest volume level
    poc_idx = np.argmax(volume_at_price)
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
    
    # Calculate Value Area (VAH/VAL)
    total_volume = volume_at_price.sum()
    target_volume = total_volume * value_area_pct
    
    # Start from POC and expand outward
    included = np.zeros(num_bins, dtype=bool)
    included[poc_idx] = True
    current_volume = volume_at_price[poc_idx]
    
    while current_volume < target_volume:
        # Look at adjacent levels
        candidates = []
        
        # Find leftmost and rightmost included indices
        included_indices = np.where(included)[0]
        left_edge = included_indices.min()
        right_edge = included_indices.max()
        
        if left_edge > 0:
            candidates.append((left_edge - 1, volume_at_price[left_edge - 1]))
        if right_edge < num_bins - 1:
            candidates.append((right_edge + 1, volume_at_price[right_edge + 1]))
        
        if not candidates:
            break
        
        # Add the candidate with higher volume
        best_candidate = max(candidates, key=lambda x: x[1])
        included[best_candidate[0]] = True
        current_volume += best_candidate[1]
    
    # Value Area High and Low
    included_indices = np.where(included)[0]
    val_price = (bins[included_indices.min()] + bins[included_indices.min() + 1]) / 2
    vah_price = (bins[included_indices.max()] + bins[included_indices.max() + 1]) / 2
    
    # Determine high/low volume areas
    median_volume = np.median(volume_at_price[volume_at_price > 0])
    
    # Create volume nodes
    nodes = []
    for i in range(num_bins):
        if volume_at_price[i] > 0:
            price_level = (bins[i] + bins[i + 1]) / 2
            nodes.append(VolumeNode(
                price_level=price_level,
                volume=volume_at_price[i],
                buy_volume=buy_volume_at_price[i],
                sell_volume=sell_volume_at_price[i],
                is_poc=(i == poc_idx),
                is_hva=(volume_at_price[i] > median_volume * 1.5),
                is_lva=(volume_at_price[i] < median_volume * 0.5)
            ))
    
    return {
        'nodes': nodes,
        'poc': poc_price,
        'vah': vah_price,
        'val': val_price,
        'total_volume': total_volume,
        'bin_size': bin_size,
        'price_range': (price_low, price_high)
    }


def get_volume_profile_df(volume_profile: dict) -> pd.DataFrame:
    """
    Convert volume profile to DataFrame for plotting.
    
    Args:
        volume_profile: Output from calculate_volume_profile()
        
    Returns:
        DataFrame with volume profile data
    """
    nodes = volume_profile['nodes']
    if not nodes:
        return pd.DataFrame(columns=['Price', 'Volume', 'BuyVolume', 'SellVolume', 'IsPOC', 'IsHVA', 'IsLVA'])
    
    return pd.DataFrame([
        {
            'Price': n.price_level,
            'Volume': n.volume,
            'BuyVolume': n.buy_volume,
            'SellVolume': n.sell_volume,
            'IsPOC': n.is_poc,
            'IsHVA': n.is_hva,
            'IsLVA': n.is_lva
        }
        for n in nodes
    ])


# ============== ORDER BOOK SIMULATION ==============

def simulate_order_book(
    df: pd.DataFrame,
    num_levels: int = 20,
    base_liquidity: float = 10000,
    volatility_factor: float = 1.0
) -> dict:
    """
    Simulate order book depth based on historical price action.
    Note: This is a simulation since we don't have real L2 data.
    
    Args:
        df: DataFrame with OHLC data
        num_levels: Number of price levels on each side
        base_liquidity: Base order size
        volatility_factor: Adjusts spread based on volatility
        
    Returns:
        Dictionary with simulated order book data
    """
    if df.empty:
        return {'bids': [], 'asks': [], 'mid_price': 0, 'spread': 0}
    
    # Get current price and volatility
    current_price = df['Close'].iloc[-1]
    
    # Calculate volatility-based spread
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() if len(returns) > 0 else 0.01
    spread_pct = max(0.001, volatility * volatility_factor)  # Min 0.1% spread
    
    # Mid price and spread
    spread = current_price * spread_pct
    best_bid = current_price - spread / 2
    best_ask = current_price + spread / 2
    
    # Price levels based on ATR
    if 'High' in df.columns and 'Low' in df.columns:
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr):
            atr = current_price * 0.01
    else:
        atr = current_price * 0.01
    
    level_spacing = atr / num_levels
    
    # Generate bid levels (buying interest)
    bids = []
    cumulative_bid = 0
    for i in range(num_levels):
        price = best_bid - (i * level_spacing)
        # Volume increases further from mid (more orders at better prices for buyers)
        distance_factor = 1 + (i * 0.1)
        # Add some randomness
        size = base_liquidity * distance_factor * (0.8 + np.random.random() * 0.4)
        cumulative_bid += size
        
        bids.append(MarketDepthLevel(
            price=price,
            bid_size=size,
            ask_size=0,
            cumulative_bid=cumulative_bid,
            cumulative_ask=0
        ))
    
    # Generate ask levels (selling interest)
    asks = []
    cumulative_ask = 0
    for i in range(num_levels):
        price = best_ask + (i * level_spacing)
        distance_factor = 1 + (i * 0.1)
        size = base_liquidity * distance_factor * (0.8 + np.random.random() * 0.4)
        cumulative_ask += size
        
        asks.append(MarketDepthLevel(
            price=price,
            bid_size=0,
            ask_size=size,
            cumulative_bid=0,
            cumulative_ask=cumulative_ask
        ))
    
    return {
        'bids': bids,
        'asks': asks,
        'mid_price': current_price,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'spread': spread,
        'spread_pct': spread_pct * 100,
        'total_bid_liquidity': cumulative_bid,
        'total_ask_liquidity': cumulative_ask
    }


def get_depth_chart_data(order_book: dict) -> pd.DataFrame:
    """
    Convert order book to DataFrame for depth chart visualization.
    
    Args:
        order_book: Output from simulate_order_book()
        
    Returns:
        DataFrame with depth chart data
    """
    bids = order_book['bids']
    asks = order_book['asks']
    
    data = []
    
    # Add bids (cumulative from best bid outward)
    for bid in bids:
        data.append({
            'Price': bid.price,
            'Side': 'Bid',
            'Size': bid.bid_size,
            'Cumulative': bid.cumulative_bid
        })
    
    # Add asks (cumulative from best ask outward)
    for ask in asks:
        data.append({
            'Price': ask.price,
            'Side': 'Ask',
            'Size': ask.ask_size,
            'Cumulative': ask.cumulative_ask
        })
    
    return pd.DataFrame(data)


# ============== ORDER FLOW ANALYSIS ==============

def calculate_order_imbalance(
    df: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:
    """
    Calculate order flow imbalance from OHLCV data.
    Uses price action to infer buying/selling pressure.
    
    Args:
        df: DataFrame with OHLC and Volume data
        window: Rolling window for analysis
        
    Returns:
        DataFrame with order imbalance data
    """
    df = df.copy()
    
    # Classify each candle as buy or sell dominated
    # Using typical price position within the range
    df['Range'] = df['High'] - df['Low']
    df['ClosePosition'] = (df['Close'] - df['Low']) / df['Range'].replace(0, np.nan)
    df['ClosePosition'] = df['ClosePosition'].fillna(0.5)
    
    # Estimate buy/sell volume split based on close position
    df['BuyVolume'] = df['Volume'] * df['ClosePosition']
    df['SellVolume'] = df['Volume'] * (1 - df['ClosePosition'])
    
    # Calculate delta (buy - sell)
    df['Delta'] = df['BuyVolume'] - df['SellVolume']
    df['CumulativeDelta'] = df['Delta'].cumsum()
    
    # Calculate imbalance ratio
    total_volume = df['BuyVolume'] + df['SellVolume']
    df['ImbalanceRatio'] = df['Delta'] / total_volume.replace(0, np.nan)
    df['ImbalanceRatio'] = df['ImbalanceRatio'].fillna(0)
    
    # Rolling metrics
    df['RollingDelta'] = df['Delta'].rolling(window).sum()
    df['RollingImbalance'] = df['ImbalanceRatio'].rolling(window).mean()
    
    return df[['Date', 'Close', 'Volume', 'BuyVolume', 'SellVolume', 
               'Delta', 'CumulativeDelta', 'ImbalanceRatio', 
               'RollingDelta', 'RollingImbalance']].dropna()


def calculate_footprint_data(
    df: pd.DataFrame,
    num_levels: int = 10
) -> dict:
    """
    Calculate footprint-style data (volume delta at price levels).
    
    Args:
        df: DataFrame with OHLC and Volume data
        num_levels: Number of price levels per candle
        
    Returns:
        Dictionary with footprint data
    """
    footprints = []
    
    for idx, row in df.iterrows():
        candle_high = row['High']
        candle_low = row['Low']
        candle_open = row['Open']
        candle_close = row['Close']
        volume = row['Volume']
        
        if candle_high == candle_low or volume == 0:
            continue
        
        is_bullish = candle_close >= candle_open
        
        # Create price levels within the candle
        levels = np.linspace(candle_low, candle_high, num_levels + 1)
        
        level_data = []
        for i in range(num_levels):
            level_price = (levels[i] + levels[i + 1]) / 2
            level_volume = volume / num_levels
            
            # Distribute buy/sell based on position relative to body
            body_low = min(candle_open, candle_close)
            body_high = max(candle_open, candle_close)
            
            if level_price >= body_low and level_price <= body_high:
                # Inside the body - bias based on direction
                buy_pct = 0.7 if is_bullish else 0.3
            elif level_price > body_high:
                # Upper wick - mostly selling
                buy_pct = 0.3
            else:
                # Lower wick - mostly buying
                buy_pct = 0.7
            
            level_data.append({
                'price': level_price,
                'volume': level_volume,
                'buy_volume': level_volume * buy_pct,
                'sell_volume': level_volume * (1 - buy_pct),
                'delta': level_volume * (2 * buy_pct - 1)
            })
        
        footprints.append({
            'date': row['Date'],
            'open': candle_open,
            'high': candle_high,
            'low': candle_low,
            'close': candle_close,
            'total_volume': volume,
            'is_bullish': is_bullish,
            'levels': level_data
        })
    
    return {'footprints': footprints}


# ============== MARKET REGIME SCORING ==============

def calculate_microstructure_score(
    df: pd.DataFrame,
    volume_profile: dict = None,
    order_book: dict = None
) -> dict:
    """
    Calculate a comprehensive microstructure score.
    
    Args:
        df: DataFrame with OHLC and Volume data
        volume_profile: Output from calculate_volume_profile()
        order_book: Output from simulate_order_book()
        
    Returns:
        Dictionary with microstructure metrics and scores
    """
    scores = {}
    
    # Calculate order imbalance
    imbalance_df = calculate_order_imbalance(df)
    if not imbalance_df.empty:
        recent_imbalance = imbalance_df['RollingImbalance'].iloc[-1]
        cumulative_delta = imbalance_df['CumulativeDelta'].iloc[-1]
        
        scores['imbalance_score'] = recent_imbalance  # -1 to 1
        scores['cumulative_delta'] = cumulative_delta
        scores['delta_trend'] = 'Bullish' if cumulative_delta > 0 else 'Bearish'
    else:
        scores['imbalance_score'] = 0
        scores['cumulative_delta'] = 0
        scores['delta_trend'] = 'Neutral'
    
    # Volume Profile analysis
    if volume_profile and volume_profile.get('nodes'):
        poc = volume_profile['poc']
        vah = volume_profile['vah']
        val = volume_profile['val']
        current_price = df['Close'].iloc[-1]
        
        # Position relative to value area
        if current_price > vah:
            scores['value_area_position'] = 'Above Value Area'
            scores['value_area_score'] = 1
        elif current_price < val:
            scores['value_area_position'] = 'Below Value Area'
            scores['value_area_score'] = -1
        else:
            scores['value_area_position'] = 'Inside Value Area'
            scores['value_area_score'] = 0
        
        # Distance from POC
        poc_distance = (current_price - poc) / poc * 100
        scores['poc_distance_pct'] = poc_distance
    else:
        scores['value_area_position'] = 'Unknown'
        scores['value_area_score'] = 0
        scores['poc_distance_pct'] = 0
    
    # Order book analysis
    if order_book:
        bid_liq = order_book.get('total_bid_liquidity', 0)
        ask_liq = order_book.get('total_ask_liquidity', 0)
        
        if bid_liq + ask_liq > 0:
            liquidity_imbalance = (bid_liq - ask_liq) / (bid_liq + ask_liq)
            scores['liquidity_imbalance'] = liquidity_imbalance
        else:
            scores['liquidity_imbalance'] = 0
        
        scores['spread_pct'] = order_book.get('spread_pct', 0)
    else:
        scores['liquidity_imbalance'] = 0
        scores['spread_pct'] = 0
    
    # Calculate overall microstructure score (-100 to 100)
    overall_score = (
        scores['imbalance_score'] * 40 +  # Order flow imbalance
        scores['value_area_score'] * 30 +  # Value area position
        scores.get('liquidity_imbalance', 0) * 30  # Order book imbalance
    )
    
    scores['overall_score'] = overall_score
    
    # Classify regime
    if overall_score > 30:
        scores['regime'] = 'Strong Buying Pressure'
        scores['regime_color'] = 'green'
    elif overall_score > 10:
        scores['regime'] = 'Mild Buying Pressure'
        scores['regime_color'] = 'lightgreen'
    elif overall_score < -30:
        scores['regime'] = 'Strong Selling Pressure'
        scores['regime_color'] = 'red'
    elif overall_score < -10:
        scores['regime'] = 'Mild Selling Pressure'
        scores['regime_color'] = 'salmon'
    else:
        scores['regime'] = 'Balanced / Consolidation'
        scores['regime_color'] = 'gray'
    
    return scores


# ============== COMPLETE MICROSTRUCTURE ANALYSIS ==============

def analyze_microstructure(
    df: pd.DataFrame,
    volume_bins: int = 50,
    depth_levels: int = 20
) -> dict:
    """
    Perform complete microstructure analysis.
    
    Args:
        df: DataFrame with OHLC and Volume data
        volume_bins: Number of bins for volume profile
        depth_levels: Number of levels for order book
        
    Returns:
        Dictionary with all microstructure data
    """
    # Calculate volume profile
    volume_profile = calculate_volume_profile(df, volume_bins)
    
    # Simulate order book
    order_book = simulate_order_book(df, depth_levels)
    
    # Calculate order imbalance
    order_imbalance = calculate_order_imbalance(df)
    
    # Calculate microstructure score
    micro_score = calculate_microstructure_score(df, volume_profile, order_book)
    
    return {
        'volume_profile': volume_profile,
        'order_book': order_book,
        'order_imbalance': order_imbalance,
        'micro_score': micro_score,
        'summary': {
            'poc': volume_profile.get('poc'),
            'vah': volume_profile.get('vah'),
            'val': volume_profile.get('val'),
            'spread_pct': order_book.get('spread_pct'),
            'regime': micro_score.get('regime'),
            'overall_score': micro_score.get('overall_score')
        }
    }
