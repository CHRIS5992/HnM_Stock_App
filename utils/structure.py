# structure.py
# Smart Money Concepts: BOS, CHoCH, Order Blocks, Fair Value Gaps
# Macrostructure Discovery for institutional-grade analysis

import pandas as pd
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass


# ============== DATA CLASSES ==============

@dataclass
class SwingPoint:
    """Represents a swing high or low point."""
    index: int
    date: pd.Timestamp
    price: float
    type: Literal['high', 'low']


@dataclass
class StructureBreak:
    """Represents a BOS or CHoCH event."""
    index: int
    date: pd.Timestamp
    price: float
    break_type: Literal['BOS', 'CHoCH']
    direction: Literal['bullish', 'bearish']
    swing_broken: SwingPoint


@dataclass
class OrderBlock:
    """Represents an Order Block (supply/demand zone)."""
    start_index: int
    end_index: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    top: float
    bottom: float
    ob_type: Literal['bullish', 'bearish']
    strength: float  # 0-1 score based on displacement strength
    tested: bool = False
    invalidated: bool = False


@dataclass 
class FairValueGap:
    """Represents a Fair Value Gap (imbalance)."""
    index: int
    date: pd.Timestamp
    top: float
    bottom: float
    fvg_type: Literal['bullish', 'bearish']
    filled: bool = False
    fill_percentage: float = 0.0


@dataclass
class LiquidityPool:
    """Represents a liquidity pool (equal highs/lows)."""
    start_index: int
    end_index: int
    price_level: float
    pool_type: Literal['equal_highs', 'equal_lows']
    strength: int  # Number of touches
    swept: bool = False


# ============== SWING DETECTION ==============

def detect_swing_points(
    df: pd.DataFrame,
    lookback: int = 5,
    min_swing_size: float = 0.001
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    Detect swing highs and lows using a rolling window approach.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Number of candles on each side to confirm swing
        min_swing_size: Minimum percentage difference for valid swing
        
    Returns:
        Tuple of (swing_highs, swing_lows)
    """
    swing_highs = []
    swing_lows = []
    
    high = df['High'].values
    low = df['Low'].values
    dates = df['Date'].values
    
    for i in range(lookback, len(df) - lookback):
        # Check for swing high
        is_swing_high = True
        for j in range(1, lookback + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_swing_high = False
                break
        
        if is_swing_high:
            # Verify minimum swing size
            local_low = min(low[i - lookback:i + lookback + 1])
            swing_size = (high[i] - local_low) / local_low
            if swing_size >= min_swing_size:
                swing_highs.append(SwingPoint(
                    index=i,
                    date=pd.Timestamp(dates[i]),
                    price=high[i],
                    type='high'
                ))
        
        # Check for swing low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_swing_low = False
                break
        
        if is_swing_low:
            # Verify minimum swing size
            local_high = max(high[i - lookback:i + lookback + 1])
            swing_size = (local_high - low[i]) / low[i]
            if swing_size >= min_swing_size:
                swing_lows.append(SwingPoint(
                    index=i,
                    date=pd.Timestamp(dates[i]),
                    price=low[i],
                    type='low'
                ))
    
    return swing_highs, swing_lows


def get_swing_structure(
    df: pd.DataFrame,
    lookback: int = 5
) -> pd.DataFrame:
    """
    Add swing point columns to DataFrame for visualization.
    
    Returns:
        DataFrame with 'swing_high', 'swing_low' columns (NaN where no swing)
    """
    df = df.copy()
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    
    swing_highs, swing_lows = detect_swing_points(df, lookback)
    
    for sh in swing_highs:
        df.loc[sh.index, 'swing_high'] = sh.price
    
    for sl in swing_lows:
        df.loc[sl.index, 'swing_low'] = sl.price
    
    return df


# ============== BOS / CHoCH DETECTION ==============

def detect_structure_breaks(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint]
) -> list[StructureBreak]:
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).
    
    BOS: Price breaks structure in the direction of the trend
    CHoCH: Price breaks structure against the trend (reversal signal)
    
    Args:
        df: DataFrame with OHLC data
        swing_highs: List of swing high points
        swing_lows: List of swing low points
        
    Returns:
        List of StructureBreak events
    """
    breaks = []
    
    if not swing_highs or not swing_lows:
        return breaks
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    dates = df['Date'].values
    
    # Combine and sort swings by index
    all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)
    
    # Track current trend (based on swing sequence)
    current_trend = None
    last_high = None
    last_low = None
    
    for i, swing in enumerate(all_swings):
        if swing.type == 'high':
            if last_high is not None:
                # Higher high = bullish structure
                if swing.price > last_high.price:
                    current_trend = 'bullish'
                elif swing.price < last_high.price:
                    # Lower high in bullish trend could signal CHoCH
                    pass
            last_high = swing
        else:  # swing.type == 'low'
            if last_low is not None:
                # Lower low = bearish structure
                if swing.price < last_low.price:
                    current_trend = 'bearish'
                elif swing.price > last_low.price:
                    # Higher low in bearish trend could signal CHoCH
                    pass
            last_low = swing
    
    # Now detect actual breaks
    last_significant_high = None
    last_significant_low = None
    
    for i in range(len(df)):
        current_close = close[i]
        
        # Update significant swing levels
        for sh in swing_highs:
            if sh.index < i:
                if last_significant_high is None or sh.price > last_significant_high.price:
                    last_significant_high = sh
        
        for sl in swing_lows:
            if sl.index < i:
                if last_significant_low is None or sl.price < last_significant_low.price:
                    last_significant_low = sl
        
        # Check for breaks
        if last_significant_high and high[i] > last_significant_high.price:
            # Broke above swing high
            # Determine if BOS or CHoCH based on recent trend
            recent_closes = close[max(0, i-20):i]
            recent_trend = 'bullish' if len(recent_closes) > 1 and recent_closes[-1] > recent_closes[0] else 'bearish'
            
            break_type = 'BOS' if recent_trend == 'bullish' else 'CHoCH'
            
            breaks.append(StructureBreak(
                index=i,
                date=pd.Timestamp(dates[i]),
                price=high[i],
                break_type=break_type,
                direction='bullish',
                swing_broken=last_significant_high
            ))
            
            # Reset after break
            last_significant_high = None
        
        if last_significant_low and low[i] < last_significant_low.price:
            # Broke below swing low
            recent_closes = close[max(0, i-20):i]
            recent_trend = 'bearish' if len(recent_closes) > 1 and recent_closes[-1] < recent_closes[0] else 'bullish'
            
            break_type = 'BOS' if recent_trend == 'bearish' else 'CHoCH'
            
            breaks.append(StructureBreak(
                index=i,
                date=pd.Timestamp(dates[i]),
                price=low[i],
                break_type=break_type,
                direction='bearish',
                swing_broken=last_significant_low
            ))
            
            # Reset after break
            last_significant_low = None
    
    return breaks


# ============== ORDER BLOCKS ==============

def detect_order_blocks(
    df: pd.DataFrame,
    displacement_threshold: float = 0.015,
    lookback_candles: int = 3
) -> list[OrderBlock]:
    """
    Detect Order Blocks (OB) - the last opposite candle before a strong move.
    
    An Order Block is formed when:
    1. There's a strong displacement move (large candle)
    2. The last opposite-colored candle before the move is the OB
    
    Args:
        df: DataFrame with OHLC data
        displacement_threshold: Minimum move % to qualify as displacement
        lookback_candles: How many candles back to look for OB
        
    Returns:
        List of OrderBlock objects
    """
    order_blocks = []
    
    open_prices = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    dates = df['Date'].values
    
    for i in range(lookback_candles + 1, len(df)):
        current_move = (close[i] - open_prices[i]) / open_prices[i]
        
        # Check for bullish displacement (strong green candle)
        if current_move > displacement_threshold:
            # Look for last bearish candle before this move
            for j in range(i - 1, max(0, i - lookback_candles - 1), -1):
                if close[j] < open_prices[j]:  # Bearish candle
                    strength = abs(current_move) / displacement_threshold
                    order_blocks.append(OrderBlock(
                        start_index=j,
                        end_index=j,
                        start_date=pd.Timestamp(dates[j]),
                        end_date=pd.Timestamp(dates[j]),
                        top=high[j],
                        bottom=low[j],
                        ob_type='bullish',
                        strength=min(strength, 3.0) / 3.0,
                        tested=False,
                        invalidated=False
                    ))
                    break
        
        # Check for bearish displacement (strong red candle)
        elif current_move < -displacement_threshold:
            # Look for last bullish candle before this move
            for j in range(i - 1, max(0, i - lookback_candles - 1), -1):
                if close[j] > open_prices[j]:  # Bullish candle
                    strength = abs(current_move) / displacement_threshold
                    order_blocks.append(OrderBlock(
                        start_index=j,
                        end_index=j,
                        start_date=pd.Timestamp(dates[j]),
                        end_date=pd.Timestamp(dates[j]),
                        top=high[j],
                        bottom=low[j],
                        ob_type='bearish',
                        strength=min(strength, 3.0) / 3.0,
                        tested=False,
                        invalidated=False
                    ))
                    break
    
    # Check for tested/invalidated OBs
    for ob in order_blocks:
        for i in range(ob.end_index + 1, len(df)):
            if ob.ob_type == 'bullish':
                # Price returning to bullish OB
                if low[i] <= ob.top:
                    ob.tested = True
                if low[i] < ob.bottom:
                    ob.invalidated = True
                    break
            else:
                # Price returning to bearish OB
                if high[i] >= ob.bottom:
                    ob.tested = True
                if high[i] > ob.top:
                    ob.invalidated = True
                    break
    
    # Return only valid (non-invalidated) OBs
    return [ob for ob in order_blocks if not ob.invalidated]


# ============== FAIR VALUE GAPS ==============

def detect_fair_value_gaps(
    df: pd.DataFrame,
    min_gap_size: float = 0.001
) -> list[FairValueGap]:
    """
    Detect Fair Value Gaps (FVG) - 3-candle imbalances.
    
    Bullish FVG: Gap between Candle 1 High and Candle 3 Low
    Bearish FVG: Gap between Candle 1 Low and Candle 3 High
    
    Args:
        df: DataFrame with OHLC data
        min_gap_size: Minimum gap size as fraction of price
        
    Returns:
        List of FairValueGap objects
    """
    fvgs = []
    
    high = df['High'].values
    low = df['Low'].values
    dates = df['Date'].values
    
    for i in range(2, len(df)):
        candle1_high = high[i - 2]
        candle1_low = low[i - 2]
        candle3_high = high[i]
        candle3_low = low[i]
        
        # Bullish FVG: Candle 3 Low > Candle 1 High (gap up)
        if candle3_low > candle1_high:
            gap_size = (candle3_low - candle1_high) / candle1_high
            if gap_size >= min_gap_size:
                fvgs.append(FairValueGap(
                    index=i - 1,  # Middle candle
                    date=pd.Timestamp(dates[i - 1]),
                    top=candle3_low,
                    bottom=candle1_high,
                    fvg_type='bullish',
                    filled=False,
                    fill_percentage=0.0
                ))
        
        # Bearish FVG: Candle 1 Low > Candle 3 High (gap down)
        if candle1_low > candle3_high:
            gap_size = (candle1_low - candle3_high) / candle3_high
            if gap_size >= min_gap_size:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    date=pd.Timestamp(dates[i - 1]),
                    top=candle1_low,
                    bottom=candle3_high,
                    fvg_type='bearish',
                    filled=False,
                    fill_percentage=0.0
                ))
    
    # Check for filled FVGs
    for fvg in fvgs:
        gap_height = fvg.top - fvg.bottom
        
        for i in range(fvg.index + 2, len(df)):
            if fvg.fvg_type == 'bullish':
                # Price filling down into the gap
                if low[i] <= fvg.top:
                    fill_depth = fvg.top - min(low[i], fvg.bottom)
                    fvg.fill_percentage = min(fill_depth / gap_height, 1.0)
                    if low[i] <= fvg.bottom:
                        fvg.filled = True
                        break
            else:
                # Price filling up into the gap
                if high[i] >= fvg.bottom:
                    fill_depth = min(high[i], fvg.top) - fvg.bottom
                    fvg.fill_percentage = min(fill_depth / gap_height, 1.0)
                    if high[i] >= fvg.top:
                        fvg.filled = True
                        break
    
    return fvgs


# ============== LIQUIDITY POOLS ==============

def detect_liquidity_pools(
    df: pd.DataFrame,
    price_tolerance: float = 0.002,
    min_touches: int = 2
) -> list[LiquidityPool]:
    """
    Detect liquidity pools (equal highs/lows where stop losses cluster).
    
    Args:
        df: DataFrame with OHLC data
        price_tolerance: How close prices need to be to count as "equal"
        min_touches: Minimum touches to qualify as liquidity pool
        
    Returns:
        List of LiquidityPool objects
    """
    pools = []
    
    high = df['High'].values
    low = df['Low'].values
    
    # Detect equal highs
    for i in range(len(df)):
        price_level = high[i]
        touches = []
        
        for j in range(len(df)):
            if abs(high[j] - price_level) / price_level <= price_tolerance:
                touches.append(j)
        
        if len(touches) >= min_touches:
            # Check if this pool is already recorded
            already_exists = False
            for pool in pools:
                if pool.pool_type == 'equal_highs':
                    if abs(pool.price_level - price_level) / price_level <= price_tolerance:
                        already_exists = True
                        break
            
            if not already_exists:
                # Check if swept
                swept = False
                end_idx = max(touches)
                for k in range(end_idx + 1, len(df)):
                    if high[k] > price_level * (1 + price_tolerance):
                        swept = True
                        break
                
                pools.append(LiquidityPool(
                    start_index=min(touches),
                    end_index=max(touches),
                    price_level=price_level,
                    pool_type='equal_highs',
                    strength=len(touches),
                    swept=swept
                ))
    
    # Detect equal lows
    for i in range(len(df)):
        price_level = low[i]
        touches = []
        
        for j in range(len(df)):
            if abs(low[j] - price_level) / price_level <= price_tolerance:
                touches.append(j)
        
        if len(touches) >= min_touches:
            already_exists = False
            for pool in pools:
                if pool.pool_type == 'equal_lows':
                    if abs(pool.price_level - price_level) / price_level <= price_tolerance:
                        already_exists = True
                        break
            
            if not already_exists:
                swept = False
                end_idx = max(touches)
                for k in range(end_idx + 1, len(df)):
                    if low[k] < price_level * (1 - price_tolerance):
                        swept = True
                        break
                
                pools.append(LiquidityPool(
                    start_index=min(touches),
                    end_index=max(touches),
                    price_level=price_level,
                    pool_type='equal_lows',
                    strength=len(touches),
                    swept=swept
                ))
    
    return pools


# ============== COMPLETE STRUCTURE ANALYSIS ==============

def analyze_market_structure(
    df: pd.DataFrame,
    swing_lookback: int = 5,
    displacement_threshold: float = 0.015,
    fvg_min_size: float = 0.001,
    liquidity_tolerance: float = 0.002
) -> dict:
    """
    Perform complete market structure analysis.
    
    Args:
        df: DataFrame with OHLC data
        swing_lookback: Lookback period for swing detection
        displacement_threshold: Minimum move % for order block detection
        fvg_min_size: Minimum gap size for FVG detection
        liquidity_tolerance: Price tolerance for liquidity pool detection
        
    Returns:
        Dictionary with all structure elements
    """
    # Detect swings
    swing_highs, swing_lows = detect_swing_points(df, swing_lookback)
    
    # Detect structure breaks
    structure_breaks = detect_structure_breaks(df, swing_highs, swing_lows)
    
    # Filter to recent breaks only
    bos_events = [b for b in structure_breaks if b.break_type == 'BOS'][-10:]
    choch_events = [b for b in structure_breaks if b.break_type == 'CHoCH'][-5:]
    
    # Detect order blocks
    order_blocks = detect_order_blocks(df, displacement_threshold)[-15:]
    
    # Detect FVGs
    fvgs = detect_fair_value_gaps(df, fvg_min_size)
    unfilled_fvgs = [f for f in fvgs if not f.filled][-10:]
    
    # Detect liquidity pools
    liquidity_pools = detect_liquidity_pools(df, liquidity_tolerance)
    active_pools = [p for p in liquidity_pools if not p.swept][-10:]
    
    # Calculate structure bias
    recent_bos = [b for b in bos_events if b.index > len(df) - 50]
    bullish_bos = len([b for b in recent_bos if b.direction == 'bullish'])
    bearish_bos = len([b for b in recent_bos if b.direction == 'bearish'])
    
    if bullish_bos > bearish_bos:
        structure_bias = 'Bullish'
    elif bearish_bos > bullish_bos:
        structure_bias = 'Bearish'
    else:
        structure_bias = 'Neutral'
    
    # Recent CHoCH signals
    recent_choch = choch_events[-1] if choch_events else None
    reversal_signal = None
    if recent_choch and recent_choch.index > len(df) - 20:
        reversal_signal = f"{recent_choch.direction.title()} reversal detected"
    
    return {
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'bos_events': bos_events,
        'choch_events': choch_events,
        'order_blocks': order_blocks,
        'unfilled_fvgs': unfilled_fvgs,
        'liquidity_pools': active_pools,
        'structure_bias': structure_bias,
        'bullish_bos_count': bullish_bos,
        'bearish_bos_count': bearish_bos,
        'reversal_signal': reversal_signal,
        'total_valid_obs': len(order_blocks),
        'total_unfilled_fvgs': len(unfilled_fvgs),
        'total_active_pools': len(active_pools)
    }


def get_structure_summary(analysis: dict) -> dict:
    """
    Generate a human-readable summary of the structure analysis.
    
    Args:
        analysis: Output from analyze_market_structure()
        
    Returns:
        Summary dictionary with key insights
    """
    return {
        'bias': analysis['structure_bias'],
        'bias_strength': abs(analysis['bullish_bos_count'] - analysis['bearish_bos_count']),
        'reversal_warning': analysis['reversal_signal'],
        'key_zones': {
            'order_blocks': analysis['total_valid_obs'],
            'fair_value_gaps': analysis['total_unfilled_fvgs'],
            'liquidity_pools': analysis['total_active_pools']
        },
        'last_bos': analysis['bos_events'][-1] if analysis['bos_events'] else None,
        'last_choch': analysis['choch_events'][-1] if analysis['choch_events'] else None
    }
