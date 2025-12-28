# analytics.py
# Technical analytics and prediction rationale generation

import pandas as pd
import numpy as np
from typing import Optional


# ============== TECHNICAL ANALYTICS ==============

def get_trend_direction(df: pd.DataFrame, short_period: int = 10, long_period: int = 30) -> dict:
    """
    Determine trend direction using MA crossover and price position.
    
    Args:
        df: DataFrame with 'Close' column
        short_period: Short MA period
        long_period: Long MA period
        
    Returns:
        Dict with trend info
    """
    if len(df) < long_period:
        return {'direction': 'Unknown', 'strength': 'Insufficient Data', 'score': 0}
    
    close = df['Close']
    
    # Moving averages
    ma_short = close.rolling(window=short_period).mean()
    ma_long = close.rolling(window=long_period).mean()
    
    current_price = close.iloc[-1]
    current_ma_short = ma_short.iloc[-1]
    current_ma_long = ma_long.iloc[-1]
    
    # Trend score: -2 to +2
    score = 0
    
    # Price vs short MA
    if current_price > current_ma_short:
        score += 1
    else:
        score -= 1
    
    # Short MA vs long MA
    if current_ma_short > current_ma_long:
        score += 1
    else:
        score -= 1
    
    # Direction
    if score >= 1:
        direction = 'Bullish'
    elif score <= -1:
        direction = 'Bearish'
    else:
        direction = 'Neutral'
    
    # Strength based on separation
    ma_separation = abs(current_ma_short - current_ma_long) / current_ma_long * 100
    if ma_separation > 3:
        strength = 'Strong'
    elif ma_separation > 1:
        strength = 'Moderate'
    else:
        strength = 'Weak'
    
    return {
        'direction': direction,
        'strength': strength,
        'score': score,
        'price_vs_ma10': round((current_price / current_ma_short - 1) * 100, 2),
        'ma10_vs_ma30': round((current_ma_short / current_ma_long - 1) * 100, 2)
    }


def get_rsi_state(df: pd.DataFrame, period: int = 14) -> dict:
    """
    Calculate RSI and classify state.
    
    States:
    - Overbought: RSI > 70
    - Oversold: RSI < 30
    - Neutral: 30-70
    
    Args:
        df: DataFrame with 'Close' column
        period: RSI period
        
    Returns:
        Dict with RSI info
    """
    if len(df) < period + 1:
        return {'value': None, 'state': 'Insufficient Data', 'signal': None}
    
    close = df['Close']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[-1]
    
    # State classification
    if current_rsi > 70:
        state = 'Overbought'
        signal = 'Potential reversal down'
    elif current_rsi < 30:
        state = 'Oversold'
        signal = 'Potential reversal up'
    elif current_rsi > 60:
        state = 'Bullish'
        signal = 'Momentum favors upside'
    elif current_rsi < 40:
        state = 'Bearish'
        signal = 'Momentum favors downside'
    else:
        state = 'Neutral'
        signal = 'No clear momentum signal'
    
    return {
        'value': round(float(current_rsi), 2),
        'state': state,
        'signal': signal
    }


def get_ma_alignment(df: pd.DataFrame) -> dict:
    """
    Check moving average alignment (5, 10, 20, 50 day).
    
    Perfect bullish: Price > MA5 > MA10 > MA20
    Perfect bearish: Price < MA5 < MA10 < MA20
    
    Args:
        df: DataFrame with 'Close' column
        
    Returns:
        Dict with MA alignment info
    """
    if len(df) < 50:
        periods = [5, 10, 20]
    else:
        periods = [5, 10, 20, 50]
    
    if len(df) < max(periods):
        return {'alignment': 'Unknown', 'bullish_count': 0, 'bearish_count': 0}
    
    close = df['Close']
    current_price = close.iloc[-1]
    
    # Calculate MAs
    mas = {}
    for p in periods:
        if len(df) >= p:
            mas[f'ma_{p}'] = close.rolling(window=p).mean().iloc[-1]
    
    # Count alignments
    values = [current_price] + [mas[f'ma_{p}'] for p in periods if f'ma_{p}' in mas]
    
    bullish_count = 0
    bearish_count = 0
    
    for i in range(len(values) - 1):
        if values[i] > values[i + 1]:
            bullish_count += 1
        elif values[i] < values[i + 1]:
            bearish_count += 1
    
    total = len(values) - 1
    
    # Alignment classification
    if bullish_count == total:
        alignment = 'Perfect Bullish'
    elif bearish_count == total:
        alignment = 'Perfect Bearish'
    elif bullish_count > bearish_count:
        alignment = 'Mostly Bullish'
    elif bearish_count > bullish_count:
        alignment = 'Mostly Bearish'
    else:
        alignment = 'Mixed'
    
    return {
        'alignment': alignment,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'total_comparisons': total,
        'mas': {k: round(v, 2) for k, v in mas.items()}
    }


def get_volatility_state(df: pd.DataFrame, period: int = 20) -> dict:
    """
    Analyze current volatility state.
    
    Args:
        df: DataFrame with 'Close' column
        period: Lookback period
        
    Returns:
        Dict with volatility info
    """
    if len(df) < period * 2:
        return {'state': 'Insufficient Data', 'current': None, 'average': None}
    
    returns = df['Close'].pct_change()
    
    # Current volatility (annualized)
    current_vol = returns.tail(period).std() * np.sqrt(252) * 100
    
    # Historical average volatility
    avg_vol = returns.rolling(window=period).std().mean() * np.sqrt(252) * 100
    
    # Ratio
    vol_ratio = current_vol / (avg_vol + 1e-10)
    
    # State classification
    if vol_ratio > 1.5:
        state = 'High'
        description = 'Volatility significantly above average'
    elif vol_ratio > 1.2:
        state = 'Elevated'
        description = 'Volatility above average'
    elif vol_ratio < 0.7:
        state = 'Low'
        description = 'Volatility below average'
    else:
        state = 'Normal'
        description = 'Volatility near average'
    
    return {
        'state': state,
        'description': description,
        'current_annual_pct': round(float(current_vol), 2),
        'average_annual_pct': round(float(avg_vol), 2),
        'ratio': round(float(vol_ratio), 2)
    }


def get_technical_summary(df: pd.DataFrame) -> dict:
    """
    Get complete technical analytics summary.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        Dict with all technical indicators
    """
    return {
        'trend': get_trend_direction(df),
        'rsi': get_rsi_state(df),
        'ma_alignment': get_ma_alignment(df),
        'volatility': get_volatility_state(df)
    }


# ============== PREDICTION RATIONALE ==============

def analyze_feature_contributions(
    df: pd.DataFrame,
    feature_cols: list[str] = None
) -> dict:
    """
    Analyze feature values to identify prediction drivers.
    Uses simple rule-based analysis (no SHAP for speed).
    
    Args:
        df: DataFrame with features
        feature_cols: Feature column names
        
    Returns:
        Dict with feature analysis
    """
    if feature_cols is None:
        feature_cols = ['returns', 'ma_5', 'ma_10', 'ma_20', 'volatility', 'rsi']
    
    # Get latest values
    latest = df[feature_cols].iloc[-1]
    close = df['Close'].iloc[-1]
    
    contributions = {}
    
    # Returns analysis
    if 'returns' in feature_cols:
        ret = latest['returns']
        if ret > 0.02:
            contributions['returns'] = {'direction': 'positive', 'strength': 'strong', 'value': ret}
        elif ret > 0.005:
            contributions['returns'] = {'direction': 'positive', 'strength': 'moderate', 'value': ret}
        elif ret < -0.02:
            contributions['returns'] = {'direction': 'negative', 'strength': 'strong', 'value': ret}
        elif ret < -0.005:
            contributions['returns'] = {'direction': 'negative', 'strength': 'moderate', 'value': ret}
        else:
            contributions['returns'] = {'direction': 'neutral', 'strength': 'weak', 'value': ret}
    
    # MA analysis (price position relative to MAs)
    for ma_col in ['ma_5', 'ma_10', 'ma_20']:
        if ma_col in feature_cols:
            ma_val = latest[ma_col]
            diff_pct = (close - ma_val) / ma_val * 100
            
            if diff_pct > 3:
                contributions[ma_col] = {'direction': 'positive', 'strength': 'strong', 'diff_pct': diff_pct}
            elif diff_pct > 1:
                contributions[ma_col] = {'direction': 'positive', 'strength': 'moderate', 'diff_pct': diff_pct}
            elif diff_pct < -3:
                contributions[ma_col] = {'direction': 'negative', 'strength': 'strong', 'diff_pct': diff_pct}
            elif diff_pct < -1:
                contributions[ma_col] = {'direction': 'negative', 'strength': 'moderate', 'diff_pct': diff_pct}
            else:
                contributions[ma_col] = {'direction': 'neutral', 'strength': 'weak', 'diff_pct': diff_pct}
    
    # Volatility analysis
    if 'volatility' in feature_cols:
        vol = latest['volatility']
        avg_vol = df['volatility'].mean()
        vol_ratio = vol / (avg_vol + 1e-10)
        
        if vol_ratio > 1.5:
            contributions['volatility'] = {'direction': 'expanding', 'strength': 'strong', 'ratio': vol_ratio}
        elif vol_ratio > 1.2:
            contributions['volatility'] = {'direction': 'expanding', 'strength': 'moderate', 'ratio': vol_ratio}
        elif vol_ratio < 0.7:
            contributions['volatility'] = {'direction': 'contracting', 'strength': 'strong', 'ratio': vol_ratio}
        else:
            contributions['volatility'] = {'direction': 'stable', 'strength': 'normal', 'ratio': vol_ratio}
    
    # RSI analysis
    if 'rsi' in feature_cols:
        rsi = latest['rsi']
        if rsi > 70:
            contributions['rsi'] = {'direction': 'negative', 'strength': 'strong', 'value': rsi, 'state': 'overbought'}
        elif rsi > 60:
            contributions['rsi'] = {'direction': 'positive', 'strength': 'moderate', 'value': rsi, 'state': 'bullish'}
        elif rsi < 30:
            contributions['rsi'] = {'direction': 'positive', 'strength': 'strong', 'value': rsi, 'state': 'oversold'}
        elif rsi < 40:
            contributions['rsi'] = {'direction': 'negative', 'strength': 'moderate', 'value': rsi, 'state': 'bearish'}
        else:
            contributions['rsi'] = {'direction': 'neutral', 'strength': 'weak', 'value': rsi, 'state': 'neutral'}
    
    return contributions


def get_top_drivers(contributions: dict) -> tuple[list, list]:
    """
    Identify top positive and negative drivers from contributions.
    
    Args:
        contributions: Dict from analyze_feature_contributions
        
    Returns:
        Tuple of (positive_drivers, negative_drivers)
    """
    positive = []
    negative = []
    
    strength_order = {'strong': 3, 'moderate': 2, 'weak': 1, 'normal': 1}
    
    for feature, info in contributions.items():
        direction = info.get('direction', 'neutral')
        strength = info.get('strength', 'weak')
        score = strength_order.get(strength, 1)
        
        if direction == 'positive':
            positive.append((feature, score, info))
        elif direction == 'negative':
            negative.append((feature, score, info))
        elif direction == 'expanding' and feature == 'volatility':
            # Expanding volatility is typically negative for prediction confidence
            negative.append((feature, score, info))
    
    # Sort by strength score
    positive.sort(key=lambda x: x[1], reverse=True)
    negative.sort(key=lambda x: x[1], reverse=True)
    
    return positive, negative


def generate_rationale(
    df: pd.DataFrame,
    feature_cols: list[str] = None,
    max_points: int = 5
) -> dict:
    """
    Generate natural-language prediction rationale.
    
    Args:
        df: DataFrame with features
        feature_cols: Feature column names
        max_points: Maximum number of bullet points
        
    Returns:
        Dict with rationale info and bullet points
    """
    if feature_cols is None:
        feature_cols = ['returns', 'ma_5', 'ma_10', 'ma_20', 'volatility', 'rsi']
    
    # Get feature contributions
    contributions = analyze_feature_contributions(df, feature_cols)
    positive_drivers, negative_drivers = get_top_drivers(contributions)
    
    # Generate bullet points
    bullets = []
    
    # Feature to description mapping
    descriptions = {
        'returns': {
            'positive_strong': 'Strong recent momentum (positive returns)',
            'positive_moderate': 'Moderate upward momentum',
            'negative_strong': 'Strong negative momentum (recent decline)',
            'negative_moderate': 'Moderate downward pressure'
        },
        'ma_5': {
            'positive_strong': 'Price well above 5-day average',
            'positive_moderate': 'Price above short-term average',
            'negative_strong': 'Price well below 5-day average',
            'negative_moderate': 'Price below short-term average'
        },
        'ma_10': {
            'positive_strong': 'Price significantly above 10-day average',
            'positive_moderate': 'Price above 10-day average',
            'negative_strong': 'Price significantly below 10-day average',
            'negative_moderate': 'Price below 10-day average'
        },
        'ma_20': {
            'positive_strong': 'Price well above key 20-day average',
            'positive_moderate': 'Price above 20-day trend line',
            'negative_strong': 'Price well below 20-day average',
            'negative_moderate': 'Price below 20-day trend line'
        },
        'volatility': {
            'expanding_strong': 'Volatility expanding significantly',
            'expanding_moderate': 'Volatility increasing',
            'contracting_strong': 'Volatility contracting (consolidation)',
            'stable': 'Volatility stable'
        },
        'rsi': {
            'positive_strong': 'RSI oversold - potential bounce',
            'positive_moderate': 'RSI showing bullish momentum',
            'negative_strong': 'RSI overbought - potential pullback',
            'negative_moderate': 'RSI showing bearish momentum'
        }
    }
    
    # Add positive factors
    for feature, score, info in positive_drivers[:max_points // 2 + 1]:
        direction = info.get('direction', 'positive')
        strength = info.get('strength', 'moderate')
        key = f"{direction}_{strength}"
        
        if feature in descriptions and key in descriptions[feature]:
            bullets.append({
                'type': 'positive',
                'text': descriptions[feature][key],
                'feature': feature
            })
    
    # Add negative factors
    for feature, score, info in negative_drivers[:max_points // 2 + 1]:
        direction = info.get('direction', 'negative')
        strength = info.get('strength', 'moderate')
        key = f"{direction}_{strength}"
        
        if feature in descriptions and key in descriptions[feature]:
            bullets.append({
                'type': 'negative',
                'text': descriptions[feature][key],
                'feature': feature
            })
    
    # Determine overall bias
    pos_score = sum(d[1] for d in positive_drivers)
    neg_score = sum(d[1] for d in negative_drivers)
    
    if pos_score > neg_score + 2:
        overall_bias = 'Bullish'
    elif neg_score > pos_score + 2:
        overall_bias = 'Bearish'
    else:
        overall_bias = 'Neutral'
    
    return {
        'overall_bias': overall_bias,
        'positive_score': pos_score,
        'negative_score': neg_score,
        'bullets': bullets[:max_points],
        'contributions': contributions
    }


def get_prediction_explanation(
    df: pd.DataFrame,
    prediction: float,
    feature_cols: list[str] = None
) -> str:
    """
    Generate a single-paragraph explanation for a prediction.
    
    Args:
        df: DataFrame with features
        prediction: Predicted price value
        feature_cols: Feature column names
        
    Returns:
        Natural language explanation string
    """
    rationale = generate_rationale(df, feature_cols)
    current_price = df['Close'].iloc[-1]
    change_pct = (prediction - current_price) / current_price * 100
    
    # Direction
    if change_pct > 0:
        direction = "upward"
        movement = f"+{change_pct:.1f}%"
    else:
        direction = "downward"
        movement = f"{change_pct:.1f}%"
    
    # Build explanation
    explanation_parts = [f"The model predicts a {direction} movement ({movement})."]
    
    # Add key drivers
    positive_bullets = [b['text'] for b in rationale['bullets'] if b['type'] == 'positive']
    negative_bullets = [b['text'] for b in rationale['bullets'] if b['type'] == 'negative']
    
    if positive_bullets:
        explanation_parts.append(f"Bullish factors: {'; '.join(positive_bullets[:2])}.")
    
    if negative_bullets:
        explanation_parts.append(f"Bearish factors: {'; '.join(negative_bullets[:2])}.")
    
    # Overall assessment
    bias = rationale['overall_bias']
    if bias == 'Bullish':
        explanation_parts.append("Overall technical signals favor upside.")
    elif bias == 'Bearish':
        explanation_parts.append("Overall technical signals favor downside.")
    else:
        explanation_parts.append("Technical signals are mixed.")
    
    return " ".join(explanation_parts)


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.data_loader import fetch_stock_data
    from utils.features import create_features
    
    # Fetch sample data
    df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2024-12-31")
    
    if not df.empty:
        df_features = create_features(df)
        
        # Technical summary
        print("=== Technical Summary ===")
        summary = get_technical_summary(df)
        print(f"Trend: {summary['trend']['direction']} ({summary['trend']['strength']})")
        print(f"RSI: {summary['rsi']['value']} - {summary['rsi']['state']}")
        print(f"MA Alignment: {summary['ma_alignment']['alignment']}")
        print(f"Volatility: {summary['volatility']['state']}")
        
        # Prediction rationale
        print("\n=== Prediction Rationale ===")
        rationale = generate_rationale(df_features)
        print(f"Overall Bias: {rationale['overall_bias']}")
        print(f"Positive Score: {rationale['positive_score']}, Negative Score: {rationale['negative_score']}")
        print("\nKey Points:")
        for bullet in rationale['bullets']:
            icon = "✅" if bullet['type'] == 'positive' else "⚠️"
            print(f"  {icon} {bullet['text']}")
        
        # Full explanation
        print("\n=== Prediction Explanation ===")
        fake_prediction = df['Close'].iloc[-1] * 1.02  # Simulate +2% prediction
        explanation = get_prediction_explanation(df_features, fake_prediction)
        print(explanation)
