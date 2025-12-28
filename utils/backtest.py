# backtest.py
# Historical backtesting for prediction models

import pandas as pd
import numpy as np
from typing import Optional, Union
from pathlib import Path


# ============== CORE BACKTESTING ==============

def rolling_backtest(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    window_size: int = 100,
    step_size: int = 1,
    horizon: int = 1
) -> dict:
    """
    Run rolling window backtest without retraining.
    Uses pre-trained model to predict at each step.
    
    Args:
        df: DataFrame with features and 'Close' column
        model: Pre-trained model with .predict() method
        feature_cols: Feature column names
        window_size: Minimum lookback window for features
        step_size: Steps between predictions (1 = every day)
        horizon: Prediction horizon in days
        
    Returns:
        Dict with predictions, actuals, and metrics
    """
    if len(df) < window_size + horizon:
        return {'error': 'Insufficient data for backtest'}
    
    predictions = []
    actuals = []
    dates = []
    
    # Start after window_size to ensure features are valid
    start_idx = window_size
    end_idx = len(df) - horizon
    
    for i in range(start_idx, end_idx, step_size):
        # Get features at current point
        X = df[feature_cols].iloc[i:i+1].values
        
        # Predict
        pred = model.predict(X)[0]
        
        # Actual value at horizon
        actual = df['Close'].iloc[i + horizon]
        
        predictions.append(pred)
        actuals.append(actual)
        dates.append(df['Date'].iloc[i])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actuals)
    
    return {
        'dates': dates,
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'horizon': horizon,
        'n_predictions': len(predictions)
    }


def multi_horizon_backtest(
    df: pd.DataFrame,
    models: dict,
    feature_cols: list[str],
    window_size: int = 100,
    step_size: int = 5
) -> dict:
    """
    Run backtest for multiple prediction horizons.
    
    Args:
        df: DataFrame with features and 'Close' column
        models: Dict of models keyed by horizon name (e.g., {'1d': model1, '5d': model5})
        feature_cols: Feature column names
        window_size: Minimum lookback window
        step_size: Steps between predictions
        
    Returns:
        Dict with results per horizon
    """
    horizon_days = {'1d': 1, '5d': 5, '20d': 20}
    results = {}
    
    for horizon_name, model in models.items():
        horizon = horizon_days.get(horizon_name, 1)
        results[horizon_name] = rolling_backtest(
            df, model, feature_cols,
            window_size=window_size,
            step_size=step_size,
            horizon=horizon
        )
    
    return results


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Calculate prediction performance metrics.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Dict with MAE, RMSE, MAPE, directional accuracy, max error
    """
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    pct_errors = abs_errors / (actuals + 1e-10) * 100
    
    # Direction accuracy (did we predict up/down correctly?)
    pred_direction = np.sign(np.diff(np.concatenate([[actuals[0]], predictions])))
    actual_direction = np.sign(np.diff(np.concatenate([[actuals[0]], actuals])))
    direction_accuracy = np.mean(pred_direction == actual_direction) * 100
    
    return {
        'mae': float(np.mean(abs_errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mape': float(np.mean(pct_errors)),
        'max_error': float(np.max(abs_errors)),
        'direction_accuracy': float(direction_accuracy),
        'mean_error': float(np.mean(errors)),  # Bias indicator
        'std_error': float(np.std(errors))
    }


# ============== DRAWDOWN ANALYSIS ==============

def calculate_drawdown(equity_curve: np.ndarray) -> tuple[np.ndarray, float, int]:
    """
    Calculate drawdown series and max drawdown.
    
    Args:
        equity_curve: Array of cumulative returns/equity values
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_dd_duration)
    """
    # Running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Drawdown at each point
    drawdown = (equity_curve - running_max) / (running_max + 1e-10) * 100
    
    # Max drawdown
    max_drawdown = float(np.min(drawdown))
    
    # Max drawdown duration (consecutive days in drawdown)
    in_drawdown = drawdown < 0
    dd_duration = 0
    max_dd_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0
    
    return drawdown, max_drawdown, max_dd_duration


def simulate_trading_equity(
    predictions: np.ndarray,
    actuals: np.ndarray,
    initial_capital: float = 100000.0,
    position_size: float = 1.0
) -> np.ndarray:
    """
    Simulate simple trading strategy equity curve.
    Strategy: Long if predicted up, flat if predicted down.
    
    Args:
        predictions: Predicted prices
        actuals: Actual prices
        initial_capital: Starting capital
        position_size: Fraction of capital to use (0-1)
        
    Returns:
        Equity curve array
    """
    equity = [initial_capital]
    
    for i in range(1, len(predictions)):
        prev_actual = actuals[i - 1]
        curr_actual = actuals[i]
        prev_pred = predictions[i - 1] if i > 0 else prev_actual
        
        # Signal: long if we predicted price would go up
        if predictions[i] > prev_actual:
            # Long position - gain/lose based on actual move
            pct_change = (curr_actual - prev_actual) / prev_actual
            new_equity = equity[-1] * (1 + pct_change * position_size)
        else:
            # Flat - no change
            new_equity = equity[-1]
        
        equity.append(new_equity)
    
    return np.array(equity)


# ============== VISUALIZATION-READY OUTPUTS ==============

def generate_backtest_df(
    backtest_result: dict,
    initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Generate DataFrame suitable for Plotly visualization.
    
    Args:
        backtest_result: Output from rolling_backtest()
        initial_capital: Starting capital for equity simulation
        
    Returns:
        DataFrame with columns for plotting
    """
    if 'error' in backtest_result:
        return pd.DataFrame()
    
    dates = backtest_result['dates']
    predictions = backtest_result['predictions']
    actuals = backtest_result['actuals']
    
    # Calculate error series
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    pct_errors = abs_errors / (actuals + 1e-10) * 100
    cumulative_error = np.cumsum(abs_errors)
    
    # Equity curve (simulated trading)
    equity = simulate_trading_equity(predictions, actuals, initial_capital)
    
    # Drawdown series
    drawdown, max_dd, max_dd_duration = calculate_drawdown(equity)
    
    # Buy & hold benchmark
    buy_hold = initial_capital * (actuals / actuals[0])
    buy_hold_dd, _, _ = calculate_drawdown(buy_hold)
    
    # Create DataFrame
    df_viz = pd.DataFrame({
        'Date': dates,
        'Predicted': predictions,
        'Actual': actuals,
        'Error': errors,
        'Abs_Error': abs_errors,
        'Pct_Error': pct_errors,
        'Cumulative_Error': cumulative_error,
        'Equity': equity,
        'Drawdown_Pct': drawdown,
        'Buy_Hold': buy_hold,
        'Buy_Hold_Drawdown': buy_hold_dd
    })
    
    # Add rolling metrics
    window = min(20, len(df_viz) // 5)
    if window > 1:
        df_viz['Rolling_MAE'] = df_viz['Abs_Error'].rolling(window=window).mean()
        df_viz['Rolling_Accuracy'] = (
            (np.sign(df_viz['Predicted'].diff()) == np.sign(df_viz['Actual'].diff()))
            .rolling(window=window).mean() * 100
        )
    
    return df_viz


def generate_multi_horizon_df(
    multi_results: dict,
    initial_capital: float = 100000.0
) -> dict[str, pd.DataFrame]:
    """
    Generate visualization DataFrames for all horizons.
    
    Args:
        multi_results: Output from multi_horizon_backtest()
        initial_capital: Starting capital
        
    Returns:
        Dict of DataFrames keyed by horizon name
    """
    dfs = {}
    for horizon_name, result in multi_results.items():
        dfs[horizon_name] = generate_backtest_df(result, initial_capital)
    return dfs


def get_backtest_summary(backtest_result: dict, initial_capital: float = 100000.0) -> dict:
    """
    Get comprehensive backtest summary with all key metrics.
    
    Args:
        backtest_result: Output from rolling_backtest()
        initial_capital: Starting capital
        
    Returns:
        Dict with complete summary
    """
    if 'error' in backtest_result:
        return {'error': backtest_result['error']}
    
    predictions = backtest_result['predictions']
    actuals = backtest_result['actuals']
    metrics = backtest_result['metrics']
    
    # Equity curve and drawdown
    equity = simulate_trading_equity(predictions, actuals, initial_capital)
    drawdown, max_dd, max_dd_duration = calculate_drawdown(equity)
    
    # Buy & hold comparison
    buy_hold_return = (actuals[-1] / actuals[0] - 1) * 100
    strategy_return = (equity[-1] / initial_capital - 1) * 100
    
    # Sharpe ratio approximation (assuming daily returns)
    daily_returns = np.diff(equity) / equity[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    
    return {
        'prediction_metrics': metrics,
        'trading_metrics': {
            'total_return_pct': float(strategy_return),
            'buy_hold_return_pct': float(buy_hold_return),
            'outperformance_pct': float(strategy_return - buy_hold_return),
            'max_drawdown_pct': float(max_dd),
            'max_drawdown_duration': int(max_dd_duration),
            'sharpe_ratio': float(sharpe),
            'final_equity': float(equity[-1]),
            'initial_capital': float(initial_capital)
        },
        'summary': {
            'n_predictions': backtest_result['n_predictions'],
            'horizon_days': backtest_result['horizon'],
            'profitable': strategy_return > 0,
            'beats_buy_hold': strategy_return > buy_hold_return
        }
    }


def compare_horizons(multi_results: dict, initial_capital: float = 100000.0) -> pd.DataFrame:
    """
    Create comparison DataFrame across all horizons.
    
    Args:
        multi_results: Output from multi_horizon_backtest()
        initial_capital: Starting capital
        
    Returns:
        DataFrame comparing metrics across horizons
    """
    rows = []
    
    for horizon_name, result in multi_results.items():
        if 'error' in result:
            continue
            
        summary = get_backtest_summary(result, initial_capital)
        
        rows.append({
            'Horizon': horizon_name,
            'MAE': summary['prediction_metrics']['mae'],
            'RMSE': summary['prediction_metrics']['rmse'],
            'MAPE (%)': summary['prediction_metrics']['mape'],
            'Direction Acc (%)': summary['prediction_metrics']['direction_accuracy'],
            'Return (%)': summary['trading_metrics']['total_return_pct'],
            'Max DD (%)': summary['trading_metrics']['max_drawdown_pct'],
            'Sharpe': summary['trading_metrics']['sharpe_ratio'],
            'Beats B&H': summary['summary']['beats_buy_hold']
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.data_loader import fetch_stock_data
    from utils.features import create_features, get_feature_columns
    from models.predictor import load_model, load_multi_horizon_models, MODELS_DIR
    
    # Load data
    df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2024-12-31")
    
    if not df.empty:
        df_features = create_features(df)
        feature_cols = get_feature_columns()
        
        # Try to load model
        try:
            model = load_model("reliance_model.joblib")
            
            print("=== Single Horizon Backtest ===")
            result = rolling_backtest(df_features, model, feature_cols, horizon=1)
            
            if 'error' not in result:
                print(f"Predictions: {result['n_predictions']}")
                print(f"Metrics: {result['metrics']}")
                
                # Get visualization DataFrame
                df_viz = generate_backtest_df(result)
                print(f"\nVisualization DataFrame shape: {df_viz.shape}")
                print(f"Columns: {list(df_viz.columns)}")
                
                # Get summary
                summary = get_backtest_summary(result)
                print(f"\nTrading Return: {summary['trading_metrics']['total_return_pct']:.2f}%")
                print(f"Max Drawdown: {summary['trading_metrics']['max_drawdown_pct']:.2f}%")
                print(f"Sharpe Ratio: {summary['trading_metrics']['sharpe_ratio']:.2f}")
            
        except FileNotFoundError:
            print("No trained model found. Train a model first.")
        
        # Multi-horizon test
        try:
            multi_models = load_multi_horizon_models("reliance_model.joblib")
            
            print("\n=== Multi-Horizon Backtest ===")
            multi_results = multi_horizon_backtest(df_features, multi_models, feature_cols)
            
            # Comparison table
            comparison = compare_horizons(multi_results)
            print("\nHorizon Comparison:")
            print(comparison.to_string(index=False))
            
        except FileNotFoundError:
            print("No multi-horizon models found.")
