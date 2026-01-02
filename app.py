# app.py
# Streamlit app for stock price prediction (optimized)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys
import concurrent.futures
import queue
import time

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from utils.data_loader import get_nse_symbols, fetch_stock_data
from utils.features import create_features, get_feature_columns
from utils.regime import detect_regime, get_regime_details
from utils.analytics import get_technical_summary, generate_rationale, get_prediction_explanation
from utils.backtest import (
    rolling_backtest, multi_horizon_backtest, 
    generate_backtest_df, get_backtest_summary, compare_horizons
)
from models.predictor import (
    load_model, predict_future, train_model, MODELS_DIR,
    train_multi_horizon_models, predict_multi_horizon,
    train_quantile_models, predict_with_confidence,
    load_multi_horizon_models, load_quantile_models,
    save_multi_horizon_models, save_quantile_models,
    predict_with_regime_adjustment, get_regime_explanation
)
from train import train_with_progress, get_last_trained_timestamp

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)


# ============== CACHED FUNCTIONS ==============

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_stock_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Cache stock data for 1 hour."""
    try:
        return fetch_stock_data(symbol, start, end)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def cached_load_symbols() -> list[str]:
    """Cache symbols for 24 hours."""
    try:
        return get_nse_symbols(yahoo_format=False)
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def cached_load_model(model_path: str):
    """Cache model in memory (persists across reruns)."""
    try:
        return load_model(model_path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def cached_load_multi_horizon_models(model_path: str):
    """Cache multi-horizon models."""
    try:
        return load_multi_horizon_models(model_path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def cached_load_quantile_models(model_path: str):
    """Cache quantile models for confidence bands."""
    try:
        return load_quantile_models(model_path)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_create_features(df_json: str) -> pd.DataFrame:
    """Cache feature engineering results."""
    df = pd.read_json(df_json)
    df['Date'] = pd.to_datetime(df['Date'])
    return create_features(df)


def get_available_models() -> list[str]:
    """Get list of trained models (not cached to reflect new trainings)."""
    try:
        models = list(MODELS_DIR.glob("*_model.joblib"))
        return [m.stem.replace("_model", "").upper() for m in models]
    except Exception:
        return []


def train_stock_model(symbol: str, df: pd.DataFrame) -> tuple[bool, str, dict]:
    """
    Train model for a stock (single, multi-horizon, and quantile models).
    Returns: (success, message, metrics)
    """
    try:
        # Create features
        df_features = create_features(df)
        
        if len(df_features) < 50:
            return False, "Not enough data (need at least 50 rows after feature engineering)", {}
        
        base_filename = f"{symbol.lower()}_model.joblib"
        
        # Train single model (original)
        model, metrics = train_model(
            df_features,
            feature_cols=get_feature_columns(),
            save_path=base_filename
        )
        
        # Train multi-horizon models
        multi_models, multi_metrics = train_multi_horizon_models(
            df_features,
            feature_cols=get_feature_columns(),
            save_path=base_filename
        )
        
        # Train quantile models for confidence bands
        q_models, q_metrics = train_quantile_models(
            df_features,
            feature_cols=get_feature_columns(),
            save_path=base_filename
        )
        
        # Clear model caches to load new models
        cached_load_model.clear()
        cached_load_multi_horizon_models.clear()
        cached_load_quantile_models.clear()
        
        return True, f"All models saved for {symbol}", metrics
        
    except Exception as e:
        return False, f"Training failed: {str(e)}", {}


# ============== HELPER FUNCTIONS ==============

def calculate_analytics(df: pd.DataFrame) -> dict:
    """Calculate basic stock analytics."""
    if df.empty or len(df) < 2:
        return {
            'current': 0, 'change_1d': 0, 'change_1w': 0,
            'change_1m': 0, 'volatility': 0
        }
    
    current = df['Close'].iloc[-1]
    
    # Safe % change calculation
    def safe_change(days_back: int) -> float:
        if len(df) > days_back:
            prev = df['Close'].iloc[-days_back - 1]
            return ((current - prev) / prev * 100) if prev != 0 else 0
        return 0
    
    # Volatility (20-day)
    returns = df['Close'].pct_change().dropna()
    volatility = returns.tail(20).std() * 100 if len(returns) >= 20 else 0
    
    return {
        'current': current,
        'change_1d': safe_change(1),
        'change_1w': safe_change(5),
        'change_1m': safe_change(21),
        'volatility': volatility
    }


def create_chart(
    df: pd.DataFrame, 
    predictions: pd.DataFrame = None,
    confidence_preds: dict = None,
    multi_horizon_preds: dict = None,
    chart_type: str = "Line"
) -> go.Figure:
    """
    Create optimized interactive Plotly chart with confidence bands.
    Supports both Line and Candlestick chart types.
    
    Args:
        df: DataFrame with OHLC data
        predictions: Day-by-day predictions DataFrame
        confidence_preds: Multi-horizon confidence band predictions
        multi_horizon_preds: Multi-horizon point predictions
        chart_type: "Line" or "Candlestick"
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Get last historical date and price for prediction overlay
    last_date = df['Date'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    
    # Downsample for performance if too many points
    plot_df = df
    if len(df) > 500:
        step = len(df) // 500
        plot_df = df.iloc[::step].copy()
        # Always include last few points for accuracy
        plot_df = pd.concat([plot_df, df.tail(10)]).drop_duplicates()
        plot_df = plot_df.sort_values('Date').reset_index(drop=True)
    
    # ============== HISTORICAL DATA ==============
    if chart_type == "Candlestick":
        # Candlestick chart for OHLC data
        fig.add_trace(go.Candlestick(
            x=plot_df['Date'],
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name='OHLC',
            increasing_line_color='#4CAF50',
            decreasing_line_color='#f44336',
            increasing_fillcolor='#4CAF50',
            decreasing_fillcolor='#f44336',
            hoverinfo='x+y+text',
        ))
    else:
        # Line chart for Close price
        fig.add_trace(go.Scattergl(
            x=plot_df['Date'],
            y=plot_df['Close'],
            mode='lines',
            name='Historical Close',
            line=dict(color='#2196F3', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>₹%{y:.2f}<extra></extra>'
        ))
    
    # ============== CONFIDENCE BANDS (render first for layering) ==============
    if confidence_preds and multi_horizon_preds:
        horizon_days = {'1d': 1, '5d': 5, '20d': 20}
        
        # Prepare data for confidence bands (starts strictly after last date)
        band_dates = [last_date]
        band_low = [last_close]
        band_mid = [last_close]
        band_high = [last_close]
        
        for horizon in ['1d', '5d', '20d']:
            if horizon in confidence_preds:
                future_date = last_date + pd.Timedelta(days=horizon_days[horizon])
                # Skip weekends
                while future_date.weekday() >= 5:
                    future_date += pd.Timedelta(days=1)
                
                band_dates.append(future_date)
                band_low.append(confidence_preds[horizon]['low'])
                band_mid.append(confidence_preds[horizon]['mid'])
                band_high.append(confidence_preds[horizon]['high'])
        
        # Confidence band (filled area)
        fig.add_trace(go.Scatter(
            x=band_dates + band_dates[::-1],
            y=band_high + band_low[::-1],
            fill='toself',
            fillcolor='rgba(76, 175, 80, 0.2)',
            line=dict(color='rgba(76, 175, 80, 0)'),
            name='Confidence Band (10%-90%)',
            hoverinfo='skip',
            showlegend=True
        ))
        
        # Mid prediction line (multi-horizon)
        fig.add_trace(go.Scatter(
            x=band_dates,
            y=band_mid,
            mode='lines+markers',
            name='Multi-Horizon Forecast',
            line=dict(color='#4CAF50', width=2),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>₹%{y:.2f} (median)<extra></extra>'
        ))
    
    # ============== DAY-BY-DAY PREDICTIONS ==============
    if predictions is not None and not predictions.empty:
        # Predictions start strictly after last historical date
        pred_dates = [last_date] + predictions['Date'].tolist()
        pred_prices = [last_close] + predictions['Predicted_Close'].tolist()
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode='lines+markers',
            name='Day-by-Day Forecast',
            line=dict(color='#FF9800', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>₹%{y:.2f} (pred)<extra></extra>'
        ))
    
    # ============== PREDICTION START MARKER ==============
    # Add vertical line to mark prediction start (using shape to avoid Timestamp issues)
    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#9E9E9E", width=1, dash="dot")
    )
    
    # Add annotation separately
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Prediction Start",
        showarrow=False,
        font=dict(size=10, color="#666666"),
        yanchor="bottom"
    )
    
    # ============== LAYOUT ==============
    chart_title_suffix = "Candlestick" if chart_type == "Candlestick" else "Line"
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02,
            xanchor='left',
            x=0
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=480,
        uirevision='constant',
        xaxis_rangeslider_visible=False,  # Disable rangeslider for candlestick
    )
    
    # Optimize hover performance
    fig.update_traces(hoverlabel=dict(namelength=-1))
    
    return fig


def create_backtest_charts(df_viz: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    """
    Create equity curve and drawdown charts for backtest visualization.
    
    Returns:
        Tuple of (equity_fig, drawdown_fig)
    """
    # Equity Curve Chart
    equity_fig = go.Figure()
    
    equity_fig.add_trace(go.Scatter(
        x=df_viz['Date'],
        y=df_viz['Equity'],
        mode='lines',
        name='Strategy',
        line=dict(color='#4CAF50', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>₹%{y:,.0f}<extra>Strategy</extra>'
    ))
    
    equity_fig.add_trace(go.Scatter(
        x=df_viz['Date'],
        y=df_viz['Buy_Hold'],
        mode='lines',
        name='Buy & Hold',
        line=dict(color='#2196F3', width=2, dash='dash'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>₹%{y:,.0f}<extra>Buy & Hold</extra>'
    ))
    
    equity_fig.update_layout(
        title='Equity Curve',
        xaxis_title='Date',
        yaxis_title='Portfolio Value (₹)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
        height=350
    )
    
    # Drawdown Chart
    dd_fig = go.Figure()
    
    dd_fig.add_trace(go.Scatter(
        x=df_viz['Date'],
        y=df_viz['Drawdown_Pct'],
        mode='lines',
        name='Strategy Drawdown',
        fill='tozeroy',
        line=dict(color='#f44336', width=1),
        fillcolor='rgba(244, 67, 54, 0.3)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra>Strategy</extra>'
    ))
    
    dd_fig.add_trace(go.Scatter(
        x=df_viz['Date'],
        y=df_viz['Buy_Hold_Drawdown'],
        mode='lines',
        name='Buy & Hold Drawdown',
        line=dict(color='#FF9800', width=1, dash='dash'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra>Buy & Hold</extra>'
    ))
    
    dd_fig.update_layout(
        title='Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
        height=300
    )
    
    return equity_fig, dd_fig


# ============== MAIN APP ==============

def main():
    st.title("📈 Stock Price Predictor")
    st.markdown("---")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # Load symbols with loading state
        symbols = cached_load_symbols()
        
        if not symbols:
            st.error("Failed to load stock symbols")
            st.stop()
        
        available_models = get_available_models()
        
        # Training safety: retrain threshold (hours)
        retrain_threshold = st.number_input("Retrain threshold (hours)", min_value=1, value=24)

        # Stock selection with search
        default_idx = symbols.index("RELIANCE") if "RELIANCE" in symbols else 0
        training_now = st.session_state.get('training', False)
        selected_symbol = st.selectbox(
            "Select Stock",
            options=symbols,
            index=default_idx,
            help="Type to search",
            disabled=training_now
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=pd.to_datetime("2023-01-01"), disabled=training_now)
        with col2:
            end_date = st.date_input("End", value=pd.to_datetime("2024-12-31"), disabled=training_now)
        
        # Validate dates
        if start_date >= end_date:
            st.error("Start date must be before end date")
            st.stop()
        
        # Prediction days
        pred_days = st.slider("Prediction Days", 1, 30, 5, disabled=training_now)
        
        # Model status & training
        st.markdown("---")
        st.subheader("🤖 Model")
        
        model_exists = selected_symbol in available_models
        if model_exists:
            st.success(f"✅ Model ready")
        else:
            st.info(f"No model for {selected_symbol}")
        
        # Last trained timestamp
        last_trained = get_last_trained_timestamp(selected_symbol)
        if last_trained:
            st.caption(f"Last trained: {last_trained.isoformat()} UTC")

        # Train button
        train_btn = st.button(
            "🚀 Train Model" if not model_exists else "🔄 Retrain Model",
            use_container_width=True,
            type="primary" if not model_exists else "secondary",
            disabled=training_now
        )
    
    # Main content
    yahoo_symbol = f"{selected_symbol}.NS"
    
    # Fetch data with loading indicator
    with st.spinner(f"Loading {selected_symbol} data..."):
        df = cached_load_stock_data(yahoo_symbol, str(start_date), str(end_date))
    
    # Handle empty/invalid data
    if df is None or df.empty:
        st.error(f"❌ No data available for {selected_symbol}")
        st.info("This could be due to:")
        st.markdown("- Invalid stock symbol\n- No data in selected date range\n- Network issues")
        st.stop()
    
    if len(df) < 25:
        st.warning(f"⚠️ Only {len(df)} data points. Consider expanding date range.")
    
    # Handle training with progress (non-blocking background thread)
    if train_btn:
        # Prevent double clicks
        st.session_state['training'] = True

        q = queue.Queue()

        def progress_cb(stage: str, pct: int):
            q.put({'stage': stage, 'pct': int(pct)})

        # Start background training
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            train_with_progress,
            f"{selected_symbol}.NS",
            str(start_date),
            str(end_date),
            progress_cb,
            retrain_threshold
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Poll for progress until done
            while not future.done():
                try:
                    while not q.empty():
                        msg = q.get_nowait()
                        pct = msg.get('pct', 0)
                        stage = msg.get('stage', '')
                        status_text.info(f"{stage} — {pct}%")
                        progress_bar.progress(pct)
                except queue.Empty:
                    pass
                time.sleep(0.1)

            # Drain any remaining messages
            try:
                while not q.empty():
                    msg = q.get_nowait()
                    pct = msg.get('pct', 0)
                    stage = msg.get('stage', '')
                    status_text.info(f"{stage} — {pct}%")
                    progress_bar.progress(pct)
            except queue.Empty:
                pass

            success, message, metrics = future.result()

            # Ensure progress bar completes
            progress_bar.progress(100)
            status_text.empty()

            if success:
                st.success(f"✅ {message}")
                if metrics:
                    st.markdown("**Training Metrics:**")
                    col1, col2, col3 = st.columns(3)
                    if 'mae' in metrics:
                        col1.metric("MAE", f"₹{metrics['mae']:.2f}")
                # Clear caches so new models load
                cached_load_model.clear()
                cached_load_multi_horizon_models.clear()
                cached_load_quantile_models.clear()
                # Update model status
                model_exists = True
                # Update last trained display
                last_trained = get_last_trained_timestamp(selected_symbol)
            else:
                st.error(f"❌ {message}")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")
        finally:
            st.session_state['training'] = False
            executor.shutdown(wait=False)
    
    # Analytics section
    analytics = calculate_analytics(df)
    
    # Market Regime Detection
    regime_details = get_regime_details(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Price", f"₹{analytics['current']:,.2f}")
    with col2:
        st.metric("1 Day", f"{analytics['change_1d']:+.2f}%")
    with col3:
        st.metric("1 Week", f"{analytics['change_1w']:+.2f}%")
    with col4:
        st.metric("1 Month", f"{analytics['change_1m']:+.2f}%")
    with col5:
        st.metric("Volatility", f"{analytics['volatility']:.2f}%")
    
    # Regime Display
    st.markdown("---")
    regime_col1, regime_col2, regime_col3, regime_col4 = st.columns(4)
    
    regime_colors = {
        "Trending": "🟢",
        "Ranging": "🟡",
        "High Volatility": "🔴",
        "Transitional": "🟠",
        "Insufficient Data": "⚪"
    }
    
    with regime_col1:
        regime_icon = regime_colors.get(regime_details['regime'], "⚪")
        st.metric("Market Regime", f"{regime_icon} {regime_details['regime']}")
    with regime_col2:
        adx_val = regime_details['adx'] if regime_details['adx'] else "N/A"
        st.metric("ADX (Trend Strength)", adx_val)
    with regime_col3:
        vol_ratio = regime_details['volatility_ratio'] if regime_details['volatility_ratio'] else "N/A"
        st.metric("Volatility Ratio", vol_ratio)
    with regime_col4:
        trend_dir = regime_details['trend_direction'] if regime_details['trend_direction'] else "N/A"
        trend_icon = "📈" if trend_dir == "Bullish" else "📉" if trend_dir == "Bearish" else "➡️"
        st.metric("Trend Direction", f"{trend_icon} {trend_dir}")
    
    st.markdown("---")
    
    # Predictions (with caching)
    predictions = None
    confidence_preds = None
    multi_horizon_preds = None
    regime_adjusted_result = None
    
    if model_exists:
        with st.spinner("Generating predictions..."):
            try:
                model = cached_load_model(f"{selected_symbol.lower()}_model.joblib")
                multi_models = cached_load_multi_horizon_models(f"{selected_symbol.lower()}_model.joblib")
                q_models = cached_load_quantile_models(f"{selected_symbol.lower()}_model.joblib")
                
                if model is not None:
                    # Cache features using JSON serialization
                    df_features = cached_create_features(df.to_json())
                    if not df_features.empty:
                        # Original iterative predictions
                        predictions = predict_future(df_features, model, days=pred_days)
                        
                        # Multi-horizon predictions
                        if multi_models is not None:
                            multi_horizon_preds = predict_multi_horizon(df_features, multi_models)
                        
                        # Regime-adjusted confidence bands
                        if q_models is not None:
                            regime_adjusted_result = predict_with_regime_adjustment(
                                df_features, 
                                q_models,
                                regime=regime_details['regime']
                            )
                            confidence_preds = regime_adjusted_result['predictions']
            except Exception as e:
                st.error(f"Prediction failed: {str(e)[:100]}")
    
    # Chart section with type toggle
    chart_header_col1, chart_header_col2 = st.columns([3, 1])
    
    with chart_header_col1:
        st.subheader(f"{selected_symbol} Price Chart")
    
    with chart_header_col2:
        chart_type = st.radio(
            "Chart Type",
            options=["Line", "Candlestick"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
    
    chart_placeholder = st.empty()
    
    with chart_placeholder:
        fig = create_chart(df, predictions, confidence_preds, multi_horizon_preds, chart_type)
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        })
    
    # Multi-Horizon Predictions with Confidence Bands
    if confidence_preds:
        st.subheader("🎯 Multi-Horizon Predictions with Confidence Bands")
        
        # Show regime adjustment info
        if regime_adjusted_result:
            adjustment = regime_adjusted_result['adjustment_factor']
            conf_level = regime_adjusted_result['confidence_level']
            explanation = get_regime_explanation(regime_details['regime'])
            
            # Confidence level colors
            conf_colors = {
                "High": "🟢",
                "Medium": "🟡", 
                "Low": "🟠",
                "Very Low": "🔴"
            }
            conf_icon = conf_colors.get(conf_level, "⚪")
            
            st.info(f"{conf_icon} **Confidence Level: {conf_level}** (Adjustment: {adjustment:.1%}) \n\n{explanation}")
        
        horizon_col1, horizon_col2, horizon_col3 = st.columns(3)
        
        for col, horizon in zip([horizon_col1, horizon_col2, horizon_col3], ['1d', '5d', '20d']):
            if horizon in confidence_preds:
                with col:
                    st.markdown(f"**{horizon.upper()} Forecast**")
                    low = confidence_preds[horizon]['low']
                    mid = confidence_preds[horizon]['mid']
                    high = confidence_preds[horizon]['high']
                    
                    st.metric("Median", f"₹{mid:,.2f}")
                    st.caption(f"Range: ₹{low:,.2f} - ₹{high:,.2f}")
                    
                    # Calculate change from current
                    current = df['Close'].iloc[-1]
                    change_pct = ((mid - current) / current) * 100
                    change_icon = "📈" if change_pct > 0 else "📉"
                    st.caption(f"{change_icon} {change_pct:+.2f}% from current")
        
        st.markdown("---")
    
    # Technical Confluence & Prediction Rationale
    if model_exists and 'df_features' in dir() and df_features is not None and not df_features.empty:
        st.subheader("📊 Technical Analysis & Prediction Rationale")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("**Technical Confluence**")
            tech_summary = get_technical_summary(df)
            
            # Trend
            trend = tech_summary['trend']
            trend_icon = "📈" if trend['direction'] == 'Bullish' else "📉" if trend['direction'] == 'Bearish' else "➡️"
            st.markdown(f"{trend_icon} **Trend:** {trend['direction']} ({trend['strength']})")
            
            # RSI
            rsi = tech_summary['rsi']
            rsi_color = "🔴" if rsi['state'] == 'Overbought' else "🟢" if rsi['state'] == 'Oversold' else "🟡"
            st.markdown(f"{rsi_color} **RSI ({rsi['value']}):** {rsi['state']}")
            st.caption(f"   {rsi['signal']}")
            
            # MA Alignment
            ma = tech_summary['ma_alignment']
            ma_icon = "✅" if 'Bullish' in ma['alignment'] else "❌" if 'Bearish' in ma['alignment'] else "⚖️"
            st.markdown(f"{ma_icon} **MA Alignment:** {ma['alignment']}")
            
            # Volatility
            vol = tech_summary['volatility']
            vol_icon = "🔥" if vol['state'] == 'High' else "📊" if vol['state'] == 'Elevated' else "😴" if vol['state'] == 'Low' else "✓"
            st.markdown(f"{vol_icon} **Volatility:** {vol['state']} ({vol['current_annual_pct']:.1f}% ann.)")
        
        with tech_col2:
            st.markdown("**Prediction Rationale**")
            rationale = generate_rationale(df_features)
            
            # Overall bias
            bias = rationale['overall_bias']
            bias_icon = "🟢" if bias == 'Bullish' else "🔴" if bias == 'Bearish' else "🟡"
            st.markdown(f"{bias_icon} **Overall Bias:** {bias}")
            st.caption(f"Positive Score: {rationale['positive_score']} | Negative Score: {rationale['negative_score']}")
            
            # Key factors
            st.markdown("**Key Factors:**")
            for bullet in rationale['bullets'][:5]:
                icon = "✅" if bullet['type'] == 'positive' else "⚠️"
                st.markdown(f"- {icon} {bullet['text']}")
            
            # Full explanation if we have a prediction
            if confidence_preds and '1d' in confidence_preds:
                pred_1d = confidence_preds['1d']['mid']
                explanation = get_prediction_explanation(df_features, pred_1d)
                with st.expander("📝 Detailed Explanation"):
                    st.write(explanation)
        
        st.markdown("---")
    
    # Predictions table (iterative)
    if predictions is not None and not predictions.empty:
        st.subheader("📊 Day-by-Day Predictions")
        pred_display = predictions.copy()
        pred_display['Date'] = pred_display['Date'].dt.strftime('%Y-%m-%d')
        pred_display['Predicted_Close'] = pred_display['Predicted_Close'].apply(lambda x: f"₹{x:,.2f}")
        pred_display.columns = ['Date', 'Predicted Price']
        st.dataframe(pred_display, use_container_width=True, hide_index=True)
    
    # ============== BACKTESTING SECTION ==============
    if model_exists:
        st.markdown("---")
        st.subheader("📈 Backtesting")
        
        backtest_col1, backtest_col2 = st.columns([1, 3])
        
        with backtest_col1:
            st.markdown("**Settings**")
            bt_horizon = st.selectbox("Horizon", ['1d', '5d', '20d'], index=0)
            bt_step = st.slider("Step Size (days)", 1, 10, 5, help="Days between predictions")
            run_backtest = st.button("🔄 Run Backtest", use_container_width=True, type="primary")
        
        with backtest_col2:
            if run_backtest:
                with st.spinner("Running backtest..."):
                    try:
                        # Get features
                        df_features_bt = cached_create_features(df.to_json())
                        feature_cols = get_feature_columns()
                        
                        # Load appropriate model
                        multi_models = cached_load_multi_horizon_models(f"{selected_symbol.lower()}_model.joblib")
                        
                        if multi_models and bt_horizon in multi_models:
                            horizon_days = {'1d': 1, '5d': 5, '20d': 20}
                            
                            # Run backtest
                            bt_result = rolling_backtest(
                                df_features_bt,
                                multi_models[bt_horizon],
                                feature_cols,
                                window_size=50,
                                step_size=bt_step,
                                horizon=horizon_days[bt_horizon]
                            )
                            
                            if 'error' not in bt_result:
                                # Get visualization data
                                df_viz = generate_backtest_df(bt_result)
                                summary = get_backtest_summary(bt_result)
                                
                                # Store in session state
                                st.session_state['backtest_result'] = bt_result
                                st.session_state['backtest_df'] = df_viz
                                st.session_state['backtest_summary'] = summary
                            else:
                                st.error(f"Backtest error: {bt_result['error']}")
                        else:
                            st.error("Multi-horizon models not found. Please retrain.")
                    except Exception as e:
                        st.error(f"Backtest failed: {str(e)[:100]}")
        
        # Display backtest results if available
        if 'backtest_summary' in st.session_state:
            summary = st.session_state['backtest_summary']
            df_viz = st.session_state['backtest_df']
            
            # Metrics row
            st.markdown("**Performance Metrics**")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            pred_metrics = summary['prediction_metrics']
            trade_metrics = summary['trading_metrics']
            
            with m1:
                st.metric("MAE", f"₹{pred_metrics['mae']:.2f}")
            with m2:
                st.metric("Direction Acc.", f"{pred_metrics['direction_accuracy']:.1f}%")
            with m3:
                ret = trade_metrics['total_return_pct']
                st.metric("Strategy Return", f"{ret:+.2f}%", delta=f"vs B&H: {trade_metrics['outperformance_pct']:+.2f}%")
            with m4:
                st.metric("Max Drawdown", f"{trade_metrics['max_drawdown_pct']:.2f}%")
            with m5:
                st.metric("Sharpe Ratio", f"{trade_metrics['sharpe_ratio']:.2f}")
            
            # Charts
            if not df_viz.empty:
                equity_fig, dd_fig = create_backtest_charts(df_viz)
                
                st.plotly_chart(equity_fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'scrollZoom': True
                })
                
                with st.expander("📉 Drawdown Chart"):
                    st.plotly_chart(dd_fig, use_container_width=True)
                
                # Prediction accuracy over time
                with st.expander("📊 Prediction Error Analysis"):
                    error_fig = go.Figure()
                    error_fig.add_trace(go.Scatter(
                        x=df_viz['Date'],
                        y=df_viz['Pct_Error'],
                        mode='lines',
                        name='% Error',
                        line=dict(color='#9C27B0', width=1),
                        fill='tozeroy',
                        fillcolor='rgba(156, 39, 176, 0.2)'
                    ))
                    if 'Rolling_MAE' in df_viz.columns:
                        error_fig.add_trace(go.Scatter(
                            x=df_viz['Date'],
                            y=df_viz['Rolling_MAE'],
                            mode='lines',
                            name='Rolling MAE',
                            line=dict(color='#FF5722', width=2)
                        ))
                    error_fig.update_layout(
                        title='Prediction Error Over Time',
                        xaxis_title='Date',
                        yaxis_title='Error',
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(error_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Historical data (lazy loaded)
    with st.expander("📋 Historical Data (Last 50 rows)"):
        display_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(50).copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        for col in ['Open', 'High', 'Low', 'Close']:
            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.2f}")
        display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()