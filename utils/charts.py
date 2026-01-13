# charts.py
# Plotly chart creation for Macrostructure and Microstructure visualization
# Professional institutional-grade dark theme charts

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


# ============== THEME CONFIGURATION ==============

INSTITUTIONAL_DARK_THEME = {
    'background': '#0e1117',
    'paper': '#1a1d24',
    'grid': '#2d3239',
    'text': '#e0e0e0',
    'muted_text': '#808080',
    'bullish': '#00d26a',
    'bearish': '#ff6b6b',
    'neutral': '#ffc107',
    'accent': '#667eea',
    'secondary': '#764ba2',
    'order_block_bull': 'rgba(0, 210, 106, 0.3)',
    'order_block_bear': 'rgba(255, 107, 107, 0.3)',
    'fvg_bull': 'rgba(102, 126, 234, 0.25)',
    'fvg_bear': 'rgba(255, 193, 7, 0.25)',
    'liquidity': 'rgba(255, 255, 255, 0.4)',
    'poc': '#ff9800',
    'vah': '#4caf50',
    'val': '#f44336'
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply institutional dark theme to a Plotly figure."""
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor=INSTITUTIONAL_DARK_THEME['background'],
        paper_bgcolor=INSTITUTIONAL_DARK_THEME['paper'],
        font=dict(
            family='Inter, -apple-system, BlinkMacSystemFont, sans-serif',
            color=INSTITUTIONAL_DARK_THEME['text']
        ),
        xaxis=dict(
            gridcolor=INSTITUTIONAL_DARK_THEME['grid'],
            zerolinecolor=INSTITUTIONAL_DARK_THEME['grid']
        ),
        yaxis=dict(
            gridcolor=INSTITUTIONAL_DARK_THEME['grid'],
            zerolinecolor=INSTITUTIONAL_DARK_THEME['grid']
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ============== MACROSTRUCTURE CHARTS ==============

def create_structure_chart(
    df: pd.DataFrame,
    structure_analysis: dict,
    show_swings: bool = True,
    show_bos: bool = True,
    show_choch: bool = True,
    show_order_blocks: bool = True,
    show_fvgs: bool = True,
    show_liquidity: bool = True
) -> go.Figure:
    """
    Create a comprehensive macrostructure chart with Smart Money concepts.
    
    Args:
        df: DataFrame with OHLC data
        structure_analysis: Output from analyze_market_structure()
        show_* : Toggle visibility of different elements
        
    Returns:
        Plotly Figure with structure overlays
    """
    fig = go.Figure()
    
    theme = INSTITUTIONAL_DARK_THEME
    
    # ============== CANDLESTICK CHART ==============
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing=dict(line=dict(color=theme['bullish']), fillcolor=theme['bullish']),
        decreasing=dict(line=dict(color=theme['bearish']), fillcolor=theme['bearish']),
        opacity=0.9
    ))
    
    # ============== SWING POINTS ==============
    if show_swings:
        swing_highs = structure_analysis.get('swing_highs', [])
        swing_lows = structure_analysis.get('swing_lows', [])
        
        # Swing Highs
        if swing_highs:
            sh_dates = [sh.date for sh in swing_highs]
            sh_prices = [sh.price for sh in swing_highs]
            fig.add_trace(go.Scatter(
                x=sh_dates,
                y=sh_prices,
                mode='markers',
                name='Swing High',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color=theme['bearish'],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Swing High</b><br>%{x|%Y-%m-%d}<br>₹%{y:.2f}<extra></extra>'
            ))
        
        # Swing Lows
        if swing_lows:
            sl_dates = [sl.date for sl in swing_lows]
            sl_prices = [sl.price for sl in swing_lows]
            fig.add_trace(go.Scatter(
                x=sl_dates,
                y=sl_prices,
                mode='markers',
                name='Swing Low',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color=theme['bullish'],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Swing Low</b><br>%{x|%Y-%m-%d}<br>₹%{y:.2f}<extra></extra>'
            ))
    
    # ============== BOS (Break of Structure) ==============
    if show_bos:
        bos_events = structure_analysis.get('bos_events', [])
        for bos in bos_events[-10:]:  # Show last 10
            color = theme['bullish'] if bos.direction == 'bullish' else theme['bearish']
            
            fig.add_annotation(
                x=bos.date,
                y=bos.price,
                text='BOS',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                font=dict(size=10, color=color, family='monospace'),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor=color,
                borderwidth=1,
                borderpad=2,
                ax=0,
                ay=-30 if bos.direction == 'bullish' else 30
            )
    
    # ============== CHoCH (Change of Character) ==============
    if show_choch:
        choch_events = structure_analysis.get('choch_events', [])
        for choch in choch_events[-5:]:  # Show last 5
            color = theme['neutral']
            
            fig.add_annotation(
                x=choch.date,
                y=choch.price,
                text='CHoCH',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=color,
                font=dict(size=11, color=color, family='monospace', weight='bold'),
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor=color,
                borderwidth=2,
                borderpad=3,
                ax=0,
                ay=-40 if choch.direction == 'bullish' else 40
            )
    
    # ============== ORDER BLOCKS ==============
    if show_order_blocks:
        order_blocks = structure_analysis.get('order_blocks', [])
        for ob in order_blocks[-15:]:  # Show last 15
            fill_color = theme['order_block_bull'] if ob.ob_type == 'bullish' else theme['order_block_bear']
            border_color = theme['bullish'] if ob.ob_type == 'bullish' else theme['bearish']
            
            # Calculate x-axis extent (from OB candle to chart end)
            end_date = df['Date'].iloc[-1]
            
            fig.add_shape(
                type='rect',
                x0=ob.start_date,
                x1=end_date,
                y0=ob.bottom,
                y1=ob.top,
                fillcolor=fill_color,
                line=dict(color=border_color, width=1, dash='dot'),
                layer='below'
            )
            
            # Add OB label
            fig.add_annotation(
                x=ob.start_date,
                y=ob.top if ob.ob_type == 'bullish' else ob.bottom,
                text=f"{'🟢' if ob.ob_type == 'bullish' else '🔴'} OB",
                showarrow=False,
                font=dict(size=9, color=border_color),
                bgcolor='rgba(0,0,0,0.6)',
                borderpad=2,
                xanchor='left',
                yanchor='bottom' if ob.ob_type == 'bullish' else 'top'
            )
    
    # ============== FAIR VALUE GAPS ==============
    if show_fvgs:
        fvgs = structure_analysis.get('unfilled_fvgs', [])
        for fvg in fvgs[-10:]:  # Show last 10 unfilled
            fill_color = theme['fvg_bull'] if fvg.fvg_type == 'bullish' else theme['fvg_bear']
            border_color = theme['accent'] if fvg.fvg_type == 'bullish' else theme['neutral']
            
            end_date = df['Date'].iloc[-1]
            
            fig.add_shape(
                type='rect',
                x0=fvg.date,
                x1=end_date,
                y0=fvg.bottom,
                y1=fvg.top,
                fillcolor=fill_color,
                line=dict(color=border_color, width=1),
                layer='below'
            )
            
            # FVG label
            fig.add_annotation(
                x=fvg.date,
                y=(fvg.top + fvg.bottom) / 2,
                text='FVG',
                showarrow=False,
                font=dict(size=8, color=border_color),
                bgcolor='rgba(0,0,0,0.5)',
                borderpad=1,
                xanchor='left'
            )
    
    # ============== LIQUIDITY POOLS ==============
    if show_liquidity:
        pools = structure_analysis.get('liquidity_pools', [])
        for pool in pools:
            line_style = 'dot' if pool.swept else 'solid'
            color = theme['liquidity']
            
            start_date = df['Date'].iloc[pool.start_index]
            end_date = df['Date'].iloc[-1]
            
            fig.add_shape(
                type='line',
                x0=start_date,
                x1=end_date,
                y0=pool.price_level,
                y1=pool.price_level,
                line=dict(color=color, width=2, dash=line_style),
                layer='below'
            )
            
            label = 'EQH' if pool.pool_type == 'equal_highs' else 'EQL'
            fig.add_annotation(
                x=end_date,
                y=pool.price_level,
                text=f'{label} ({pool.strength})',
                showarrow=False,
                font=dict(size=8, color='white'),
                bgcolor='rgba(255,255,255,0.2)',
                borderpad=2,
                xanchor='right'
            )
    
    # ============== LAYOUT ==============
    fig.update_layout(
        title=dict(
            text='📊 Market Structure Analysis',
            font=dict(size=18)
        ),
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        hovermode='x unified'
    )
    
    apply_dark_theme(fig)
    
    return fig


# ============== MICROSTRUCTURE CHARTS ==============

def create_volume_profile_chart(
    df: pd.DataFrame,
    volume_profile: dict,
    chart_type: str = 'horizontal'
) -> go.Figure:
    """
    Create a Volume Profile (VPVR) chart.
    
    Args:
        df: DataFrame with OHLC data
        volume_profile: Output from calculate_volume_profile()
        chart_type: 'horizontal' (bars on price axis) or 'separate'
        
    Returns:
        Plotly Figure with volume profile
    """
    theme = INSTITUTIONAL_DARK_THEME
    
    if chart_type == 'horizontal':
        # Two subplots: Price chart with horizontal volume profile
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.85, 0.15],
            shared_yaxes=True,
            horizontal_spacing=0.01
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing=dict(line=dict(color=theme['bullish'])),
            decreasing=dict(line=dict(color=theme['bearish']))
        ), row=1, col=1)
        
        # Volume Profile (horizontal bars)
        nodes = volume_profile.get('nodes', [])
        if nodes:
            prices = [n.price_level for n in nodes]
            volumes = [n.volume for n in nodes]
            max_vol = max(volumes) if volumes else 1
            
            # Normalize volumes for display
            norm_volumes = [v / max_vol * 100 for v in volumes]
            
            # Color based on buy/sell dominance
            colors = []
            for n in nodes:
                if n.is_poc:
                    colors.append(theme['poc'])
                elif n.buy_volume > n.sell_volume:
                    colors.append(theme['bullish'])
                else:
                    colors.append(theme['bearish'])
            
            fig.add_trace(go.Bar(
                x=norm_volumes,
                y=prices,
                orientation='h',
                name='Volume Profile',
                marker=dict(color=colors, opacity=0.7),
                hovertemplate='₹%{y:.2f}<br>Vol: %{customdata:,.0f}<extra></extra>',
                customdata=volumes
            ), row=1, col=2)
            
            # POC, VAH, VAL lines
            poc = volume_profile.get('poc')
            vah = volume_profile.get('vah')
            val = volume_profile.get('val')
            
            if poc:
                fig.add_hline(
                    y=poc, 
                    line=dict(color=theme['poc'], width=2, dash='dash'),
                    annotation_text='POC',
                    annotation_position='left',
                    row=1, col=1
                )
            
            if vah:
                fig.add_hline(
                    y=vah,
                    line=dict(color=theme['vah'], width=1, dash='dot'),
                    annotation_text='VAH',
                    annotation_position='left',
                    row=1, col=1
                )
            
            if val:
                fig.add_hline(
                    y=val,
                    line=dict(color=theme['val'], width=1, dash='dot'),
                    annotation_text='VAL',
                    annotation_position='left',
                    row=1, col=1
                )
        
        fig.update_layout(
            title='📊 Volume Profile (VPVR)',
            height=550,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            xaxis2=dict(showticklabels=False, showgrid=False)
        )
    
    else:
        # Separate chart view
        fig = go.Figure()
        
        nodes = volume_profile.get('nodes', [])
        if nodes:
            prices = [n.price_level for n in nodes]
            volumes = [n.volume for n in nodes]
            
            colors = []
            for n in nodes:
                if n.is_poc:
                    colors.append(theme['poc'])
                elif n.is_hva:
                    colors.append(theme['accent'])
                elif n.is_lva:
                    colors.append(theme['muted_text'])
                else:
                    colors.append(theme['text'])
            
            fig.add_trace(go.Bar(
                x=volumes,
                y=prices,
                orientation='h',
                marker=dict(color=colors, opacity=0.8),
                hovertemplate='Price: ₹%{y:.2f}<br>Volume: %{x:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='📊 Volume Profile Distribution',
            xaxis_title='Volume',
            yaxis_title='Price (₹)',
            height=400
        )
    
    apply_dark_theme(fig)
    return fig


def create_depth_chart(order_book: dict) -> go.Figure:
    """
    Create a market depth chart (cumulative order book).
    
    Args:
        order_book: Output from simulate_order_book()
        
    Returns:
        Plotly Figure with depth visualization
    """
    theme = INSTITUTIONAL_DARK_THEME
    fig = go.Figure()
    
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])
    mid_price = order_book.get('mid_price', 0)
    
    if bids:
        bid_prices = [b.price for b in bids]
        bid_cumulative = [b.cumulative_bid for b in bids]
        
        fig.add_trace(go.Scatter(
            x=bid_prices,
            y=bid_cumulative,
            mode='lines',
            name='Bids',
            fill='tozeroy',
            line=dict(color=theme['bullish'], width=2),
            fillcolor='rgba(0, 210, 106, 0.3)',
            hovertemplate='Price: ₹%{x:.2f}<br>Cumulative: %{y:,.0f}<extra>Bids</extra>'
        ))
    
    if asks:
        ask_prices = [a.price for a in asks]
        ask_cumulative = [a.cumulative_ask for a in asks]
        
        fig.add_trace(go.Scatter(
            x=ask_prices,
            y=ask_cumulative,
            mode='lines',
            name='Asks',
            fill='tozeroy',
            line=dict(color=theme['bearish'], width=2),
            fillcolor='rgba(255, 107, 107, 0.3)',
            hovertemplate='Price: ₹%{x:.2f}<br>Cumulative: %{y:,.0f}<extra>Asks</extra>'
        ))
    
    # Add mid price line
    if mid_price:
        fig.add_vline(
            x=mid_price,
            line=dict(color=theme['neutral'], width=2, dash='dash'),
            annotation_text=f'Mid: ₹{mid_price:.2f}',
            annotation_position='top'
        )
    
    fig.update_layout(
        title='📈 Market Depth (Simulated Order Book)',
        xaxis_title='Price (₹)',
        yaxis_title='Cumulative Size',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    
    apply_dark_theme(fig)
    return fig


def create_order_flow_chart(order_imbalance_df: pd.DataFrame) -> go.Figure:
    """
    Create an order flow / delta chart.
    
    Args:
        order_imbalance_df: Output from calculate_order_imbalance()
        
    Returns:
        Plotly Figure with order flow visualization
    """
    theme = INSTITUTIONAL_DARK_THEME
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Cumulative Delta', 'Volume Delta', 'Imbalance Ratio')
    )
    
    df = order_imbalance_df
    
    # Price chart
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close',
        line=dict(color=theme['accent'], width=2)
    ), row=1, col=1)
    
    # Cumulative Delta (secondary y-axis simulated with scaling)
    delta_scaled = df['CumulativeDelta'] / df['CumulativeDelta'].abs().max() * df['Close'].max() * 0.3
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'].mean() + delta_scaled,
        mode='lines',
        name='Cum. Delta',
        line=dict(color=theme['neutral'], width=1.5, dash='dot'),
        opacity=0.7
    ), row=1, col=1)
    
    # Volume Delta bars
    colors = [theme['bullish'] if d > 0 else theme['bearish'] for d in df['Delta']]
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Delta'],
        name='Delta',
        marker=dict(color=colors, opacity=0.7)
    ), row=2, col=1)
    
    # Imbalance Ratio
    imbalance_colors = [theme['bullish'] if i > 0 else theme['bearish'] for i in df['ImbalanceRatio']]
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['ImbalanceRatio'],
        mode='lines',
        name='Imbalance',
        line=dict(color=theme['secondary'], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(118, 75, 162, 0.2)'
    ), row=3, col=1)
    
    # Add zero line to imbalance
    fig.add_hline(y=0, line=dict(color=theme['muted_text'], width=1), row=3, col=1)
    
    fig.update_layout(
        title='📊 Order Flow Analysis',
        height=600,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text='Price (₹)', row=1, col=1)
    fig.update_yaxes(title_text='Delta', row=2, col=1)
    fig.update_yaxes(title_text='Ratio', row=3, col=1)
    
    apply_dark_theme(fig)
    return fig


def create_microstructure_gauge(micro_score: dict) -> go.Figure:
    """
    Create a gauge chart showing overall microstructure score.
    
    Args:
        micro_score: Output from calculate_microstructure_score()
        
    Returns:
        Plotly Figure with gauge visualization
    """
    theme = INSTITUTIONAL_DARK_THEME
    
    overall_score = micro_score.get('overall_score', 0)
    regime = micro_score.get('regime', 'Unknown')
    
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=overall_score,
        title=dict(
            text=f"Microstructure Score<br><span style='font-size:0.8em;color:{theme['muted_text']}'>{regime}</span>",
            font=dict(size=16)
        ),
        delta=dict(reference=0, increasing=dict(color=theme['bullish']), decreasing=dict(color=theme['bearish'])),
        gauge=dict(
            axis=dict(range=[-100, 100], tickwidth=1, tickcolor=theme['text']),
            bar=dict(color=theme['accent']),
            bgcolor=theme['paper'],
            borderwidth=2,
            bordercolor=theme['grid'],
            steps=[
                dict(range=[-100, -30], color='rgba(255, 107, 107, 0.3)'),
                dict(range=[-30, -10], color='rgba(255, 193, 7, 0.2)'),
                dict(range=[-10, 10], color='rgba(128, 128, 128, 0.2)'),
                dict(range=[10, 30], color='rgba(144, 238, 144, 0.2)'),
                dict(range=[30, 100], color='rgba(0, 210, 106, 0.3)')
            ],
            threshold=dict(
                line=dict(color=theme['neutral'], width=4),
                thickness=0.75,
                value=overall_score
            )
        )
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    apply_dark_theme(fig)
    return fig


def create_heatmap_calendar(df: pd.DataFrame) -> go.Figure:
    """
    Create a calendar heatmap showing daily returns.
    
    Args:
        df: DataFrame with Date and Close columns
        
    Returns:
        Plotly Figure with calendar heatmap
    """
    theme = INSTITUTIONAL_DARK_THEME
    
    df = df.copy()
    df['Returns'] = df['Close'].pct_change() * 100
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Week'] = pd.to_datetime(df['Date']).dt.isocalendar().week
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    
    # Create pivot table
    pivot = df.pivot_table(
        values='Returns',
        index='DayOfWeek',
        columns='Week',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale=[
            [0, theme['bearish']],
            [0.5, theme['paper']],
            [1, theme['bullish']]
        ],
        zmid=0,
        hovertemplate='Week %{x}<br>%{y}<br>Return: %{z:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='📅 Returns Calendar',
        xaxis_title='Week',
        yaxis_title='Day',
        height=250
    )
    
    apply_dark_theme(fig)
    return fig
