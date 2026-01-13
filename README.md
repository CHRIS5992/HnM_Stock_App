# 📈 Stock Price Predictor (Market Genesis)

An interactive institutional-grade stock analysis platform for NSE (National Stock Exchange) equities. Built with Python, Streamlit, and Plotly.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

### 📈 Tab 1: ML Predictions
- **Multi-horizon forecasts**: 1-day, 5-day, and 20-day price predictions
- **Confidence bands**: Quantile regression for 10%-90% prediction intervals
- **Regime-adjusted predictions**: Adapts to market conditions (trending, ranging, high volatility)
- **Backtesting engine**: Rolling backtests with equity curves and drawdown analysis

### 🌍 Tab 2: Macrostructure Discovery (Smart Money Concepts)
- **BOS (Break of Structure)**: Detects when price breaks swing highs/lows in trend direction
- **CHoCH (Change of Character)**: Identifies potential reversal signals when structure breaks against trend
- **Order Blocks (OB)**: Highlights supply/demand zones preceding displacement moves
- **Fair Value Gaps (FVG)**: 3-candle imbalances showing unfilled price gaps
- **Liquidity Pools**: Equal highs/lows where stop losses cluster
- **Structure Bias Analysis**: Overall bullish/bearish market structure assessment

### � Tab 3: Microstructure Discovery
- **Volume Profile (VPVR)**: Volume at price analysis with POC, VAH, VAL levels
- **Simulated Order Book**: Market depth visualization with bid/ask liquidity
- **Order Flow Analysis**: Buy/sell volume delta and cumulative delta tracking
- **Imbalance Ratio**: Real-time buying/selling pressure measurement
- **Microstructure Score**: Combined metric (-100 to +100) indicating market regime

### �📊 Technical Analysis
- **Market regime detection**: Identifies trending, ranging, or volatile market conditions
- **Technical confluence**: RSI, Moving Average alignment, volatility analysis
- **Prediction rationale**: AI-generated explanations for forecast decisions

### 📉 Visualization
- **Interactive charts**: Toggle between Line and Candlestick views
- **Dark institutional theme**: Professional-grade chart styling
- **Real-time data**: Fetches live stock data from Yahoo Finance
- **Portfolio analytics**: Current price, daily/weekly/monthly changes, volatility metrics

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/HnM_Stock_App.git
   cd HnM_Stock_App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:8501`

---

## 📁 Project Structure

```
market genesis/
├── app.py                 # Main Streamlit application (tabbed interface)
├── train.py               # Model training utilities
├── requirements.txt       # Python dependencies
├── EQUITY_L.csv           # NSE equity symbols list
├── models/                # Trained model files (.joblib)
├── data/                  # Data storage and cache
└── utils/
    ├── analytics.py       # Technical analysis & rationale generation
    ├── backtest.py        # Backtesting engine
    ├── charts.py          # Plotly chart creation (institutional dark theme)
    ├── config.py          # Configuration settings
    ├── data_loader.py     # Stock data fetching (daily + intraday)
    ├── features.py        # Feature engineering
    ├── helpers.py         # Utility functions
    ├── microstructure.py  # Volume Profile, Order Book, Order Flow
    ├── regime.py          # Market regime detection
    └── structure.py       # BOS/CHoCH, Order Blocks, FVGs, Liquidity
```

---

## 📖 Usage Guide

### Training a Model

1. Select a stock from the dropdown (e.g., RELIANCE)
2. Choose a date range for training data
3. Click **🚀 Train Model**
4. Wait for training to complete (includes single, multi-horizon, and quantile models)

### Making Predictions

1. Ensure a model is trained for your selected stock (✅ Model ready)
2. Adjust the **Prediction Days** slider (1-30 days)
3. View predictions on the interactive chart
4. Check **Multi-Horizon Predictions with Confidence Bands** for detailed forecasts

### Reading the Analysis

- **Market Regime**: Current market condition (Trending 🟢, Ranging 🟡, High Volatility 🔴)
- **Technical Confluence**: RSI, MA alignment, volatility signals
- **Prediction Rationale**: Key factors driving the forecast

---

## 📊 Model Details

### Prediction Horizons
| Horizon | Description |
|---------|-------------|
| **1-Day** | Short-term trading signals |
| **5-Day** | Weekly outlook |
| **20-Day** | Monthly trend projection |

### Confidence Levels
| Level | Regime | Reliability |
|-------|--------|-------------|
| 🟢 High | Trending | Most reliable predictions |
| 🟡 Medium | Ranging | Moderate reliability |
| 🟠 Low | Transitional | Exercise caution |
| 🔴 Very Low | High Volatility | Wide uncertainty bands |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct your own research and consult with financial professionals before making investment decisions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for stock data
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Plotly](https://plotly.com/) for interactive visualizations
