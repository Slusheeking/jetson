# Jetson Trading System

Production-ready ML4Trading system optimized for NVIDIA Jetson Orin 16GB edge computing platform.

## Overview

This system implements a complete machine learning-based trading pipeline optimized for NVIDIA Jetson Orin hardware. It leverages 100 TOPS of AI performance, ARM64 architecture, and CUDA acceleration to run a fully standalone trading system locally.

### Key Features

- **Edge Computing**: Fully standalone system with no cloud dependencies
- **ML4Trading Methodology**: Proven algorithmic trading strategies
- **GPU Acceleration**: Optimized for Jetson Orin 16GB with CUDA support
- **Real-time Processing**: <100ms signal generation latency
- **Risk Management**: Comprehensive portfolio protection
- **Multiple Markets**: Support for stocks, ETFs, and major indices
- **Paper Trading**: Safe testing environment included

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Jetson Orin 16GB Edge System                    │
├─────────────────────────────────────────────────────────────────────┤
│  Data Ingestion → Feature Engineering → ML Pipeline → Signals      │
│       ↓                                                            │
│  Risk Management → Trade Execution → Portfolio Management          │
│       ↓                                                            │
│  Performance Monitoring ← Backtesting Engine                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

### NVIDIA Jetson Orin 16GB Specifications
- **CPU**: 8-core ARM64 Cortex-A78AE @ 2.2GHz
- **GPU**: 1024-core NVIDIA Ampere GPU
- **Memory**: 16GB LPDDR5
- **AI Performance**: 100 TOPS
- **Storage**: NVMe SSD (high-speed local storage)

### Minimum Requirements
- 8GB+ RAM (16GB recommended)
- 50GB+ free disk space
- Ubuntu 20.04+ or JetPack 5.0+
- Python 3.8+

## Quick Start

### 1. Installation

Clone the repository:
```bash
git clone <repository-url>
cd jetson-trading-system
```

Install dependencies:
```bash
python3 jetson_trading_system/scripts/install_dependencies.py
```

### 2. Configuration

Create environment file:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```bash
# Required API Keys
POLYGON_API_KEY=your_polygon_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Trading Configuration
PAPER_TRADING=true
INITIAL_CAPITAL=25000
```

### 3. System Check

Run diagnostic to verify setup:
```bash
python3 jetson_trading_system/scripts/run_system.py --mode diagnostic
```

### 4. Training Models

Train ML models on historical data:
```bash
python3 main.py --mode train --retrain-days 60
```

### 5. Backtesting

Test strategy performance:
```bash
python3 main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

### 6. Live Trading

Start paper trading:
```bash
python3 main.py --mode live
```

## Usage

### Command Line Interface

The main application supports multiple operating modes:

```bash
# Live trading mode
python3 main.py --mode live

# Backtesting mode
python3 main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31

# Model training mode
python3 main.py --mode train --symbols AAPL MSFT GOOGL --retrain-days 30

# System launcher with health checks
python3 jetson_trading_system/scripts/run_system.py --mode live
```

### System Components

#### Data Pipeline
- **Polygon API**: Real-time and historical market data
- **Local Cache**: High-performance data caching
- **SQLite Database**: Local data storage

#### Machine Learning
- **LightGBM**: GPU-accelerated gradient boosting
- **Feature Engineering**: 8 core technical indicators
- **Model Registry**: Versioned model management

#### Risk Management
- **Portfolio Limits**: Position sizing and exposure controls
- **Stop Loss**: Automatic loss protection
- **Kelly Criterion**: Optimal position sizing

#### Execution
- **Alpaca API**: Commission-free trade execution
- **Order Management**: Lifecycle tracking
- **Portfolio Monitoring**: Real-time P&L tracking

## Trading Strategy

### Technical Indicators (8 Core Factors)
1. **RSI(14)** - Relative Strength Index
2. **MACD** - Moving Average Convergence Divergence
3. **Bollinger Bands** - Price bands with standard deviation
4. **SMA Ratio** - Close/SMA(20) ratio
5. **Volume Ratio** - Volume/SMA(Volume, 20) ratio
6. **ATR(14)** - Average True Range (volatility)
7. **Stochastic %K** - Momentum oscillator
8. **Williams %R** - Momentum indicator

### Risk Management Rules
- **Starting Capital**: $25,000
- **Max Positions**: 20 stocks
- **Position Size**: 5% of capital per stock
- **Stop Loss**: -2%
- **Profit Target**: +3%
- **Max Hold Period**: 10 days
- **Max Sector Exposure**: 30%
- **Cash Buffer**: 20% minimum
- **Daily Loss Limit**: 1% of capital

### Trading Universe
- **US Mid-cap Equities**: $2B - $10B market cap
- **US ETFs**: Sector, broad market, and thematic
- **Major Indices**: SPY, QQQ, IWM, VIX for market context
- **Liquidity Filter**: Min $5M average daily volume
- **Price Filter**: >$10/share
- **Universe Size**: 400-600 securities

## Performance Targets

### Model Performance
- **Information Coefficient (IC)**: >0.05
- **IC t-statistic**: >2.0
- **Prediction Accuracy**: >52% directional accuracy

### Trading Performance
- **Annual Return**: 15-20%
- **Sharpe Ratio**: >1.0
- **Maximum Drawdown**: <15%
- **Win Rate**: >60%

### System Performance
- **99.5%+ uptime**
- **<100ms signal latency**
- **<12GB memory usage**
- **<25W power consumption**

## File Structure

```
jetson-trading-system/
├── main.py                          # Application entry point
├── README.md                        # This file
├── requirements_jetson.txt          # Dependencies
├── .env                            # Environment variables
├── jetson_trading_system/
│   ├── __init__.py
│   ├── config/                     # Configuration
│   │   ├── jetson_settings.py
│   │   └── trading_params.py
│   ├── data/                       # Data pipeline
│   │   ├── polygon_client.py
│   │   ├── data_pipeline.py
│   │   └── cache_manager.py
│   ├── features/                   # Feature engineering
│   │   ├── technical_indicators.py
│   │   ├── ml4t_factors.py
│   │   └── feature_engine.py
│   ├── models/                     # ML models
│   │   ├── lightgbm_trainer.py
│   │   ├── model_predictor.py
│   │   └── model_registry.py
│   ├── risk/                       # Risk management
│   │   ├── risk_manager.py
│   │   └── position_sizer.py
│   ├── execution/                  # Trade execution
│   │   ├── trading_engine.py
│   │   ├── order_manager.py
│   │   └── portfolio_tracker.py
│   ├── backtesting/               # Backtesting
│   │   ├── zipline_engine.py
│   │   └── performance_analyzer.py
│   ├── monitoring/                # System monitoring
│   │   ├── system_monitor.py
│   │   └── trading_monitor.py
│   ├── utils/                     # Utilities
│   │   ├── jetson_utils.py
│   │   ├── ml4t_utils.py
│   │   ├── database.py
│   │   └── logger.py
│   └── scripts/                   # Setup scripts
│       ├── setup_jetson.py
│       ├── install_dependencies.py
│       └── run_system.py
```

## Development

### Adding New Features

1. Create feature branch
2. Implement changes following existing patterns
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

### Testing

```bash
# Run system diagnostic
python3 jetson_trading_system/scripts/run_system.py --mode diagnostic

# Test individual components
python3 -m pytest tests/

# Run backtests for validation
python3 main.py --mode backtest --start-date 2023-01-01 --end-date 2023-03-31
```

### Monitoring

The system includes comprehensive monitoring:

- **Hardware**: GPU/CPU utilization, memory, temperature
- **Trading**: P&L, positions, risk metrics
- **System**: Uptime, latency, errors
- **Data**: API connectivity, data quality

## Troubleshooting

### Common Issues

**Memory Issues**
```bash
# Check memory usage
free -h

# Reduce cache size in .env
CACHE_MEMORY_MB=512
CACHE_DISK_MB=2048
```

**GPU Issues**
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
python3 -c "import cupy; print(cupy.cuda.device.Device().compute_capability)"
```

**API Connection Issues**
```bash
# Test API connectivity
python3 -c "
from jetson_trading_system.data.polygon_client import PolygonDataClient
import os
client = PolygonDataClient(os.getenv('POLYGON_API_KEY'))
print('API test:', client.test_connection())
"
```

### Performance Optimization

1. **Memory**: Adjust cache sizes in configuration
2. **CPU**: Modify parallel worker count
3. **GPU**: Enable/disable GPU acceleration based on workload
4. **Disk**: Use NVMe SSD for best performance

## Security

- **Local-only**: No cloud dependencies
- **Encrypted**: API keys stored securely
- **Isolated**: Sandboxed execution environment
- **Backed up**: Local backup procedures included

## Support

For issues and questions:

1. Check troubleshooting section
2. Run diagnostic mode for system health
3. Check logs in `logs/` directory
4. Review configuration in `.env` file

## License

[Add your license information here]

## Disclaimer

This software is for educational and research purposes. Trading involves risk and past performance does not guarantee future results. Use at your own risk.