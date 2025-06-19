"""
Trading Strategy Parameters
Based on ML4Trading proven methodologies with Jetson optimizations
"""

import os
from datetime import timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class TradingParams:
    """Core trading strategy configuration with environment variable support"""
    
    @classmethod
    def get_env_var(cls, key, default=None):
        """Get environment variable with optional default"""
        return os.getenv(key, default)
    
    @classmethod
    def get_env_bool(cls, key, default=False):
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @classmethod
    def get_env_int(cls, key, default=0):
        """Get integer environment variable"""
        try:
            return int(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def get_env_float(cls, key, default=0.0):
        """Get float environment variable"""
        try:
            return float(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
    
    # Capital Management - with environment overrides
    INITIAL_CAPITAL = property(lambda self: self.get_env_float('INITIAL_CAPITAL', 25000.0))
    MAX_POSITIONS = 20  # Maximum concurrent positions
    POSITION_SIZE_PCT = 5.0  # 5% of capital per position
    DEFAULT_POSITION_SIZE = 2500.0  # Default $2,500 per position
    CASH_BUFFER_PCT = 20.0  # 20% minimum cash buffer
    
    # Risk Management - with environment overrides
    STOP_LOSS_PCT = property(lambda self: self.get_env_float('STOP_LOSS_PCT', 5.0))  # 5% stop loss
    PROFIT_TARGET_PCT = 10.0  # +10% profit target
    MAX_HOLD_DAYS = 10  # Maximum holding period
    MAX_SECTOR_EXPOSURE = property(lambda self: self.get_env_float('MAX_SECTOR_EXPOSURE', 0.30))  # 30% max sector exposure
    MAX_PORTFOLIO_RISK = property(lambda self: self.get_env_float('MAX_PORTFOLIO_RISK', 0.15))  # 15% max portfolio risk
    MAX_POSITION_SIZE_PCT = property(lambda self: self.get_env_float('MAX_POSITION_SIZE_PCT', 0.05))  # 5% max position size
    DAILY_LOSS_LIMIT_PCT = 1.0  # 1% daily loss limit
    
    # Trading Cycle - with environment overrides
    TRADING_CYCLE_SECONDS = property(lambda self: self.get_env_int('TRADING_CYCLE_SECONDS', 30))
    KELLY_LOOKBACK_DAYS = property(lambda self: self.get_env_int('KELLY_LOOKBACK_DAYS', 60))
    
    # Paper Trading - with environment overrides
    PAPER_TRADING = property(lambda self: self.get_env_bool('PAPER_TRADING', True))
    
    # Dynamic Symbol Discovery Configuration
    SYMBOL_DISCOVERY_SOURCES = {
        'polygon_gainers': {
            'enabled': True,
            'limit': 25,
            'min_volume': 1_000_000,
            'weight': 0.15
        },
        'polygon_losers': {
            'enabled': True,
            'limit': 25,
            'min_volume': 1_000_000,
            'weight': 0.10
        },
        'polygon_most_active': {
            'enabled': True,
            'limit': 50,
            'min_volume': 5_000_000,
            'weight': 0.20
        },
        'yahoo_gainers': {
            'enabled': True,
            'limit': 25,
            'weight': 0.15
        },
        'yahoo_losers': {
            'enabled': True,
            'limit': 25,
            'weight': 0.10
        },
        'yahoo_most_active': {
            'enabled': True,
            'limit': 50,
            'weight': 0.20
        },
        'yahoo_trending': {
            'enabled': True,
            'limit': 20,
            'weight': 0.05
        },
        'unusual_volume': {
            'enabled': True,
            'volume_threshold_multiplier': 2.0,  # 2x average volume
            'limit': 30,
            'weight': 0.05
        }
    }
    
    # Core Market Indices (Always included for market context)
    CORE_MARKET_SYMBOLS = [
        'SPY',  # S&P 500 - broad market
        'QQQ',  # Nasdaq 100 - tech heavy
        'IWM',  # Russell 2000 - small cap context
        'VIX',  # Volatility index - market fear
    ]
    
    # Symbol Universe Filters
    SYMBOL_FILTERS = {
        'min_price': 5.0,           # Minimum $5 stock price
        'max_price': 1000.0,        # Maximum $1000 stock price
        'min_volume': 1_000_000,    # Minimum $1M daily volume
        'min_market_cap': 100_000_000,  # $100M minimum market cap
        'max_market_cap': 500_000_000_000,  # $500B maximum market cap
        'exclude_penny_stocks': True,
        'exclude_otc': True,
        'include_etfs': True,
        'max_symbols': 200,         # Maximum symbols in universe
    }
    
    # Symbol Discovery Schedule
    DISCOVERY_SCHEDULE = {
        'premarket_scan': (8, 0),   # 8:00 AM - premarket scan
        'market_open_scan': (9, 35), # 9:35 AM - post-open scan
        'midday_scan': (12, 0),     # 12:00 PM - midday momentum
        'afternoon_scan': (14, 0),  # 2:00 PM - afternoon scan
        'update_interval_minutes': 30,  # Update every 30 minutes during market hours
    }
    
    # Performance Targets
    TARGET_WIN_RATE_PCT = 60.0  # >60% win rate target
    TARGET_ANNUAL_RETURN_PCT = 17.5  # 15-20% annual return target
    TARGET_SHARPE_RATIO = 1.0  # >1.0 Sharpe ratio
    MAX_DRAWDOWN_PCT = 15.0  # <15% maximum drawdown
    
    # ML Model Parameters
    MODEL_LOOKBACK_DAYS = 252  # 1 year training data
    PREDICTION_HORIZON_DAYS = 5  # 5-day return predictions
    MODEL_UPDATE_FREQUENCY_HOURS = 24  # Daily model updates
    MIN_PREDICTION_CONFIDENCE = 0.55  # Minimum prediction confidence
    MIN_PROFIT_THRESHOLD = 0.005  # 0.5% minimum return threshold for positive label
    
    # Technical Indicators Configuration
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    SMA_SHORT_PERIOD = 20
    SMA_LONG_PERIOD = 50
    ATR_PERIOD = 14
    STOCH_K_PERIOD = 14
    STOCH_D_PERIOD = 3
    WILLIAMS_R_PERIOD = 14
    
    # Universe Configuration
    MIN_PRICE = 10.0  # Minimum stock price
    MIN_VOLUME = 5_000_000  # Minimum $5M daily volume
    MIN_MARKET_CAP = 2_000_000_000  # $2B minimum market cap
    MAX_MARKET_CAP = 10_000_000_000  # $10B maximum market cap
    
    # Market Context Indices
    MARKET_INDICES = [
        'SPY',  # S&P 500 - broad market
        'QQQ',  # Nasdaq 100 - tech heavy
        'IWM',  # Russell 2000 - small cap context
        'VIX',  # Volatility index - market fear
    ]
    
    # ETF Categories to Include
    SECTOR_ETFS = [
        'XLF',  # Financial Select Sector
        'XLK',  # Technology Select Sector
        'XLE',  # Energy Select Sector
        'XLV',  # Health Care Select Sector
        'XLI',  # Industrial Select Sector
        'XLP',  # Consumer Staples Select Sector
        'XLY',  # Consumer Discretionary Select Sector
        'XLU',  # Utilities Select Sector
        'XLB',  # Materials Select Sector
        'XLRE', # Real Estate Select Sector
        'XLC',  # Communication Services Select Sector
    ]
    
    BROAD_MARKET_ETFS = [
        'SPY',  # SPDR S&P 500 ETF
        'QQQ',  # Invesco QQQ Trust
        'IWM',  # iShares Russell 2000 ETF
        'VTI',  # Vanguard Total Stock Market ETF
        'VTV',  # Vanguard Value ETF
        'VUG',  # Vanguard Growth ETF
    ]
    
    # Trading Schedule (Market Hours Eastern Time)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0
    
    # Signal Generation Timing
    SIGNAL_GENERATION_TIMES = [
        (9, 35),   # 5 minutes after market open
        (10, 0),   # 10:00 AM
        (12, 0),   # 12:00 PM (lunch)
        (14, 0),   # 2:00 PM
        (15, 30),  # 30 minutes before close
    ]
    
    # Data Update Schedule
    EOD_DATA_UPDATE_TIME = (17, 0)  # 5:00 PM
    PREMARKET_UPDATE_TIME = (8, 0)   # 8:00 AM
    
    @classmethod
    def get_position_size(cls, capital: float) -> float:
        """Calculate position size based on current capital"""
        return capital * (cls.POSITION_SIZE_PCT / 100.0)
    
    @classmethod
    def get_max_portfolio_value(cls, capital: float) -> float:
        """Calculate maximum portfolio value (excluding cash buffer)"""
        return capital * ((100.0 - cls.CASH_BUFFER_PCT) / 100.0)
    
    @classmethod
    def get_daily_loss_limit(cls, capital: float) -> float:
        """Calculate daily loss limit in dollars"""
        return capital * (cls.DAILY_LOSS_LIMIT_PCT / 100.0)

class BacktestConfig:
    """Backtesting specific configuration"""
    
    # Backtest Period
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    
    # Transaction Costs
    COMMISSION_PER_TRADE = 0.0  # Commission-free with Alpaca
    BID_ASK_SPREAD_PCT = 0.05  # 5 basis points spread
    SLIPPAGE_PCT = 0.02  # 2 basis points slippage
    
    # Benchmark
    BENCHMARK_SYMBOL = "SPY"
    
    # Cross-Validation
    CV_FOLDS = 5
    TRAIN_PERIOD_MONTHS = 12  # 1 year training
    TEST_PERIOD_MONTHS = 3   # 3 months testing
    WALK_FORWARD_MONTHS = 1  # 1 month walk forward

class DataConfig:
    """Data source and processing configuration"""
    
    # Polygon API Configuration
    POLYGON_BASE_URL = "https://api.polygon.io"
    POLYGON_RATE_LIMIT_PER_MINUTE = 1000  # Adjust based on plan
    
    # Alpaca API Configuration
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading initially
    ALPACA_DATA_URL = "https://data.alpaca.markets"
    
    # Database Configuration
    DATABASE_PATH = "./data/trading_system.db"
    BACKUP_INTERVAL_HOURS = 6  # Backup every 6 hours
    
    # Cache Configuration
    CACHE_EXPIRY_MINUTES = 5  # Cache data for 5 minutes
    MAX_CACHE_SIZE_MB = 512  # 512MB cache limit
    
    # Data Quality Checks
    MAX_MISSING_DATA_PCT = 5.0  # Max 5% missing data
    MIN_DATA_POINTS = 100  # Minimum data points for analysis
    OUTLIER_THRESHOLD_STD = 3.0  # 3 standard deviation outlier threshold

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'technical_indicators': [
        'rsi_14',
        'macd_line',
        'macd_signal',
        'bb_upper',
        'bb_lower',
        'bb_ratio',
        'sma_ratio_20',
        'volume_ratio_20',
        'atr_14',
        'stoch_k_14',
        'williams_r_14'
    ],
    'market_context_features': [
        'spy_rsi',
        'spy_sma_ratio',
        'qqq_rsi',
        'qqq_sma_ratio',
        'iwm_rsi',
        'iwm_sma_ratio',
        'vix_level',
        'vix_change'
    ],
    'lag_features': [1, 2, 3, 5],  # 1, 2, 3, 5-day lags
    'rolling_features': [5, 10, 20],  # 5, 10, 20-day rolling windows
}

# Alias for backward compatibility (many files expect TradingConfig)
TradingConfig = TradingParams