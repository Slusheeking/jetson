"""Utilities module for Jetson Trading System"""

from jetson_trading_system.utils.jetson_utils import JetsonMonitor
from jetson_trading_system.utils.database import TradingDatabase, trading_db
from jetson_trading_system.utils.logger import setup_logging, get_logger

# Import all functions from ml4t_utils module
from jetson_trading_system.utils.ml4t_utils import (
    MultipleTimeSeriesCV,
    information_coefficient,
    ic_statistics,
    directional_accuracy,
    calculate_returns,
    calculate_log_returns,
    winsorize_series,
    standardize_series,
    rolling_rank,
    cross_sectional_rank,
    neutralize_by_factor,
    PerformanceMetrics,
    format_time,
    get_business_days,
    align_data_to_trading_calendar,
    create_forward_returns,
    optimize_dataframe_memory,
    ML4TradingUtils
)

__all__ = [
    "JetsonMonitor",
    "TradingDatabase",
    "trading_db",
    "setup_logging",
    "get_logger",
    # ML4T utilities
    "MultipleTimeSeriesCV",
    "information_coefficient",
    "ic_statistics",
    "directional_accuracy",
    "calculate_returns",
    "calculate_log_returns",
    "winsorize_series",
    "standardize_series",
    "rolling_rank",
    "cross_sectional_rank",
    "neutralize_by_factor",
    "PerformanceMetrics",
    "format_time",
    "get_business_days",
    "align_data_to_trading_calendar",
    "create_forward_returns",
    "optimize_dataframe_memory",
    "ML4TradingUtils"
]
