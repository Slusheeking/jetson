"""
Jetson Trading System
Production-ready ML4Trading system optimized for NVIDIA Jetson Orin 16GB
"""

__version__ = "1.0.0"
__author__ = "Jetson Trading Team"
__description__ = "ML4Trading system optimized for NVIDIA Jetson Orin edge computing"

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingParams

__all__ = [
    "JetsonConfig",
    "TradingParams"
]
