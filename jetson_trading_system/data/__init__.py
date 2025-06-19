"""Data module for Jetson Trading System"""

from jetson_trading_system.data.polygon_client import PolygonClient
from jetson_trading_system.data.data_pipeline import DataPipeline
from jetson_trading_system.data.cache_manager import CacheManager
from jetson_trading_system.data.symbol_discovery import SymbolDiscoveryEngine, SymbolCandidate, DiscoverySource
from jetson_trading_system.data.yahoo_client import YahooFinanceClient

# Export under both names for backward compatibility
PolygonDataClient = PolygonClient

__all__ = [
    "PolygonClient",
    "PolygonDataClient",  # Alias for backward compatibility
    "DataPipeline",
    "CacheManager",
    "SymbolDiscoveryEngine",
    "SymbolCandidate",
    "DiscoverySource",
    "YahooFinanceClient"
]
