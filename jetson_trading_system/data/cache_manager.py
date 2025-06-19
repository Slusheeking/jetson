"""
Cache Manager for Jetson Trading System
High-performance caching system using Redis for scalability and persistence
"""

import redis
import pickle
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.utils.logger import get_data_logger

@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_keys: int
    memory_usage_mb: float
    hit_rate: float
    total_hits: int
    total_misses: int
    redis_version: str
    uptime_days: int

class CacheManager:
    """
    Redis-based caching system for trading data.
    - Handles serialization of complex objects like DataFrames.
    - Manages TTLs for automatic data expiration.
    - Provides specialized methods for common trading data types.
    """

    def __init__(self, 
                 host: str = None, 
                 port: int = None, 
                 db: int = 0,
                 default_ttl_seconds: int = 3600):
        """
        Initialize the Redis-based cache manager.

        Connection details are retrieved from environment variables if not provided:
        - REDIS_HOST (default: 'localhost')
        - REDIS_PORT (default: 6379)

        Args:
            host (str, optional): Redis server host. Defaults to None.
            port (int, optional): Redis server port. Defaults to None.
            db (int, optional): Redis database number. Defaults to 0.
            default_ttl_seconds (int, optional): Default TTL for new cache entries. Defaults to 3600.
        """
        self.logger = get_data_logger()
        self.default_ttl = default_ttl_seconds

        # Get Redis connection details from environment or use defaults
        redis_host = host or JetsonConfig.get_env_var('REDIS_HOST', 'localhost')
        redis_port = port or JetsonConfig.get_env_int('REDIS_PORT', 6379)
        
        try:
            self.redis_client = redis.StrictRedis(
                host=redis_host, 
                port=redis_port, 
                db=db,
                decode_responses=False  # Store bytes to handle pickled objects
            )
            # Check connection
            self.redis_client.ping()
            self.logger.info(f"CacheManager connected to Redis at {redis_host}:{redis_port}")
        except redis.exceptions.ConnectionError as e:
            self.logger.error(f"Could not connect to Redis at {redis_host}:{redis_port}. Caching will be disabled. Error: {e}")
            self.redis_client = None

        # Statistics tracking
        self.hits = 0
        self.misses = 0

    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage in Redis."""
        # Use pickle for general Python objects, including pandas DataFrames
        return pickle.dumps(data)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data retrieved from Redis."""
        if data is None:
            return None
        return pickle.loads(data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get data from cache.

        Args:
            key (str): The cache key.
            default (Any, optional): Default value to return if key not found.

        Returns:
            Any: The cached data or the default value.
        """
        if not self.redis_client:
            self.misses += 1
            return default
            
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                self.hits += 1
                return self._deserialize(cached_data)
            else:
                self.misses += 1
                return default
        except Exception as e:
            self.logger.error(f"Error getting cache entry '{key}': {e}")
            self.misses += 1
            return default

    def put(self, key: str, data: Any, ttl_seconds: int = None) -> bool:
        """
        Store data in the cache.

        Args:
            key (str): The cache key.
            data (Any): The data to store.
            ttl_seconds (int, optional): Time-to-live in seconds. Uses default if None.

        Returns:
            bool: True if successfully cached, False otherwise.
        """
        if not self.redis_client:
            return False

        try:
            ttl = ttl_seconds or self.default_ttl
            serialized_data = self._serialize(data)
            self.redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            self.logger.error(f"Error storing cache entry '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        if not self.redis_client:
            return False
            
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            self.logger.error(f"Error deleting cache entry '{key}': {e}")
            return False

    def clear(self):
        """Clear all entries from the current Redis database."""
        if not self.redis_client:
            return
            
        try:
            self.redis_client.flushdb()
            self.logger.info("Cache (Redis DB) cleared.")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            
    def clear_symbol_cache(self, symbol: str):
        """Clear all cache entries related to a specific symbol."""
        if not self.redis_client:
            return

        try:
            # Find all keys related to the symbol and delete them
            keys_to_delete = []
            for key in self.redis_client.scan_iter(match=f"*:{symbol}"):
                keys_to_delete.append(key)
            
            if keys_to_delete:
                self.redis_client.delete(*keys_to_delete)
                self.logger.info(f"Cleared {len(keys_to_delete)} cache entries for symbol {symbol}")
        except Exception as e:
            self.logger.error(f"Error clearing symbol cache for '{symbol}': {e}")

    def get_cache_stats(self) -> Optional[CacheStats]:
        """Get comprehensive cache statistics from Redis."""
        if not self.redis_client:
            return None
            
        try:
            info = self.redis_client.info()
            total_requests = self.hits + self.misses
            
            return CacheStats(
                total_keys=info.get('db0', {}).get('keys', 0),
                memory_usage_mb=info.get('used_memory', 0) / (1024 * 1024),
                hit_rate=(self.hits / total_requests * 100) if total_requests > 0 else 0,
                total_hits=self.hits,
                total_misses=self.misses,
                redis_version=info.get('redis_version'),
                uptime_days=info.get('uptime_in_days', 0)
            )
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return None

    def shutdown(self):
        """Shutdown the cache manager and close the Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
                self.logger.info("CacheManager Redis connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing Redis connection: {e}")

    # Specialized methods for common trading data types
    def cache_price_data(self, symbol: str, data: Union[pd.DataFrame, Dict[str, Any]], ttl_seconds: int = 300):
        key = f"price_data:{symbol}"
        return self.put(key, data, ttl_seconds)

    def get_price_data(self, symbol: str) -> Optional[Union[pd.DataFrame, Dict[str, Any]]]:
        key = f"price_data:{symbol}"
        return self.get(key)
        
    def cache_technical_indicators(self, symbol: str, indicators: pd.DataFrame, ttl_seconds: int = 600):
        key = f"indicators:{symbol}"
        return self.put(key, indicators, ttl_seconds)

    def get_technical_indicators(self, symbol: str) -> Optional[pd.DataFrame]:
        key = f"indicators:{symbol}"
        return self.get(key)

    def cache_model_prediction(self, symbol: str, prediction: Dict[str, Any], ttl_seconds: int = 300):
        key = f"prediction:{symbol}"
        return self.put(key, prediction, ttl_seconds)

    def get_model_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = f"prediction:{symbol}"
        return self.get(key)

# Example usage
if __name__ == "__main__":
    import os
    
    print("--- Running CacheManager Redis Demo ---")
    
    # This demo requires a running Redis instance on localhost:6379
    # You can start one with: docker run -d -p 6379:6379 redis
    
    cache_manager = CacheManager()

    if not cache_manager.redis_client:
        print("\nCould not connect to Redis. Aborting demo.")
        print("Please ensure Redis is running on localhost:6379.")
    else:
        try:
            print("\n1. Clearing cache for a fresh start...")
            cache_manager.clear()
            
            # 2. Basic put/get
            print("\n2. Testing basic put/get...")
            cache_manager.put("my_key", "my_value", ttl_seconds=10)
            retrieved = cache_manager.get("my_key")
            print(f"Retrieved 'my_key': {retrieved}")
            assert retrieved == "my_value"

            # 3. DataFrame caching
            print("\n3. Testing with pandas DataFrame...")
            df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
            cache_manager.cache_price_data("TEST_DF", df)
            retrieved_df = cache_manager.get_price_data("TEST_DF")
            print("Retrieved DataFrame:")
            print(retrieved_df)
            assert isinstance(retrieved_df, pd.DataFrame)

            # 4. TTL expiration
            print("\n4. Testing TTL...")
            cache_manager.put("expire_me", "I will vanish", ttl_seconds=2)
            print("Waiting 3 seconds for key to expire...")
            time.sleep(3)
            expired = cache_manager.get("expire_me")
            print(f"Retrieved 'expire_me': {expired}")
            assert expired is None

            # 5. Get and print stats
            print("\n5. Cache Stats:")
            stats = cache_manager.get_cache_stats()
            if stats:
                print(json.dumps(asdict(stats), indent=2))
            
            # 6. Test symbol-specific clearing
            print("\n6. Clearing cache for symbol AAPL...")
            cache_manager.put("price_data:AAPL", "some_data")
            cache_manager.put("indicators:AAPL", "some_indicators")
            cache_manager.clear_symbol_cache("AAPL")
            assert cache_manager.get("price_data:AAPL") is None
            print("AAPL cache cleared successfully.")

        except Exception as e:
            print(f"\nAn error occurred during the demo: {e}")
        finally:
            cache_manager.shutdown()
            print("\n--- CacheManager Redis Demo Complete ---")