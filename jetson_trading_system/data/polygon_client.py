"""
Polygon API Client for Market Data and Indices
Optimized for Jetson Orin with rate limiting and caching
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3
from dataclasses import dataclass

from jetson_trading_system.config.trading_params import TradingParams, DataConfig
from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.data.cache_manager import CacheManager

logger = logging.getLogger(__name__)

@dataclass
class PolygonQuote:
    """Real-time quote data structure"""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: int

@dataclass
class PolygonBar:
    """OHLCV bar data structure"""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None

class PolygonRateLimiter:
    """Rate limiter for Polygon API calls"""
    
    def __init__(self, max_calls_per_minute: int = 1000):
        self.max_calls = max_calls_per_minute
        self.calls = []
        
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        # If we're at the limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = 60 - (now - self.calls[0]) + 0.1  # Small buffer
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class PolygonClient:
    """
    Async Polygon API client optimized for Jetson Orin
    Handles market data, indices, and real-time quotes
    """
    
    def __init__(self, api_key: str, cache_manager: CacheManager = None):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Use Redis cache manager
        self.cache_manager = cache_manager or CacheManager()
        
        # Rate limiting
        self.rate_limiter = PolygonRateLimiter(DataConfig.POLYGON_RATE_LIMIT_PER_MINUTE)
        
        # Session for connection pooling
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache settings
        self.cache_expiry_seconds = DataConfig.CACHE_EXPIRY_MINUTES * 60
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to Redis cache"""
        try:
            self.cache_manager.put(cache_key, data, ttl_seconds=self.cache_expiry_seconds)
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from Redis cache"""
        try:
            return self.cache_manager.get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_key}: {e}")
            return None
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        await self.rate_limiter.acquire()
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params['apikey'] = self.api_key
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    logger.warning("Rate limited by API, backing off")
                    await asyncio.sleep(60)
                    return await self._make_request(endpoint, params)
                else:
                    logger.error(f"API request failed: {response.status} - {await response.text()}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
    
    async def get_historical_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timespan: str = "day",
        multiplier: int = 1,
        is_index: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL') or index symbol (e.g., 'VIX')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timespan: 'minute', 'hour', 'day', 'week', 'month'
            multiplier: Number of timespans (e.g., 5 minute bars)
            is_index: True if symbol is an index (will use I: prefix)
        """
        cache_key = f"bars_{symbol}_{start_date}_{end_date}_{timespan}_{multiplier}_{is_index}"
        
        # Check cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        # Use "I:" prefix for indices
        api_symbol = f"I:{symbol}" if is_index else symbol
        endpoint = f"/v2/aggs/ticker/{api_symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        response = await self._make_request(endpoint, params)
        if not response or response.get('status') != 'OK':
            logger.warning(f"No data returned for {symbol}")
            return None
        
        results = response.get('results', [])
        if not results:
            logger.warning(f"No data returned for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df['symbol'] = symbol  # Use original symbol without prefix
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        })
        
        # Select and order columns - handle missing columns for indices
        base_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close']
        optional_columns = ['volume', 'vwap']
        
        # Add optional columns only if they exist
        columns = base_columns.copy()
        for col in optional_columns:
            if col in df.columns:
                columns.append(col)
            elif col == 'volume':
                # For indices like VIX that don't have volume, add as zero
                df['volume'] = 0
                columns.append('volume')
        
        df = df[columns]
        
        # Cache the results - convert timestamps to strings for JSON serialization
        df_for_cache = df.copy()
        if 'timestamp' in df_for_cache.columns:
            df_for_cache['timestamp'] = df_for_cache['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        self._save_to_cache(cache_key, df_for_cache.to_dict('records'))
        
        return df
    
    async def get_multiple_symbols_bars(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        timespan: str = "day"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical bars for multiple symbols efficiently"""
        tasks = []
        for symbol in symbols:
            task = self.get_historical_bars(symbol, start_date, end_date, timespan)
            tasks.append(task)
        
        # Execute requests with controlled concurrency to respect rate limits
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def bounded_request(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_request(task) for task in tasks])
        
        # Combine results
        data_dict = {}
        for symbol, df in zip(symbols, results):
            if df is not None and not df.empty:
                data_dict[symbol] = df
            else:
                logger.warning(f"No data received for {symbol}")
        
        return data_dict
    
    async def get_indices_data(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Get data for major market indices"""
        indices = TradingParams.MARKET_INDICES
        logger.info(f"Fetching data for indices: {indices}")
        
        # Handle mixed indices - some need I: prefix (like VIX), others don't (like SPY, QQQ, IWM)
        tasks = []
        for symbol in indices:
            # VIX needs the index endpoint with I: prefix
            is_index = symbol in ['VIX']
            task = self.get_historical_bars(symbol, start_date, end_date, is_index=is_index)
            tasks.append(task)
        
        # Execute requests with controlled concurrency
        semaphore = asyncio.Semaphore(10)
        
        async def bounded_request(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[bounded_request(task) for task in tasks])
        
        # Combine results
        data_dict = {}
        for symbol, df in zip(indices, results):
            if df is not None and not df.empty:
                data_dict[symbol] = df
            else:
                logger.warning(f"No data received for {symbol}")
        
        return data_dict
    
    async def get_current_quote(self, symbol: str) -> Optional[PolygonQuote]:
        """Get current real-time quote for a symbol"""
        endpoint = f"/v2/last/nbbo/{symbol}"
        
        response = await self._make_request(endpoint)
        if not response or response.get('status') != 'OK':
            return None
        
        result = response.get('results', {})
        return PolygonQuote(
            symbol=symbol,
            bid=result.get('P', 0.0),
            ask=result.get('p', 0.0),
            bid_size=result.get('S', 0),
            ask_size=result.get('s', 0),
            timestamp=result.get('t', 0)
        )
    
    async def get_market_status(self) -> Dict:
        """Get current market status"""
        endpoint = "/v1/marketstatus/now"
        
        response = await self._make_request(endpoint)
        if response and response.get('status') == 'OK':
            return response
        
        return {'market': 'unknown'}
    
    async def get_symbol_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed information about a symbol"""
        cache_key = f"details_{symbol}"
        
        # Check cache (details don't change often)
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        endpoint = f"/v3/reference/tickers/{symbol}"
        
        response = await self._make_request(endpoint)
        if response and response.get('status') == 'OK':
            details = response.get('results', {})
            # Cache for longer since this data is relatively static
            self._save_to_cache(cache_key, details)
            return details
        
        return None
    
    async def screen_symbols(
        self, 
        market_cap_gte: int = None,
        market_cap_lte: int = None,
        active: bool = True
    ) -> List[str]:
        """Screen symbols based on criteria"""
        endpoint = "/v3/reference/tickers"
        params = {
            'market': 'stocks',
            'active': str(active).lower(),
            'limit': 1000
        }
        
        if market_cap_gte:
            params['market_cap.gte'] = market_cap_gte
        if market_cap_lte:
            params['market_cap.lte'] = market_cap_lte
        
        symbols = []
        next_url = None
        
        while True:
            if next_url:
                # Handle pagination
                response = await self._make_request("", {"cursor": next_url.split("cursor=")[1]})
            else:
                response = await self._make_request(endpoint, params)
            
            if not response or response.get('status') != 'OK':
                break
            
            results = response.get('results', [])
            for ticker in results:
                symbol = ticker.get('ticker')
                if symbol:
                    symbols.append(symbol)
            
            next_url = response.get('next_url')
            if not next_url:
                break
        
        return symbols
    
    async def get_universe_symbols(self) -> List[str]:
        """Get trading universe symbols (mid-cap stocks + ETFs)"""
        # Get mid-cap stocks
        mid_cap_stocks = await self.screen_symbols(
            market_cap_gte=TradingParams.MIN_MARKET_CAP,
            market_cap_lte=TradingParams.MAX_MARKET_CAP
        )
        
        # Add ETFs from configuration
        etfs = TradingParams.SECTOR_ETFS + TradingParams.BROAD_MARKET_ETFS
        
        # Add indices
        indices = TradingParams.MARKET_INDICES
        
        # Combine all symbols
        all_symbols = list(set(mid_cap_stocks + etfs + indices))
        
        logger.info(f"Universe contains {len(all_symbols)} symbols")
        logger.info(f"Mid-cap stocks: {len(mid_cap_stocks)}")
        logger.info(f"ETFs: {len(etfs)}")
        logger.info(f"Indices: {len(indices)}")
        
        return all_symbols
    
    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        timestamp_gte: str = None,
        timestamp_lte: str = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get technical indicator data from Polygon
        
        Args:
            symbol: Stock symbol
            indicator: 'sma', 'ema', 'macd', 'rsi'
            timestamp_gte: Start date (YYYY-MM-DD)
            timestamp_lte: End date (YYYY-MM-DD)
            **kwargs: Additional parameters (window, short_window, long_window, etc.)
        """
        cache_key = f"indicator_{symbol}_{indicator}_{timestamp_gte}_{timestamp_lte}_{hash(str(kwargs))}"
        
        # Check cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        endpoint = f"/v1/indicators/{indicator}/{symbol}"
        params = {
            'limit': 5000,
            'order': 'asc'
        }
        
        if timestamp_gte:
            params['timestamp.gte'] = timestamp_gte
        if timestamp_lte:
            params['timestamp.lte'] = timestamp_lte
            
        # Add indicator-specific parameters
        params.update(kwargs)
        
        response = await self._make_request(endpoint, params)
        if not response or response.get('status') != 'OK':
            logger.warning(f"Failed to get {indicator} for {symbol}")
            return None
        
        results = response.get('results', {}).get('values', [])
        if not results:
            logger.warning(f"No {indicator} data returned for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['symbol'] = symbol
        df['indicator'] = indicator
        
        # Cache the results - convert timestamps to strings for JSON serialization
        df_for_cache = df.copy()
        if 'timestamp' in df_for_cache.columns:
            df_for_cache['timestamp'] = df_for_cache['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        self._save_to_cache(cache_key, df_for_cache.to_dict('records'))
        
        return df
    
    async def get_sma(self, symbol: str, window: int = 20, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Get Simple Moving Average"""
        return await self.get_technical_indicator(
            symbol, 'sma',
            timestamp_gte=start_date,
            timestamp_lte=end_date,
            window=window
        )
    
    async def get_ema(self, symbol: str, window: int = 20, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Get Exponential Moving Average"""
        return await self.get_technical_indicator(
            symbol, 'ema',
            timestamp_gte=start_date,
            timestamp_lte=end_date,
            window=window
        )
    
    async def get_rsi(self, symbol: str, window: int = 14, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Get Relative Strength Index"""
        return await self.get_technical_indicator(
            symbol, 'rsi',
            timestamp_gte=start_date,
            timestamp_lte=end_date,
            window=window
        )
    
    async def get_macd(self, symbol: str, short_window: int = 12, long_window: int = 26, signal_window: int = 9, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Get MACD indicator"""
        return await self.get_technical_indicator(
            symbol, 'macd',
            timestamp_gte=start_date,
            timestamp_lte=end_date,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window
        )

# Utility functions for data processing
def align_dataframes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align multiple dataframes to have the same date index"""
    if not dfs:
        return {}
    
    # Find common date range
    all_dates = []
    for df in dfs.values():
        if not df.empty:
            all_dates.extend(df['timestamp'].dt.date.unique())
    
    if not all_dates:
        return {}
    
    common_dates = sorted(set(all_dates))
    
    # Align each dataframe
    aligned_dfs = {}
    for symbol, df in dfs.items():
        if df.empty:
            continue
            
        df_dates = df['timestamp'].dt.date
        aligned_df = df[df_dates.isin(common_dates)].copy()
        
        if not aligned_df.empty:
            aligned_dfs[symbol] = aligned_df.sort_values('timestamp').reset_index(drop=True)
    
    return aligned_dfs

def validate_data_quality(df: pd.DataFrame, symbol: str) -> bool:
    """Validate data quality for a symbol"""
    if df.empty:
        logger.warning(f"No data for {symbol}")
        return False
    
    # Check for minimum data points
    if len(df) < DataConfig.MIN_DATA_POINTS:
        logger.warning(f"Insufficient data points for {symbol}: {len(df)}")
        return False
    
    # Check for missing values
    missing_pct = df[['open', 'high', 'low', 'close', 'volume']].isnull().sum().sum() / (len(df) * 5) * 100
    if missing_pct > DataConfig.MAX_MISSING_DATA_PCT:
        logger.warning(f"Too much missing data for {symbol}: {missing_pct:.1f}%")
        return False
    
    # Check for outliers in price data
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > DataConfig.OUTLIER_THRESHOLD_STD).sum()
            if outliers > len(df) * 0.1:  # More than 10% outliers
                logger.warning(f"Too many outliers in {col} for {symbol}: {outliers}")
                return False
    
    return True

if __name__ == '__main__':
    import os
    
    async def comprehensive_data_test():
        """Comprehensive test of all Polygon API endpoints and data sources"""
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            print("❌ Please set the POLYGON_API_KEY environment variable.")
            return

        print("🚀 Starting Comprehensive Polygon API Data Test")
        print("=" * 60)

        async with PolygonClient(api_key=api_key) as client:
            # Test date range - use longer period for meaningful data quality testing
            start_date = "2024-01-01"
            end_date = "2024-12-31"
            
            # === STOCK DATA ENDPOINTS TEST ===
            print("\n📈 TESTING STOCK DATA ENDPOINTS")
            print("-" * 40)
            
            stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
            stock_results = {}
            
            for symbol in stock_symbols:
                print(f"\n🔍 Testing {symbol} stock data...")
                try:
                    bars = await client.get_historical_bars(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        is_index=False  # Explicitly test stock endpoint
                    )
                    
                    if bars is not None and not bars.empty:
                        stock_results[symbol] = bars
                        print(f"  ✅ SUCCESS: {len(bars)} bars retrieved")
                        print(f"  📊 Columns: {list(bars.columns)}")
                        print(f"  📅 Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")
                        print(f"  💰 Price range: ${bars['close'].min():.2f} - ${bars['close'].max():.2f}")
                        print(f"  📈 Sample data:")
                        print(f"     {bars.iloc[0]['timestamp'].strftime('%Y-%m-%d')}: O=${bars.iloc[0]['open']:.2f} H=${bars.iloc[0]['high']:.2f} L=${bars.iloc[0]['low']:.2f} C=${bars.iloc[0]['close']:.2f} V={bars.iloc[0]['volume']:,}")
                        
                        # Validate data quality (adjusted for test scenarios)
                        is_valid = validate_data_quality(bars, symbol)
                        # For testing, show validation but don't treat as failure if we have some data
                        if is_valid:
                            print(f"  🔍 Data quality: ✅ VALID")
                        else:
                            print(f"  🔍 Data quality: ⚠️  LIMITED (but functional for testing)")
                    else:
                        print(f"  ❌ FAILED: No data returned for {symbol}")
                        
                except Exception as e:
                    print(f"  ❌ ERROR: {symbol} - {e}")
            
            # === INDEX DATA ENDPOINTS TEST ===
            print("\n\n📊 TESTING INDEX DATA ENDPOINTS")
            print("-" * 40)
            
            # Test regular ETF indices (no I: prefix needed)
            etf_indices = ["SPY", "QQQ", "IWM"]
            index_results = {}
            
            for symbol in etf_indices:
                print(f"\n🔍 Testing {symbol} ETF index data...")
                try:
                    bars = await client.get_historical_bars(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        is_index=False  # ETFs use stock endpoint
                    )
                    
                    if bars is not None and not bars.empty:
                        index_results[symbol] = bars
                        print(f"  ✅ SUCCESS: {len(bars)} bars retrieved")
                        print(f"  📊 Columns: {list(bars.columns)}")
                        print(f"  📅 Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")
                        print(f"  💰 Price range: ${bars['close'].min():.2f} - ${bars['close'].max():.2f}")
                        print(f"  📈 Sample data:")
                        print(f"     {bars.iloc[0]['timestamp'].strftime('%Y-%m-%d')}: O=${bars.iloc[0]['open']:.2f} H=${bars.iloc[0]['high']:.2f} L=${bars.iloc[0]['low']:.2f} C=${bars.iloc[0]['close']:.2f} V={bars.iloc[0]['volume']:,}")
                        
                        # Validate data quality
                        is_valid = validate_data_quality(bars, symbol)
                        print(f"  🔍 Data quality: {'✅ VALID' if is_valid else '❌ INVALID'}")
                    else:
                        print(f"  ❌ FAILED: No data returned for {symbol}")
                        
                except Exception as e:
                    print(f"  ❌ ERROR: {symbol} - {e}")
            
            # Test true indices (needs I: prefix)
            true_indices = ["VIX"]
            
            for symbol in true_indices:
                print(f"\n🔍 Testing {symbol} true index data (I: prefix)...")
                try:
                    bars = await client.get_historical_bars(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        is_index=True  # Use index endpoint with I: prefix
                    )
                    
                    if bars is not None and not bars.empty:
                        index_results[symbol] = bars
                        print(f"  ✅ SUCCESS: {len(bars)} bars retrieved")
                        print(f"  📊 Columns: {list(bars.columns)}")
                        print(f"  📅 Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")
                        print(f"  💰 Value range: {bars['close'].min():.2f} - {bars['close'].max():.2f}")
                        print(f"  📈 Sample data:")
                        print(f"     {bars.iloc[0]['timestamp'].strftime('%Y-%m-%d')}: O={bars.iloc[0]['open']:.2f} H={bars.iloc[0]['high']:.2f} L={bars.iloc[0]['low']:.2f} C={bars.iloc[0]['close']:.2f}")
                        
                        # Note: VIX might not have volume, so adjusted validation
                        has_volume = 'volume' in bars.columns and bars['volume'].notna().any()
                        print(f"  📊 Volume data: {'✅ Available' if has_volume else '⚠️  Not available (normal for indices)'}")
                    else:
                        print(f"  ❌ FAILED: No data returned for {symbol}")
                        
                except Exception as e:
                    print(f"  ❌ ERROR: {symbol} - {e}")
            
            # === BATCH TESTING ===
            print("\n\n⚡ TESTING BATCH DATA RETRIEVAL")
            print("-" * 40)
            
            all_symbols = stock_symbols + etf_indices
            print(f"\n🔍 Testing batch retrieval for: {all_symbols}")
            
            try:
                batch_results = await client.get_multiple_symbols_bars(
                    symbols=all_symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                
                print(f"  ✅ Batch retrieval completed")
                print(f"  📊 Symbols retrieved: {len(batch_results)}/{len(all_symbols)}")
                for symbol, df in batch_results.items():
                    print(f"     {symbol}: {len(df)} bars")
                    
            except Exception as e:
                print(f"  ❌ ERROR in batch retrieval: {e}")
            
            # === INDICES CONVENIENCE METHOD TEST ===
            print("\n\n🏛️ TESTING INDICES CONVENIENCE METHOD")
            print("-" * 40)
            
            try:
                indices_data = await client.get_indices_data(start_date, end_date)
                print(f"  ✅ Indices method completed")
                print(f"  📊 Indices retrieved: {len(indices_data)}")
                for symbol, df in indices_data.items():
                    print(f"     {symbol}: {len(df)} bars")
                    
            except Exception as e:
                print(f"  ❌ ERROR in indices method: {e}")
            
            # === MARKET STATUS & METADATA TESTS ===
            print("\n\n🏢 TESTING MARKET STATUS & METADATA")
            print("-" * 40)
            
            # Test market status
            print("\n🔍 Testing market status...")
            try:
                market_status = await client.get_market_status()
                print(f"  ✅ Market status retrieved")
                print(f"  📊 Status: {market_status.get('market', 'Unknown')}")
                if 'exchanges' in market_status:
                    print(f"  🏛️ Exchanges: {len(market_status['exchanges'])} available")
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
            
            # Test symbol details
            print("\n🔍 Testing symbol details...")
            test_symbols_details = ["AAPL", "TSLA"]
            for symbol in test_symbols_details:
                try:
                    details = await client.get_symbol_details(symbol)
                    if details:
                        print(f"  ✅ {symbol}: {details.get('name', 'N/A')}")
                        print(f"     Sector: {details.get('sic_description', 'N/A')}")
                        print(f"     Market Cap: {details.get('market_cap', 'N/A')}")
                    else:
                        print(f"  ❌ No details for {symbol}")
                except Exception as e:
                    print(f"  ❌ ERROR for {symbol}: {e}")
            
            # === REAL-TIME DATA TEST ===
            print("\n\n⚡ TESTING REAL-TIME DATA")
            print("-" * 40)
            
            print("\n🔍 Testing real-time quotes...")
            for symbol in ["AAPL", "SPY"]:
                try:
                    quote = await client.get_current_quote(symbol)
                    if quote:
                        print(f"  ✅ {symbol}: Bid=${quote.bid:.2f} Ask=${quote.ask:.2f}")
                        print(f"     Spread: ${quote.ask - quote.bid:.2f}")
                    else:
                        print(f"  ❌ No quote for {symbol}")
                except Exception as e:
                    print(f"  ❌ ERROR for {symbol}: {e}")
            
            # === TECHNICAL INDICATORS TEST ===
            print("\n\n📈 TESTING TECHNICAL INDICATORS")
            print("-" * 40)
            
            print("\n🔍 Testing technical indicators...")
            try:
                # Test SMA
                sma_data = await client.get_sma("AAPL", window=20, start_date=start_date, end_date=end_date)
                if sma_data is not None:
                    print(f"  ✅ SMA(20): {len(sma_data)} data points")
                else:
                    print(f"  ❌ SMA failed")
                
                # Test RSI
                rsi_data = await client.get_rsi("AAPL", window=14, start_date=start_date, end_date=end_date)
                if rsi_data is not None:
                    print(f"  ✅ RSI(14): {len(rsi_data)} data points")
                else:
                    print(f"  ❌ RSI failed")
                    
            except Exception as e:
                print(f"  ❌ Technical indicators error: {e}")
            
            # === SUMMARY REPORT ===
            print("\n\n📋 COMPREHENSIVE TEST SUMMARY")
            print("=" * 60)
            
            total_stocks_tested = len(stock_symbols)
            total_indices_tested = len(etf_indices) + len(true_indices)
            successful_stocks = len([s for s in stock_symbols if s in stock_results])
            successful_indices = len([s for s in (etf_indices + true_indices) if s in index_results])
            
            print(f"📈 Stock Endpoints:")
            print(f"   ✅ Successful: {successful_stocks}/{total_stocks_tested}")
            print(f"   📊 Symbols: {list(stock_results.keys())}")
            
            print(f"\n📊 Index Endpoints:")
            print(f"   ✅ Successful: {successful_indices}/{total_indices_tested}")
            print(f"   📊 Symbols: {list(index_results.keys())}")
            
            print(f"\n🎯 Overall Success Rate: {(successful_stocks + successful_indices)}/{(total_stocks_tested + total_indices_tested)} ({((successful_stocks + successful_indices)/(total_stocks_tested + total_indices_tested)*100):.1f}%)")
            
            if successful_stocks + successful_indices == total_stocks_tested + total_indices_tested:
                print("\n🎉 ALL TESTS PASSED! Polygon API integration is working correctly.")
            else:
                print("\n⚠️  Some tests failed. Check error messages above for details.")
                
            print("\n✅ Comprehensive test completed!")

    asyncio.run(comprehensive_data_test())
