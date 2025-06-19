"""
Dynamic Symbol Discovery System
Discovers trading opportunities from market movers, gainers, losers, and volume leaders
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from jetson_trading_system.config.trading_params import TradingParams
from jetson_trading_system.utils.logger import get_data_logger
from jetson_trading_system.data.polygon_client import PolygonClient
from jetson_trading_system.data.yahoo_client import YahooFinanceClient
from jetson_trading_system.data.cache_manager import CacheManager

class DiscoverySource(Enum):
    """Symbol discovery sources"""
    POLYGON_GAINERS = "polygon_gainers"
    POLYGON_LOSERS = "polygon_losers"
    POLYGON_MOST_ACTIVE = "polygon_most_active"
    YAHOO_GAINERS = "yahoo_gainers"
    YAHOO_LOSERS = "yahoo_losers"
    YAHOO_MOST_ACTIVE = "yahoo_most_active"
    YAHOO_TRENDING = "yahoo_trending"
    UNUSUAL_VOLUME = "unusual_volume"
    MOMENTUM_BREAKOUT = "momentum_breakout"

@dataclass
class SymbolCandidate:
    """Symbol candidate with discovery metadata"""
    symbol: str
    source: DiscoverySource
    score: float
    price: float
    volume: int
    market_cap: Optional[float]
    change_percent: float
    discovery_time: datetime
    reason: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'source': self.source.value,
            'score': self.score,
            'price': self.price,
            'volume': self.volume,
            'market_cap': self.market_cap,
            'change_percent': self.change_percent,
            'discovery_time': self.discovery_time.isoformat(),
            'reason': self.reason
        }

class SymbolDiscoveryEngine:
    """
    Dynamic symbol discovery engine
    Finds trading opportunities from multiple market data sources
    """
    
    def __init__(self,
                 trading_params: TradingParams,
                 polygon_client: PolygonClient = None,
                 yahoo_client: YahooFinanceClient = None,
                 cache_manager: CacheManager = None
                 ):
        """
        Initialize symbol discovery engine
        
        Args:
            trading_params: Trading parameters and filters
            polygon_client: Polygon API client, optional.
            yahoo_client: Yahoo Finance API client, optional.
            cache_manager: Cache manager for storing results, optional.
        """
        self.trading_params = trading_params
        self.polygon_client = polygon_client or PolygonClient()
        self.yahoo_client = yahoo_client or YahooFinanceClient()
        self.cache_manager = cache_manager or CacheManager()
        self.logger = get_data_logger()
        
        # Discovery configuration
        self.discovery_config = trading_params.SYMBOL_DISCOVERY_SOURCES
        self.filters = trading_params.SYMBOL_FILTERS
        self.core_symbols = set(trading_params.CORE_MARKET_SYMBOLS)
        
        # Cache keys
        self.cache_prefix = "symbol_discovery"
        self.universe_cache_key = f"{self.cache_prefix}:universe"
        self.candidates_cache_key = f"{self.cache_prefix}:candidates"
        
    async def discover_symbols(self, max_symbols: int = None) -> List[SymbolCandidate]:
        """
        Discover trading symbols from multiple sources
        
        Args:
            max_symbols: Maximum number of symbols to return
            
        Returns:
            List of symbol candidates sorted by score
        """
        try:
            self.logger.info("Starting dynamic symbol discovery...")
            
            # Check cache first
            cached_candidates = self.cache_manager.get(self.candidates_cache_key)
            if cached_candidates:
                self.logger.info(f"Retrieved {len(cached_candidates)} candidates from cache")
                return cached_candidates[:max_symbols] if max_symbols else cached_candidates
            
            # Discover from all enabled sources
            all_candidates = []
            
            # Polygon market movers
            if self.discovery_config['polygon_gainers']['enabled']:
                gainers = await self._discover_polygon_gainers()
                all_candidates.extend(gainers)
                
            if self.discovery_config['polygon_losers']['enabled']:
                losers = await self._discover_polygon_losers()
                all_candidates.extend(losers)
                
            if self.discovery_config['polygon_most_active']['enabled']:
                most_active = await self._discover_polygon_most_active()
                all_candidates.extend(most_active)
                
            # Volume-based discovery
            if self.discovery_config['unusual_volume']['enabled']:
                unusual_volume = await self._discover_unusual_volume()
                all_candidates.extend(unusual_volume)
            
            # Filter and deduplicate candidates
            filtered_candidates = self._filter_and_rank_candidates(all_candidates)
            
            # Add core market symbols
            core_candidates = await self._get_core_market_candidates()
            filtered_candidates.extend(core_candidates)
            
            # Final ranking and limiting
            final_candidates = self._final_ranking(filtered_candidates, max_symbols)
            
            # Cache results
            self.cache_manager.put(self.candidates_cache_key, final_candidates, ttl_seconds=1800)  # 30 minutes
            
            self.logger.info(f"Discovered {len(final_candidates)} trading candidates")
            return final_candidates
            
        except Exception as e:
            self.logger.error(f"Error in symbol discovery: {e}")
            return await self._get_fallback_symbols()
    
    async def _discover_polygon_gainers(self) -> List[SymbolCandidate]:
        """Discover top gainers from Polygon"""
        try:
            config = self.discovery_config['polygon_gainers']
            
            # Use Polygon snapshot endpoint to get gainers
            gainers_data = await self.polygon_client.get_market_snapshot_gainers(
                include_otc=False
            )
            
            candidates = []
            for ticker_data in gainers_data.get('results', [])[:config['limit']]:
                try:
                    ticker = ticker_data.get('ticker', '')
                    value = ticker_data.get('value', {})
                    
                    if not value:
                        continue
                        
                    price = value.get('c', 0)  # Close price
                    volume = value.get('v', 0)  # Volume
                    change_pct = value.get('dp', 0)  # Daily change percent
                    
                    if volume < config['min_volume']:
                        continue
                        
                    candidate = SymbolCandidate(
                        symbol=ticker,
                        source=DiscoverySource.POLYGON_GAINERS,
                        score=change_pct * config['weight'],
                        price=price,
                        volume=volume,
                        market_cap=None,  # Would need additional API call
                        change_percent=change_pct,
                        discovery_time=datetime.now(),
                        reason=f"Top gainer: +{change_pct:.2f}%"
                    )
                    
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing gainer {ticker_data}: {e}")
                    continue
            
            self.logger.info(f"Found {len(candidates)} gainer candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error discovering gainers: {e}")
            return []
    
    async def _discover_polygon_losers(self) -> List[SymbolCandidate]:
        """Discover top losers from Polygon"""
        try:
            config = self.discovery_config['polygon_losers']
            
            # Use Polygon snapshot endpoint to get losers
            losers_data = await self.polygon_client.get_market_snapshot_losers(
                include_otc=False
            )
            
            candidates = []
            for ticker_data in losers_data.get('results', [])[:config['limit']]:
                try:
                    ticker = ticker_data.get('ticker', '')
                    value = ticker_data.get('value', {})
                    
                    if not value:
                        continue
                        
                    price = value.get('c', 0)  # Close price
                    volume = value.get('v', 0)  # Volume
                    change_pct = value.get('dp', 0)  # Daily change percent
                    
                    if volume < config['min_volume']:
                        continue
                        
                    # For losers, we might want to look for oversold bounces
                    # Score based on how oversold they are (negative change)
                    oversold_score = abs(change_pct) * config['weight']
                    
                    candidate = SymbolCandidate(
                        symbol=ticker,
                        source=DiscoverySource.POLYGON_LOSERS,
                        score=oversold_score,
                        price=price,
                        volume=volume,
                        market_cap=None,
                        change_percent=change_pct,
                        discovery_time=datetime.now(),
                        reason=f"Potential oversold bounce: {change_pct:.2f}%"
                    )
                    
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing loser {ticker_data}: {e}")
                    continue
            
            self.logger.info(f"Found {len(candidates)} loser candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error discovering losers: {e}")
            return []
    
    async def _discover_polygon_most_active(self) -> List[SymbolCandidate]:
        """Discover most active stocks from Polygon"""
        try:
            config = self.discovery_config['polygon_most_active']
            
            # Use Polygon snapshot endpoint to get most active
            active_data = await self.polygon_client.get_market_snapshot_direction('*')
            
            # Sort by volume to get most active
            if 'results' in active_data:
                sorted_by_volume = sorted(
                    active_data['results'], 
                    key=lambda x: x.get('value', {}).get('v', 0), 
                    reverse=True
                )
            else:
                sorted_by_volume = []
            
            candidates = []
            for ticker_data in sorted_by_volume[:config['limit']]:
                try:
                    ticker = ticker_data.get('ticker', '')
                    value = ticker_data.get('value', {})
                    
                    if not value:
                        continue
                        
                    price = value.get('c', 0)  # Close price
                    volume = value.get('v', 0)  # Volume
                    change_pct = value.get('dp', 0)  # Daily change percent
                    
                    if volume < config['min_volume']:
                        continue
                        
                    # Score based on volume and volatility
                    volume_score = (volume / 10_000_000) * config['weight']  # Normalize volume
                    volatility_bonus = abs(change_pct) * 0.1  # Bonus for volatility
                    total_score = volume_score + volatility_bonus
                    
                    candidate = SymbolCandidate(
                        symbol=ticker,
                        source=DiscoverySource.POLYGON_MOST_ACTIVE,
                        score=total_score,
                        price=price,
                        volume=volume,
                        market_cap=None,
                        change_percent=change_pct,
                        discovery_time=datetime.now(),
                        reason=f"High volume: {volume:,} shares"
                    )
                    
                    candidates.append(candidate)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing active stock {ticker_data}: {e}")
                    continue
            
            self.logger.info(f"Found {len(candidates)} most active candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error discovering most active: {e}")
            return []
    
    async def _discover_unusual_volume(self) -> List[SymbolCandidate]:
        """Discover stocks with unusual volume patterns"""
        try:
            config = self.discovery_config['unusual_volume']
            
            # This would require comparing current volume to historical averages
            # For now, we'll identify high-volume stocks from the current snapshot
            snapshot_data = await self.polygon_client.get_market_snapshot_direction('*')
            
            candidates = []
            if 'results' in snapshot_data:
                for ticker_data in snapshot_data['results']:
                    try:
                        ticker = ticker_data.get('ticker', '')
                        value = ticker_data.get('value', {})
                        
                        if not value:
                            continue
                            
                        price = value.get('c', 0)
                        volume = value.get('v', 0)
                        change_pct = value.get('dp', 0)
                        
                        # Simple unusual volume detection: very high volume
                        if volume > 50_000_000:  # 50M+ shares is unusual for most stocks
                            score = (volume / 100_000_000) * config['weight']
                            
                            candidate = SymbolCandidate(
                                symbol=ticker,
                                source=DiscoverySource.UNUSUAL_VOLUME,
                                score=score,
                                price=price,
                                volume=volume,
                                market_cap=None,
                                change_percent=change_pct,
                                discovery_time=datetime.now(),
                                reason=f"Unusual volume: {volume:,} shares"
                            )
                            
                            candidates.append(candidate)
                            
                    except Exception as e:
                        continue
            
            # Limit results
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)[:config['limit']]
            
            self.logger.info(f"Found {len(candidates)} unusual volume candidates")
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error discovering unusual volume: {e}")
            return []
    
    async def _get_core_market_candidates(self) -> List[SymbolCandidate]:
        """Get core market symbols that are always included"""
        candidates = []
        
        for symbol in self.core_symbols:
            try:
                # Get current quote for core symbols
                quote_data = await self.polygon_client.get_last_quote(symbol)
                
                if quote_data and 'results' in quote_data:
                    results = quote_data['results']
                    price = (results.get('P', 0) + results.get('p', 0)) / 2  # Average bid/ask
                    
                    candidate = SymbolCandidate(
                        symbol=symbol,
                        source=DiscoverySource.POLYGON_MOST_ACTIVE,  # Use as default
                        score=100.0,  # High score for core symbols
                        price=price,
                        volume=0,  # Would need additional call
                        market_cap=None,
                        change_percent=0,
                        discovery_time=datetime.now(),
                        reason="Core market index"
                    )
                    
                    candidates.append(candidate)
                    
            except Exception as e:
                self.logger.debug(f"Error getting core symbol {symbol}: {e}")
                continue
        
        return candidates
    
    def _filter_and_rank_candidates(self, candidates: List[SymbolCandidate]) -> List[SymbolCandidate]:
        """Filter and rank symbol candidates"""
        filtered = []
        seen_symbols = set()
        
        for candidate in candidates:
            try:
                # Skip duplicates
                if candidate.symbol in seen_symbols:
                    continue
                    
                # Apply filters
                if not self._passes_filters(candidate):
                    continue
                    
                filtered.append(candidate)
                seen_symbols.add(candidate.symbol)
                
            except Exception as e:
                self.logger.debug(f"Error filtering candidate {candidate.symbol}: {e}")
                continue
        
        # Sort by score
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        return filtered
    
    def _passes_filters(self, candidate: SymbolCandidate) -> bool:
        """Check if candidate passes all filters"""
        try:
            # Price filters
            if candidate.price < self.filters['min_price']:
                return False
            if candidate.price > self.filters['max_price']:
                return False
                
            # Volume filter
            if candidate.volume > 0 and candidate.volume < self.filters['min_volume']:
                return False
                
            # Exclude penny stocks
            if self.filters['exclude_penny_stocks'] and candidate.price < 5.0:
                return False
                
            # Basic symbol validation
            if not candidate.symbol or len(candidate.symbol) > 5:
                return False
                
            # Exclude obvious OTC symbols
            if self.filters['exclude_otc'] and ('.' in candidate.symbol or candidate.symbol.endswith('F')):
                return False
                
            return True
            
        except Exception as e:
            self.logger.debug(f"Error checking filters for {candidate.symbol}: {e}")
            return False
    
    def _final_ranking(self, candidates: List[SymbolCandidate], max_symbols: int = None) -> List[SymbolCandidate]:
        """Final ranking and selection of candidates"""
        try:
            # Remove duplicates and re-rank
            seen_symbols = set()
            unique_candidates = []
            
            for candidate in candidates:
                if candidate.symbol not in seen_symbols:
                    unique_candidates.append(candidate)
                    seen_symbols.add(candidate.symbol)
            
            # Sort by score
            unique_candidates.sort(key=lambda x: x.score, reverse=True)
            
            # Apply max symbols limit
            if max_symbols:
                max_limit = min(max_symbols, self.filters['max_symbols'])
            else:
                max_limit = self.filters['max_symbols']
                
            final_candidates = unique_candidates[:max_limit]
            
            return final_candidates
            
        except Exception as e:
            self.logger.error(f"Error in final ranking: {e}")
            return candidates[:100]  # Fallback limit
    
    async def _get_fallback_symbols(self) -> List[SymbolCandidate]:
        """Get fallback symbols if discovery fails"""
        fallback_symbols = ['SPY', 'QQQ', 'IWM', 'VIX', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        candidates = []
        
        for symbol in fallback_symbols:
            candidate = SymbolCandidate(
                symbol=symbol,
                source=DiscoverySource.POLYGON_MOST_ACTIVE,
                score=50.0,  # Medium score
                price=100.0,  # Placeholder
                volume=1_000_000,  # Placeholder
                market_cap=None,
                change_percent=0.0,
                discovery_time=datetime.now(),
                reason="Fallback symbol"
            )
            candidates.append(candidate)
        
        return candidates
    
    def get_current_universe(self) -> List[str]:
        """Get current trading universe as list of symbols"""
        candidates = self.cache_manager.get(self.candidates_cache_key, [])
        return [c.symbol for c in candidates]
    
    async def update_universe(self) -> List[str]:
        """Update the trading universe and return new symbol list"""
        candidates = await self.discover_symbols()
        universe = [c.symbol for c in candidates]
        
        # Cache the universe
        self.cache_manager.put(self.universe_cache_key, universe, ttl_seconds=1800)
        
        return universe
    
    def get_discovery_stats(self) -> Dict[str, int]:
        """Get statistics about symbol discovery"""
        candidates = self.cache_manager.get(self.candidates_cache_key, [])
        
        stats = {
            'total_candidates': len(candidates),
            'sources': {}
        }
        
        for candidate in candidates:
            source = candidate.source.value
            stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        return stats

# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv

    # The __main__ block remains the same, but imports are now cleaner
    # as they rely on the Python path being correctly set (e.g., via workspace settings)
    load_dotenv()

    async def test_discovery():
        print("--- Running SymbolDiscoveryEngine Demo ---")
        
        # Initialize components
        trading_params = TradingParams()
        cache_manager = CacheManager()
        
        # Use a real API key if available, otherwise it will gracefully fail
        polygon_api_key = os.getenv("POLYGON_API_KEY")
        if not polygon_api_key:
            print("\nWARNING: POLYGON_API_KEY not found. Discovery will rely on fallback symbols.")
            polygon_client = None
        else:
            print("\nFound POLYGON_API_KEY. Using Polygon client for discovery.")
            polygon_client = PolygonClient(api_key=polygon_api_key)

        yahoo_client = YahooFinanceClient()

        # Create discovery engine
        discovery_engine = SymbolDiscoveryEngine(
            trading_params=trading_params,
            polygon_client=polygon_client,
            yahoo_client=yahoo_client,
            cache_manager=cache_manager
        )

        # Discover symbols
        print("\nDiscovering symbols (this may take a moment)...")
        candidates = await discovery_engine.discover_symbols(max_symbols=25)
        
        if candidates:
            print(f"\nDiscovered {len(candidates)} trading candidates:")
            for i, candidate in enumerate(candidates[:15]):  # Show top 15
                print(f"  {i+1}. {candidate.symbol:<6s}: {candidate.reason} (Score: {candidate.score:.2f}, Source: {candidate.source.value})")
        else:
            print("\nNo candidates discovered. This may be due to missing API keys or market conditions.")

        # Get current universe
        universe = discovery_engine.get_current_universe()
        print(f"\nCurrent trading universe: {universe}")

        # Get discovery stats
        stats = discovery_engine.get_discovery_stats()
        print(f"\nDiscovery stats: {stats}")

        print("\n--- SymbolDiscoveryEngine Demo Complete ---")
        if polygon_client:
            await polygon_client.close()
        
        cache_manager.shutdown()

    asyncio.run(test_discovery())
