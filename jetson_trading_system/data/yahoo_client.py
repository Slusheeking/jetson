"""
Yahoo Finance API Client
Simple client for market movers: most active, top gainers, top losers, trending
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging

from ..utils.logger import get_data_logger

class YahooFinanceClient:
    """
    Yahoo Finance API client for market data endpoints
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """Initialize Yahoo Finance client"""
        self.session = session
        self.logger = get_data_logger()
        
        # Yahoo Finance API endpoints
        self.base_url = "https://query1.finance.yahoo.com/v1/finance"
        
        # Headers to mimic browser requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
        return self.session
    
    async def _make_request(self, url: str) -> Dict:
        """Make HTTP request to Yahoo Finance API"""
        try:
            session = await self._get_session()
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    self.logger.error(f"Yahoo API error {response.status}: {url}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error making request to {url}: {e}")
            return {}
    
    async def get_trending_tickers(self, region: str = "US") -> List[Dict]:
        """
        Get trending tickers from Yahoo Finance
        
        Args:
            region: Market region (US, GB, etc.)
            
        Returns:
            List of trending ticker data
        """
        try:
            url = f"{self.base_url}/trending/{region}"
            data = await self._make_request(url)
            
            trending_stocks = []
            
            if 'finance' in data and 'result' in data['finance']:
                results = data['finance']['result']
                if results and 'quotes' in results[0]:
                    for quote in results[0]['quotes']:
                        stock_data = {
                            'symbol': quote.get('symbol', ''),
                            'name': quote.get('longName', quote.get('shortName', '')),
                            'price': quote.get('regularMarketPrice', 0),
                            'change': quote.get('regularMarketChange', 0),
                            'change_percent': quote.get('regularMarketChangePercent', 0),
                            'volume': quote.get('regularMarketVolume', 0),
                            'market_cap': quote.get('marketCap'),
                            'source': 'yahoo_trending'
                        }
                        trending_stocks.append(stock_data)
            
            self.logger.info(f"Retrieved {len(trending_stocks)} trending stocks")
            return trending_stocks
            
        except Exception as e:
            self.logger.error(f"Error getting trending tickers: {e}")
            return []
    
    async def get_most_active(self, count: int = 25) -> List[Dict]:
        """
        Get most active stocks from Yahoo Finance
        
        Args:
            count: Number of stocks to return
            
        Returns:
            List of most active stocks
        """
        try:
            url = f"{self.base_url}/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=most_actives&count={count}"
            data = await self._make_request(url)
            
            active_stocks = []
            
            if 'finance' in data and 'result' in data['finance']:
                results = data['finance']['result']
                if results and 'quotes' in results[0]:
                    for quote in results[0]['quotes']:
                        stock_data = {
                            'symbol': quote.get('symbol', ''),
                            'name': quote.get('longName', quote.get('shortName', '')),
                            'price': quote.get('regularMarketPrice', 0),
                            'change': quote.get('regularMarketChange', 0),
                            'change_percent': quote.get('regularMarketChangePercent', 0),
                            'volume': quote.get('regularMarketVolume', 0),
                            'avg_volume': quote.get('averageDailyVolume3Month', 0),
                            'market_cap': quote.get('marketCap'),
                            'source': 'yahoo_most_active'
                        }
                        active_stocks.append(stock_data)
            
            self.logger.info(f"Retrieved {len(active_stocks)} most active stocks")
            return active_stocks
            
        except Exception as e:
            self.logger.error(f"Error getting most active stocks: {e}")
            return []
    
    async def get_top_gainers(self, count: int = 25) -> List[Dict]:
        """
        Get top gaining stocks from Yahoo Finance
        
        Args:
            count: Number of stocks to return
            
        Returns:
            List of top gaining stocks
        """
        try:
            url = f"{self.base_url}/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=day_gainers&count={count}"
            data = await self._make_request(url)
            
            gainer_stocks = []
            
            if 'finance' in data and 'result' in data['finance']:
                results = data['finance']['result']
                if results and 'quotes' in results[0]:
                    for quote in results[0]['quotes']:
                        stock_data = {
                            'symbol': quote.get('symbol', ''),
                            'name': quote.get('longName', quote.get('shortName', '')),
                            'price': quote.get('regularMarketPrice', 0),
                            'change': quote.get('regularMarketChange', 0),
                            'change_percent': quote.get('regularMarketChangePercent', 0),
                            'volume': quote.get('regularMarketVolume', 0),
                            'market_cap': quote.get('marketCap'),
                            'source': 'yahoo_gainers'
                        }
                        gainer_stocks.append(stock_data)
            
            self.logger.info(f"Retrieved {len(gainer_stocks)} top gainers")
            return gainer_stocks
            
        except Exception as e:
            self.logger.error(f"Error getting top gainers: {e}")
            return []
    
    async def get_top_losers(self, count: int = 25) -> List[Dict]:
        """
        Get top losing stocks from Yahoo Finance
        
        Args:
            count: Number of stocks to return
            
        Returns:
            List of top losing stocks
        """
        try:
            url = f"{self.base_url}/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=day_losers&count={count}"
            data = await self._make_request(url)
            
            loser_stocks = []
            
            if 'finance' in data and 'result' in data['finance']:
                results = data['finance']['result']
                if results and 'quotes' in results[0]:
                    for quote in results[0]['quotes']:
                        stock_data = {
                            'symbol': quote.get('symbol', ''),
                            'name': quote.get('longName', quote.get('shortName', '')),
                            'price': quote.get('regularMarketPrice', 0),
                            'change': quote.get('regularMarketChange', 0),
                            'change_percent': quote.get('regularMarketChangePercent', 0),
                            'volume': quote.get('regularMarketVolume', 0),
                            'market_cap': quote.get('marketCap'),
                            'source': 'yahoo_losers'
                        }
                        loser_stocks.append(stock_data)
            
            self.logger.info(f"Retrieved {len(loser_stocks)} top losers")
            return loser_stocks
            
        except Exception as e:
            self.logger.error(f"Error getting top losers: {e}")
            return []
    
    async def get_all_market_movers(self, count_per_category: int = 50) -> Dict[str, List[Dict]]:
        """
        Get all market movers: trending, most active, gainers, losers
        
        Args:
            count_per_category: Number of stocks per category
            
        Returns:
            Dictionary with all market mover categories
        """
        try:
            # Run all requests concurrently
            tasks = [
                self.get_trending_tickers('US'),
                self.get_most_active(count_per_category),
                self.get_top_gainers(count_per_category),
                self.get_top_losers(count_per_category)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_movers = {
                'trending': results[0] if not isinstance(results[0], Exception) else [],
                'most_active': results[1] if not isinstance(results[1], Exception) else [],
                'gainers': results[2] if not isinstance(results[2], Exception) else [],
                'losers': results[3] if not isinstance(results[3], Exception) else []
            }
            
            total_symbols = sum(len(category) for category in market_movers.values())
            self.logger.info(f"Retrieved {total_symbols} total market mover symbols")
            
            return market_movers
            
        except Exception as e:
            self.logger.error(f"Error getting all market movers: {e}")
            return {'trending': [], 'most_active': [], 'gainers': [], 'losers': []}
    
    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get quote data for a single symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Quote data or None if error
        """
        try:
            url = f"{self.base_url}/quote?symbols={symbol}&formatted=true&lang=en-US&region=US"
            data = await self._make_request(url)
            
            if 'quoteResponse' in data and 'result' in data['quoteResponse']:
                results = data['quoteResponse']['result']
                if results:
                    quote = results[0]
                    return {
                        'symbol': quote.get('symbol', ''),
                        'name': quote.get('longName', quote.get('shortName', '')),
                        'price': quote.get('regularMarketPrice', 0),
                        'change': quote.get('regularMarketChange', 0),
                        'change_percent': quote.get('regularMarketChangePercent', 0),
                        'volume': quote.get('regularMarketVolume', 0),
                        'market_cap': quote.get('marketCap'),
                        'source': 'yahoo_quote'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_yahoo_client():
        client = YahooFinanceClient()
        
        try:
            # Test all market movers
            market_movers = await client.get_all_market_movers(count_per_category=10)
            
            print("=== YAHOO FINANCE MARKET MOVERS ===")
            
            for category, stocks in market_movers.items():
                print(f"\n{category.upper()} ({len(stocks)} stocks):")
                for stock in stocks[:5]:  # Show top 5 in each category
                    print(f"  {stock['symbol']}: ${stock['price']:.2f} ({stock['change_percent']:.2f}%)")
            
            # Test individual quote
            quote = await client.get_quote('AAPL')
            if quote:
                print(f"\nAAPL Quote: ${quote['price']:.2f} ({quote['change_percent']:.2f}%)")
        
        finally:
            await client.close()
    
    asyncio.run(test_yahoo_client())