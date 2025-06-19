"""
ML4Trading Factor Analysis for Jetson Trading System
Advanced factor generation and analysis based on ML4Trading methodology
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingParams
from jetson_trading_system.utils.logger import get_data_logger
from jetson_trading_system.utils.database import TradingDatabase

class ML4TFactors:
    """
    ML4Trading factor analysis and generation
    Implements advanced factor models for systematic trading
    """
    
    def __init__(self):
        """Initialize ML4Trading factors analyzer"""
        self.logger = get_data_logger()
        self.db_manager = TradingDatabase()
        
        # Factor categories
        self.factor_categories = {
            'momentum': ['returns_1m', 'returns_3m', 'returns_6m', 'returns_12m', 'rsi_momentum'],
            'mean_reversion': ['rsi_mean_reversion', 'bollinger_position', 'price_vs_ma'],
            'volatility': ['realized_vol', 'garch_vol', 'vol_ratio', 'atr_normalized'],
            'volume': ['volume_momentum', 'volume_mean_reversion', 'vwap_ratio', 'money_flow'],
            'technical': ['macd_signal', 'stoch_signal', 'williams_signal', 'adx_trend'],
            'market_structure': ['market_correlation', 'sector_correlation', 'beta_factor'],
            'sentiment': ['vix_factor', 'term_structure', 'credit_spread']
        }
        
        # Factor calculation parameters
        self.factor_params = {
            'lookback_periods': [5, 10, 20, 60, 120, 252],
            'momentum_windows': [20, 60, 120, 252],
            'volatility_windows': [20, 60, 120],
            'correlation_window': 60,
            'min_observations': 30
        }
        
        self.logger.info("ML4TFactors initialized")
    
    def generate_all_factors(self, 
                           symbols: List[str], 
                           start_date: str, 
                           end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive factor dataset for all symbols
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for factor generation
            end_date: End date for factor generation
            
        Returns:
            Dictionary of symbol -> factors DataFrame
        """
        try:
            self.logger.info(f"Generating ML4T factors for {len(symbols)} symbols")
            
            all_factors = {}
            
            for symbol in symbols:
                try:
                    factors = self.generate_symbol_factors(symbol, start_date, end_date)
                    if factors is not None and not factors.empty:
                        all_factors[symbol] = factors
                        self.logger.info(f"Generated {len(factors.columns)} factors for {symbol}")
                    else:
                        self.logger.warning(f"No factors generated for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error generating factors for {symbol}: {e}")
            
            self.logger.info(f"Factor generation completed for {len(all_factors)} symbols")
            return all_factors
            
        except Exception as e:
            self.logger.error(f"Error in generate_all_factors: {e}")
            return {}
    
    def generate_symbol_factors(self, 
                               symbol: str, 
                               start_date: str, 
                               end_date: str) -> Optional[pd.DataFrame]:
        """
        Generate comprehensive factors for a single symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with all factors
        """
        try:
            # Get extended price data for factor calculation
            extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=400)).strftime('%Y-%m-%d')
            
            price_data = self.db_manager.get_market_data(symbol, extended_start, end_date)
            if price_data is None or len(price_data) < self.factor_params['min_observations']:
                return None
            
            # Get market data for relative factors
            market_data = self._get_market_data(extended_start, end_date)
            
            # Initialize factors DataFrame
            factors = pd.DataFrame(index=price_data.index)
            
            # Generate factor categories
            momentum_factors = self._generate_momentum_factors(price_data)
            mean_reversion_factors = self._generate_mean_reversion_factors(price_data)
            volatility_factors = self._generate_volatility_factors(price_data)
            volume_factors = self._generate_volume_factors(price_data)
            market_structure_factors = self._generate_market_structure_factors(price_data, market_data)

            # technical_factors are now handled by the TechnicalIndicators class
            
            # Combine all factors
            factor_dfs = [
                momentum_factors,
                mean_reversion_factors,
                volatility_factors,
                volume_factors,
                market_structure_factors
            ]
            
            for factor_df in factor_dfs:
                if factor_df is not None and not factor_df.empty:
                    factors = factors.join(factor_df, how='outer')

            # --- Composite Factor Generation ---
            composite_factors = pd.DataFrame(index=factors.index)
            for category, factor_names in self.factor_categories.items():
                # Get the factors that were actually generated for this category
                available_factors = [f for f in factor_names if f in factors.columns]
                if available_factors:
                    # Calculate the composite score (e.g., mean of the z-scores of the factors)
                    category_factors = self._clean_factors(factors[available_factors])
                    composite_factors[f'factor_{category}'] = category_factors.mean(axis=1)

            # Clean the final composite factors
            composite_factors = self._clean_factors(composite_factors)

            # Filter to the requested date range
            composite_factors = composite_factors.loc[start_date:end_date]

            return composite_factors
            
        except Exception as e:
            self.logger.error(f"Error generating factors for {symbol}: {e}")
            return None
    
    def _get_market_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get market data for factor calculations"""
        try:
            market_data = {}
            
            for index_symbol in TradingParams.MARKET_INDICES:
                try:
                    data = self.db_manager.get_market_data(index_symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        market_data[index_symbol] = data
                except Exception as e:
                    self.logger.warning(f"Could not get market data for {index_symbol}: {e}")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    def _generate_momentum_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based factors"""
        try:
            factors = pd.DataFrame(index=price_data.index)
            
            # Price momentum factors
            for window in self.factor_params['momentum_windows']:
                if len(price_data) > window:
                    factors[f'returns_{window}d'] = price_data['close'].pct_change(window)
                    factors[f'log_returns_{window}d'] = np.log(price_data['close'] / price_data['close'].shift(window))
            
            # Relative strength momentum
            factors['rsi_momentum'] = self._calculate_rsi_momentum(price_data)
            
            # Price acceleration
            factors['price_acceleration'] = price_data['close'].pct_change().diff()
            
            # Momentum persistence
            factors['momentum_persistence'] = self._calculate_momentum_persistence(price_data)
            
            # Volume-weighted momentum
            factors['vwap_momentum'] = self._calculate_vwap_momentum(price_data)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error generating momentum factors: {e}")
            return pd.DataFrame()
    
    def _generate_mean_reversion_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion factors"""
        try:
            factors = pd.DataFrame(index=price_data.index)
            
            # RSI-based mean reversion
            factors['rsi_mean_reversion'] = self._calculate_rsi_mean_reversion(price_data)
            
            # Bollinger Band position
            factors['bollinger_position'] = self._calculate_bollinger_position(price_data)
            
            # Price vs moving averages
            for window in [20, 50, 200]:
                if len(price_data) > window:
                    ma = price_data['close'].rolling(window).mean()
                    factors[f'price_vs_ma_{window}'] = (price_data['close'] - ma) / ma
            
            # Deviation from VWAP
            factors['vwap_deviation'] = self._calculate_vwap_deviation(price_data)
            
            # Z-score of returns
            for window in [20, 60]:
                if len(price_data) > window:
                    returns = price_data['close'].pct_change()
                    rolling_mean = returns.rolling(window).mean()
                    rolling_std = returns.rolling(window).std()
                    factors[f'return_zscore_{window}'] = (returns - rolling_mean) / rolling_std
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion factors: {e}")
            return pd.DataFrame()
    
    def _generate_volatility_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based factors"""
        try:
            factors = pd.DataFrame(index=price_data.index)
            
            # Realized volatility
            returns = price_data['close'].pct_change()
            for window in self.factor_params['volatility_windows']:
                if len(price_data) > window:
                    factors[f'realized_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
            
            # High-low volatility
            factors['hl_volatility'] = self._calculate_hl_volatility(price_data)
            
            # Volatility ratio
            factors['vol_ratio'] = self._calculate_volatility_ratio(price_data)
            
            # ATR-based volatility
            factors['atr_normalized'] = self._calculate_normalized_atr(price_data)
            
            # Volatility skew
            factors['vol_skew'] = self._calculate_volatility_skew(price_data)
            
            # GARCH volatility estimate
            factors['garch_vol'] = self._calculate_garch_volatility(price_data)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error generating volatility factors: {e}")
            return pd.DataFrame()
    
    def _generate_volume_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based factors"""
        try:
            factors = pd.DataFrame(index=price_data.index)
            
            # Volume momentum
            for window in [5, 20, 60]:
                if len(price_data) > window:
                    factors[f'volume_momentum_{window}d'] = price_data['volume'].pct_change(window)
            
            # Volume ratio
            for window in [20, 60]:
                if len(price_data) > window:
                    factors[f'volume_ratio_{window}d'] = price_data['volume'] / price_data['volume'].rolling(window).mean()
            
            # Money flow
            factors['money_flow'] = self._calculate_money_flow(price_data)
            
            # Volume-price trend
            factors['vpt'] = self._calculate_volume_price_trend(price_data)
            
            # On-balance volume
            factors['obv'] = self._calculate_obv(price_data)
            
            # Volume volatility
            factors['volume_volatility'] = price_data['volume'].rolling(20).std() / price_data['volume'].rolling(20).mean()
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error generating volume factors: {e}")
            return pd.DataFrame()
    
    def _generate_technical_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator factors"""
        try:
            # This method is disabled as these are generated by TechnicalIndicators
            # to avoid duplicate feature columns.
            factors = pd.DataFrame(index=price_data.index)
            
            # # MACD signal
            # factors['macd_signal'] = self._calculate_macd_signal(price_data)
            
            # # Stochastic signal
            # factors['stoch_signal'] = self._calculate_stochastic_signal(price_data)
            
            # # Williams %R signal
            # factors['williams_signal'] = self._calculate_williams_signal(price_data)
            
            # # ADX trend strength
            # factors['adx_trend'] = self._calculate_adx_trend(price_data)
            
            # # Commodity Channel Index
            # factors['cci'] = self._calculate_cci(price_data)
            
            # # Rate of Change
            # for window in [10, 20, 50]:
            #     if len(price_data) > window:
            #         factors[f'roc_{window}'] = ((price_data['close'] / price_data['close'].shift(window)) - 1) * 100
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error generating technical factors: {e}")
            return pd.DataFrame()
    
    def _generate_market_structure_factors(self, price_data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate market structure factors"""
        try:
            factors = pd.DataFrame(index=price_data.index)
            
            # Beta calculations
            if 'SPY' in market_data:
                factors['beta_spy'] = self._calculate_rolling_beta(price_data, market_data['SPY'])
            
            # Correlation with market indices
            for index_name, index_data in market_data.items():
                if not index_data.empty:
                    correlation = self._calculate_rolling_correlation(price_data, index_data)
                    factors[f'correlation_{index_name.lower()}'] = correlation
            
            # Sector relative strength (simplified)
            factors['sector_relative_strength'] = self._calculate_sector_relative_strength(price_data, market_data)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error generating market structure factors: {e}")
            return pd.DataFrame()
    
    # Helper methods for factor calculations
    def _calculate_rsi_momentum(self, price_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI momentum factor"""
        try:
            delta = price_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # RSI momentum: rate of change of RSI
            return rsi.pct_change(5)
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_rsi_mean_reversion(self, price_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI mean reversion factor"""
        try:
            delta = price_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Mean reversion signal: deviation from 50
            return (50 - rsi) / 50
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_bollinger_position(self, price_data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.Series:
        """Calculate Bollinger Band position"""
        try:
            ma = price_data['close'].rolling(window).mean()
            std = price_data['close'].rolling(window).std()
            
            upper_band = ma + (std * num_std)
            lower_band = ma - (std * num_std)
            
            # Position within bands (-1 to 1)
            return (price_data['close'] - ma) / (upper_band - ma)
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_vwap_momentum(self, price_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate VWAP momentum"""
        try:
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            vwap = (typical_price * price_data['volume']).rolling(window).sum() / price_data['volume'].rolling(window).sum()
            
            return (price_data['close'] / vwap - 1) * 100
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_vwap_deviation(self, price_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate deviation from VWAP"""
        try:
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            vwap = (typical_price * price_data['volume']).rolling(window).sum() / price_data['volume'].rolling(window).sum()
            
            return (price_data['close'] - vwap) / vwap
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_momentum_persistence(self, price_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate momentum persistence"""
        try:
            returns = price_data['close'].pct_change()
            positive_returns = (returns > 0).rolling(window).sum()
            
            return positive_returns / window
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_hl_volatility(self, price_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate high-low volatility"""
        try:
            hl_ratio = np.log(price_data['high'] / price_data['low'])
            return hl_ratio.rolling(window).mean()
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_volatility_ratio(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate volatility ratio"""
        try:
            returns = price_data['close'].pct_change()
            short_vol = returns.rolling(10).std()
            long_vol = returns.rolling(60).std()
            
            return short_vol / long_vol
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_normalized_atr(self, price_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate normalized Average True Range"""
        try:
            high_low = price_data['high'] - price_data['low']
            high_close = np.abs(price_data['high'] - price_data['close'].shift())
            low_close = np.abs(price_data['low'] - price_data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window).mean()
            
            return atr / price_data['close']
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_volatility_skew(self, price_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate volatility skew"""
        try:
            returns = price_data['close'].pct_change()
            return returns.rolling(window).skew()
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_garch_volatility(self, price_data: pd.DataFrame, window: int = 60) -> pd.Series:
        """Calculate GARCH volatility estimate (simplified)"""
        try:
            returns = price_data['close'].pct_change()
            
            # Simplified GARCH(1,1) using rolling estimates
            alpha = 0.1
            beta = 0.85
            omega = 0.05
            
            variance = returns.rolling(window).var()
            garch_var = omega + alpha * (returns ** 2).shift(1) + beta * variance.shift(1)
            
            return np.sqrt(garch_var) * np.sqrt(252)
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_money_flow(self, price_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        try:
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            money_flow = typical_price * price_data['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window).sum()
            negative_mf = negative_flow.rolling(window).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            
            return mfi
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_volume_price_trend(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        try:
            price_change = price_data['close'].pct_change()
            vpt = (price_change * price_data['volume']).cumsum()
            
            return vpt
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_obv(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            price_change = price_data['close'].diff()
            volume_direction = np.where(price_change > 0, price_data['volume'],
                                      np.where(price_change < 0, -price_data['volume'], 0))
            
            return pd.Series(volume_direction, index=price_data.index).cumsum()
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_macd_signal(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate MACD signal"""
        try:
            ema12 = price_data['close'].ewm(span=12).mean()
            ema26 = price_data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            return macd - signal
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_stochastic_signal(self, price_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Stochastic oscillator signal"""
        try:
            lowest_low = price_data['low'].rolling(window).min()
            highest_high = price_data['high'].rolling(window).max()
            
            stoch_k = 100 * (price_data['close'] - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(3).mean()
            
            return stoch_k - stoch_d
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_williams_signal(self, price_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R signal"""
        try:
            highest_high = price_data['high'].rolling(window).max()
            lowest_low = price_data['low'].rolling(window).min()
            
            williams_r = -100 * (highest_high - price_data['close']) / (highest_high - lowest_low)
            
            return williams_r
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_adx_trend(self, price_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ADX trend strength (simplified)"""
        try:
            # Simplified ADX calculation
            high_diff = price_data['high'].diff()
            low_diff = price_data['low'].diff().abs()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = np.maximum(price_data['high'] - price_data['low'],
                           np.maximum(abs(price_data['high'] - price_data['close'].shift(1)),
                                    abs(price_data['low'] - price_data['close'].shift(1))))
            
            plus_di = 100 * pd.Series(plus_dm).rolling(window).mean() / pd.Series(tr).rolling(window).mean()
            minus_di = 100 * pd.Series(minus_dm).rolling(window).mean() / pd.Series(tr).rolling(window).mean()
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window).mean()
            
            return adx
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_cci(self, price_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
            sma = typical_price.rolling(window).mean()
            mad = typical_price.rolling(window).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma) / (0.015 * mad)
            
            return cci
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_rolling_beta(self, price_data: pd.DataFrame, market_data: pd.DataFrame, window: int = 60) -> pd.Series:
        """Calculate rolling beta"""
        try:
            stock_returns = price_data['close'].pct_change()
            market_returns = market_data['close'].pct_change()
            
            # Align data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
            
            if aligned_data.empty or len(aligned_data.columns) != 2:
                return pd.Series(index=price_data.index, dtype=float)
            
            aligned_data.columns = ['stock', 'market']
            
            # Calculate rolling beta
            def calc_beta(data):
                if len(data) < 10:
                    return np.nan
                try:
                    covariance = np.cov(data['stock'], data['market'])[0, 1]
                    variance = np.var(data['market'])
                    return covariance / variance if variance > 0 else np.nan
                except:
                    return np.nan
            
            beta = aligned_data.rolling(window).apply(lambda x: calc_beta(x), raw=False)['stock']
            
            return beta.reindex(price_data.index)
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_rolling_correlation(self, price_data: pd.DataFrame, market_data: pd.DataFrame, window: int = 60) -> pd.Series:
        """Calculate rolling correlation"""
        try:
            stock_returns = price_data['close'].pct_change()
            market_returns = market_data['close'].pct_change()
            
            # Align data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
            
            if aligned_data.empty or len(aligned_data.columns) != 2:
                return pd.Series(index=price_data.index, dtype=float)
            
            aligned_data.columns = ['stock', 'market']
            correlation = aligned_data['stock'].rolling(window).corr(aligned_data['market'])
            
            return correlation.reindex(price_data.index)
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _calculate_sector_relative_strength(self, price_data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculate sector relative strength (simplified)"""
        try:
            if 'SPY' not in market_data:
                return pd.Series(index=price_data.index, dtype=float)
            
            stock_returns = price_data['close'].pct_change(20)
            market_returns = market_data['SPY']['close'].pct_change(20)
            
            # Align data
            aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
            
            if aligned_data.empty:
                return pd.Series(index=price_data.index, dtype=float)
            
            relative_strength = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
            
            return relative_strength.reindex(price_data.index)
            
        except Exception:
            return pd.Series(index=price_data.index, dtype=float)
    
    def _clean_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate factor data"""
        try:
            # Remove infinite values
            factors = factors.replace([np.inf, -np.inf], np.nan)
            
            # Winsorize extreme values
            for col in factors.columns:
                if factors[col].dtype in [np.float64, np.float32]:
                    q01 = factors[col].quantile(0.01)
                    q99 = factors[col].quantile(0.99)
                    factors[col] = factors[col].clip(lower=q01, upper=q99)
            
            # Forward fill missing values (limited)
            factors = factors.fillna(method='ffill', limit=5)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error cleaning factors: {e}")
            return factors
    
    def calculate_factor_scores(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate cross-sectional factor scores"""
        try:
            if not factors:
                return pd.DataFrame()
            
            # Combine all factors into a single panel
            all_dates = set()
            for symbol_factors in factors.values():
                all_dates.update(symbol_factors.index)
            
            all_dates = sorted(all_dates)
            
            # Calculate cross-sectional scores for each date
            factor_scores = []
            
            for date in all_dates:
                date_data = {}
                
                for symbol, symbol_factors in factors.items():
                    if date in symbol_factors.index:
                        date_data[symbol] = symbol_factors.loc[date]
                
                if len(date_data) < 3:  # Need minimum symbols
                    continue
                
                # Create cross-sectional DataFrame
                cross_section = pd.DataFrame(date_data).T
                
                # Calculate z-scores (cross-sectional standardization)
                z_scores = cross_section.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x, axis=0)
                
                # Add date and reshape
                z_scores['date'] = date
                z_scores['symbol'] = z_scores.index
                
                factor_scores.append(z_scores.reset_index(drop=True))
            
            if factor_scores:
                return pd.concat(factor_scores, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error calculating factor scores: {e}")
            return pd.DataFrame()
    
    def perform_factor_analysis(self, factors: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform comprehensive factor analysis"""
        try:
            self.logger.info("Performing factor analysis...")
            
            # Calculate factor scores
            factor_scores = self.calculate_factor_scores(factors)
            
            if factor_scores.empty:
                return {}
            
            # Factor correlation analysis
            factor_correlations = self._analyze_factor_correlations(factor_scores)
            
            # Principal component analysis
            pca_results = self._perform_pca_analysis(factor_scores)
            
            # Factor performance attribution
            performance_attribution = self._analyze_factor_performance(factors)
            
            return {
                'factor_correlations': factor_correlations,
                'pca_results': pca_results,
                'performance_attribution': performance_attribution,
                'factor_coverage': {
                    'total_symbols': len(factors),
                    'total_factors': len(factor_scores.columns) - 2,  # Exclude date and symbol
                    'date_range': f"{factor_scores['date'].min()} to {factor_scores['date'].max()}"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in factor analysis: {e}")
            return {}
    
    def _analyze_factor_correlations(self, factor_scores: pd.DataFrame) -> pd.DataFrame:
        """Analyze factor correlations"""
        try:
            numeric_cols = factor_scores.select_dtypes(include=[np.number]).columns
            correlations = factor_scores[numeric_cols].corr()
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error analyzing factor correlations: {e}")
            return pd.DataFrame()
    
    def _perform_pca_analysis(self, factor_scores: pd.DataFrame, n_components: int = 10) -> Dict[str, Any]:
        """Perform Principal Component Analysis on factors"""
        try:
            numeric_cols = factor_scores.select_dtypes(include=[np.number]).columns
            data = factor_scores[numeric_cols].fillna(0)
            
            if len(data.columns) < 3:
                return {}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Perform PCA
            pca = PCA(n_components=min(n_components, len(data.columns)))
            principal_components = pca.fit_transform(scaled_data)
            
            # Create results
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=numeric_cols
            )
            
            return {
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance_ratio': cumulative_variance.tolist(),
                'component_loadings': loadings.to_dict(),
                'n_components': pca.n_components_
            }
            
        except Exception as e:
            self.logger.error(f"Error in PCA analysis: {e}")
            return {}
    
    def _analyze_factor_performance(self, factors: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze factor performance and attribution"""
        try:
            # Calculate factor returns (simplified)
            factor_returns = {}
            
            for category, factor_names in self.factor_categories.items():
                category_returns = []
                
                for symbol, symbol_factors in factors.items():
                    symbol_category_factors = [f for f in factor_names if f in symbol_factors.columns]
                    
                    if symbol_category_factors:
                        category_factor_values = symbol_factors[symbol_category_factors].mean(axis=1)
                        category_returns.append(category_factor_values)
                
                if category_returns:
                    avg_category_return = pd.concat(category_returns, axis=1).mean(axis=1)
                    factor_returns[category] = {
                        'mean_return': avg_category_return.mean(),
                        'volatility': avg_category_return.std(),
                        'sharpe_ratio': avg_category_return.mean() / avg_category_return.std() if avg_category_return.std() > 0 else 0
                    }
            
            return factor_returns
            
        except Exception as e:
            self.logger.error(f"Error analyzing factor performance: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    print("--- Running ML4TFactors Demo in Live Mode ---")
    
    # This demo will use the live TradingDatabase to pull real data.
    # Make sure you have some data in your database first.
    # You can run the data_pipeline.py demo to populate it.
    
    # Initialize ML4TFactors with the actual database manager
    ml4t_factors = ML4TFactors()
    
    # Define symbols and date range for the demo
    symbols_to_test = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-06-01"
    end_date = "2023-12-31"

    print(f"\n1. Generating factors for {symbols_to_test} from {start_date} to {end_date}...")
    
    try:
        factors = ml4t_factors.generate_all_factors(
            symbols=symbols_to_test,
            start_date=start_date,
            end_date=end_date
        )

        if factors:
            print(f"  OK: Generated factors for {len(factors)} symbols.")
            aapl_factors = factors.get("AAPL")
            if aapl_factors is not None and not aapl_factors.empty:
                print("\n--- Sample of generated factors for AAPL ---")
                print(aapl_factors.head().iloc[:, :5])
                print("\n--- Factor data descriptions ---")
                print(aapl_factors.describe())
            else:
                print("  WARN: No factors were generated for AAPL.")

            print("\n2. Performing factor analysis...")
            analysis = ml4t_factors.perform_factor_analysis(factors)
            
            if analysis:
                print("  OK: Factor analysis completed.")
                print("\n--- Factor Analysis Summary ---")
                print(f"  - Factor coverage: {analysis.get('factor_coverage', 'N/A')}")
                
                if 'pca_results' in analysis and analysis['pca_results']:
                    print("\n--- Top 5 PCA Component Loadings for PC1 ---")
                    pc1_loadings = pd.Series(analysis['pca_results']['component_loadings']['PC1'])
                    print(pc1_loadings.abs().nlargest(5))
                else:
                    print("  WARN: PCA results not available.")

            else:
                print("  ERROR: Factor analysis failed to produce results.")
                
        else:
            print("  ERROR: No factors were generated. Ensure the database contains price data for the specified symbols and date range.")

    except Exception as e:
        print(f"\nAn error occurred during the demo: {e}")
        print("Please ensure your database is populated and accessible.")

    print("\n--- Demo Finished ---")
