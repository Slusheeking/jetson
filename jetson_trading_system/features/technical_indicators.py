"""
Technical Indicators using pandas_ta
Implementing the 8 proven indicators from ML4Trading methodology
Optimized for Jetson Orin ARM64 architecture
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor

from jetson_trading_system.config.trading_params import TradingParams
from jetson_trading_system.config.jetson_settings import JetsonConfig

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Technical Indicators Calculator using pandas_ta
    
    Implements the 8 core proven indicators:
    1. RSI(14) - Relative Strength Index
    2. MACD - Moving Average Convergence Divergence
    3. Bollinger Bands - Price bands with standard deviation
    4. SMA Ratio - Close/SMA(20) ratio
    5. Volume Ratio - Volume/SMA(Volume, 20) ratio
    6. ATR - Average True Range (volatility)
    7. Stochastic %K - Momentum oscillator
    8. Williams %R - Momentum indicator
    
    All calculations use pandas_ta for maximum ARM64 compatibility
    """
    
    def __init__(self, polygon_client=None, use_polygon_api: bool = True):
        """
        Initialize technical indicators calculator
        
        Args:
            polygon_client: Polygon API client for fetching pre-calculated indicators
            use_polygon_api: Whether to try Polygon API first
        """
        self.polygon_client = polygon_client
        self.use_polygon_api = use_polygon_api
        
        # Configuration from trading parameters
        self.rsi_period = TradingParams.RSI_PERIOD
        self.macd_fast = TradingParams.MACD_FAST
        self.macd_slow = TradingParams.MACD_SLOW
        self.macd_signal = TradingParams.MACD_SIGNAL
        self.bb_period = TradingParams.BOLLINGER_PERIOD
        self.bb_std = TradingParams.BOLLINGER_STD
        self.sma_period = TradingParams.SMA_SHORT_PERIOD
        self.atr_period = TradingParams.ATR_PERIOD
        self.stoch_k_period = TradingParams.STOCH_K_PERIOD
        self.stoch_d_period = TradingParams.STOCH_D_PERIOD
        self.williams_r_period = TradingParams.WILLIAMS_R_PERIOD
    
    def calculate_rsi(self, close: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            close: Close price series
            period: RSI period (default from config)
            
        Returns:
            RSI values (0-100)
        """
        if period is None:
            period = self.rsi_period
        
        try:
            rsi = ta.rsi(close, length=period)
            rsi.name = f'rsi_{period}'
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(np.nan, index=close.index, name=f'rsi_{period}')
    
    def calculate_macd(self, close: pd.Series,
                      fast: int = None, slow: int = None, signal: int = None) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            close: Close price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Dictionary with 'macd_line', 'macd_signal', and 'macd_histogram'
        """
        if fast is None:
            fast = self.macd_fast
        if slow is None:
            slow = self.macd_slow
        if signal is None:
            signal = self.macd_signal
        
        try:
            # Use copy of close series to prevent pandas_ta from modifying original
            close_copy = close.copy()
            macd_df = ta.macd(close_copy, fast=fast, slow=slow, signal=signal)
            
            if macd_df is None or macd_df.empty:
                raise ValueError("pandas_ta.macd() returned empty DataFrame")
            
            # Find actual column names - pandas_ta may use different formats
            available_cols = macd_df.columns.tolist()
            
            macd_col = None
            signal_col = None
            hist_col = None
            
            # Look for MACD columns with flexible pattern matching
            for col in available_cols:
                if col.startswith(f'MACD_{fast}_{slow}_{signal}') and not col.startswith('MACDs') and not col.startswith('MACDh'):
                    macd_col = col
                elif col.startswith(f'MACDs_{fast}_{slow}_{signal}'):
                    signal_col = col
                elif col.startswith(f'MACDh_{fast}_{slow}_{signal}'):
                    hist_col = col
            
            # Fallback: look for any MACD* columns
            if not all([macd_col, signal_col, hist_col]):
                for col in available_cols:
                    if 'MACD' in col and not 'MACDs' in col and not 'MACDh' in col and not macd_col:
                        macd_col = col
                    elif 'MACDs' in col and not signal_col:
                        signal_col = col
                    elif 'MACDh' in col and not hist_col:
                        hist_col = col
            
            # Check if we found all required columns
            missing_cols = []
            if not macd_col:
                missing_cols.append('MACD (line)')
            if not signal_col:
                missing_cols.append('MACDs (signal)')
            if not hist_col:
                missing_cols.append('MACDh (histogram)')
                
            if missing_cols:
                raise KeyError(f"Missing MACD columns: {missing_cols}. Available: {available_cols}")
            
            return {
                'macd_line': macd_df[macd_col].rename('macd_line'),
                'macd_signal': macd_df[signal_col].rename('macd_signal'),
                'macd_histogram': macd_df[hist_col].rename('macd_histogram')
            }
                
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            nan_series = pd.Series(np.nan, index=close.index)
            return {
                'macd_line': nan_series.copy().rename('macd_line'),
                'macd_signal': nan_series.copy().rename('macd_signal'),
                'macd_histogram': nan_series.copy().rename('macd_histogram')
            }
    
    def calculate_bollinger_bands(self, close: pd.Series,
                                period: int = None, std_dev: float = None) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            close: Close price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with 'bb_upper', 'bb_middle', 'bb_lower', and 'bb_ratio'
        """
        if period is None:
            period = self.bb_period
        if std_dev is None:
            std_dev = self.bb_std
        
        try:
            bb_df = ta.bbands(close, length=period, std=std_dev)
            
            if bb_df is None or bb_df.empty:
                raise ValueError("pandas_ta.bbands() returned empty DataFrame")
            
            # Find actual column names - pandas_ta may use different formats
            available_cols = bb_df.columns.tolist()
            
            # Try different column name patterns
            lower_col = None
            middle_col = None
            upper_col = None
            percent_col = None
            
            # Pattern 1: BBL_20_2.0 format
            for col in available_cols:
                if col.startswith(f'BBL_{period}_'):
                    lower_col = col
                elif col.startswith(f'BBM_{period}_'):
                    middle_col = col
                elif col.startswith(f'BBU_{period}_'):
                    upper_col = col
                elif col.startswith(f'BBP_{period}_'):
                    percent_col = col
            
            # Fallback: look for any BB* columns
            if not all([lower_col, middle_col, upper_col]):
                for col in available_cols:
                    if 'BBL' in col and not lower_col:
                        lower_col = col
                    elif 'BBM' in col and not middle_col:
                        middle_col = col
                    elif 'BBU' in col and not upper_col:
                        upper_col = col
                    elif 'BBP' in col and not percent_col:
                        percent_col = col
            
            # Check if we found all required columns
            missing_cols = []
            if not lower_col:
                missing_cols.append('BBL (lower)')
            if not middle_col:
                missing_cols.append('BBM (middle)')
            if not upper_col:
                missing_cols.append('BBU (upper)')
            if not percent_col:
                missing_cols.append('BBP (percent)')
                
            if missing_cols:
                raise KeyError(f"Missing Bollinger Band columns: {missing_cols}. Available: {available_cols}")
            
            return {
                'bb_upper': bb_df[upper_col].rename('bb_upper'),
                'bb_middle': bb_df[middle_col].rename('bb_middle'),
                'bb_lower': bb_df[lower_col].rename('bb_lower'),
                'bb_ratio': bb_df[percent_col].rename('bb_ratio')  # BBP is the %B indicator (position within bands)
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            nan_series = pd.Series(np.nan, index=close.index)
            return {
                'bb_upper': nan_series.copy().rename('bb_upper'),
                'bb_middle': nan_series.copy().rename('bb_middle'),
                'bb_lower': nan_series.copy().rename('bb_lower'),
                'bb_ratio': nan_series.copy().rename('bb_ratio')
            }
    
    def calculate_sma_ratio(self, close: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Simple Moving Average Ratio (Close / SMA)
        
        Args:
            close: Close price series
            period: SMA period
            
        Returns:
            SMA ratio values
        """
        if period is None:
            period = self.sma_period
        
        try:
            sma = ta.sma(close, length=period)
            sma_ratio = close / sma
            sma_ratio.name = f'sma_ratio_{period}'
            
            return sma_ratio
        except Exception as e:
            logger.error(f"Error calculating SMA ratio: {e}")
            return pd.Series(np.nan, index=close.index, name=f'sma_ratio_{period}')
    
    def calculate_volume_ratio(self, volume: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Volume Ratio (Volume / SMA(Volume))
        
        Args:
            volume: Volume series
            period: SMA period for volume
            
        Returns:
            Volume ratio values
        """
        if period is None:
            period = self.sma_period
        
        try:
            volume_sma = ta.sma(volume, length=period)
            volume_ratio = volume / volume_sma
            volume_ratio.name = f'volume_ratio_{period}'
            
            return volume_ratio
        except Exception as e:
            logger.error(f"Error calculating volume ratio: {e}")
            return pd.Series(np.nan, index=volume.index, name=f'volume_ratio_{period}')
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = None) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high: High price series
            low: Low price series  
            close: Close price series
            period: ATR period
            
        Returns:
            ATR values
        """
        if period is None:
            period = self.atr_period
        
        try:
            atr = ta.atr(high=high, low=low, close=close, length=period)
            atr.name = f'atr_{period}'
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series(np.nan, index=close.index, name=f'atr_{period}')
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = None, d_period: int = None) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: %K period
            d_period: %D period
            
        Returns:
            Dictionary with 'stoch_k' and 'stoch_d'
        """
        if k_period is None:
            k_period = self.stoch_k_period
        if d_period is None:
            d_period = self.stoch_d_period
        
        try:
            stoch_df = ta.stoch(high=high, low=low, close=close, k=k_period, d=d_period)
            
            # pandas_ta returns DataFrame with columns: STOCHk_14_3_3, STOCHd_14_3_3
            k_col = f'STOCHk_{k_period}_{d_period}_{d_period}'
            d_col = f'STOCHd_{k_period}_{d_period}_{d_period}'
            
            return {
                'stoch_k': stoch_df[k_col].rename(f'stoch_k_{k_period}'),
                'stoch_d': stoch_df[d_col].rename(f'stoch_d_{d_period}')
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            nan_series = pd.Series(np.nan, index=close.index)
            return {
                'stoch_k': nan_series.copy().rename(f'stoch_k_{k_period}'),
                'stoch_d': nan_series.copy().rename(f'stoch_d_{d_period}')
            }
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = None) -> pd.Series:
        """
        Calculate Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Williams %R period
            
        Returns:
            Williams %R values
        """
        if period is None:
            period = self.williams_r_period
        
        try:
            williams_r = ta.willr(high=high, low=low, close=close, length=period)
            williams_r.name = f'williams_r_{period}'
            
            return williams_r
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
            return pd.Series(np.nan, index=close.index, name=f'williams_r_{period}')
    
    def calculate_all_indicators(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all 8 technical indicators for OHLCV data
        
        Args:
            ohlcv_df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with all technical indicators added
        """
        result_df = ohlcv_df.copy()
        
        # Check if we have enough data
        min_periods_needed = 30  # Need at least 30 periods for reliable indicators
        if len(ohlcv_df) < min_periods_needed:
            logger.warning(f"Insufficient data for technical indicators: {len(ohlcv_df)} rows (need {min_periods_needed})")
            # Return original dataframe with NaN columns for indicators
            indicator_columns = [
                'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_ratio',
                'sma_ratio_20', 'volume_ratio_20', 'atr_14',
                'stoch_k_14', 'stoch_d_3', 'williams_r_14'
            ]
            for col in indicator_columns:
                result_df[col] = np.nan
            return result_df
        
        # Extract price series
        high = ohlcv_df['high']
        low = ohlcv_df['low']
        close = ohlcv_df['close']
        volume = ohlcv_df['volume']
        
        try:
            # 1. RSI(14)
            result_df['rsi_14'] = self.calculate_rsi(close)
            
            # 2. MACD
            macd_dict = self.calculate_macd(close)
            for key, series in macd_dict.items():
                result_df[key] = series
            
            # 3. Bollinger Bands
            bb_dict = self.calculate_bollinger_bands(close)
            for key, series in bb_dict.items():
                result_df[key] = series
            
            # 4. SMA Ratio
            result_df['sma_ratio_20'] = self.calculate_sma_ratio(close)
            
            # 5. Volume Ratio
            result_df['volume_ratio_20'] = self.calculate_volume_ratio(volume)
            
            # 6. ATR
            result_df['atr_14'] = self.calculate_atr(high, low, close)
            
            # 7. Stochastic
            stoch_dict = self.calculate_stochastic(high, low, close)
            for key, series in stoch_dict.items():
                result_df[key] = series
            
            # 8. Williams %R
            result_df['williams_r_14'] = self.calculate_williams_r(high, low, close)
            
            logger.info(f"Calculated all technical indicators for {len(result_df)} rows using pandas_ta")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Add NaN columns if calculation fails
            indicator_columns = [
                'rsi_14', 'macd_line', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_ratio',
                'sma_ratio_20', 'volume_ratio_20', 'atr_14',
                'stoch_k_14', 'stoch_d_3', 'williams_r_14'
            ]
            for col in indicator_columns:
                if col not in result_df.columns:
                    result_df[col] = np.nan
        
        return result_df
    
    def calculate_indicators_for_multiple_symbols(self, 
                                                data_dict: Dict[str, pd.DataFrame],
                                                parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Calculate technical indicators for multiple symbols efficiently
        
        Args:
            data_dict: Dictionary of {symbol: ohlcv_dataframe}
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary of {symbol: dataframe_with_indicators}
        """
        if not parallel or len(data_dict) < 4:
            # Sequential processing for small datasets
            result_dict = {}
            for symbol, df in data_dict.items():
                try:
                    result_dict[symbol] = self.calculate_all_indicators(df)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    result_dict[symbol] = df  # Return original data if calculation fails
            return result_dict
        
        # Parallel processing for larger datasets
        def process_symbol(symbol_df_tuple):
            symbol, df = symbol_df_tuple
            try:
                return symbol, self.calculate_all_indicators(df)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return symbol, df
        
        # Use thread pool for I/O bound operations
        max_workers = min(JetsonConfig.PARALLEL_WORKERS, len(data_dict))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_symbol, data_dict.items()))
        
        return dict(results)

def create_market_context_features(indices_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create market context features from indices data
    
    Args:
        indices_data: Dictionary of {index_symbol: ohlcv_dataframe}
        
    Returns:
        DataFrame with market context features
    """
    indicator_calc = TechnicalIndicators()
    context_features = []
    
    for symbol, df in indices_data.items():
        if df.empty:
            continue
        
        try:
            # Calculate basic indicators for each index
            df_with_indicators = indicator_calc.calculate_all_indicators(df)
            
            # Select key features with symbol prefix
            features_df = pd.DataFrame(index=df.index)
            features_df['date'] = df.get('timestamp', df.index)
            
            # Key market context features
            features_df[f'{symbol.lower()}_rsi'] = df_with_indicators['rsi_14']
            features_df[f'{symbol.lower()}_sma_ratio'] = df_with_indicators['sma_ratio_20']
            features_df[f'{symbol.lower()}_bb_ratio'] = df_with_indicators['bb_ratio']
            
            # Special handling for VIX (volatility level and change)
            if symbol == 'VIX':
                features_df['vix_level'] = df['close']
                features_df['vix_change'] = df['close'].pct_change()
            
            context_features.append(features_df)
            
        except Exception as e:
            logger.error(f"Error creating context features for {symbol}: {e}")
    
    if not context_features:
        return pd.DataFrame()
    
    # Merge all context features on date
    result = context_features[0]
    for df in context_features[1:]:
        result = result.merge(df, on='date', how='outer', suffixes=('', '_dup'))
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.str.endswith('_dup')]
    
    return result.sort_values('date').reset_index(drop=True)

# Pre-configured indicator sets for different use cases
MOMENTUM_INDICATORS = ['rsi_14', 'stoch_k_14', 'williams_r_14']
TREND_INDICATORS = ['sma_ratio_20', 'macd_line', 'macd_signal']
VOLATILITY_INDICATORS = ['atr_14', 'bb_ratio']
VOLUME_INDICATORS = ['volume_ratio_20']

ALL_CORE_INDICATORS = MOMENTUM_INDICATORS + TREND_INDICATORS + VOLATILITY_INDICATORS + VOLUME_INDICATORS

def get_indicator_subset(df: pd.DataFrame, indicator_type: str) -> pd.DataFrame:
    """
    Get subset of indicators by type
    
    Args:
        df: DataFrame with all indicators
        indicator_type: 'momentum', 'trend', 'volatility', 'volume', or 'all'
        
    Returns:
        DataFrame with selected indicators
    """
    if indicator_type == 'momentum':
        cols = MOMENTUM_INDICATORS
    elif indicator_type == 'trend':
        cols = TREND_INDICATORS
    elif indicator_type == 'volatility':
        cols = VOLATILITY_INDICATORS
    elif indicator_type == 'volume':
        cols = VOLUME_INDICATORS
    elif indicator_type == 'all':
        cols = ALL_CORE_INDICATORS
    else:
        raise ValueError(f"Unknown indicator type: {indicator_type}")
    
    # Include base columns plus requested indicators
    base_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    available_base_cols = [col for col in base_cols if col in df.columns]
    available_indicator_cols = [col for col in cols if col in df.columns]
    
    return df[available_base_cols + available_indicator_cols]

if __name__ == '__main__':
    print("--- Running TechnicalIndicators Demo ---")

    # Create sample OHLCV data
    data = {
        'open': np.random.uniform(98, 102, 100),
        'high': np.random.uniform(102, 105, 100),
        'low': np.random.uniform(95, 98, 100),
        'close': np.random.uniform(99, 104, 100),
        'volume': np.random.randint(1_000_000, 5_000_000, 100)
    }
    sample_df = pd.DataFrame(data)
    sample_df['high'] = sample_df[['open', 'high']].max(axis=1)
    sample_df['low'] = sample_df[['open', 'low']].min(axis=1)

    print("Sample OHLCV Data:")
    print(sample_df.head())
    
    # Initialize indicator calculator
    ti = TechnicalIndicators()
    
    # Calculate all indicators
    print("\n--- Calculating All Indicators ---")
    indicators_df = ti.calculate_all_indicators(sample_df)
    print(indicators_df.tail())
    
    print("\n--- Indicator Columns ---")
    print(indicators_df.columns)
    
    print("\n--- Getting Indicator Subsets ---")
    momentum = get_indicator_subset(indicators_df, 'momentum')
    print("\nMomentum Indicators:")
    print(momentum.tail())
    
    trend = get_indicator_subset(indicators_df, 'trend')
    print("\nTrend Indicators:")
    print(trend.tail())

    print("\n--- Demo Finished ---")
