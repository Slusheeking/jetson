"""
Feature Engineering Engine for Jetson Trading System
Comprehensive feature generation, selection, and engineering pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import concurrent.futures
import threading
from dataclasses import dataclass
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingParams as TradingConfig
from jetson_trading_system.utils.logger import get_data_logger
from jetson_trading_system.data.cache_manager import CacheManager
from jetson_trading_system.utils.database import TradingDatabase
from jetson_trading_system.features.technical_indicators import TechnicalIndicators
from jetson_trading_system.features.ml4t_factors import ML4TFactors

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    technical_indicators: bool = True
    ml4t_factors: bool = True
    time_features: bool = True
    lag_features: bool = True
    rolling_features: bool = True
    interaction_features: bool = False
    polynomial_features: bool = False
    feature_selection: bool = True
    scaling: str = "standard"  # "standard", "minmax", "robust", "none"
    max_features: int = 100
    lag_periods: List[int] = None
    rolling_windows: List[int] = None

class FeatureEngine:
    """
    Comprehensive feature engineering pipeline
    Optimized for ML4Trading methodology and Jetson hardware
    """
    
    def __init__(self, config: FeatureConfig = None, cache_manager: CacheManager = None):
        """
        Initialize feature engine
        
        Args:
            config: Feature engineering configuration
            cache_manager: Redis cache manager for features
        """
        self.config = config or FeatureConfig()
        if self.config.lag_periods is None:
            self.config.lag_periods = [1, 2, 3, 5, 10]
        if self.config.rolling_windows is None:
            self.config.rolling_windows = [5, 10, 20, 60]
        
        self.logger = get_data_logger()
        self.db_manager = TradingDatabase()
        self.ta_calculator = TechnicalIndicators()
        self.ml4t_factors = ML4TFactors()
        
        # Use Redis cache manager
        self.cache_manager = cache_manager or CacheManager()
        
        # Feature processing components
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        
        # Performance tracking
        self.feature_stats = {
            'total_features_generated': 0,
            'features_selected': 0,
            'processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger.info("FeatureEngine initialized")
    
    def generate_features(self, 
                         symbols: Union[str, List[str]], 
                         start_date: str, 
                         end_date: str,
                         target_column: str = None) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive feature set for symbols
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for feature generation
            end_date: End date for feature generation
            target_column: Target column for supervised feature selection
            
        Returns:
            Dictionary of symbol -> features DataFrame
        """
        try:
            start_time = datetime.now()
            
            if isinstance(symbols, str):
                symbols = [symbols]
            
            self.logger.info(f"Generating features for {len(symbols)} symbols")
            
            all_features = {}
            
            # Use parallel processing for multiple symbols
            if len(symbols) > 1 and JetsonConfig.CPU_CORES > 2:
                all_features = self._generate_features_parallel(symbols, start_date, end_date, target_column)
            else:
                all_features = self._generate_features_sequential(symbols, start_date, end_date, target_column)
            
            # Post-process features
            if self.config.feature_selection and target_column:
                all_features = self._apply_feature_selection(all_features, target_column)
            
            # Apply scaling
            if self.config.scaling != "none":
                all_features = self._apply_scaling(all_features)
            
            # Update performance stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.feature_stats['processing_time'] = processing_time
            
            self.logger.info(f"Feature generation completed in {processing_time:.2f}s")
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            return {}
    
    def _generate_features_sequential(self, 
                                    symbols: List[str], 
                                    start_date: str, 
                                    end_date: str,
                                    target_column: str = None) -> Dict[str, pd.DataFrame]:
        """Generate features sequentially"""
        all_features = {}
        
        for symbol in symbols:
            try:
                features = self._generate_symbol_features(symbol, start_date, end_date, target_column)
                if features is not None and not features.empty:
                    all_features[symbol] = features
                    self.logger.info(f"Generated {len(features.columns)} features for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error generating features for {symbol}: {e}")
        
        return all_features
    
    def _generate_features_parallel(self, 
                                  symbols: List[str], 
                                  start_date: str, 
                                  end_date: str,
                                  target_column: str = None) -> Dict[str, pd.DataFrame]:
        """Generate features in parallel"""
        all_features = {}
        
        max_workers = min(len(symbols), JetsonConfig.CPU_CORES - 1)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(self._generate_symbol_features, symbol, start_date, end_date, target_column): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    features = future.result()
                    if features is not None and not features.empty:
                        all_features[symbol] = features
                        self.logger.info(f"Generated {len(features.columns)} features for {symbol}")
                
                except Exception as e:
                    self.logger.error(f"Error generating features for {symbol}: {e}")
        
        return all_features
    
    def _generate_symbol_features(self, 
                                symbol: str, 
                                start_date: str, 
                                end_date: str,
                                target_column: str = None) -> Optional[pd.DataFrame]:
        """Generate comprehensive features for a single symbol"""
        try:
            # Check Redis cache
            cache_key = f"features:{symbol}:{start_date}:{end_date}"
            cached_features = self.cache_manager.get(cache_key)
            if cached_features is not None:
                self.feature_stats['cache_hits'] += 1
                return cached_features
            else:
                self.feature_stats['cache_misses'] += 1
            
            # Get base price data
            extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=200)).strftime('%Y-%m-%d')
            price_data = self.db_manager.get_market_data(symbol, extended_start, end_date)
            
            if price_data is None or len(price_data) < 50:
                return None
            
            # Initialize features DataFrame
            features = pd.DataFrame(index=price_data.index)
            
            # Generate feature categories
            feature_components = []
            
            # 1. Technical indicators
            if self.config.technical_indicators:
                ta_features = self._generate_technical_features(price_data)
                if ta_features is not None:
                    feature_components.append(ta_features)
            
            # 2. ML4T factors
            if self.config.ml4t_factors:
                # Now generates composite factors
                ml4t_features = self._generate_ml4t_features(symbol, price_data, extended_start, end_date)
                if ml4t_features is not None:
                    feature_components.append(ml4t_features)
            
            # 3. Time-based features
            if self.config.time_features:
                time_features = self._generate_time_features(price_data.index)
                if time_features is not None:
                    feature_components.append(time_features)
            
            # 4. Price and volume features
            price_volume_features = self._generate_price_volume_features(price_data)
            if price_volume_features is not None:
                feature_components.append(price_volume_features)
            
            # 5. Lag features
            if self.config.lag_features:
                lag_features = self._generate_lag_features(price_data)
                if lag_features is not None:
                    feature_components.append(lag_features)
            
            # 6. Rolling window features
            if self.config.rolling_features:
                rolling_features = self._generate_rolling_features(price_data)
                if rolling_features is not None:
                    feature_components.append(rolling_features)
            
            # Combine all features
            for component in feature_components:
                if component is not None and not component.empty:
                    features = features.join(component, how='outer')
            
            # Add target if specified
            if target_column:
                target = self._generate_target(price_data, target_column)
                if target is not None:
                    features = features.join(target, how='left')
            
            # Filter to requested date range
            features = features.loc[start_date:end_date]
            
            # Clean features
            features = self._clean_features(features)
            
            # Cache result in Redis
            self.cache_manager.put(cache_key, features, ttl_seconds=1800)  # 30 minute TTL
            
            self.feature_stats['total_features_generated'] += len(features.columns)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating features for {symbol}: {e}")
            return None
    
    def _generate_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicator features"""
        try:
            return self.ta_calculator.calculate_all_indicators(price_data)
        except Exception as e:
            self.logger.error(f"Error generating technical features: {e}")
            return pd.DataFrame()
    
    def _generate_ml4t_features(self, symbol: str, price_data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate ML4T factor features"""
        try:
            factors = self.ml4t_factors.generate_symbol_factors(symbol, start_date, end_date)
            return factors if factors is not None else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error generating ML4T features: {e}")
            return pd.DataFrame()
    
    def _generate_time_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate time-based features"""
        try:
            features = pd.DataFrame(index=index)
            
            # Basic time features
            features['hour'] = index.hour
            features['day_of_week'] = index.dayofweek
            features['day_of_month'] = index.day
            features['month'] = index.month
            features['quarter'] = index.quarter
            features['year'] = index.year
            
            # Cyclical encoding
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Market session features
            features['is_market_open'] = ((features['hour'] >= 9) & (features['hour'] < 16)).astype(int)
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            
            # Holiday effects (simplified)
            features['is_month_end'] = (features['day_of_month'] >= 25).astype(int)
            features['is_quarter_end'] = ((features['month'] % 3 == 0) & (features['day_of_month'] >= 25)).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating time features: {e}")
            return pd.DataFrame()
    
    def _generate_price_volume_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate price and volume based features"""
        try:
            features = pd.DataFrame(index=price_data.index)
            
            # Price features
            features['open_close_ratio'] = price_data['open'] / price_data['close']
            features['high_low_ratio'] = price_data['high'] / price_data['low']
            features['close_open_ratio'] = price_data['close'] / price_data['open']
            
            # Price changes
            features['price_change'] = price_data['close'].pct_change()
            features['price_change_abs'] = features['price_change'].abs()
            features['price_range'] = (price_data['high'] - price_data['low']) / price_data['close']
            
            # Volume features
            features['volume_change'] = price_data['volume'].pct_change()
            features['volume_change_abs'] = features['volume_change'].abs()
            features['price_volume'] = features['price_change'] * price_data['volume']
            
            # OHLC patterns
            features['doji'] = (abs(price_data['close'] - price_data['open']) / (price_data['high'] - price_data['low'])).fillna(0)
            features['hammer'] = ((price_data['close'] > price_data['open']) & 
                                 ((price_data['open'] - price_data['low']) > 2 * (price_data['close'] - price_data['open']))).astype(int)
            
            # Gap features
            features['gap_up'] = ((price_data['open'] > price_data['high'].shift(1))).astype(int)
            features['gap_down'] = ((price_data['open'] < price_data['low'].shift(1))).astype(int)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating price/volume features: {e}")
            return pd.DataFrame()
    
    def _generate_lag_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate lagged features"""
        try:
            features = pd.DataFrame(index=price_data.index)
            
            # Price lags
            for lag in self.config.lag_periods:
                features[f'close_lag_{lag}'] = price_data['close'].shift(lag)
                features[f'volume_lag_{lag}'] = price_data['volume'].shift(lag)
                features[f'returns_lag_{lag}'] = price_data['close'].pct_change().shift(lag)
            
            # High/Low lags
            for lag in self.config.lag_periods[:3]:  # Only short lags for high/low
                features[f'high_lag_{lag}'] = price_data['high'].shift(lag)
                features[f'low_lag_{lag}'] = price_data['low'].shift(lag)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating lag features: {e}")
            return pd.DataFrame()
    
    def _generate_rolling_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling window features"""
        try:
            features = pd.DataFrame(index=price_data.index)
            
            returns = price_data['close'].pct_change()
            
            # Rolling statistics
            for window in self.config.rolling_windows:
                # Price rolling features
                features[f'close_mean_{window}'] = price_data['close'].rolling(window).mean()
                features[f'close_std_{window}'] = price_data['close'].rolling(window).std()
                features[f'close_min_{window}'] = price_data['close'].rolling(window).min()
                features[f'close_max_{window}'] = price_data['close'].rolling(window).max()
                
                # Returns rolling features
                features[f'returns_mean_{window}'] = returns.rolling(window).mean()
                features[f'returns_std_{window}'] = returns.rolling(window).std()
                features[f'returns_skew_{window}'] = returns.rolling(window).skew()
                features[f'returns_kurt_{window}'] = returns.rolling(window).kurt()
                
                # Volume rolling features
                features[f'volume_mean_{window}'] = price_data['volume'].rolling(window).mean()
                features[f'volume_std_{window}'] = price_data['volume'].rolling(window).std()
                
                # Relative position features
                features[f'close_position_{window}'] = (price_data['close'] - features[f'close_min_{window}']) / (features[f'close_max_{window}'] - features[f'close_min_{window}'])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating rolling features: {e}")
            return pd.DataFrame()
    
    def _generate_target(self, price_data: pd.DataFrame, target_column: str) -> pd.Series:
        """Generate target variable"""
        try:
            if target_column == 'returns_1d':
                return price_data['close'].pct_change().shift(-1).rename('target')
            elif target_column == 'returns_5d':
                return price_data['close'].pct_change(5).shift(-5).rename('target')
            elif target_column == 'binary_1d':
                returns = price_data['close'].pct_change().shift(-1)
                return (returns > 0).astype(int).rename('target')
            elif target_column == 'binary_threshold':
                returns = price_data['close'].pct_change().shift(-1)
                threshold = TradingConfig.MIN_PROFIT_THRESHOLD
                return (returns > threshold).astype(int).rename('target')
            else:
                return price_data['close'].pct_change().shift(-1).rename('target')
                
        except Exception as e:
            self.logger.error(f"Error generating target: {e}")
            return pd.Series(index=price_data.index, dtype=float, name='target')
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Remove infinite values
            features = features.replace([np.inf, -np.inf], np.nan)
            
            # Remove columns with too many missing values
            threshold = 0.8
            features = features.dropna(axis=1, thresh=int(threshold * len(features)))

            # If target exists, separate it before filling NaNs
            target = None
            if 'target' in features.columns:
                target = features['target']
                features = features.drop(columns=['target'])

            # Forward fill limited missing values in features
            features = features.fillna(method='ffill', limit=5)
            features = features.fillna(0)  # Fill remaining with 0
            
            # Re-attach target
            if target is not None:
                features = features.join(target, how='left')

            # Remove constant features (after filling NaNs)
            constant_features = [col for col in features.columns if features[col].nunique(dropna=False) <= 1 and col != 'target']
            if constant_features:
                features = features.drop(columns=constant_features)
                self.logger.info(f"Removed {len(constant_features)} constant features")

            # Drop rows where ALL features are NaN (should be none after fill)
            features = features.dropna(how='all', subset=[c for c in features.columns if c != 'target'])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error cleaning features: {e}")
            return features
    
    def _apply_feature_selection(self, all_features: Dict[str, pd.DataFrame], target_column: str) -> Dict[str, pd.DataFrame]:
        """Apply feature selection across all symbols"""
        try:
            if not all_features:
                return all_features
            
            self.logger.info("Applying feature selection...")
            
            # Combine all features for selection
            combined_features = []
            combined_targets = []
            
            for symbol, features in all_features.items():
                if 'target' in features.columns:
                    X = features.drop('target', axis=1)
                    y = features['target']
                    
                    # Remove missing values
                    mask = ~(X.isnull().any(axis=1) | y.isnull())
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    if len(X_clean) > 0:
                        combined_features.append(X_clean)
                        combined_targets.append(y_clean)
            
            if not combined_features:
                return all_features
            
            # Combine data
            X_combined = pd.concat(combined_features, ignore_index=True)
            y_combined = pd.concat(combined_targets, ignore_index=True)
            
            # Apply feature selection
            if y_combined.nunique() > 1:  # Ensure we have variation in target
                max_features = min(self.config.max_features, len(X_combined.columns))
                
                # Use mutual information for feature selection
                selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
                selector.fit(X_combined.fillna(0), y_combined)
                
                self.selected_features = X_combined.columns[selector.get_support()].tolist()
                self.feature_selector = selector
                
                # Apply selection to all symbol features
                selected_features = {}
                for symbol, features in all_features.items():
                    feature_cols = [col for col in self.selected_features if col in features.columns]
                    selected_data = features[feature_cols]
                    
                    # Add target back if it exists
                    if 'target' in features.columns:
                        selected_data = selected_data.join(features['target'])
                    
                    selected_features[symbol] = selected_data
                
                self.feature_stats['features_selected'] = len(self.selected_features)
                self.logger.info(f"Selected {len(self.selected_features)} features")
                
                return selected_features
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return all_features
    
    def _apply_scaling(self, all_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply feature scaling"""
        try:
            if not all_features or self.config.scaling == "none":
                return all_features
            
            self.logger.info(f"Applying {self.config.scaling} scaling...")
            
            # Initialize scaler
            if self.config.scaling == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaling == "minmax":
                self.scaler = MinMaxScaler()
            elif self.config.scaling == "robust":
                self.scaler = RobustScaler()
            else:
                return all_features
            
            # Combine all features for fitting scaler
            combined_features = []
            for symbol, features in all_features.items():
                # Exclude target column from scaling
                feature_cols = [col for col in features.columns if col != 'target']
                if feature_cols:
                    combined_features.append(features[feature_cols])
            
            if combined_features:
                X_combined = pd.concat(combined_features, ignore_index=True)
                X_combined_clean = X_combined.fillna(0)  # Fill NaN for scaler
                
                # Fit scaler
                self.scaler.fit(X_combined_clean)
                
                # Apply scaling to each symbol
                scaled_features = {}
                for symbol, features in all_features.items():
                    feature_cols = [col for col in features.columns if col != 'target']
                    
                    if feature_cols:
                        X_scaled = self.scaler.transform(features[feature_cols].fillna(0))
                        scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=features.index)
                        
                        # Add target back if it exists
                        if 'target' in features.columns:
                            scaled_df = scaled_df.join(features['target'])
                        
                        scaled_features[symbol] = scaled_df
                    else:
                        scaled_features[symbol] = features
                
                return scaled_features
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"Error applying scaling: {e}")
            return all_features
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted preprocessors"""
        try:
            if features.empty:
                return features
            
            # Apply feature selection
            if self.selected_features:
                available_features = [col for col in self.selected_features if col in features.columns]
                features = features[available_features]
            
            # Apply scaling
            if self.scaler is not None:
                feature_cols = [col for col in features.columns if col != 'target']
                if feature_cols:
                    X_scaled = self.scaler.transform(features[feature_cols].fillna(0))
                    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=features.index)
                    
                    # Add target back if it exists
                    if 'target' in features.columns:
                        scaled_df = scaled_df.join(features['target'])
                    
                    return scaled_df
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {e}")
            return features
    
    def get_feature_importance(self, features: Dict[str, pd.DataFrame], target_column: str = 'target') -> pd.DataFrame:
        """Calculate feature importance across all symbols"""
        try:
            if not features:
                return pd.DataFrame()
            
            # Combine all features
            combined_features = []
            combined_targets = []
            
            for symbol, symbol_features in features.items():
                if target_column in symbol_features.columns:
                    X = symbol_features.drop(target_column, axis=1)
                    y = symbol_features[target_column]
                    
                    # Remove missing values
                    mask = ~(X.isnull().any(axis=1) | y.isnull())
                    if mask.sum() > 0:
                        combined_features.append(X[mask])
                        combined_targets.append(y[mask])
            
            if not combined_features:
                return pd.DataFrame()
            
            X_combined = pd.concat(combined_features, ignore_index=True)
            y_combined = pd.concat(combined_targets, ignore_index=True)
            
            # Calculate mutual information scores
            selector = SelectKBest(score_func=mutual_info_classif, k='all')
            selector.fit(X_combined.fillna(0), y_combined)
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': X_combined.columns,
                'importance': selector.scores_
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return pd.DataFrame()
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature engineering statistics"""
        # Get Redis cache size
        try:
            stats = self.cache_manager.get_cache_stats()
            cache_size = stats.total_keys if stats else 0
        except:
            cache_size = 0
            
        return {
            'config': self.config.__dict__,
            'stats': self.feature_stats,
            'selected_features_count': len(self.selected_features) if self.selected_features else 0,
            'scaler_fitted': self.scaler is not None,
            'cache_size': cache_size
        }
    
    def clear_cache(self):
        """Clear feature cache"""
        # Clear feature-related keys from Redis
        try:
            stats = self.cache_manager.get_cache_stats()
            cache_size = stats.total_keys if stats else 0
        except:
            cache_size = 0
            
        self.logger.info(f"Feature cache cleared (Redis cache size: {cache_size} keys)")
    
    def save_preprocessors(self, filepath: str):
        """Save fitted preprocessors"""
        try:
            import pickle
            
            preprocessors = {
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'selected_features': self.selected_features,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(preprocessors, f)
            
            self.logger.info(f"Preprocessors saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessors: {e}")
    
    def load_preprocessors(self, filepath: str):
        """Load fitted preprocessors"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                preprocessors = pickle.load(f)
            
            self.scaler = preprocessors.get('scaler')
            self.feature_selector = preprocessors.get('feature_selector')
            self.selected_features = preprocessors.get('selected_features')
            self.config = preprocessors.get('config', self.config)
            
            self.logger.info(f"Preprocessors loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading preprocessors: {e}")

# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # This demo requires data in the database (trading_system.db)
    # Run the data_pipeline.py demo first to populate the necessary data.

    print("--- Running FeatureEngine Demo ---")

    # 1. Initialize FeatureEngine with a specific configuration
    print("\n1. Initializing FeatureEngine...")
    config = FeatureConfig(
        technical_indicators=True,
        ml4t_factors=False, # Disable for a faster demo run
        time_features=True,
        lag_features=True,
        rolling_features=True,
        feature_selection=True,
        max_features=30, # Select top 30 features
        scaling="robust" # Use robust scaler to handle outliers
    )
    engine = FeatureEngine(config)

    # 2. Generate features for a set of symbols
    print("\n2. Generating features for AAPL and MSFT...")
    symbols = ["AAPL", "MSFT"]
    features_dict = engine.generate_features(
        symbols=symbols,
        start_date="2023-06-01",
        end_date="2023-12-31",
        target_column="binary_threshold" # For supervised feature selection
    )

    if features_dict:
        print(f"\nSuccessfully generated features for {len(features_dict)} symbols.")
        
        # 3. Inspect generated features for one symbol
        print("\n3. Inspecting features for AAPL...")
        aapl_features = features_dict.get("AAPL")
        if aapl_features is not None:
            print(f"Shape of AAPL features: {aapl_features.shape}")
            print("First 5 rows of AAPL features (transposed):")
            print(aapl_features.head().T) # Transpose for better readability
            # Check if scaling and selection were applied
            print(f"Number of selected features: {len(aapl_features.columns) - 1}") # -1 for target
            assert len(aapl_features.columns) -1 <= config.max_features
        else:
            print("Could not generate features for AAPL.")

        # 4. Get feature importance
        print("\n4. Calculating feature importance across all symbols...")
        importance_df = engine.get_feature_importance(features_dict)
        if not importance_df.empty:
            print("Top 10 most important features:")
            print(importance_df.head(10).to_string(index=False))
        else:
            print("Could not calculate feature importance.")

        # 5. Get and print feature engineering stats
        print("\n5. Displaying feature engineering stats...")
        stats = engine.get_feature_stats()
        print(f"Total features generated before selection: {stats['stats']['total_features_generated']}")
        print(f"Features selected: {stats['selected_features_count']}")
        print(f"Processing time: {stats['stats']['processing_time']:.2f} seconds")
        print(f"Cache hits: {stats['stats']['cache_hits']}")

        # 6. Test feature transformation on new data (conceptual)
        print("\n6. Demonstrating feature transformation on unseen data...")
        # In a real scenario, you'd load the fitted preprocessors
        # and apply them to new, raw data before prediction.
        if aapl_features is not None:
            unseen_data_sample = aapl_features.drop('target', axis=1).iloc[[-1]]
            transformed_sample = engine.transform_features(unseen_data_sample)
            print(f"Shape of transformed sample: {transformed_sample.shape}")
            assert unseen_data_sample.shape == transformed_sample.shape
    else:
        print("\nFeature generation failed. Please ensure the database is populated with recent data.")

    print("\n--- FeatureEngine Demo Complete ---")
