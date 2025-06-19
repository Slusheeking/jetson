"""
Model Prediction Engine for Jetson Trading System
Real-time inference with ML4Trading methodology
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from pathlib import Path
import time
from datetime import datetime, timedelta
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingConfig
from jetson_trading_system.utils.logger import get_model_logger
from jetson_trading_system.data.cache_manager import CacheManager
from jetson_trading_system.utils.database import TradingDatabase
from jetson_trading_system.features.technical_indicators import TechnicalIndicators
from jetson_trading_system.models.model_registry import ModelRegistry

class ModelPredictor:
    """
    Real-time prediction engine optimized for Jetson Orin
    Handles inference for multiple symbols with performance monitoring
    """
    
    def __init__(self, 
                 models_dir: str = "./models",
                 max_workers: int = None,
                 cache_manager: CacheManager = None):
        """
        Initialize model predictor
        
        Args:
            models_dir: Directory containing trained models
            max_workers: Max threads for parallel prediction
            cache_manager: Redis cache manager for predictions
        """
        self.models_dir = Path(models_dir)
        self.max_workers = max_workers or min(JetsonConfig.CPU_CORES, 4)
        
        self.logger = get_model_logger()
        self.db_manager = TradingDatabase()
        self.ta_calculator = TechnicalIndicators()
        self.model_registry = ModelRegistry()
        
        # Use Redis cache manager
        self.cache_manager = cache_manager or CacheManager()
        
        # Model storage
        self.loaded_models = {}  # symbol -> model_info
        self.model_metadata = {}  # symbol -> metadata
        self.feature_columns = {}  # symbol -> feature_list
        
        # Performance tracking
        self.prediction_times = {}  # symbol -> [times]
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.prediction_queue = queue.Queue()
        self.prediction_lock = threading.Lock()
        
        # Initialize
        self._load_available_models()
        
        self.logger.info(f"ModelPredictor initialized with {len(self.loaded_models)} models")
    
    def _load_available_models(self):
        """Load all available trained models"""
        if not self.models_dir.exists():
            self.logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        model_files = list(self.models_dir.glob("*_jetson_trading_v1.pkl"))
        
        for model_file in model_files:
            try:
                # Extract symbol from filename
                symbol = model_file.stem.replace("_jetson_trading_v1", "")
                self._load_model(symbol)
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_file}: {e}")
    
    def _load_model(self, symbol: str) -> bool:
        """Load a specific model and its metadata"""
        try:
            model_path = self.models_dir / f"{symbol}_jetson_trading_v1.pkl"
            metadata_path = self.models_dir / f"{symbol}_jetson_trading_v1_metadata.json"
            
            if not model_path.exists():
                return False
            
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata
            metadata = {}
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Store model info
            self.loaded_models[symbol] = {
                'model': model,
                'loaded_time': datetime.now(),
                'model_path': model_path,
                'prediction_count': 0
            }
            
            self.model_metadata[symbol] = metadata
            
            # Extract feature columns from metadata
            if 'features' in metadata:
                self.feature_columns[symbol] = metadata['features']
            else:
                # Fallback: try to get from model if available
                try:
                    self.feature_columns[symbol] = model.feature_name()
                except:
                    self.logger.warning(f"Could not determine features for {symbol}")
                    self.feature_columns[symbol] = []
            
            self.logger.info(f"Loaded model for {symbol} with {len(self.feature_columns[symbol])} features")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model for {symbol}: {e}")
            return False
    
    def reload_model(self, symbol: str) -> bool:
        """Reload a model (useful for model updates)"""
        if symbol in self.loaded_models:
            del self.loaded_models[symbol]
            del self.model_metadata[symbol]
            del self.feature_columns[symbol]
        
        return self._load_model(symbol)
    
    def get_latest_features(self,
                          symbol: str,
                          lookback_days: int = 60) -> Optional[pd.DataFrame]:
        """
        Get latest features for prediction using pre-loaded database indicators
        
        Args:
            symbol: Stock symbol
            lookback_days: Days of historical data for feature calculation
            
        Returns:
            Latest features DataFrame or None if insufficient data
        """
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Get price data from database
            price_data = self.db_manager.get_market_data(symbol, start_date, end_date)
            
            if len(price_data) < 20:
                self.logger.warning(f"Insufficient price data for {symbol}: {len(price_data)} bars")
                return None
            
            # Get pre-calculated technical indicators from database (pivot format)
            indicators = self.db_manager.get_technical_indicators(symbol, start_date, end_date, pivot=True)
            
            if len(indicators) < 10:
                self.logger.warning(f"Insufficient indicator data for {symbol}: {len(indicators)} bars")
                return None
            
            # Get market data for context features
            market_data = {}
            for index in TradingConfig.MARKET_INDICES:
                try:
                    index_data = self.db_manager.get_market_data(index, start_date, end_date)
                    if not index_data.empty:
                        market_data[index] = index_data
                except Exception:
                    pass
            
            # Create features using database indicators
            features = self._create_prediction_features_from_db(price_data, indicators, market_data, symbol)
            
            if features.empty:
                return None
            
            # Return only the latest row
            return features.iloc[[-1]]
            
        except Exception as e:
            self.logger.error(f"Error getting features for {symbol}: {e}")
            return None
    
    def _create_prediction_features_from_db(self,
                                           price_data: pd.DataFrame,
                                           indicators: pd.DataFrame,
                                           market_data: Dict[str, pd.DataFrame],
                                           symbol: str) -> pd.DataFrame:
        """Create features for prediction using database indicators (matches training)"""
        
        try:
            # Join indicators to price_data, ensuring all price_data rows are kept
            if not indicators.empty:
                # Reindex indicators to match price_data's index, filling missing forward
                aligned_indicators = indicators.reindex(price_data.index, method='ffill')
                merged_data = price_data.join(aligned_indicators, how='left', rsuffix='_ind')
            else:
                merged_data = price_data.copy()

            if merged_data.empty:
                self.logger.warning(f"No data for {symbol} after joining indicators")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=merged_data.index)
            
            # Price-based features (use price data columns)
            features['returns_1d'] = merged_data['close'].pct_change()
            features['returns_5d'] = merged_data['close'].pct_change(5)
            features['returns_20d'] = merged_data['close'].pct_change(20)
            
            # Volume features (use price data columns)
            features['volume_ratio'] = (
                merged_data['volume'] / merged_data['volume'].rolling(20).mean()
            )
            features['volume_trend'] = merged_data['volume'].pct_change(5)
            
            # Volatility features
            features['volatility_5d'] = features['returns_1d'].rolling(5).std()
            features['volatility_20d'] = features['returns_1d'].rolling(20).std()
            
            # Technical indicators from database (pre-calculated)
            if not indicators.empty:
                # RSI features - use database column names
                if 'rsi_14' in merged_data.columns:
                    features['rsi'] = merged_data['rsi_14']
                    features['rsi_oversold'] = (merged_data['rsi_14'] < 30).astype(int)
                    features['rsi_overbought'] = (merged_data['rsi_14'] > 70).astype(int)
                
                # MACD features
                if 'macd_line' in merged_data.columns:
                    features['macd'] = merged_data['macd_line']
                if 'macd_signal' in merged_data.columns:
                    features['macd_signal'] = merged_data['macd_signal']
                if 'macd_histogram' in merged_data.columns:
                    features['macd_histogram'] = merged_data['macd_histogram']
                if 'macd_line' in merged_data.columns and 'macd_signal' in merged_data.columns:
                    features['macd_bullish'] = (merged_data['macd_line'] > merged_data['macd_signal']).astype(int)
                
                # Bollinger Bands
                if 'bb_ratio' in merged_data.columns:
                    features['bb_position'] = merged_data['bb_ratio']
                if 'bb_upper' in merged_data.columns and 'bb_lower' in merged_data.columns:
                    features['bb_width'] = merged_data['bb_upper'] - merged_data['bb_lower']
                    features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).quantile(0.2)).astype(int)
                
                # Moving averages
                if 'sma_ratio_20' in merged_data.columns:
                    features['sma_ratio'] = merged_data['sma_ratio_20']
                    features['price_above_sma'] = (merged_data['sma_ratio_20'] > 1).astype(int)
                
                # Stochastic
                if 'stoch_k' in merged_data.columns:
                    features['stoch_k'] = merged_data['stoch_k']
                    features['stoch_oversold'] = (merged_data['stoch_k'] < 20).astype(int)
                    features['stoch_overbought'] = (merged_data['stoch_k'] > 80).astype(int)
                
                # Williams %R
                if 'williams_r_14' in merged_data.columns:
                    features['williams_r'] = merged_data['williams_r_14']
                
                # ATR (normalized)
                if 'atr_14' in merged_data.columns:
                    features['atr_ratio'] = merged_data['atr_14'] / merged_data['close']
            
            # Market context features with proper index alignment
            for index_name, index_data in market_data.items():
                if not index_data.empty:
                    try:
                        prefix = index_name.lower()
                        
                        # Align index data with merged_data timestamps
                        aligned_index_data = index_data.reindex(merged_data.index, method='ffill')
                        
                        if not aligned_index_data.empty and aligned_index_data['close'].notna().sum() > 0:
                            index_returns = aligned_index_data['close'].pct_change()
                            index_returns_5d = aligned_index_data['close'].pct_change(5)
                            
                            features[f'{prefix}_return'] = index_returns
                            features[f'{prefix}_return_5d'] = index_returns_5d
                            
                            # Correlation with market (only if sufficient data)
                            if len(features['returns_1d'].dropna()) >= 20 and len(index_returns.dropna()) >= 20:
                                features[f'{prefix}_correlation'] = (
                                    features['returns_1d'].rolling(20).corr(index_returns)
                                )
                            else:
                                features[f'{prefix}_correlation'] = 0.0
                    except Exception as e:
                        self.logger.warning(f"Error processing market data for {index_name}: {e}")
                        # Set default values if market data fails
                        prefix = index_name.lower()
                        features[f'{prefix}_return'] = 0.0
                        features[f'{prefix}_return_5d'] = 0.0
                        features[f'{prefix}_correlation'] = 0.0
            
            # Time-based features
            features['hour'] = features.index.hour
            features['day_of_week'] = features.index.dayofweek
            features['month'] = features.index.month
            features['quarter'] = features.index.quarter
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                features[f'return_lag_{lag}'] = features['returns_1d'].shift(lag)
                if 'rsi' in features.columns:
                    features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
            
            return features.dropna()
            
        except Exception as e:
            self.logger.error(f"Error creating features for {symbol}: {e}")
            return pd.DataFrame()
    
    def predict(self,
               symbol: str,
               features_df: Optional[pd.DataFrame] = None,
               use_cache: bool = True,
               return_probability: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generate prediction for a symbol
        
        Args:
            symbol: Stock symbol
            use_cache: Use cached predictions if available
            return_probability: Return probability score instead of binary
            
        Returns:
            Prediction dictionary with signal, confidence, etc.
        """
        try:
            start_time = time.time()
            
            # Check if model is loaded
            if symbol not in self.loaded_models:
                if not self._load_model(symbol):
                    self.logger.warning(f"No model available for {symbol}")
                    return None
            
            # Generate a cache key that is specific to the date of the features
            cache_key = None
            if use_cache and features_df is not None and not features_df.empty:
                # Use the timestamp from the features DataFrame for the cache key
                feature_date = features_df.index[0].strftime('%Y-%m-%d')
                cache_key = f"prediction:{symbol}:{feature_date}"
                
                cached_prediction = self.cache_manager.get(cache_key)
                if cached_prediction:
                    self.cache_hits += 1
                    return cached_prediction
            
            self.cache_misses += 1
            
            # If features are not provided, get the latest from the database
            if features_df is None:
                features = self.get_latest_features(symbol)
            else:
                features = features_df

            if features is None or features.empty:
                self.logger.warning(f"No features available for {symbol}, cannot predict.")
                return None
            
            # Align features with model expectations
            model_features = self.feature_columns[symbol]
            
            # Select only features that model expects
            available_features = [f for f in model_features if f in features.columns]
            missing_features = [f for f in model_features if f not in features.columns]
            
            if missing_features and use_cache:  # In backtesting (use_cache=False), it's expected to have missing features for early dates
                self.logger.warning(f"Missing features for {symbol}: {missing_features}")
            
            if not available_features:
                self.logger.error(f"No matching features for {symbol}")
                return None
            
            # Prepare feature vector
            X = features[available_features].fillna(0)  # Fill missing with 0
            
            # Add missing features as zeros
            for missing_feature in missing_features:
                X[missing_feature] = 0
            
            # Reorder to match model training order
            X = X[model_features]
            
            # Make prediction
            model = self.loaded_models[symbol]['model']
            
            # Get probability prediction
            prob_prediction = model.predict(X)[0]
            
            # Convert to binary signal
            binary_prediction = 1 if prob_prediction > 0.5 else 0
            
            # Calculate confidence (distance from 0.5)
            confidence = abs(prob_prediction - 0.5) * 2
            
            # Determine signal strength
            if prob_prediction > 0.7:
                signal_strength = "STRONG_BUY"
            elif prob_prediction > 0.6:
                signal_strength = "BUY"
            elif prob_prediction > 0.4:
                signal_strength = "HOLD"
            elif prob_prediction > 0.3:
                signal_strength = "SELL"
            else:
                signal_strength = "STRONG_SELL"
            
            # Create prediction result
            prediction_result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'signal': binary_prediction,
                'probability': prob_prediction,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'features_used': len(available_features),
                'features_missing': len(missing_features),
                'model_info': {
                    'loaded_time': self.loaded_models[symbol]['loaded_time'].isoformat(),
                    'prediction_count': self.loaded_models[symbol]['prediction_count'] + 1
                }
            }
            
            # Update counters
            self.loaded_models[symbol]['prediction_count'] += 1
            
            # Track performance
            prediction_time = time.time() - start_time
            if symbol not in self.prediction_times:
                self.prediction_times[symbol] = []
            self.prediction_times[symbol].append(prediction_time)
            
            # Keep only last 100 times
            if len(self.prediction_times[symbol]) > 100:
                self.prediction_times[symbol] = self.prediction_times[symbol][-100:]
            
            prediction_result['prediction_time'] = prediction_time
            
            # Cache result in Redis
            if use_cache:
                self.cache_manager.put(cache_key, prediction_result, ttl_seconds=300)  # 5 minute TTL
            
            self.logger.info(f"Prediction for {symbol}: {signal_strength} (prob={prob_prediction:.3f}, conf={confidence:.3f})")
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            return None
    
    def batch_predict(self, 
                     symbols: List[str],
                     use_cache: bool = True) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Generate predictions for multiple symbols in parallel
        
        Args:
            symbols: List of stock symbols
            use_cache: Use cached predictions if available
            
        Returns:
            Dictionary of symbol -> prediction results
        """
        try:
            # Use ThreadPoolExecutor for parallel predictions
            future_to_symbol = {
                self.executor.submit(self.predict, symbol, use_cache): symbol 
                for symbol in symbols
            }
            
            results = {}
            
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results[symbol] = result
                except Exception as e:
                    self.logger.error(f"Batch prediction failed for {symbol}: {e}")
                    results[symbol] = None
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return {}
    
    async def async_predict(self, 
                          symbol: str,
                          use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Async wrapper for prediction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.predict, symbol, use_cache)
    
    def get_prediction_performance(self, symbol: str = None) -> Dict[str, Any]:
        """Get prediction performance metrics"""
        if symbol and symbol in self.prediction_times:
            times = self.prediction_times[symbol]
            return {
                'symbol': symbol,
                'avg_prediction_time': np.mean(times),
                'min_prediction_time': np.min(times),
                'max_prediction_time': np.max(times),
                'total_predictions': len(times),
                'model_loaded_time': self.loaded_models.get(symbol, {}).get('loaded_time')
            }
        else:
            # Overall performance
            all_times = []
            for times in self.prediction_times.values():
                all_times.extend(times)
            
            # Get Redis cache size
            try:
                stats = self.cache_manager.get_cache_stats()
                cache_size = stats.total_keys if stats else 0
            except:
                cache_size = 0
                
            return {
                'total_predictions': len(all_times),
                'avg_prediction_time': np.mean(all_times) if all_times else 0,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                'cache_size': cache_size,
                'loaded_models': len(self.loaded_models),
                'symbols': list(self.loaded_models.keys())
            }
    
    def clear_cache(self):
        """Clear prediction cache"""
        # Clear prediction-related keys from Redis
        try:
            # Get cache stats from Redis
            stats = self.cache_manager.get_cache_stats()
            if stats:
                cache_size = stats.total_keys
            else:
                cache_size = 0
        except:
            cache_size = 0
            
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger.info(f"Prediction cache cleared (Redis cache size: {cache_size} keys)")
    
    def get_model_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get information about loaded models"""
        if symbol:
            if symbol in self.loaded_models:
                info = self.loaded_models[symbol].copy()
                info.update(self.model_metadata.get(symbol, {}))
                return info
            return {}
        else:
            return {
                symbol: {
                    **model_info,
                    **self.model_metadata.get(symbol, {})
                }
                for symbol, model_info in self.loaded_models.items()
            }
    
    def validate_features(self, symbol: str) -> Dict[str, Any]:
        """Validate that current features match model expectations"""
        if symbol not in self.loaded_models:
            return {'valid': False, 'error': 'Model not loaded'}
        
        try:
            features = self.get_latest_features(symbol)
            if features is None:
                return {'valid': False, 'error': 'Cannot get current features'}
            
            model_features = set(self.feature_columns[symbol])
            current_features = set(features.columns)
            
            missing = model_features - current_features
            extra = current_features - model_features
            
            return {
                'valid': len(missing) == 0,
                'model_features': len(model_features),
                'current_features': len(current_features),
                'missing_features': list(missing),
                'extra_features': list(extra),
                'match_ratio': len(model_features & current_features) / len(model_features)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def close(self):
        """Clean shutdown"""
        self.executor.shutdown(wait=True)
        self.logger.info("ModelPredictor shutdown complete")

# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv

    load_dotenv()
    
    # This demo requires a trained model to exist in the ./models directory
    # You can generate one by running the lightgbm_trainer.py script
    
    print("--- Running ModelPredictor Demo ---")
    
    # Initialize predictor
    # Ensure you have models like 'AAPL_jetson_trading_v1.pkl' in the './models' directory
    predictor = ModelPredictor(models_dir="./models")

    # --- Test Case 1: Single Synchronous Prediction ---
    print("\n--- 1. Single Synchronous Prediction for AAPL ---")
    # This requires data in the database for AAPL.
    # The data_pipeline.py demo can be used to populate it.
    aapl_prediction = predictor.predict("AAPL")
    if aapl_prediction:
        print(f"AAPL Prediction: {aapl_prediction['signal_strength']} (Prob: {aapl_prediction['probability']:.3f})")
        print(f"Prediction generated in {aapl_prediction.get('prediction_time', 0):.4f} seconds.")
    else:
        print("Could not generate prediction for AAPL. Ensure model exists and DB has data.")

    # --- Test Case 2: Batch Prediction ---
    print("\n--- 2. Batch Prediction for multiple symbols ---")
    symbols = ["AAPL", "MSFT", "GOOGL"] # Assumes models exist for these
    batch_results = predictor.batch_predict(symbols)
    
    for symbol, result in batch_results.items():
        if result:
            print(f"  - {symbol}: {result['signal_strength']} (Confidence: {result['confidence']:.3f})")
        else:
            print(f"  - {symbol}: Prediction failed (check model/data)")

    # --- Test Case 3: Asynchronous Prediction ---
    async def run_async_prediction():
        print("\n--- 3. Asynchronous Prediction for TSLA ---")
        tsla_prediction = await predictor.async_predict("TSLA")
        if tsla_prediction:
            print(f"TSLA Async Prediction: {tsla_prediction['signal_strength']}")
        else:
            print("Could not generate async prediction for TSLA.")
    
    asyncio.run(run_async_prediction())

    # --- Test Case 4: Performance and Model Info ---
    print("\n--- 4. Performance & Model Information ---")
    # Run a few more predictions to gather stats
    for _ in range(5):
        predictor.predict("AAPL", use_cache=False)
        
    perf = predictor.get_prediction_performance("AAPL")
    print(f"AAPL Avg. Prediction Time: {perf.get('avg_prediction_time', 0):.4f}s")
    
    overall_perf = predictor.get_prediction_performance()
    print(f"Overall Cache Hit Rate: {overall_perf.get('cache_hit_rate', 0):.1%}")

    model_info = predictor.get_model_info("AAPL")
    if model_info:
        print(f"AAPL Model Info - Trained: {model_info.get('training_date', 'N/A')}, Accuracy: {model_info.get('cv_accuracy', 0):.3f}")

    # --- Test Case 5: Feature Validation ---
    print("\n--- 5. Feature Validation for MSFT ---")
    validation = predictor.validate_features("MSFT")
    if validation['valid']:
        print("MSFT features are valid.")
    else:
        print(f"MSFT feature mismatch: {validation.get('error') or validation.get('missing_features')}")

    # --- Shutdown ---
    predictor.close()
    print("\n--- ModelPredictor Demo Complete ---")
