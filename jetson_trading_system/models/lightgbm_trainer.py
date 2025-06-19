"""
LightGBM Model Training for Jetson Orin
Optimized ML4Trading implementation with GPU acceleration
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import gc

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingConfig
from jetson_trading_system.utils.ml4t_utils import ML4TradingUtils
from jetson_trading_system.utils.logger import get_model_logger
from jetson_trading_system.utils.database import TradingDatabase

class LightGBMTrainer:
    """
    LightGBM training optimized for Jetson Orin hardware
    Implements ML4Trading methodology with time series validation
    """
    
    def __init__(self, 
                 model_name: str = "jetson_trading_v1",
                 use_gpu: bool = True,
                 models_dir: str = "./models"):
        """
        Initialize LightGBM trainer
        
        Args:
            model_name: Name for the model
            use_gpu: Use GPU acceleration if available
            models_dir: Directory to save models
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and JetsonConfig().USE_GPU_ACCELERATION
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_model_logger()
        self.ml4t_utils = ML4TradingUtils()
        self.db_manager = TradingDatabase()
        
        # Model storage
        self.models = {}  # symbol -> model
        self.feature_importance = {}
        self.training_history = {}
        self.validation_scores = {}
        
        # Robust GPU detection - actually test model training
        if self.use_gpu:
            try:
                # Test actual GPU model training capability
                test_data = lgb.Dataset(np.random.random((100, 5)), np.random.randint(0, 2, 100))
                test_params = {
                    'objective': 'binary',
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'verbose': -1
                }
                # Actually try to train a model
                lgb.train(test_params, test_data, num_boost_round=1, valid_sets=[test_data])
                self.logger.info("GPU acceleration enabled for LightGBM")
            except Exception as e:
                self.logger.warning(f"GPU not available, falling back to CPU: {e}")
                self.use_gpu = False
    
    def _get_base_params(self) -> Dict[str, Any]:
        """Get base LightGBM parameters optimized for Jetson"""
        base_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,  # Conservative for Jetson memory
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': JetsonConfig.CPU_CORES,
            'max_depth': 8,  # Limit depth for Jetson
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        
        # Add GPU parameters if available
        if self.use_gpu:
            gpu_params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_bin': 63,  # GPU optimization
            }
            base_params.update(gpu_params)
        
        return base_params
    
    def prepare_training_data(self,
                            symbol: str,
                            start_date: str,
                            end_date: str,
                            min_samples: int = 200) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from database
        
        Args:
            symbol: Stock symbol
            start_date: Start date for training data
            end_date: End date for training data
            min_samples: Minimum samples required
            
        Returns:
            Features DataFrame and target Series
        """
        try:
            # Get price data
            price_data = self.db_manager.get_market_data(
                symbol=symbol, start_date=start_date, end_date=end_date
            )
            
            if len(price_data) < min_samples:
                raise ValueError(f"Insufficient data: {len(price_data)} < {min_samples}")
            
            # Get technical indicators (use pivot=True for ML-ready format)
            indicators = self.db_manager.get_technical_indicators(
                symbol=symbol, start_date=start_date, end_date=end_date, pivot=True
            )
            
            # Get market data (indices)
            market_data = {}
            for index in TradingConfig.MARKET_INDICES:
                try:
                    index_data = self.db_manager.get_market_data(
                        symbol=index, start_date=start_date, end_date=end_date
                    )
                    market_data[index] = index_data
                except Exception:
                    self.logger.warning(f"No market data for {index}")
            
            # Create features
            features = self._create_features(
                price_data, indicators, market_data, symbol
            )
            
            # Create target (next day return > threshold)
            returns = price_data['close'].pct_change().shift(-1)  # Next day return
            target = (returns > TradingConfig.MIN_PROFIT_THRESHOLD).astype(int)
            
            # Align features and target
            joined_data = features.join(target.rename('target'))
            
            # Drop problematic columns that are mostly NaN
            nan_ratios = joined_data.isna().sum() / len(joined_data)
            problematic_cols = nan_ratios[nan_ratios > 0.5].index.tolist()  # Drop cols >50% NaN
            if problematic_cols:
                self.logger.warning(f"Dropping problematic columns: {problematic_cols}")
                joined_data = joined_data.drop(columns=problematic_cols)
            
            # Use more intelligent NaN handling - keep rows with target values
            # Drop rows where target is NaN, then forward fill remaining features
            aligned_data = joined_data.dropna(subset=['target'])  # Only drop if target is NaN
            
            # Forward fill remaining NaN values in features (common for time series)
            feature_cols = [col for col in aligned_data.columns if col != 'target']
            aligned_data[feature_cols] = aligned_data[feature_cols].fillna(method='ffill')
            
            # Drop any remaining rows with NaN values (shouldn't be many)
            aligned_data = aligned_data.dropna()
            
            X = aligned_data.drop('target', axis=1)
            y = aligned_data['target']
            
            self.logger.info(f"Prepared {len(X)} samples for {symbol}")
            self.logger.info(f"Feature columns: {list(X.columns)}")
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data for {symbol}: {e}")
            raise
    
    def _create_features(self, 
                        price_data: pd.DataFrame,
                        indicators: pd.DataFrame,
                        market_data: Dict[str, pd.DataFrame],
                        symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Price-based features
        features['returns_1d'] = price_data['close'].pct_change()
        features['returns_5d'] = price_data['close'].pct_change(5)
        features['returns_20d'] = price_data['close'].pct_change(20)
        
        # Volume features
        features['volume_ratio'] = (
            price_data['volume'] / price_data['volume'].rolling(20).mean()
        )
        features['volume_trend'] = price_data['volume'].pct_change(5)
        
        # Volatility features
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        
        # Technical indicators (now in pivoted format)
        if not indicators.empty:
            # Align indicators with price data by timestamp
            indicators_aligned = indicators.reindex(price_data.index)
            
            # RSI features (using actual column name: rsi_14)
            if 'rsi_14' in indicators_aligned.columns:
                features['rsi'] = indicators_aligned['rsi_14']
                features['rsi_oversold'] = (indicators_aligned['rsi_14'] < 30).astype(int)
                features['rsi_overbought'] = (indicators_aligned['rsi_14'] > 70).astype(int)
            
            # MACD features (using actual column names: macd_line, macd_signal, macd_histogram)
            if all(col in indicators_aligned.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
                features['macd'] = indicators_aligned['macd_line']
                features['macd_signal'] = indicators_aligned['macd_signal']
                features['macd_histogram'] = indicators_aligned['macd_histogram']
                features['macd_bullish'] = (indicators_aligned['macd_line'] > indicators_aligned['macd_signal']).astype(int)
            
            # Bollinger Bands (using actual column names: bb_upper, bb_middle, bb_lower, bb_ratio)
            if all(col in indicators_aligned.columns for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_ratio']):
                features['bb_position'] = indicators_aligned['bb_ratio']  # bb_ratio is the %B indicator (position within bands)
                bb_width = (indicators_aligned['bb_upper'] - indicators_aligned['bb_lower']) / indicators_aligned['bb_middle']
                features['bb_width'] = bb_width
                features['bb_squeeze'] = (bb_width < bb_width.rolling(20).quantile(0.2)).astype(int)
            
            # Moving averages (using actual column name: sma_ratio_20)
            if 'sma_ratio_20' in indicators_aligned.columns:
                features['sma_ratio'] = indicators_aligned['sma_ratio_20']
                features['price_above_sma'] = (indicators_aligned['sma_ratio_20'] > 1).astype(int)
            
            # Stochastic (using actual column name: stoch_k)
            if 'stoch_k' in indicators_aligned.columns:
                features['stoch_k'] = indicators_aligned['stoch_k']
                features['stoch_oversold'] = (indicators_aligned['stoch_k'] < 20).astype(int)
                features['stoch_overbought'] = (indicators_aligned['stoch_k'] > 80).astype(int)
            
            # Williams %R (using actual column name: williams_r_14)
            if 'williams_r_14' in indicators_aligned.columns:
                features['williams_r'] = indicators_aligned['williams_r_14']
            
            # ATR (using actual column name: atr_14)
            if 'atr_14' in indicators_aligned.columns:
                features['atr_ratio'] = indicators_aligned['atr_14'] / price_data['close']
        
        # Market context features with improved data alignment
        for index_name, index_data in market_data.items():
            if not index_data.empty:
                prefix = index_name.lower()
                
                # Align index data with price data timestamps
                index_aligned = index_data.reindex(price_data.index, method='ffill')
                
                if not index_aligned.empty and 'close' in index_aligned.columns:
                    index_returns = index_aligned['close'].pct_change()
                    index_returns_5d = index_aligned['close'].pct_change(5)
                    
                    # Only add features if we have sufficient valid data (>50% coverage)
                    if index_returns.notna().sum() > len(price_data) * 0.5:
                        features[f'{prefix}_return'] = index_returns
                        features[f'{prefix}_return_5d'] = index_returns_5d
                        
                        # Correlation with market (only if both series have enough data)
                        correlation = features['returns_1d'].rolling(20).corr(index_returns)
                        if correlation.notna().sum() > len(price_data) * 0.3:  # At least 30% valid correlations
                            features[f'{prefix}_correlation'] = correlation
        
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
        
        return features
    
    def train_model(self, 
                   symbol: str,
                   start_date: str,
                   end_date: str,
                   cv_folds: int = 5,
                   early_stopping_rounds: int = 100) -> Dict[str, Any]:
        """
        Train LightGBM model with time series cross-validation
        
        Args:
            symbol: Stock symbol
            start_date: Training start date
            end_date: Training end date
            cv_folds: Number of CV folds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Training results dictionary
        """
        try:
            self.logger.info(f"Starting training for {symbol}")
            
            # Prepare data
            X, y = self.prepare_training_data(symbol, start_date, end_date)
            
            if len(X) < 100:
                raise ValueError(f"Insufficient training data: {len(X)} samples")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = []
            feature_importance_scores = []
            
            best_model = None
            best_score = 0
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                self.logger.info(f"Training fold {fold + 1}/{cv_folds}")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create datasets
                train_set = lgb.Dataset(X_train, label=y_train)
                val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
                
                # Train model
                params = self._get_base_params()
                model = lgb.train(
                    params,
                    train_set,
                    valid_sets=[train_set, val_set],
                    valid_names=['train', 'val'],
                    num_boost_round=1000,
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds),
                        lgb.log_evaluation(period=0)  # Suppress output
                    ]
                )
                
                # Validate
                y_pred = model.predict(X_val)
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_val, y_pred_binary)
                precision = precision_score(y_val, y_pred_binary, zero_division=0)
                recall = recall_score(y_val, y_pred_binary, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_val, y_pred)
                except ValueError:
                    auc = 0.5  # If only one class in validation
                
                fold_score = {
                    'fold': fold + 1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                }
                
                cv_scores.append(fold_score)
                feature_importance_scores.append(model.feature_importance(importance_type='gain'))
                
                self.logger.info(f"Fold {fold + 1} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
                
                # Keep best model
                if auc > best_score:
                    best_score = auc
                    best_model = model
                
                # Memory cleanup
                del model, train_set, val_set
                gc.collect()
            
            # Calculate average feature importance
            avg_feature_importance = np.mean(feature_importance_scores, axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': avg_feature_importance
            }).sort_values('importance', ascending=False)
            
            # Calculate CV metrics
            cv_results = {
                'mean_accuracy': np.mean([s['accuracy'] for s in cv_scores]),
                'std_accuracy': np.std([s['accuracy'] for s in cv_scores]),
                'mean_auc': np.mean([s['auc'] for s in cv_scores]),
                'std_auc': np.std([s['auc'] for s in cv_scores]),
                'mean_precision': np.mean([s['precision'] for s in cv_scores]),
                'mean_recall': np.mean([s['recall'] for s in cv_scores]),
                'cv_scores': cv_scores
            }
            
            # Train final model on all data
            self.logger.info("Training final model on all data")
            final_train_set = lgb.Dataset(X, label=y)
            
            params = self._get_base_params()
            final_model = lgb.train(
                params,
                final_train_set,
                num_boost_round=best_model.num_trees(),
                callbacks=[lgb.log_evaluation(period=0)]
            )
            
            # Store results
            self.models[symbol] = final_model
            self.feature_importance[symbol] = feature_importance_df
            self.validation_scores[symbol] = cv_results
            
            training_result = {
                'symbol': symbol,
                'model': final_model,
                'cv_results': cv_results,
                'feature_importance': feature_importance_df,
                'training_samples': len(X),
                'features': list(X.columns),
                'trained_date': datetime.now().isoformat(),
                'model_path': str(self.models_dir / f"{symbol}_{self.model_name}.pkl")
            }
            
            # Save model
            self.save_model(symbol, training_result)
            
            self.logger.info(f"Training completed for {symbol}")
            self.logger.info(f"CV Results - Accuracy: {cv_results['mean_accuracy']:.3f}±{cv_results['std_accuracy']:.3f}")
            self.logger.info(f"CV Results - AUC: {cv_results['mean_auc']:.3f}±{cv_results['std_auc']:.3f}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Training failed for {symbol}: {e}")
            raise
    
    def save_model(self, symbol: str, training_result: Dict[str, Any]):
        """Save trained model and metadata"""
        try:
            model_path = self.models_dir / f"{symbol}_{self.model_name}.pkl"
            metadata_path = self.models_dir / f"{symbol}_{self.model_name}_metadata.json"
            
            # Save model
            joblib.dump(training_result['model'], model_path)
            
            # Save metadata (excluding the model object)
            metadata = training_result.copy()
            del metadata['model']  # Can't serialize LightGBM model in JSON
            
            # Convert DataFrames to dict for JSON serialization
            if 'feature_importance' in metadata:
                metadata['feature_importance'] = metadata['feature_importance'].to_dict('records')
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model for {symbol}: {e}")
            raise
    
    def load_model(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load trained model and metadata"""
        try:
            model_path = self.models_dir / f"{symbol}_{self.model_name}.pkl"
            metadata_path = self.models_dir / f"{symbol}_{self.model_name}_metadata.json"
            
            if not model_path.exists():
                return None
            
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Convert feature importance back to DataFrame
                if 'feature_importance' in metadata:
                    metadata['feature_importance'] = pd.DataFrame(metadata['feature_importance'])
            
            result = metadata.copy()
            result['model'] = model
            
            # Store in memory
            self.models[symbol] = model
            if 'feature_importance' in result:
                self.feature_importance[symbol] = result['feature_importance']
            if 'cv_results' in result:
                self.validation_scores[symbol] = result['cv_results']
            
            self.logger.info(f"Model loaded: {model_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading model for {symbol}: {e}")
            return None
    
    def get_feature_importance(self, symbol: str, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importance for symbol"""
        if symbol in self.feature_importance:
            return self.feature_importance[symbol].head(top_n)
        return pd.DataFrame()
    
    def batch_train(self, 
                   symbols: List[str],
                   start_date: str,
                   end_date: str) -> Dict[str, Dict[str, Any]]:
        """Train models for multiple symbols"""
        results = {}
        
        for i, symbol in enumerate(symbols):
            try:
                self.logger.info(f"Training {symbol} ({i+1}/{len(symbols)})")
                result = self.train_model(symbol, start_date, end_date)
                results[symbol] = result
                
                # Memory cleanup between symbols
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Failed to train {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results

# Example usage
if __name__ == "__main__":
    print("--- Running LightGBMTrainer Demo in Live Mode ---")
    
    # This demo will use the live TradingDatabase to pull real data.
    # Make sure you have some data in your database first for symbols like 'AAPL' and 'SPY'.
    # You can run the data_pipeline.py demo to populate it.

    # Initialize the trainer, which will use the live TradingDatabase
    trainer = LightGBMTrainer(models_dir="./models_demo", use_gpu=False) # Recommend CPU for local demo

    try:
        # Train a model using live data
        print("\nTraining a model for AAPL...")
        result = trainer.train_model(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        if result:
            print("\n--- Training Results ---")
            print(f"Training completed for symbol: {result['symbol']}")
            print(f"CV Mean AUC: {result['cv_results']['mean_auc']:.4f}")
            print(f"Model saved to: {result['model_path']}")

            print("\n--- Top 10 Feature Importances ---")
            print(result['feature_importance'].head(10))

        else:
            print("\nModel training failed. This can happen if there is insufficient data in the database for the specified date range.")
            print("Please run the data_pipeline.py demo to populate the database.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during the demo: {e}")
        traceback.print_exc()
        print("\nPlease ensure your database is populated and accessible.")
    
    finally:
        # Clean up the demo models directory
        import shutil
        demo_dir = Path("./models_demo")
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"\nCleaned up demo directory: {demo_dir}")

    print("\n--- Demo Finished ---")
