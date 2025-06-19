"""
Zipline-style Backtesting Engine for Jetson Trading System
Custom backtesting framework optimized for ML4Trading methodology
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingConfig
from jetson_trading_system.utils.logger import get_model_logger
from jetson_trading_system.utils.database import TradingDatabase
from jetson_trading_system.models.lightgbm_trainer import LightGBMTrainer
from jetson_trading_system.models.model_predictor import ModelPredictor
from jetson_trading_system.risk.position_sizer import PositionSizer, SizingInput
from jetson_trading_system.features.technical_indicators import TechnicalIndicators
from jetson_trading_system.features.feature_engine import FeatureEngine

class BacktestState(Enum):
    """Backtest execution states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BacktestTrade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    duration_days: int
    signal_strength: float
    model_confidence: float
    commission: float

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    portfolio_history: List[Dict[str, Any]]
    trades: List[BacktestTrade]
    daily_returns: List[float]
    benchmark_return: Optional[float] = None
    information_ratio: Optional[float] = None

class ZiplineEngine:
    """
    Custom backtesting engine implementing Zipline-like functionality
    Optimized for ML4Trading strategies on Jetson hardware
    """
    
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 25000.0,
                 commission_per_share: float = 0.0,
                 benchmark_symbol: str = "SPY"):
        """
        Initialize backtesting engine
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
            commission_per_share: Commission per share
            benchmark_symbol: Benchmark symbol for comparison
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.benchmark_symbol = benchmark_symbol
        
        self.logger = get_model_logger()
        self.db_manager = TradingDatabase()
        
        # Strategy components
        self.model_trainer = None
        self.model_predictor = None
        self.position_sizer = PositionSizer()
        self.feature_engine = FeatureEngine()
        
        # Backtest state
        self.state = BacktestState.INITIALIZING
        self.current_date = self.start_date
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.open_trades = {}  # symbol -> trade_info
        
        # Results tracking
        self.trades = []
        self.portfolio_history = []
        self.daily_returns = []
        self.benchmark_data = None
        
        # Performance metrics
        self.high_water_mark = initial_capital
        self.drawdown_history = []
        
        # Data cache - pre-loaded at initialization
        self.price_data = {}  # symbol -> DataFrame
        self.all_features = {} # symbol -> DataFrame of all historical features
        
        self.logger.info(f"ZiplineEngine initialized: {start_date} to {end_date}")
    
    def add_strategy(self, 
                    symbols: List[str],
                    model_name: str = "jetson_trading_v1",
                    retrain_frequency: int = 30,
                    signal_threshold: float = 0.6):
        """
        Add ML trading strategy
        
        Args:
            symbols: List of symbols to trade
            model_name: Model name for predictions
            retrain_frequency: Days between model retraining
            signal_threshold: Minimum signal confidence threshold
        """
        try:
            self.symbols = symbols
            self.model_name = model_name
            self.retrain_frequency = retrain_frequency
            self.signal_threshold = signal_threshold
            
            # Initialize components (disable GPU for stability)
            self.model_trainer = LightGBMTrainer(model_name=model_name, use_gpu=False)
            self.model_predictor = ModelPredictor()
            
            # Load and prepare data
            self._prepare_data()
            
            # Train initial models
            self._train_initial_models()
            
            self.logger.info(f"Strategy added for {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error adding strategy: {e}")
            raise
    
    def _prepare_data(self):
        """Prepare historical data and generate all features for the backtest period."""
        try:
            self.logger.info("Preparing historical data for backtest...")

            # Define date ranges
            backtest_start_str = self.start_date.strftime("%Y-%m-%d")
            backtest_end_str = self.end_date.strftime("%Y-%m-%d")
            # Fetch a bit more historical data for feature calculation lookback
            features_start_str = (self.start_date - timedelta(days=365)).strftime("%Y-%m-%d")

            # 1. Load price data for backtest period
            for symbol in self.symbols:
                try:
                    # Load data for the full range needed for features + backtest
                    price_data = self.db_manager.get_market_data(symbol, features_start_str, backtest_end_str)
                    if price_data is not None and not price_data.empty:
                        self.price_data[symbol] = price_data
                        self.logger.info(f"Loaded {len(price_data)} price bars for {symbol}")
                    else:
                        self.logger.warning(f"No price data available for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error loading price data for {symbol}: {e}")

            # 2. Generate features for the backtesting period (no target needed)
            self.logger.info("Generating features for the backtest period...")
            self.all_features = self.feature_engine.generate_features(
                symbols=self.symbols,
                start_date=features_start_str,
                end_date=backtest_end_str,
                target_column=None  # No target needed for prediction features
            )

            for symbol, features in self.all_features.items():
                if features is not None and not features.empty:
                    self.logger.info(f"Generated {len(features)} feature rows for {symbol} for backtesting")
                else:
                    self.logger.warning(f"Backtest feature generation returned empty for {symbol}")

            # 3. Load benchmark data
            if self.benchmark_symbol:
                try:
                    self.benchmark_data = self.db_manager.get_market_data(
                        self.benchmark_symbol, features_start_str, backtest_end_str
                    )
                except Exception as e:
                    self.logger.error(f"Error loading benchmark data for {self.benchmark_symbol}: {e}")

            self.logger.info(f"Backtest data preparation completed for {len(self.price_data)} symbols")
            
        except Exception as e:
            self.logger.error(f"Critical error in _prepare_data: {e}")
            raise
    
    def _prepare_training_data(self, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Prepares training data by generating features and the target variable."""
        self.logger.info(f"Preparing training data up to {end_date.strftime('%Y-%m-%d')}...")
        
        training_start_str = (end_date - timedelta(days=4 * 365)).strftime('%Y-%m-%d') # 4 years of data for training
        training_end_str = end_date.strftime('%Y-%m-%d')

        training_features = self.feature_engine.generate_features(
            symbols=self.symbols,
            start_date=training_start_str,
            end_date=training_end_str,
            target_column='binary_threshold'  # Generate target for training
        )
        return training_features

    def _train_initial_models(self):
        """Train initial models using a dedicated training data preparation step."""
        try:
            self.logger.info("Training initial models...")
            training_end_date = self.start_date - timedelta(days=1)
            
            # 1. Prepare training data explicitly
            training_data_dict = self._prepare_training_data(training_end_date)
            
            models_trained = 0
            for symbol in self.symbols:
                if symbol in training_data_dict and training_data_dict[symbol] is not None:
                    try:
                        training_features = training_data_dict[symbol]
                        
                        if not training_features.empty and 'target' in training_features.columns:
                            # Drop rows where the target is NaN (expected for last few rows)
                            training_data = training_features.dropna(subset=['target'])
                            
                            if not training_data.empty:
                                X = training_data.drop(columns=['target'])
                                y = training_data['target']
                                
                                self.logger.info(f"Training initial model for {symbol} with {len(X)} samples.")
                                result = self.model_trainer.train_model(symbol=symbol, X=X, y=y)
                                
                                if result:
                                    self.logger.info(f"Model trained for {symbol} - AUC: {result.get('cv_results',{}).get('mean_auc', 'N/A'):.3f}")
                                    models_trained += 1
                                else:
                                    self.logger.warning(f"Failed to train model for {symbol}")
                            else:
                                self.logger.warning(f"No training data with a valid target for {symbol} before {training_end_date}")
                        else:
                            self.logger.warning(f"No target column found in training data for {symbol}")

                    except Exception as e:
                        self.logger.error(f"Error training initial model for {symbol}: {e}", exc_info=True)
            
            # 2. Load all available models into the predictor for the backtest
            self.model_predictor._load_available_models()

            if models_trained == 0:
                self.logger.warning("No models were successfully trained.")
            else:
                self.logger.info(f"Initial model training completed: {models_trained}/{len(self.symbols)} models trained.")

        except Exception as e:
            self.logger.critical(f"A critical error occurred during initial model training: {e}", exc_info=True)
    
    def run_backtest(self) -> BacktestResults:
        """
        Execute the backtest
        
        Returns:
            Comprehensive backtest results
        """
        try:
            self.logger.info("Starting backtest execution...")
            self.state = BacktestState.RUNNING
            
            # Initialize portfolio tracking
            self.portfolio_history.append({
                'date': self.current_date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions': self.positions.copy()
            })
            
            # Main backtest loop
            while self.current_date <= self.end_date:
                try:
                    self._process_day()
                    self.current_date += timedelta(days=1)
                    
                    # Skip weekends (simplified)
                    while self.current_date.weekday() >= 5:
                        self.current_date += timedelta(days=1)
                        
                except Exception as e:
                    self.logger.error(f"Error processing day {self.current_date}: {e}")
                    continue
            
            # Final portfolio valuation
            self._update_portfolio_value()
            
            # Generate results
            results = self._generate_results()
            
            self.state = BacktestState.COMPLETED
            self.logger.info(f"Backtest completed - Final value: ${self.portfolio_value:,.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            self.state = BacktestState.FAILED
            raise
    
    def _process_day(self):
        """Process a single trading day"""
        try:
            # Check if we have data for this date
            if not self._has_data_for_date(self.current_date):
                return
            
            # Update portfolio valuation with current prices
            self._update_portfolio_value()
            
            # Generate signals
            signals = self._generate_signals()
            
            # Process signals and execute trades
            for symbol, signal in signals.items():
                self._process_signal(symbol, signal)
            
            # Check for exits
            self._check_exits()
            
            # Record portfolio state
            self._record_portfolio_state()
            
            # Retrain models if needed
            if self._should_retrain_models():
                self._retrain_models()
            
        except Exception as e:
            self.logger.error(f"Error processing day {self.current_date}: {e}")
    
    def _has_data_for_date(self, date: datetime) -> bool:
        """Check if we have price data for the given date"""
        for symbol in self.symbols:
            if symbol in self.price_data:
                symbol_data = self.price_data[symbol]
                if date.strftime("%Y-%m-%d") in symbol_data.index.strftime("%Y-%m-%d"):
                    return True
        return False
    
    def _update_portfolio_value(self):
        """Update portfolio valuation with current market prices"""
        try:
            equity_value = 0.0
            
            for symbol, quantity in self.positions.items():
                if quantity != 0 and symbol in self.price_data:
                    current_price = self._get_price(symbol, self.current_date)
                    if current_price is not None:
                        equity_value += int(quantity) * current_price
            
            self.portfolio_value = self.cash + equity_value
            
            # Update high water mark and drawdown
            if self.portfolio_value > self.high_water_mark:
                self.high_water_mark = self.portfolio_value
            
            drawdown = (self.high_water_mark - self.portfolio_value) / self.high_water_mark
            self.drawdown_history.append({
                'date': self.current_date,
                'value': self.portfolio_value,
                'drawdown': drawdown
            })
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def _generate_signals(self) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals for current date"""
        signals = {}
        
        try:
            for symbol in self.symbols:
                if symbol not in self.price_data:
                    continue
                
                # Get features for prediction
                features = self._get_features_for_date(symbol, self.current_date)
                if features is None:
                    continue
                
                # Get model prediction
                prediction = self._get_model_prediction(symbol, features)
                if prediction is None:
                    continue
                
                # Convert prediction to signal
                signal = self._prediction_to_signal(symbol, prediction)
                if signal:
                    signals[symbol] = signal
        
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _get_features_for_date(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Get pre-calculated features for a specific symbol and date."""
        try:
            if symbol in self.all_features:
                symbol_features = self.all_features[symbol]
                
                # Find the most recent available features on or before the given date
                # This handles non-trading days by using the last available data
                available_features = symbol_features[symbol_features.index <= date]
                
                if not available_features.empty:
                    # Return the last row as a new DataFrame
                    return available_features.tail(1)

            self.logger.warning(f"No features found for {symbol} on or before {date.strftime('%Y-%m-%d')}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting features for {symbol} on {date}: {e}")
            return None
    
    def _get_model_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get model prediction for features using actual ModelPredictor"""
        try:
            if features.empty:
                return None
            
            # Use actual model predictor for inference
            if self.model_predictor:
                # Pass the dynamically generated features to the predictor
                # Caching is now handled by the predictor with date-specific keys
                prediction = self.model_predictor.predict(symbol, features_df=features, use_cache=True)
                
                if prediction:
                    # Convert ModelPredictor output to expected format
                    return {
                        'signal_strength': prediction.get('signal_strength', 'HOLD'),
                        'probability': prediction.get('probability', 0.5),
                        'confidence': prediction.get('confidence', 0.5),
                        'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                        'features_used': prediction.get('features_used', 0),
                        'prediction_time': prediction.get('prediction_time', 0.0)
                    }
            
            # Fallback to technical analysis if model prediction fails
            self.logger.warning(f"Model prediction failed for {symbol}, using technical analysis fallback")
            return self._fallback_technical_prediction(symbol, features)
            
        except Exception as e:
            self.logger.error(f"Error getting prediction for {symbol}: {e}")
            return self._fallback_technical_prediction(symbol, features)
    
    def _fallback_technical_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Fallback prediction based on technical indicators"""
        try:
            if features.empty:
                return None
            
            # Technical analysis based on RSI and MACD
            rsi = features.get('rsi', pd.Series([50])).iloc[0] if 'rsi' in features.columns else 50
            macd = features.get('macd', pd.Series([0])).iloc[0] if 'macd' in features.columns else 0
            macd_signal = features.get('macd_signal', pd.Series([0])).iloc[0] if 'macd_signal' in features.columns else 0
            bb_position = features.get('bb_position', pd.Series([0.5])).iloc[0] if 'bb_position' in features.columns else 0.5
            volume_ratio = features.get('volume_ratio', pd.Series([1.0])).iloc[0] if 'volume_ratio' in features.columns else 1.0
            
            # Multi-factor signal logic
            signals = []
            
            # RSI signals
            if rsi < 30:
                signals.append(('BUY', 0.7))
            elif rsi > 70:
                signals.append(('SELL', 0.7))
            
            # MACD signals
            if macd > macd_signal and macd > 0:
                signals.append(('BUY', 0.6))
            elif macd < macd_signal and macd < 0:
                signals.append(('SELL', 0.6))
            
            # Bollinger Band signals
            if bb_position < 0.2:
                signals.append(('BUY', 0.5))
            elif bb_position > 0.8:
                signals.append(('SELL', 0.5))
            
            # Volume confirmation
            volume_boost = min(volume_ratio / 1.5, 1.2) if volume_ratio > 1.5 else 1.0
            
            # Aggregate signals
            if signals:
                buy_signals = [s for s in signals if s[0] == 'BUY']
                sell_signals = [s for s in signals if s[0] == 'SELL']
                
                if len(buy_signals) >= 2:
                    signal_strength = "BUY"
                    probability = min(np.mean([s[1] for s in buy_signals]) * volume_boost, 0.8)
                    confidence = min(len(buy_signals) / 3.0 * volume_boost, 0.7)
                elif len(sell_signals) >= 2:
                    signal_strength = "SELL"
                    probability = max(1 - np.mean([s[1] for s in sell_signals]) * volume_boost, 0.2)
                    confidence = min(len(sell_signals) / 3.0 * volume_boost, 0.7)
                elif len(buy_signals) > len(sell_signals):
                    signal_strength = "BUY"
                    probability = 0.6
                    confidence = 0.4
                elif len(sell_signals) > len(buy_signals):
                    signal_strength = "SELL"
                    probability = 0.4
                    confidence = 0.4
                else:
                    signal_strength = "HOLD"
                    probability = 0.5
                    confidence = 0.3
            else:
                signal_strength = "HOLD"
                probability = 0.5
                confidence = 0.3
            
            return {
                'signal_strength': signal_strength,
                'probability': probability,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'fallback_mode': True,
                'technical_signals': len(signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error in fallback prediction for {symbol}: {e}")
            return None
    
    def _prediction_to_signal(self, symbol: str, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert model prediction to trading signal"""
        try:
            confidence = prediction.get('confidence', 0.0)
            
            # Check confidence threshold
            if confidence < self.signal_threshold:
                return None
            
            signal_strength = prediction.get('signal_strength', 'HOLD')
            probability = prediction.get('probability', 0.5)
            
            # Get current price
            current_price = self._get_price(symbol, self.current_date)
            if current_price is None:
                return None
            
            # Calculate position size
            if signal_strength in ['BUY', 'STRONG_BUY']:
                # Calculate actual volatility and expected return
                symbol_volatility = self._calculate_symbol_volatility(symbol)
                symbol_expected_return = self._calculate_expected_return(symbol, prediction)
                
                sizing_input = SizingInput(
                    symbol=symbol,
                    current_price=current_price,
                    signal_probability=probability,
                    signal_strength=confidence,
                    volatility=symbol_volatility,
                    expected_return=symbol_expected_return,
                    portfolio_value=self.portfolio_value,
                    current_position=self.positions.get(symbol, 0)
                )
                
                sizing_result = self.position_sizer.calculate_position_size(sizing_input)
                position_size = sizing_result.recommended_shares
            else:
                position_size = 0
            
            return {
                'action': signal_strength,
                'confidence': confidence,
                'probability': probability,
                'position_size': position_size,
                'price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error converting prediction to signal for {symbol}: {e}")
            return None
    
    def _process_signal(self, symbol: str, signal: Dict[str, Any]):
        """Process trading signal and execute trade if appropriate"""
        try:
            action = signal.get('action', 'HOLD')
            position_size = signal.get('position_size', 0)
            price = signal.get('price', 0.0)
            confidence = signal.get('confidence', 0.0)
            
            current_position = self.positions.get(symbol, 0)
            
            if action in ['BUY', 'STRONG_BUY'] and position_size > 0:
                # Check if we can afford the trade
                position_size = int(position_size)  # Ensure position_size is integer
                trade_value = position_size * price
                commission = position_size * self.commission_per_share
                
                if self.cash >= (trade_value + commission):
                    # Execute buy
                    self._execute_trade(symbol, position_size, price, 'long', confidence)
                    
            elif action in ['SELL', 'STRONG_SELL'] and current_position > 0:
                # Sell current position
                self._execute_trade(symbol, -current_position, price, 'exit', confidence)
            
        except Exception as e:
            self.logger.error(f"Error processing signal for {symbol}: {e}")
    
    def _execute_trade(self, symbol: str, quantity: int, price: float, side: str, confidence: float):
        """Execute a trade"""
        try:
            commission = abs(quantity) * self.commission_per_share
            trade_value = abs(quantity) * price
            
            if quantity > 0:  # Buy
                self.cash -= (trade_value + commission)
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                # Record open trade
                self.open_trades[symbol] = {
                    'entry_date': self.current_date,
                    'entry_price': price,
                    'quantity': int(quantity),  # Ensure quantity is stored as integer
                    'side': side,
                    'confidence': confidence,
                    'commission': commission
                }
                
            else:  # Sell
                self.cash += (trade_value - commission)
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                # Close trade if we have an open trade
                if symbol in self.open_trades:
                    self._close_trade(symbol, price, commission)
            
            self.logger.info(f"Trade executed: {symbol} {quantity}@${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def _close_trade(self, symbol: str, exit_price: float, commission: float):
        """Close an open trade and record results"""
        try:
            if symbol not in self.open_trades:
                return
            
            trade_info = self.open_trades[symbol]
            
            # Calculate P&L
            quantity = int(trade_info['quantity'])  # Ensure quantity is an integer
            entry_price = trade_info['entry_price']
            
            if trade_info['side'] == 'long':
                pnl = quantity * (exit_price - entry_price) - trade_info['commission'] - commission
                pnl_pct = (exit_price / entry_price - 1) * 100 if entry_price != 0 else 0.0
            else:  # short side
                pnl = quantity * (entry_price - exit_price) - trade_info['commission'] - commission
                pnl_pct = (entry_price / exit_price - 1) * 100 if exit_price != 0 else 0.0
            
            duration = (self.current_date - trade_info['entry_date']).days
            
            # Create trade record
            trade = BacktestTrade(
                symbol=symbol,
                entry_date=trade_info['entry_date'],
                exit_date=self.current_date,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                side=trade_info['side'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration_days=duration,
                signal_strength=0.0,  # Would get from signal
                model_confidence=trade_info['confidence'],
                commission=trade_info['commission'] + commission
            )
            
            self.trades.append(trade)
            
            # Remove from open trades
            del self.open_trades[symbol]
            
            self.logger.info(f"Trade closed: {symbol} P&L=${pnl:.2f} ({pnl_pct:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Error closing trade for {symbol}: {e}")
    
    def _check_exits(self):
        """Check for position exits based on multiple criteria"""
        try:
            # Get trading parameters from an instance to correctly resolve properties
            trade_config = TradingConfig()
            max_hold_days = getattr(trade_config, 'MAX_HOLD_DAYS', 30)
            stop_loss_pct = getattr(trade_config, 'STOP_LOSS_PCT', 2.0) / 100.0
            profit_target_pct = getattr(trade_config, 'PROFIT_TARGET_PCT', 3.0) / 100.0
            
            for symbol in list(self.open_trades.keys()):
                trade_info = self.open_trades[symbol]
                
                # Skip checking exits on the day of entry to avoid immediate closure
                if self.current_date <= trade_info['entry_date']:
                    continue

                days_held = (self.current_date - trade_info['entry_date']).days
                current_price = self._get_price(symbol, self.current_date)
                
                if current_price is None:
                    self.logger.warning(f"Could not find price for {symbol} on {self.current_date}, skipping exit check.")
                    continue
                
                entry_price = trade_info['entry_price']
                
                # Ensure entry price is not zero to avoid division errors
                if entry_price == 0:
                    continue

                quantity = int(trade_info['quantity'])
                side = trade_info['side']
                
                # Calculate current P&L percentage
                if side == 'long':
                    pnl_pct = (current_price / entry_price) - 1.0
                else:  # short
                    pnl_pct = (entry_price / current_price) - 1.0
                
                exit_reason = None
                
                # 1. Stop Loss Check
                if pnl_pct <= -stop_loss_pct and days_held > 0:
                    exit_reason = f"stop_loss_{stop_loss_pct:.1%}"
                
                # 2. Profit Target Check
                elif pnl_pct >= profit_target_pct:
                    exit_reason = f"profit_target_{profit_target_pct:.1%}"
                
                # 3. Maximum Hold Period Check
                elif days_held >= max_hold_days:
                    exit_reason = f"max_hold_{max_hold_days}d"
                
                # 4. Technical Signal Reversal Check
                elif self._check_signal_reversal(symbol, trade_info):
                    exit_reason = "signal_reversal"
                
                # 5. Risk Management - Portfolio Heat Check
                elif self._check_portfolio_heat_exit(symbol, current_price, trade_info):
                    exit_reason = "portfolio_heat"
                
                # Execute exit if criteria met
                if exit_reason:
                    self.logger.info(f"Exiting {symbol} position: {exit_reason} (P&L: {pnl_pct:.1%})")
                    self._execute_trade(symbol, -int(quantity), current_price, 'exit', 0.0)
            
        except Exception as e:
            self.logger.error(f"Error checking exits: {e}")
    
    def _check_signal_reversal(self, symbol: str, trade_info: Dict[str, Any]) -> bool:
        """Check if trading signal has reversed"""
        try:
            # Get current features and prediction
            features = self._get_features_for_date(symbol, self.current_date)
            if features is None:
                return False
            
            prediction = self._get_model_prediction(symbol, features)
            if prediction is None:
                return False
            
            signal_strength = prediction.get('signal_strength', 'HOLD')
            confidence = prediction.get('confidence', 0.0)
            
            # Check for signal reversal with sufficient confidence
            if confidence > 0.6:  # Only act on high-confidence reversals
                side = trade_info['side']
                
                if side == 'long' and signal_strength in ['SELL', 'STRONG_SELL']:
                    return True
                elif side == 'short' and signal_strength in ['BUY', 'STRONG_BUY']:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking signal reversal for {symbol}: {e}")
            return False
    
    def _check_portfolio_heat_exit(self, symbol: str, current_price: float, trade_info: Dict[str, Any]) -> bool:
        """Check if position should be exited due to portfolio heat/risk"""
        try:
            # Calculate current position value
            quantity = int(trade_info['quantity'])  # Ensure quantity is an integer
            position_value = abs(quantity) * current_price
            
            # Check if position is too large relative to portfolio
            trade_config = TradingConfig()
            max_position_pct = getattr(trade_config, 'MAX_POSITION_SIZE_PCT', 0.05)  # 5%
            current_position_pct = position_value / self.portfolio_value
            
            if current_position_pct > max_position_pct * 1.5:  # 50% buffer before exit
                return True
            
            # Check total portfolio exposure
            total_exposure = sum(
                abs(int(info['quantity'])) * self._get_price(sym, self.current_date) or 0
                for sym, info in self.open_trades.items()
                if self._get_price(sym, self.current_date) is not None
            )
            
            max_exposure_pct = 0.95  # Max 95% exposure
            if total_exposure / self.portfolio_value > max_exposure_pct:
                # Exit smallest positions first (by value)
                position_values = {}
                for sym, info in self.open_trades.items():
                    price = self._get_price(sym, self.current_date)
                    if price:
                        position_values[sym] = abs(int(info['quantity'])) * price
                
                # If this is among the smallest 20% of positions, exit it
                sorted_positions = sorted(position_values.items(), key=lambda x: x[1])
                bottom_20_pct = int(len(sorted_positions) * 0.2) + 1
                
                if symbol in [pos[0] for pos in sorted_positions[:bottom_20_pct]]:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio heat for {symbol}: {e}")
            return False
    
    def _record_portfolio_state(self):
        """Record current portfolio state"""
        try:
            portfolio_record = {
                'date': self.current_date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions': self.positions.copy(),
                'open_trades': len(self.open_trades)
            }
            
            self.portfolio_history.append(portfolio_record)
            
            # Calculate daily return
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2]['portfolio_value']
                daily_return = (self.portfolio_value / prev_value - 1) if prev_value > 0 else 0.0
                self.daily_returns.append(daily_return)
            
        except Exception as e:
            self.logger.error(f"Error recording portfolio state: {e}")
    
    def _should_retrain_models(self) -> bool:
        """Check if models should be retrained"""
        days_since_start = (self.current_date - self.start_date).days
        return days_since_start > 0 and days_since_start % self.retrain_frequency == 0
    
    def _retrain_models(self):
        """Retrain models with updated data using pre-generated features."""
        try:
            self.logger.info(f"Retraining models on {self.current_date.strftime('%Y-%m-%d')}...")
            
            # 1. Prepare fresh training data up to the current backtest date
            training_data_dict = self._prepare_training_data(self.current_date)
            
            for symbol in self.symbols:
                if symbol in training_data_dict and training_data_dict[symbol] is not None:
                    try:
                        training_features = training_data_dict[symbol]
                        
                        if not training_features.empty and 'target' in training_features.columns:
                            training_data = training_features.dropna(subset=['target'])

                            if not training_data.empty:
                                X = training_data.drop(columns=['target'])
                                y = training_data['target']

                                self.logger.info(f"Retraining model for {symbol} with {len(X)} samples.")
                                result = self.model_trainer.train_model(symbol=symbol, X=X, y=y)

                                if result:
                                    self.model_predictor.reload_model(symbol)
                                    self.logger.info(f"Successfully retrained and reloaded model for {symbol}.")
                                else:
                                    self.logger.warning(f"Retraining failed for {symbol}, continuing with old model.")
                            else:
                                self.logger.warning(f"No valid training data for retraining {symbol} up to {self.current_date.strftime('%Y-%m-%d')}")
                        else:
                             self.logger.warning(f"No target column found in retraining data for {symbol}")

                    except Exception as e:
                        self.logger.error(f"Error during model retraining for {symbol}: {e}", exc_info=True)

        except Exception as e:
            self.logger.error(f"A critical error occurred during the retraining cycle: {e}", exc_info=True)
    
    def _get_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get price for symbol on specific date"""
        try:
            if symbol not in self.price_data:
                return None
            
            data = self.price_data[symbol]
            date_str = date.strftime("%Y-%m-%d")
            
            # Find closest date
            available_dates = data.index.strftime("%Y-%m-%d")
            if date_str in available_dates:
                return data.loc[data.index.strftime("%Y-%m-%d") == date_str]['close'].iloc[0]
            
            # Find nearest previous date
            mask = data.index <= date
            if mask.any():
                return data[mask]['close'].iloc[-1]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol} on {date}: {e}")
            return None
    
    def _generate_results(self) -> BacktestResults:
        """Generate comprehensive backtest results"""
        try:
            # Basic metrics
            total_return = (self.portfolio_value / self.initial_capital - 1) * 100
            
            # Time-based metrics
            days = (self.end_date - self.start_date).days
            annualized_return = ((self.portfolio_value / self.initial_capital) ** (365.25 / days) - 1) * 100 if days > 0 else 0.0
            
            # Risk metrics
            if len(self.daily_returns) > 1:
                volatility = np.std(self.daily_returns) * np.sqrt(252) * 100
                sharpe_ratio = (np.mean(self.daily_returns) * 252 - 0.02) / (volatility / 100) if volatility > 0 else 0.0
                
                # Sortino ratio
                negative_returns = [r for r in self.daily_returns if r < 0]
                downside_deviation = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0.0
                sortino_ratio = (np.mean(self.daily_returns) * 252 - 0.02) / downside_deviation if downside_deviation > 0 else 0.0
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
            
            # Drawdown metrics
            max_drawdown = max([d['drawdown'] for d in self.drawdown_history]) * 100 if self.drawdown_history else 0.0
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Trade metrics
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0.0
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = sum(t.pnl for t in losing_trades)
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
            
            largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0.0
            largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0.0
            
            avg_trade_duration = np.mean([t.duration_days for t in self.trades]) if self.trades else 0.0
            
            # Benchmark comparison
            benchmark_return = self._calculate_benchmark_return()
            information_ratio = self._calculate_information_ratio(benchmark_return)
            
            return BacktestResults(
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                final_capital=self.portfolio_value,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(self.trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_trade_duration=avg_trade_duration,
                portfolio_history=self.portfolio_history,
                trades=self.trades,
                daily_returns=self.daily_returns,
                benchmark_return=benchmark_return,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            self.logger.error(f"Error generating results: {e}")
            raise
    
    def _calculate_benchmark_return(self) -> Optional[float]:
        """Calculate benchmark return for comparison"""
        try:
            if self.benchmark_data is None or self.benchmark_data.empty:
                return None
            
            # Ensure timestamp column exists and is datetime
            if 'timestamp' in self.benchmark_data.columns:
                # Convert timestamp to datetime if needed
                self.benchmark_data['timestamp'] = pd.to_datetime(self.benchmark_data['timestamp'])
                # Set timestamp as index for filtering
                benchmark_data = self.benchmark_data.set_index('timestamp')
            else:
                # Use existing index
                benchmark_data = self.benchmark_data.copy()
                # Ensure index is datetime
                benchmark_data.index = pd.to_datetime(benchmark_data.index)
            
            # Filter benchmark data to backtest period
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            
            benchmark_subset = benchmark_data[
                (benchmark_data.index >= start_date) &
                (benchmark_data.index <= end_date)
            ]
            
            if len(benchmark_subset) < 2:
                self.logger.warning(f"Insufficient benchmark data: {len(benchmark_subset)} records")
                return None
            
            start_price = benchmark_subset['close'].iloc[0]
            end_price = benchmark_subset['close'].iloc[-1]
            
            return (end_price / start_price - 1) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark return: {e}")
            return None
    
    def _calculate_information_ratio(self, benchmark_return: Optional[float]) -> Optional[float]:
        """Calculate information ratio vs benchmark"""
        try:
            if benchmark_return is None or not self.daily_returns:
                return None
            
            # Calculate tracking error (simplified)
            portfolio_return = (self.portfolio_value / self.initial_capital - 1) * 100
            excess_return = portfolio_return - benchmark_return
            
            # Simplified calculation - would need daily benchmark returns for proper IR
            tracking_error = np.std(self.daily_returns) * np.sqrt(252) * 100
            
            return excess_return / tracking_error if tracking_error > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating information ratio: {e}")
            return None
    
    def _calculate_symbol_volatility(self, symbol: str, lookback_days: int = 60) -> float:
        """Calculate symbol volatility from historical data"""
        try:
            if symbol not in self.price_data:
                return 0.25  # Default fallback
            
            # Get price data up to current date
            price_data = self.price_data[symbol]
            mask = price_data.index <= self.current_date
            historical_data = price_data[mask]
            
            if len(historical_data) < 20:
                return 0.25  # Default fallback
            
            # Use last N days for volatility calculation
            recent_data = historical_data.tail(lookback_days)
            
            # Calculate daily returns
            returns = recent_data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return 0.25  # Default fallback
            
            # Annualized volatility
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            # Apply reasonable bounds
            return max(0.05, min(annual_vol, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.25  # Default fallback
    
    def _calculate_expected_return(self, symbol: str, prediction: Dict[str, Any]) -> float:
        """Calculate expected return based on model prediction and historical data"""
        try:
            # Get base expected return from historical performance
            historical_return = self._get_historical_return(symbol)
            
            # Get signal-based adjustment
            signal_adjustment = self._get_signal_return_adjustment(prediction)
            
            # Combine historical and signal-based returns
            # Weight more heavily on model prediction if confidence is high
            confidence = prediction.get('confidence', 0.5)
            probability = prediction.get('probability', 0.5)
            
            # Calculate signal strength multiplier
            if probability > 0.7:
                signal_multiplier = 1.5  # Strong positive signal
            elif probability > 0.6:
                signal_multiplier = 1.2  # Moderate positive signal
            elif probability < 0.3:
                signal_multiplier = -1.5  # Strong negative signal
            elif probability < 0.4:
                signal_multiplier = -1.2  # Moderate negative signal
            else:
                signal_multiplier = 0.0  # Neutral signal
            
            # Base expected return (annualized)
            base_return = historical_return
            
            # Signal-based return adjustment
            signal_return = signal_adjustment * signal_multiplier * confidence
            
            # Combine with confidence weighting
            expected_return = (
                base_return * (1 - confidence) +
                (base_return + signal_return) * confidence
            )
            
            # Apply reasonable bounds (-50% to +100% annualized)
            return max(-0.5, min(expected_return, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating expected return for {symbol}: {e}")
            return 0.08  # Default 8% expected return
    
    def _get_historical_return(self, symbol: str, lookback_days: int = 252) -> float:
        """Get historical annualized return for symbol"""
        try:
            if symbol not in self.price_data:
                return 0.08  # Market average fallback
            
            # Get price data up to current date
            price_data = self.price_data[symbol]
            mask = price_data.index <= self.current_date
            historical_data = price_data[mask]
            
            if len(historical_data) < 30:
                return 0.08  # Default fallback
            
            # Use specified lookback period
            recent_data = historical_data.tail(lookback_days)
            
            if len(recent_data) < 30:
                return 0.08  # Default fallback
            
            # Calculate total return over period
            start_price = recent_data['close'].iloc[0]
            end_price = recent_data['close'].iloc[-1]
            
            total_return = (end_price / start_price) - 1
            
            # Annualize the return
            days_in_period = len(recent_data)
            annual_return = (1 + total_return) ** (252 / days_in_period) - 1
            
            return annual_return
            
        except Exception as e:
            self.logger.error(f"Error getting historical return for {symbol}: {e}")
            return 0.08  # Default fallback
    
    def _get_signal_return_adjustment(self, prediction: Dict[str, Any]) -> float:
        """Get expected return adjustment based on signal characteristics"""
        try:
            signal_strength = prediction.get('signal_strength', 'HOLD')
            probability = prediction.get('probability', 0.5)
            
            # Base return expectations by signal type
            signal_returns = {
                'STRONG_BUY': 0.15,    # Expect 15% return
                'BUY': 0.10,           # Expect 10% return
                'HOLD': 0.0,           # No adjustment
                'SELL': -0.10,         # Expect -10% return
                'STRONG_SELL': -0.15   # Expect -15% return
            }
            
            base_signal_return = signal_returns.get(signal_strength, 0.0)
            
            # Adjust for prediction time horizon (assume signals are for 5-day horizon)
            horizon_days = getattr(TradingConfig, 'PREDICTION_HORIZON_DAYS', 5)
            annualized_adjustment = base_signal_return * (252 / horizon_days)
            
            return annualized_adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating signal return adjustment: {e}")
            return 0.0

# Example usage
if __name__ == "__main__":
    # Initialize backtest engine with proper date range for technical indicators
    # Using period that matches available data in database
    engine = ZiplineEngine(
        start_date="2024-06-01",  # Start after enough lookback data available
        end_date="2024-12-31",    # Use available data range
        initial_capital=25000.0
    )
    
    # Add strategy with symbols that have data
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    engine.add_strategy(symbols)
    
    # Run backtest
    results = engine.run_backtest()
    
    # Display results
    print(f"Total Return: {results.total_return:.2f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2f}%")
    print(f"Win Rate: {results.win_rate:.1f}%")
    print(f"Total Trades: {results.total_trades}")