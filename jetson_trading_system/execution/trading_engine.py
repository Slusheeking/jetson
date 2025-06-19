"""
Trading Engine for Jetson Trading System
High-performance execution engine with ML signal integration
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import queue
import json

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingConfig
from jetson_trading_system.utils.logger import get_trading_logger
from jetson_trading_system.utils.database import TradingDatabase
from jetson_trading_system.models.model_predictor import ModelPredictor
from jetson_trading_system.risk.risk_manager import RiskManager
from jetson_trading_system.risk.position_sizer import PositionSizer, SizingInput
from jetson_trading_system.execution.order_manager import OrderManager
from jetson_trading_system.execution.portfolio_tracker import PortfolioTracker

class TradingState(Enum):
    """Trading engine states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class SignalType(Enum):
    """Signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    strength: float
    confidence: float
    probability: float
    price: float
    timestamp: datetime
    model_prediction: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    position_size: int
    reasoning: str

@dataclass
class TradingPerformance:
    """Trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    current_positions: int
    portfolio_value: float
    daily_pnl: float

class TradingEngine:
    """
    Main trading engine orchestrating ML predictions, risk management, and execution
    Optimized for Jetson Orin with real-time performance monitoring
    """
    
    def __init__(self,
                 symbols: List[str] = None,
                 paper_trading: bool = True,
                 max_positions: int = 10,
                 min_signal_confidence: float = 0.6,
                 model_predictor: ModelPredictor = None,
                 risk_manager: RiskManager = None,
                 position_sizer: PositionSizer = None,
                 order_manager: OrderManager = None,
                 portfolio_tracker: PortfolioTracker = None):
        """
        Initialize trading engine with dependency injection.
        
        Args:
            symbols: List of symbols to trade
            paper_trading: Use paper trading mode
            max_positions: Maximum number of concurrent positions
            min_signal_confidence: Minimum confidence threshold for signals
            model_predictor: The model prediction engine.
            risk_manager: The risk management engine.
            position_sizer: The position sizing engine.
            order_manager: The order management engine.
            portfolio_tracker: The portfolio tracking engine.
        """
        self.symbols = symbols or TradingConfig.SYMBOLS
        self.paper_trading = paper_trading
        self.max_positions = max_positions
        self.min_signal_confidence = min_signal_confidence
        
        self.logger = get_trading_logger()
        self.db_manager = TradingDatabase()
        
        # Initialize components, allowing for dependency injection
        self.model_predictor = model_predictor or ModelPredictor()
        self.risk_manager = risk_manager or RiskManager()
        self.position_sizer = position_sizer or PositionSizer()
        self.order_manager = order_manager or OrderManager(paper_trading=paper_trading)
        self.portfolio_tracker = portfolio_tracker or PortfolioTracker()
        
        # Engine state
        self.state = TradingState.STOPPED
        self.last_run_time = None
        self.run_count = 0
        self.error_count = 0
        
        # Signal processing
        self.signal_queue = queue.Queue()
        self.processed_signals = []
        self.signal_callbacks = []
        
        # Performance tracking
        self.performance_metrics = TradingPerformance(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_pnl=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            max_drawdown=0.0, sharpe_ratio=0.0, current_positions=0,
            portfolio_value=25000.0, daily_pnl=0.0
        )
        
        # Threading
        self.main_thread = None
        self.signal_thread = None
        self.shutdown_event = threading.Event()
        self.pause_event = threading.Event()
        
        # Configuration
        self.config = {
            'signal_generation_interval': 300,  # 5 minutes
            'portfolio_update_interval': 60,    # 1 minute
            'risk_check_interval': 30,          # 30 seconds
            'max_daily_trades': 20,
            'max_daily_loss': 0.05,             # 5% max daily loss
            'position_timeout': 3600,           # 1 hour position timeout
        }
        
        self.logger.info(f"TradingEngine initialized for {len(self.symbols)} symbols")
        self.logger.info(f"Paper trading: {paper_trading}, Max positions: {max_positions}")
    
    def start(self) -> bool:
        """Start the trading engine"""
        try:
            if self.state != TradingState.STOPPED:
                self.logger.warning(f"Cannot start engine in state: {self.state}")
                return False
            
            self.state = TradingState.STARTING
            self.logger.info("Starting trading engine...")
            
            # Reset shutdown event
            self.shutdown_event.clear()
            self.pause_event.clear()
            
            # Validate components
            if not self._validate_components():
                self.state = TradingState.ERROR
                return False
            
            # Start main trading thread
            self.main_thread = threading.Thread(target=self._main_trading_loop, daemon=True)
            self.main_thread.start()
            
            # Start signal processing thread
            self.signal_thread = threading.Thread(target=self._signal_processing_loop, daemon=True)
            self.signal_thread.start()
            
            # Wait for startup
            time.sleep(2)
            
            if self.state == TradingState.RUNNING:
                self.logger.info("Trading engine started successfully")
                return True
            else:
                self.logger.error("Failed to start trading engine")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting trading engine: {e}")
            self.state = TradingState.ERROR
            return False
    
    def stop(self) -> bool:
        """Stop the trading engine"""
        try:
            if self.state == TradingState.STOPPED:
                return True
            
            self.logger.info("Stopping trading engine...")
            self.state = TradingState.STOPPING
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for threads to finish
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=10)
            
            if self.signal_thread and self.signal_thread.is_alive():
                self.signal_thread.join(timeout=5)
            
            # Close all positions if needed
            if not self.paper_trading:
                self._close_all_positions()
            
            self.state = TradingState.STOPPED
            self.logger.info("Trading engine stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trading engine: {e}")
            return False
    
    def pause(self):
        """Pause trading engine"""
        if self.state == TradingState.RUNNING:
            self.pause_event.set()
            self.state = TradingState.PAUSED
            self.logger.info("Trading engine paused")
    
    def resume(self):
        """Resume trading engine"""
        if self.state == TradingState.PAUSED:
            self.pause_event.clear()
            self.state = TradingState.RUNNING
            self.logger.info("Trading engine resumed")
    
    def _validate_components(self) -> bool:
        """Validate that all trading components are properly initialized."""
        try:
            if self.model_predictor is None:
                self.logger.error("ModelPredictor is not initialized.")
                return False
            if self.risk_manager is None:
                self.logger.error("RiskManager is not initialized.")
                return False
            if self.position_sizer is None:
                self.logger.error("PositionSizer is not initialized.")
                return False
            if self.order_manager is None:
                self.logger.error("OrderManager is not initialized.")
                return False
            if self.portfolio_tracker is None:
                self.logger.error("PortfolioTracker is not initialized.")
                return False

            # Check order manager connection
            if not self.order_manager.is_connected():
                self.logger.error("OrderManager is not connected.")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Component validation failed: {e}")
            return False
    
    def _main_trading_loop(self):
        """Main trading loop running in separate thread"""
        try:
            self.state = TradingState.RUNNING
            self.logger.info("Main trading loop started")
            
            last_signal_time = 0
            last_portfolio_update = 0
            last_risk_check = 0
            
            while not self.shutdown_event.is_set():
                try:
                    current_time = time.time()
                    
                    # Wait if paused
                    if self.pause_event.is_set():
                        time.sleep(1)
                        continue
                    
                    # Generate signals
                    if current_time - last_signal_time >= self.config['signal_generation_interval']:
                        self._generate_signals()
                        last_signal_time = current_time
                    
                    # Update portfolio
                    if current_time - last_portfolio_update >= self.config['portfolio_update_interval']:
                        self._update_portfolio()
                        last_portfolio_update = current_time
                    
                    # Risk checks
                    if current_time - last_risk_check >= self.config['risk_check_interval']:
                        self._perform_risk_checks()
                        last_risk_check = current_time
                    
                    # Update run metrics
                    self.last_run_time = datetime.now()
                    self.run_count += 1
                    
                    # Sleep briefly
                    time.sleep(1)
                    
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"Error in main trading loop: {e}")
                    
                    if self.error_count > 10:
                        self.logger.critical("Too many errors, stopping engine")
                        break
                    
                    time.sleep(5)  # Wait before retrying
            
        except Exception as e:
            self.logger.critical(f"Fatal error in main trading loop: {e}")
            self.state = TradingState.ERROR
        
        finally:
            self.logger.info("Main trading loop ended")
    
    def _signal_processing_loop(self):
        """Signal processing loop running in separate thread"""
        try:
            self.logger.info("Signal processing loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Get signal from queue (blocking with timeout)
                    try:
                        signal = self.signal_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    # Process signal
                    if signal:
                        self._process_trading_signal(signal)
                        self.signal_queue.task_done()
                    
                except Exception as e:
                    self.logger.error(f"Error processing signal: {e}")
                    time.sleep(1)
        
        except Exception as e:
            self.logger.error(f"Error in signal processing loop: {e}")
        
        finally:
            self.logger.info("Signal processing loop ended")
    
    def _generate_signals(self):
        """Generate trading signals for all symbols"""
        try:
            self.logger.info("Generating trading signals...")
            
            # Get predictions for all symbols
            predictions = self.model_predictor.batch_predict(self.symbols)
            
            for symbol, prediction in predictions.items():
                if prediction is None:
                    continue
                
                try:
                    # Create trading signal
                    signal = self._create_trading_signal(symbol, prediction)
                    
                    if signal and signal.confidence >= self.min_signal_confidence:
                        # Add to queue for processing
                        self.signal_queue.put(signal)
                        self.logger.info(f"Generated signal: {symbol} {signal.signal_type.value} (conf: {signal.confidence:.3f})")
                
                except Exception as e:
                    self.logger.error(f"Error creating signal for {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
    
    def _create_trading_signal(self, symbol: str, prediction: Dict[str, Any]) -> Optional[TradingSignal]:
        """Create trading signal from ML prediction"""
        try:
            # Extract prediction data
            probability = prediction.get('probability', 0.5)
            confidence = prediction.get('confidence', 0.0)
            signal_strength_raw = prediction.get('signal_strength', 'HOLD')
            
            # Map signal strength to signal type and strength value
            if signal_strength_raw in ['STRONG_BUY', 'BUY']:
                signal_type = SignalType.BUY
                strength = 0.8 if signal_strength_raw == 'STRONG_BUY' else 0.6
            elif signal_strength_raw in ['STRONG_SELL', 'SELL']:
                signal_type = SignalType.SELL
                strength = 0.8 if signal_strength_raw == 'STRONG_SELL' else 0.6
            else:
                signal_type = SignalType.HOLD
                strength = 0.0
            
            # Get current price (simplified - would get from market data)
            current_price = 150.0  # Placeholder
            
            # Calculate position size
            if signal_type in [SignalType.BUY, SignalType.SELL]:
                sizing_input = SizingInput(
                    symbol=symbol,
                    current_price=current_price,
                    signal_probability=probability,
                    signal_strength=strength,
                    volatility=0.25,  # Placeholder
                    expected_return=0.10,  # Placeholder
                    portfolio_value=self.portfolio_tracker.get_total_value(),
                    current_position=self.portfolio_tracker.get_position_size(symbol)
                )
                
                sizing_result = self.position_sizer.calculate_position_size(sizing_input)
                position_size = sizing_result.recommended_shares
            else:
                position_size = 0
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                probability=probability,
                price=current_price,
                timestamp=datetime.now(),
                model_prediction=prediction,
                risk_metrics={},  # Would populate with risk metrics
                position_size=position_size,
                reasoning=f"ML signal: {signal_strength_raw} (prob: {probability:.3f})"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating trading signal for {symbol}: {e}")
            return None
    
    def _process_trading_signal(self, signal: TradingSignal):
        """Process a trading signal and execute if approved"""
        try:
            self.logger.info(f"Processing signal: {signal.symbol} {signal.signal_type.value}")
            
            # Risk check
            approved, reason = self.risk_manager.check_trade_approval(
                signal.symbol,
                signal.position_size,
                signal.price,
                signal.signal_type.value
            )
            
            if not approved:
                self.logger.warning(f"Signal rejected by risk management: {reason}")
                return
            
            # Check position limits
            current_positions = self.portfolio_tracker.get_position_count()
            if current_positions >= self.max_positions and signal.signal_type == SignalType.BUY:
                self.logger.warning("Maximum positions reached, skipping BUY signal")
                return
            
            # Execute trade
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                success = self._execute_trade(signal)
                if success:
                    self.processed_signals.append(signal)
                    
                    # Update performance
                    self._update_performance_metrics(signal)
                    
                    # Notify callbacks
                    for callback in self.signal_callbacks:
                        try:
                            callback(signal)
                        except Exception as e:
                            self.logger.error(f"Error in signal callback: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing trading signal: {e}")
    
    def _execute_trade(self, signal: TradingSignal) -> bool:
        """Execute a trade based on signal"""
        try:
            if signal.signal_type == SignalType.BUY:
                order_id = self.order_manager.place_market_buy_order(
                    signal.symbol,
                    signal.position_size,
                    f"ML Signal Buy - {signal.reasoning}"
                )
            elif signal.signal_type == SignalType.SELL:
                # Determine quantity to sell
                current_position = self.portfolio_tracker.get_position_size(signal.symbol)
                sell_quantity = min(abs(signal.position_size), current_position)
                
                if sell_quantity > 0:
                    order_id = self.order_manager.place_market_sell_order(
                        signal.symbol,
                        sell_quantity,
                        f"ML Signal Sell - {signal.reasoning}"
                    )
                else:
                    self.logger.warning(f"No position to sell for {signal.symbol}")
                    return False
            else:
                return False
            
            if order_id:
                self.logger.info(f"Trade executed: {signal.symbol} {signal.signal_type.value} {signal.position_size} shares (Order: {order_id})")
                return True
            else:
                self.logger.error(f"Failed to execute trade: {signal.symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def _update_portfolio(self):
        """Update portfolio positions and values"""
        try:
            self.portfolio_tracker.update_positions()
            
            # Update risk manager capital
            portfolio_value = self.portfolio_tracker.get_total_value()
            self.risk_manager.update_capital(portfolio_value)
            
            # Update performance metrics
            self.performance_metrics.portfolio_value = portfolio_value
            self.performance_metrics.current_positions = self.portfolio_tracker.get_position_count()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    def _perform_risk_checks(self):
        """Perform periodic risk checks"""
        try:
            # Check portfolio risk limits
            risk_report = self.risk_manager.get_risk_report()
            
            if risk_report.get('circuit_breaker_active'):
                self.logger.critical("Circuit breaker active - pausing trading")
                self.pause()
                return
            
            # Check daily loss limit
            daily_pnl = self.portfolio_tracker.get_daily_pnl()
            if daily_pnl < -self.config['max_daily_loss'] * self.performance_metrics.portfolio_value:
                self.logger.critical(f"Daily loss limit exceeded: ${daily_pnl:,.2f}")
                self.risk_manager.activate_circuit_breaker("Daily loss limit exceeded")
                self.pause()
            
        except Exception as e:
            self.logger.error(f"Error in risk checks: {e}")
    
    def _update_performance_metrics(self, signal: TradingSignal):
        """Update trading performance metrics"""
        try:
            self.performance_metrics.total_trades += 1
            
            # This is simplified - would need actual P&L calculation
            # For now, just update trade counts
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                self.performance_metrics.total_trades += 1
            
            # Calculate win rate
            if self.performance_metrics.total_trades > 0:
                self.performance_metrics.win_rate = (
                    self.performance_metrics.winning_trades / self.performance_metrics.total_trades
                )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.portfolio_tracker.get_all_positions()
            
            for symbol, quantity in positions.items():
                if quantity > 0:
                    self.order_manager.place_market_sell_order(
                        symbol, quantity, "Engine shutdown - close all positions"
                    )
                    self.logger.info(f"Closing position: {symbol} {quantity} shares")
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
    
    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """Add callback for signal processing notifications"""
        self.signal_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'state': self.state.value,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'run_count': self.run_count,
            'error_count': self.error_count,
            'symbols': len(self.symbols),
            'signals_processed': len(self.processed_signals),
            'signals_pending': self.signal_queue.qsize(),
            'performance': asdict(self.performance_metrics),
            'config': self.config
        }
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processed signals"""
        recent_signals = self.processed_signals[-limit:] if self.processed_signals else []
        return [asdict(signal) for signal in recent_signals]
    
    def force_signal_generation(self):
        """Force immediate signal generation (for testing)"""
        threading.Thread(target=self._generate_signals, daemon=True).start()

# Example usage
if __name__ == "__main__":
    print("--- Running TradingEngine Demo in Live Mode ---")

    # Initialize the full trading engine with live components
    engine = TradingEngine(
        symbols=["AAPL", "GOOG", "MSFT", "TSLA"],
        paper_trading=True,  # Use paper trading for safety
        max_positions=10,
        min_signal_confidence=0.55
    )
    
    # Configure shorter intervals for a quick demonstration
    engine.config['signal_generation_interval'] = 15  # seconds
    engine.config['portfolio_update_interval'] = 30 # seconds
    engine.config['risk_check_interval'] = 20      # seconds

    # Add a callback to print processed signals
    def live_signal_callback(signal):
        print(f"\n>>> LIVE SIGNAL PROCESSED: {signal.symbol} "
              f"{signal.signal_type.value} for {signal.position_size} shares. "
              f"Confidence: {signal.confidence:.2f}\n")
    
    engine.add_signal_callback(live_signal_callback)

    # Start the engine
    if engine.start():
        print("\nTrading engine started successfully in live paper trading mode.")
        print("The engine will run for 2 minutes to generate and process live signals.")
        print("Press Ctrl+C to stop the demo early.")
        
        try:
            # Let the engine run and demonstrate its functionality
            time.sleep(120)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down demo...")
        
        finally:
            # Stop the engine
            print("\nStopping trading engine...")
            engine.stop()
            
            print("\n--- Final Engine Status ---")
            status = engine.get_status()
            print(f"  - State: {status['state']}")
            print(f"  - Signals Processed: {status['signals_processed']}")
            print(f"  - Total Runs: {status['run_count']}")
            print(f"  - Errors: {status['error_count']}")

            print("\n--- Final Performance Metrics ---")
            perf = status['performance']
            print(f"  - Portfolio Value: ${perf['portfolio_value']:,.2f}")
            print(f"  - Total Trades: {perf['total_trades']}")
            print(f"  - Current Positions: {perf['current_positions']}")
            
            print("\n--- Last 10 Processed Signals ---")
            recent_signals = engine.get_recent_signals()
            if not recent_signals:
                print("  No signals were processed during the demo.")
            for signal in recent_signals:
                print(f"  - {signal['timestamp']}: {signal['symbol']} {signal['signal_type']} "
                      f"({signal['confidence']:.2f}) for {signal['position_size']} shares.")

    else:
        print("\nFailed to start trading engine.")
            
    print("\n--- Demo Finished ---")
