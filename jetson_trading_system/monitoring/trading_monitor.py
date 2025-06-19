"""
Trading Performance Monitor for Jetson Trading System
Real-time trading performance tracking and analysis
"""

import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json

from ..config.jetson_settings import JetsonConfig
from ..config.trading_params import TradingConfig
from ..utils.logger import get_trading_logger
from ..utils.database import TradingDatabase

@dataclass
class TradingMetrics:
    """Real-time trading performance metrics"""
    timestamp: datetime
    portfolio_value: float
    daily_pnl: float
    daily_pnl_pct: float
    total_pnl: float
    total_pnl_pct: float
    active_positions: int
    cash_balance: float
    exposure: float
    leverage: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    trades_today: int
    signals_generated: int
    signals_acted: int
    avg_prediction_time: float
    model_accuracy: float

@dataclass
class TradePerformance:
    """Individual trade performance tracking"""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str
    pnl: float
    pnl_pct: float
    duration_minutes: int
    signal_confidence: float
    model_prediction: float
    slippage: float
    commission: float
    is_open: bool

@dataclass
class ModelPerformance:
    """Model prediction performance tracking"""
    symbol: str
    timestamp: datetime
    prediction: float
    confidence: float
    actual_return: Optional[float]
    prediction_accuracy: Optional[float]
    prediction_time_ms: float
    features_used: int

@dataclass
class TradingAlert:
    """Trading performance alert"""
    timestamp: datetime
    alert_type: str  # 'performance', 'risk', 'model', 'system'
    severity: str    # 'info', 'warning', 'critical'
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: Optional[float]

class TradingMonitor:
    """
    Comprehensive trading performance monitoring
    Tracks strategy performance, model accuracy, and risk metrics
    """
    
    def __init__(self, 
                 monitoring_interval: int = 30,
                 history_size: int = 1000):
        """
        Initialize trading monitor
        
        Args:
            monitoring_interval: Seconds between performance calculations
            history_size: Number of historical records to keep
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        self.logger = get_trading_logger()
        self.db_manager = TradingDatabase()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.shutdown_event = threading.Event()
        
        # Data storage
        self.trading_metrics_history = deque(maxlen=history_size)
        self.trade_performance_history = deque(maxlen=history_size)
        self.model_performance_history = deque(maxlen=history_size)
        self.alerts_history = deque(maxlen=100)
        
        # Performance tracking
        self.open_trades = {}  # trade_id -> TradePerformance
        self.daily_stats = defaultdict(dict)  # date -> stats
        self.model_predictions = defaultdict(list)  # symbol -> predictions
        
        # Alert system
        self.alert_thresholds = self._setup_trading_thresholds()
        self.alert_callbacks = []
        self.last_alert_times = {}
        
        # Reference values
        self.initial_portfolio_value = 25000.0
        self.high_water_mark = 25000.0
        self.session_start_time = datetime.now()
        
        # Performance statistics
        self.monitoring_stats = {
            'total_trades_monitored': 0,
            'total_signals_tracked': 0,
            'alerts_generated': 0,
            'uptime_start': datetime.now()
        }
        
        self.logger.info("TradingMonitor initialized")
    
    def _setup_trading_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup trading performance alert thresholds"""
        return {
            'daily_pnl_pct': {'warning': -2.0, 'critical': -5.0},
            'max_drawdown': {'warning': -5.0, 'critical': -10.0},
            'win_rate': {'warning': 40.0, 'critical': 30.0},
            'sharpe_ratio': {'warning': 0.5, 'critical': 0.0},
            'model_accuracy': {'warning': 55.0, 'critical': 50.0},
            'leverage': {'warning': 1.5, 'critical': 2.0},
            'exposure': {'warning': 90.0, 'critical': 95.0}
        }
    
    def start_monitoring(self, initial_portfolio_value: float = 25000.0) -> bool:
        """Start trading performance monitoring"""
        try:
            if self.is_monitoring:
                self.logger.warning("Trading monitoring is already running")
                return True
            
            self.initial_portfolio_value = initial_portfolio_value
            self.high_water_mark = initial_portfolio_value
            self.session_start_time = datetime.now()
            
            self.logger.info("Starting trading performance monitoring...")
            
            # Reset shutdown event
            self.shutdown_event.clear()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.is_monitoring = True
            self.monitoring_stats['uptime_start'] = datetime.now()
            
            self.logger.info("Trading monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start trading monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop trading performance monitoring"""
        try:
            if not self.is_monitoring:
                return True
            
            self.logger.info("Stopping trading performance monitoring...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            
            self.is_monitoring = False
            
            self.logger.info("Trading monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping trading monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main trading monitoring loop"""
        try:
            self.logger.info("Trading monitoring loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Calculate current trading metrics
                    metrics = self._calculate_trading_metrics()
                    
                    if metrics:
                        # Store metrics
                        self.trading_metrics_history.append(metrics)
                        
                        # Check for alerts
                        self._check_trading_alerts(metrics)
                        
                        # Update daily stats
                        self._update_daily_stats(metrics)
                    
                    # Wait for next interval
                    self.shutdown_event.wait(timeout=self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in trading monitoring loop: {e}")
                    time.sleep(5)
        
        except Exception as e:
            self.logger.critical(f"Fatal error in trading monitoring loop: {e}")
        
        finally:
            self.logger.info("Trading monitoring loop ended")
    
    def _calculate_trading_metrics(self) -> Optional[TradingMetrics]:
        """Calculate comprehensive trading performance metrics"""
        try:
            current_time = datetime.now()
            
            # Get current portfolio data (would interface with portfolio tracker)
            portfolio_value = self._get_current_portfolio_value()
            cash_balance = self._get_cash_balance()
            active_positions = len(self.open_trades)
            
            # Calculate P&L metrics
            total_pnl = portfolio_value - self.initial_portfolio_value
            total_pnl_pct = (total_pnl / self.initial_portfolio_value) * 100
            
            # Daily P&L
            daily_pnl, daily_pnl_pct = self._calculate_daily_pnl(portfolio_value)
            
            # Risk metrics
            exposure = ((portfolio_value - cash_balance) / portfolio_value) * 100 if portfolio_value > 0 else 0
            leverage = (portfolio_value - cash_balance) / cash_balance if cash_balance > 0 else 0
            
            # Performance metrics
            win_rate = self._calculate_win_rate()
            profit_factor = self._calculate_profit_factor()
            sharpe_ratio = self._calculate_sharpe_ratio()
            max_drawdown = self._calculate_max_drawdown(portfolio_value)
            
            # Trading activity
            trades_today = self._count_trades_today()
            signals_generated, signals_acted = self._get_signal_stats()
            
            # Model performance
            avg_prediction_time = self._calculate_avg_prediction_time()
            model_accuracy = self._calculate_model_accuracy()
            
            return TradingMetrics(
                timestamp=current_time,
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                active_positions=active_positions,
                cash_balance=cash_balance,
                exposure=exposure,
                leverage=leverage,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                trades_today=trades_today,
                signals_generated=signals_generated,
                signals_acted=signals_acted,
                avg_prediction_time=avg_prediction_time,
                model_accuracy=model_accuracy
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating trading metrics: {e}")
            return None
    
    def _get_current_portfolio_value(self) -> float:
        """Get current portfolio value"""
        # Placeholder - would interface with actual portfolio tracker
        return 25000.0 + np.random.normal(0, 100)  # Simulated for example
    
    def _get_cash_balance(self) -> float:
        """Get current cash balance"""
        # Placeholder - would interface with actual portfolio tracker
        return 5000.0 + np.random.normal(0, 50)  # Simulated for example
    
    def _calculate_daily_pnl(self, current_value: float) -> tuple[float, float]:
        """Calculate daily P&L"""
        try:
            today = datetime.now().date()
            
            # Get yesterday's closing value
            yesterday_metrics = None
            for metrics in reversed(self.trading_metrics_history):
                if metrics.timestamp.date() < today:
                    yesterday_metrics = metrics
                    break
            
            if yesterday_metrics:
                daily_pnl = current_value - yesterday_metrics.portfolio_value
                daily_pnl_pct = (daily_pnl / yesterday_metrics.portfolio_value) * 100
            else:
                daily_pnl = current_value - self.initial_portfolio_value
                daily_pnl_pct = (daily_pnl / self.initial_portfolio_value) * 100
            
            return daily_pnl, daily_pnl_pct
            
        except Exception as e:
            self.logger.error(f"Error calculating daily P&L: {e}")
            return 0.0, 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed trades"""
        try:
            closed_trades = [t for t in self.trade_performance_history if not t.is_open]
            
            if not closed_trades:
                return 50.0  # Default
            
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            return (len(winning_trades) / len(closed_trades)) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 50.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            closed_trades = [t for t in self.trade_performance_history if not t.is_open]
            
            if not closed_trades:
                return 1.0
            
            gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
            
            return gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            return 1.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent returns"""
        try:
            if len(self.trading_metrics_history) < 30:
                return 0.0
            
            # Calculate daily returns
            returns = []
            prev_value = None
            
            for metrics in self.trading_metrics_history:
                if prev_value is not None:
                    daily_return = (metrics.portfolio_value / prev_value - 1)
                    returns.append(daily_return)
                prev_value = metrics.portfolio_value
            
            if len(returns) < 10:
                return 0.0
            
            returns = np.array(returns)
            excess_returns = returns - (0.02 / 252)  # Risk-free rate
            
            return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, current_value: float) -> float:
        """Calculate maximum drawdown"""
        try:
            # Update high water mark
            if current_value > self.high_water_mark:
                self.high_water_mark = current_value
            
            # Calculate current drawdown
            drawdown = ((self.high_water_mark - current_value) / self.high_water_mark) * 100
            
            return drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _count_trades_today(self) -> int:
        """Count trades executed today"""
        today = datetime.now().date()
        return len([t for t in self.trade_performance_history 
                   if t.entry_time.date() == today])
    
    def _get_signal_stats(self) -> tuple[int, int]:
        """Get signal generation and action statistics"""
        # Placeholder - would interface with actual trading engine
        return 10, 7  # signals_generated, signals_acted
    
    def _calculate_avg_prediction_time(self) -> float:
        """Calculate average model prediction time"""
        try:
            recent_predictions = [p for p in self.model_performance_history 
                                if (datetime.now() - p.timestamp).total_seconds() < 3600]
            
            if not recent_predictions:
                return 0.0
            
            return np.mean([p.prediction_time_ms for p in recent_predictions])
            
        except Exception as e:
            self.logger.error(f"Error calculating avg prediction time: {e}")
            return 0.0
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate model prediction accuracy"""
        try:
            recent_predictions = [p for p in self.model_performance_history 
                                if p.prediction_accuracy is not None and
                                (datetime.now() - p.timestamp).total_seconds() < 86400]  # Last 24 hours
            
            if not recent_predictions:
                return 50.0
            
            return np.mean([p.prediction_accuracy for p in recent_predictions]) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating model accuracy: {e}")
            return 50.0
    
    def _check_trading_alerts(self, metrics: TradingMetrics):
        """Check trading metrics against alert thresholds"""
        try:
            current_time = datetime.now()
            
            # Define checks
            checks = [
                ('daily_pnl_pct', metrics.daily_pnl_pct, 'lower'),
                ('max_drawdown', metrics.max_drawdown, 'lower'),
                ('win_rate', metrics.win_rate, 'lower'),
                ('sharpe_ratio', metrics.sharpe_ratio, 'lower'),
                ('model_accuracy', metrics.model_accuracy, 'lower'),
                ('leverage', metrics.leverage, 'upper'),
                ('exposure', metrics.exposure, 'upper')
            ]
            
            for metric_name, current_value, direction in checks:
                if metric_name not in self.alert_thresholds:
                    continue
                
                thresholds = self.alert_thresholds[metric_name]
                
                # Check cooldown
                last_alert_time = self.last_alert_times.get(metric_name)
                if last_alert_time and (current_time - last_alert_time).total_seconds() < 300:  # 5 min cooldown
                    continue
                
                # Determine alert level
                alert_level = None
                threshold_value = None
                
                if direction == 'lower':
                    if current_value <= thresholds['critical']:
                        alert_level = 'critical'
                        threshold_value = thresholds['critical']
                    elif current_value <= thresholds['warning']:
                        alert_level = 'warning'
                        threshold_value = thresholds['warning']
                else:  # upper
                    if current_value >= thresholds['critical']:
                        alert_level = 'critical'
                        threshold_value = thresholds['critical']
                    elif current_value >= thresholds['warning']:
                        alert_level = 'warning'
                        threshold_value = thresholds['warning']
                
                if alert_level:
                    self._generate_trading_alert(
                        alert_type='performance',
                        severity=alert_level,
                        title=f"{metric_name.replace('_', ' ').title()} Alert",
                        message=f"{metric_name} is {current_value:.2f} (threshold: {threshold_value:.2f})",
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=threshold_value
                    )
                    self.last_alert_times[metric_name] = current_time
        
        except Exception as e:
            self.logger.error(f"Error checking trading alerts: {e}")
    
    def _generate_trading_alert(self, alert_type: str, severity: str, title: str, 
                              message: str, metric_name: str, current_value: float,
                              threshold_value: Optional[float] = None):
        """Generate trading alert"""
        try:
            alert = TradingAlert(
                timestamp=datetime.now(),
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value
            )
            
            # Store alert
            self.alerts_history.append(alert)
            self.monitoring_stats['alerts_generated'] += 1
            
            # Log alert
            if severity == 'critical':
                self.logger.critical(f"TRADING ALERT: {title} - {message}")
            elif severity == 'warning':
                self.logger.warning(f"TRADING ALERT: {title} - {message}")
            else:
                self.logger.info(f"TRADING ALERT: {title} - {message}")
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in trading alert callback: {e}")
        
        except Exception as e:
            self.logger.error(f"Error generating trading alert: {e}")
    
    def _update_daily_stats(self, metrics: TradingMetrics):
        """Update daily statistics"""
        try:
            today = datetime.now().date().isoformat()
            
            self.daily_stats[today] = {
                'portfolio_value': metrics.portfolio_value,
                'daily_pnl': metrics.daily_pnl,
                'daily_pnl_pct': metrics.daily_pnl_pct,
                'trades_count': metrics.trades_today,
                'win_rate': metrics.win_rate,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'timestamp': metrics.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating daily stats: {e}")
    
    # Public interface methods
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a new trade for monitoring"""
        try:
            trade = TradePerformance(
                trade_id=trade_data.get('trade_id', ''),
                symbol=trade_data.get('symbol', ''),
                entry_time=trade_data.get('entry_time', datetime.now()),
                exit_time=trade_data.get('exit_time'),
                entry_price=trade_data.get('entry_price', 0.0),
                exit_price=trade_data.get('exit_price'),
                quantity=trade_data.get('quantity', 0),
                side=trade_data.get('side', ''),
                pnl=trade_data.get('pnl', 0.0),
                pnl_pct=trade_data.get('pnl_pct', 0.0),
                duration_minutes=trade_data.get('duration_minutes', 0),
                signal_confidence=trade_data.get('signal_confidence', 0.0),
                model_prediction=trade_data.get('model_prediction', 0.0),
                slippage=trade_data.get('slippage', 0.0),
                commission=trade_data.get('commission', 0.0),
                is_open=trade_data.get('is_open', True)
            )
            
            self.trade_performance_history.append(trade)
            
            if trade.is_open:
                self.open_trades[trade.trade_id] = trade
            else:
                # Remove from open trades if closed
                if trade.trade_id in self.open_trades:
                    del self.open_trades[trade.trade_id]
            
            self.monitoring_stats['total_trades_monitored'] += 1
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def record_model_prediction(self, prediction_data: Dict[str, Any]):
        """Record model prediction for performance tracking"""
        try:
            prediction = ModelPerformance(
                symbol=prediction_data.get('symbol', ''),
                timestamp=prediction_data.get('timestamp', datetime.now()),
                prediction=prediction_data.get('prediction', 0.0),
                confidence=prediction_data.get('confidence', 0.0),
                actual_return=prediction_data.get('actual_return'),
                prediction_accuracy=prediction_data.get('prediction_accuracy'),
                prediction_time_ms=prediction_data.get('prediction_time_ms', 0.0),
                features_used=prediction_data.get('features_used', 0)
            )
            
            self.model_performance_history.append(prediction)
            self.monitoring_stats['total_signals_tracked'] += 1
            
        except Exception as e:
            self.logger.error(f"Error recording model prediction: {e}")
    
    def get_current_metrics(self) -> Optional[TradingMetrics]:
        """Get most recent trading metrics"""
        return self.trading_metrics_history[-1] if self.trading_metrics_history else None
    
    def get_metrics_history(self, hours: int = 24) -> List[TradingMetrics]:
        """Get trading metrics history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.trading_metrics_history if m.timestamp >= cutoff_time]
    
    def get_recent_trades(self, count: int = 10) -> List[TradePerformance]:
        """Get recent trades"""
        return list(self.trade_performance_history)[-count:] if self.trade_performance_history else []
    
    def get_open_trades(self) -> List[TradePerformance]:
        """Get currently open trades"""
        return list(self.open_trades.values())
    
    def get_recent_alerts(self, count: int = 10) -> List[TradingAlert]:
        """Get recent alerts"""
        return list(self.alerts_history)[-count:] if self.alerts_history else []
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive trading summary"""
        try:
            current_metrics = self.get_current_metrics()
            uptime = datetime.now() - self.monitoring_stats['uptime_start']
            
            return {
                'status': 'running' if self.is_monitoring else 'stopped',
                'session_uptime_hours': uptime.total_seconds() / 3600,
                'current_metrics': asdict(current_metrics) if current_metrics else None,
                'open_trades_count': len(self.open_trades),
                'total_trades_today': self._count_trades_today(),
                'recent_alerts_count': len([a for a in self.alerts_history if 
                                          (datetime.now() - a.timestamp).total_seconds() < 3600]),
                'monitoring_stats': self.monitoring_stats,
                'daily_stats': dict(self.daily_stats),
                'alert_thresholds': self.alert_thresholds
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading summary: {e}")
            return {}
    
    def add_alert_callback(self, callback: Callable[[TradingAlert], None]):
        """Add callback for trading alert notifications"""
        self.alert_callbacks.append(callback)
    
    def update_alert_threshold(self, metric_name: str, warning_level: float, critical_level: float):
        """Update alert threshold for a trading metric"""
        if metric_name in self.alert_thresholds:
            self.alert_thresholds[metric_name]['warning'] = warning_level
            self.alert_thresholds[metric_name]['critical'] = critical_level
            self.logger.info(f"Updated trading alert threshold for {metric_name}")
    
    def export_trading_data(self, filename: str, hours: int = 24) -> bool:
        """Export trading data to JSON file"""
        try:
            export_data = {
                'export_time': datetime.now().isoformat(),
                'time_period_hours': hours,
                'trading_metrics': [asdict(m) for m in self.get_metrics_history(hours)],
                'recent_trades': [asdict(t) for t in self.get_recent_trades(50)],
                'model_performance': [asdict(p) for p in self.model_performance_history if 
                                    (datetime.now() - p.timestamp).total_seconds() < hours * 3600],
                'alerts': [asdict(a) for a in self.alerts_history],
                'daily_stats': dict(self.daily_stats),
                'monitoring_stats': self.monitoring_stats
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Trading data exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting trading data: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize trading monitor
    monitor = TradingMonitor(monitoring_interval=10)
    
    # Add alert callback
    def trading_alert_handler(alert):
        print(f"TRADING ALERT [{alert.severity}]: {alert.title} - {alert.message}")
    
    monitor.add_alert_callback(trading_alert_handler)
    
    # Start monitoring
    if monitor.start_monitoring(initial_portfolio_value=25000.0):
        print("Trading monitoring started")
        
        try:
            # Simulate some trading activity
            time.sleep(5)
            
            # Record a sample trade
            monitor.record_trade({
                'trade_id': 'TEST001',
                'symbol': 'AAPL',
                'entry_time': datetime.now(),
                'entry_price': 150.0,
                'quantity': 100,
                'side': 'long',
                'signal_confidence': 0.75,
                'model_prediction': 0.72,
                'is_open': True
            })
            
            # Record a model prediction
            monitor.record_model_prediction({
                'symbol': 'AAPL',
                'prediction': 0.72,
                'confidence': 0.75,
                'prediction_time_ms': 15.5,
                'features_used': 25
            })
            
            time.sleep(10)
            
            # Get current metrics
            current = monitor.get_current_metrics()
            if current:
                print(f"Portfolio Value: ${current.portfolio_value:,.2f}")
                print(f"Daily P&L: {current.daily_pnl_pct:.2f}%")
                print(f"Win Rate: {current.win_rate:.1f}%")
                print(f"Active Positions: {current.active_positions}")
            
            # Get summary
            summary = monitor.get_trading_summary()
            print(f"Total trades monitored: {summary.get('monitoring_stats', {}).get('total_trades_monitored', 0)}")
            
        except KeyboardInterrupt:
            print("Stopping...")
        
        finally:
            monitor.stop_monitoring()
    else:
        print("Failed to start trading monitoring")
