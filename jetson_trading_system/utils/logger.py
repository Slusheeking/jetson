"""
Enhanced Logging System
Optimized logging for Jetson Orin with performance monitoring
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional
import threading
import queue
import time

from ..config.jetson_settings import JetsonConfig
from ..config.trading_params import DataConfig

class JetsonFormatter(logging.Formatter):
    """Custom formatter with enhanced information for trading system"""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def format(self, record):
        # Add custom fields
        record.uptime = time.time() - self.start_time
        record.thread_name = threading.current_thread().name
        
        # Create base format
        base_format = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | "
            "T:%(thread_name)-10s | UP:%(uptime)6.1fs | "
            "%(message)s"
        )
        
        # Add extra context for trading-specific logs
        if hasattr(record, 'symbol'):
            base_format += " | SYM:%(symbol)s"
        if hasattr(record, 'pnl'):
            base_format += " | PNL:%(pnl).2f"
        if hasattr(record, 'signal_strength'):
            base_format += " | SIG:%(signal_strength).3f"
        
        formatter = logging.Formatter(base_format)
        return formatter.format(record)

class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def __init__(self):
        super().__init__()
        self._call_count = 0
        self._start_time = time.time()
    
    def filter(self, record):
        self._call_count += 1
        record.call_count = self._call_count
        record.system_time = time.time() - self._start_time
        return True

class TradingLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with trading-specific context"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add trading context to log messages
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Merge adapter's extra with call's extra
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs
    
    def trade(self, symbol: str, action: str, quantity: int, price: float, msg: str = ""):
        """Log trading action"""
        self.info(
            f"TRADE: {action} {quantity} {symbol} @ ${price:.2f} {msg}",
            extra={'symbol': symbol, 'action': action, 'quantity': quantity, 'price': price}
        )
    
    def signal(self, symbol: str, signal_type: str, strength: float, msg: str = ""):
        """Log trading signal"""
        self.info(
            f"SIGNAL: {signal_type} {symbol} strength={strength:.3f} {msg}",
            extra={'symbol': symbol, 'signal_type': signal_type, 'signal_strength': strength}
        )
    
    def pnl(self, symbol: str, pnl_value: float, msg: str = ""):
        """Log P&L information"""
        level = logging.INFO if pnl_value >= 0 else logging.WARNING
        self.log(
            level,
            f"PNL: {symbol} ${pnl_value:+.2f} {msg}",
            extra={'symbol': symbol, 'pnl': pnl_value}
        )
    
    def performance(self, metric: str, value: float, msg: str = ""):
        """Log performance metrics"""
        self.info(
            f"PERF: {metric}={value:.3f} {msg}",
            extra={'metric': metric, 'value': value}
        )

class AsyncFileHandler(logging.handlers.RotatingFileHandler):
    """Asynchronous file handler for better performance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Background worker to write log records"""
        while not self._stop_event.is_set():
            try:
                record = self.queue.get(timeout=1.0)
                if record is None:  # Sentinel to stop
                    break
                super().emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Fallback to stderr if logging fails
                print(f"Logging error: {e}", file=sys.stderr)
    
    def emit(self, record):
        """Queue record for async processing"""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Drop the record if queue is full
            pass
    
    def close(self):
        """Clean shutdown of async handler"""
        self._stop_event.set()
        self.queue.put(None)  # Sentinel
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        super().close()

class SystemMetricsHandler(logging.Handler):
    """Handler to capture system metrics with log records"""
    
    def __init__(self):
        super().__init__()
        self.metrics_file = Path("./logs/system_metrics.jsonl")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to import Jetson utilities
        try:
            from .jetson_utils import jetson_monitor
            self.jetson_monitor = jetson_monitor
        except ImportError:
            self.jetson_monitor = None
    
    def emit(self, record):
        """Emit log record with system metrics"""
        if self.jetson_monitor and record.levelno >= logging.WARNING:
            try:
                # Get system stats for warnings and errors
                stats = self.jetson_monitor.get_complete_system_stats()
                
                metrics_record = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'system_stats': stats
                }
                
                with open(self.metrics_file, 'a') as f:
                    f.write(json.dumps(metrics_record) + '\n')
                    
            except Exception:
                # Don't let metrics collection break logging
                pass

def setup_logging(log_level: str = "INFO", 
                 log_dir: str = "./logs",
                 enable_async: bool = True,
                 enable_metrics: bool = True) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging system for Jetson trading
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        enable_async: Use async file handlers
        enable_metrics: Enable system metrics logging
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create custom formatter
    formatter = JetsonFormatter()
    
    # Console handler with colors (if supported)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(PerformanceFilter())
    
    # Main log file handler
    if enable_async:
        main_handler = AsyncFileHandler(
            log_path / "jetson_trading.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
    else:
        main_handler = logging.handlers.RotatingFileHandler(
            log_path / "jetson_trading.log",
            maxBytes=50*1024*1024,
            backupCount=5
        )
    
    main_handler.setLevel(getattr(logging, log_level.upper()))
    main_handler.setFormatter(formatter)
    main_handler.addFilter(PerformanceFilter())
    
    # Error log file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # Trading-specific log handler
    trading_handler = logging.handlers.RotatingFileHandler(
        log_path / "trading.log",
        maxBytes=25*1024*1024,  # 25MB
        backupCount=5
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    
    # System metrics handler
    if enable_metrics:
        metrics_handler = SystemMetricsHandler()
        metrics_handler.setLevel(logging.WARNING)
        root_logger.addHandler(metrics_handler)
    
    # Create specialized loggers
    loggers = {}
    
    # Trading logger
    trading_logger = logging.getLogger('trading')
    trading_logger.addHandler(trading_handler)
    trading_logger.setLevel(logging.INFO)
    loggers['trading'] = TradingLoggerAdapter(trading_logger)
    
    # Data logger
    data_logger = logging.getLogger('data')
    loggers['data'] = TradingLoggerAdapter(data_logger)
    
    # Model logger
    model_logger = logging.getLogger('model')
    loggers['model'] = TradingLoggerAdapter(model_logger)
    
    # System logger
    system_logger = logging.getLogger('system')
    loggers['system'] = TradingLoggerAdapter(system_logger)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    loggers['performance'] = TradingLoggerAdapter(perf_logger)
    
    # Risk logger
    risk_logger = logging.getLogger('risk')
    loggers['risk'] = TradingLoggerAdapter(risk_logger)
    
    logging.info(f"Logging system initialized - Level: {log_level}, Dir: {log_dir}")
    
    return loggers

def get_logger(name: str) -> TradingLoggerAdapter:
    """Get a trading logger with the specified name"""
    logger = logging.getLogger(name)
    return TradingLoggerAdapter(logger)

class LoggingContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, logger: TradingLoggerAdapter, **context):
        self.logger = logger
        self.context = context
        self.original_extra = logger.extra.copy()
    
    def __enter__(self):
        self.logger.extra.update(self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.extra = self.original_extra

# Performance timing decorator
def log_performance(logger: TradingLoggerAdapter = None):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_logger = logger or get_logger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                func_logger.performance(
                    f"{func.__name__}_time", 
                    execution_time,
                    f"Function {func.__name__} completed"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                func_logger.error(
                    f"Function {func.__name__} failed after {execution_time:.3f}s: {e}",
                    extra={'execution_time': execution_time, 'error': str(e)}
                )
                raise
        
        return wrapper
    return decorator

# Initialize default logging
default_loggers = setup_logging()

# Convenience functions
def get_trading_logger() -> TradingLoggerAdapter:
    """Get the main trading logger"""
    return default_loggers.get('trading', get_logger('trading'))

def get_data_logger() -> TradingLoggerAdapter:
    """Get the data logger"""
    return default_loggers.get('data', get_logger('data'))

def get_model_logger() -> TradingLoggerAdapter:
    """Get the model logger"""
    return default_loggers.get('model', get_logger('model'))

def get_system_logger() -> TradingLoggerAdapter:
    """Get the system logger"""
    return default_loggers.get('system', get_logger('system'))

def get_risk_logger() -> TradingLoggerAdapter:
    """Get the risk logger"""
    return default_loggers.get('risk', get_logger('risk'))

# Example usage
if __name__ == "__main__":
    print("--- Running Logger Demo ---")
    
    # Get different loggers
    trading_logger = get_trading_logger()
    data_logger = get_data_logger()
    model_logger = get_model_logger()
    system_logger = get_system_logger()
    
    print("\n--- Testing Standard Logs ---")
    trading_logger.info("System startup")
    data_logger.debug("Fetching initial data...") # This may not appear depending on log level
    model_logger.warning("Model performance is slightly below threshold.")
    system_logger.error("System configuration value is missing.")
    
    print("\n--- Testing TradingLoggerAdapter Special Methods ---")
    trading_logger.trade("AAPL", "BUY", 100, 150.50, "Opening new position based on signal.")
    trading_logger.signal("MSFT", "BULLISH", 0.85, "MACD crossover confirmed.")
    trading_logger.pnl("GOOG", -120.25, "Position closed at a loss.")
    trading_logger.performance("sharpe_ratio", 1.25, "Portfolio Sharpe ratio updated.")

    print("\n--- Testing LoggingContext ---")
    with LoggingContext(trading_logger, symbol="TSLA", strategy="mean_reversion"):
        trading_logger.info("Executing mean reversion strategy.")
        trading_logger.trade("TSLA", "SELL", 50, 205.10, "Closing over-extended position.")
    
    print("\n--- Testing Performance Decorator ---")
    @log_performance(system_logger)
    def sample_task(duration):
        print(f"Executing a sample task that takes {duration}s...")
        time.sleep(duration)
        print("Sample task finished.")
        
    sample_task(0.5)
    
    print("\n--- Logger Demo Finished ---")
