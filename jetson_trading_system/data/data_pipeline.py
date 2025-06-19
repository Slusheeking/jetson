"""
Data Pipeline for Jetson Trading System
Comprehensive data ingestion, processing, and management pipeline
"""

import asyncio
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import json

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingParams
from jetson_trading_system.utils.logger import get_data_logger
from jetson_trading_system.utils.database import TradingDatabase
from jetson_trading_system.data.polygon_client import PolygonClient
from jetson_trading_system.data.cache_manager import CacheManager
from jetson_trading_system.features.technical_indicators import TechnicalIndicators

@dataclass
class DataPipelineConfig:
    """Data pipeline configuration"""
    market_data_interval: int = 60  # seconds
    indicator_update_interval: int = 300  # seconds
    cache_cleanup_interval: int = 3600  # seconds
    max_concurrent_requests: int = 5
    enable_real_time: bool = True
    enable_historical_backfill: bool = True
    data_quality_checks: bool = True
    auto_recovery: bool = True

@dataclass
class DataQualityMetrics:
    """Data quality metrics"""
    timestamp: datetime
    symbol: str
    total_records: int
    missing_records: int
    duplicate_records: int
    outlier_records: int
    data_completeness: float
    data_freshness_seconds: float
    quality_score: float

class DataPipeline:
    """
    Comprehensive data pipeline for Jetson trading system
    Handles real-time and historical data ingestion, processing, and quality control
    """
    
    def __init__(self, config: DataPipelineConfig = None, cache_manager: CacheManager = None):
        """
        Initialize data pipeline
        
        Args:
            config: Pipeline configuration
            cache_manager: An instance of CacheManager. If None, a new one is created.
        """
        self.config = config or DataPipelineConfig()
        
        self.logger = get_data_logger()
        self.db_manager = TradingDatabase()
        self.polygon_client = PolygonClient()
        self.cache_manager = cache_manager or CacheManager()
        self.ta_calculator = TechnicalIndicators()
        
        # Pipeline state
        self.is_running = False
        self.pipeline_threads = {}
        self.shutdown_event = threading.Event()
        
        # Data processing queues
        self.market_data_queue = queue.Queue(maxsize=1000)
        self.indicator_queue = queue.Queue(maxsize=500)
        self.error_queue = queue.Queue(maxsize=100)
        
        # Performance tracking
        self.pipeline_stats = {
            'data_points_processed': 0,
            'indicators_calculated': 0,
            'errors_encountered': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'uptime_start': None,
            'last_data_update': None
        }
        
        # Data quality tracking
        self.quality_metrics = {}  # symbol -> DataQualityMetrics
        self.data_callbacks = []
        
        # Symbols to monitor
        self.monitored_symbols = TradingParams.CORE_MARKET_SYMBOLS + TradingParams.MARKET_INDICES
        
        self.logger.info("DataPipeline initialized")
    
    def start_pipeline(self) -> bool:
        """Start the data pipeline"""
        try:
            if self.is_running:
                self.logger.warning("Data pipeline is already running")
                return True
            
            self.logger.info("Starting data pipeline...")
            
            # Reset shutdown event
            self.shutdown_event.clear()
            self.pipeline_stats['uptime_start'] = datetime.now()
            
            # Start pipeline threads
            self._start_pipeline_threads()
            
            # Schedule periodic tasks
            self._schedule_periodic_tasks()
            
            self.is_running = True
            
            self.logger.info("Data pipeline started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start data pipeline: {e}")
            return False
    
    def stop_pipeline(self) -> bool:
        """Stop the data pipeline"""
        try:
            if not self.is_running:
                return True
            
            self.logger.info("Stopping data pipeline...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for threads to finish
            for thread_name, thread in self.pipeline_threads.items():
                if thread.is_alive():
                    thread.join(timeout=10)
                    if thread.is_alive():
                        self.logger.warning(f"Thread {thread_name} did not shut down gracefully")
            
            self.is_running = False
            
            self.logger.info("Data pipeline stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping data pipeline: {e}")
            return False
    
    def _start_pipeline_threads(self):
        """Start all pipeline threads"""
        try:
            # Market data ingestion thread
            if self.config.enable_real_time:
                self.pipeline_threads['market_data'] = threading.Thread(
                    target=self._market_data_loop, daemon=True
                )
                self.pipeline_threads['market_data'].start()
            
            # Data processing thread
            self.pipeline_threads['data_processor'] = threading.Thread(
                target=self._data_processing_loop, daemon=True
            )
            self.pipeline_threads['data_processor'].start()
            
            # Indicator calculation thread
            self.pipeline_threads['indicators'] = threading.Thread(
                target=self._indicator_calculation_loop, daemon=True
            )
            self.pipeline_threads['indicators'].start()
            
            # Data quality monitoring thread
            if self.config.data_quality_checks:
                self.pipeline_threads['quality_monitor'] = threading.Thread(
                    target=self._quality_monitoring_loop, daemon=True
                )
                self.pipeline_threads['quality_monitor'].start()
            
            # Error handling thread
            self.pipeline_threads['error_handler'] = threading.Thread(
                target=self._error_handling_loop, daemon=True
            )
            self.pipeline_threads['error_handler'].start()
            
            self.logger.info(f"Started {len(self.pipeline_threads)} pipeline threads")
            
        except Exception as e:
            self.logger.error(f"Error starting pipeline threads: {e}")
            raise
    
    def _schedule_periodic_tasks(self):
        """Schedule periodic maintenance tasks"""
        try:
            # Schedule data quality reports
            schedule.every().day.at("09:00").do(self._generate_quality_report)
            
            # Schedule historical data backfill
            if self.config.enable_historical_backfill:
                schedule.every().day.at("06:00").do(self._backfill_historical_data)
            
            # Start scheduler thread
            self.pipeline_threads['scheduler'] = threading.Thread(
                target=self._scheduler_loop, daemon=True
            )
            self.pipeline_threads['scheduler'].start()
            
        except Exception as e:
            self.logger.error(f"Error scheduling periodic tasks: {e}")
    
    def _market_data_loop(self):
        """Main market data ingestion loop"""
        try:
            self.logger.info("Market data ingestion loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Check if market is open
                    if self._is_market_open():
                        # Fetch real-time data for all symbols
                        self._fetch_real_time_data()
                    
                    # Wait for next interval
                    self.shutdown_event.wait(timeout=self.config.market_data_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in market data loop: {e}")
                    self.error_queue.put(('market_data', str(e), datetime.now()))
                    time.sleep(30)  # Wait before retrying
        
        except Exception as e:
            self.logger.critical(f"Fatal error in market data loop: {e}")
        
        finally:
            self.logger.info("Market data ingestion loop ended")
    
    def _data_processing_loop(self):
        """Main data processing loop"""
        try:
            self.logger.info("Data processing loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Process queued market data
                    try:
                        data_batch = self.market_data_queue.get(timeout=1)
                        self._process_data_batch(data_batch)
                        self.market_data_queue.task_done()
                        
                    except queue.Empty:
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error in data processing loop: {e}")
                    self.error_queue.put(('data_processing', str(e), datetime.now()))
        
        except Exception as e:
            self.logger.critical(f"Fatal error in data processing loop: {e}")
        
        finally:
            self.logger.info("Data processing loop ended")
    
    def _indicator_calculation_loop(self):
        """Technical indicator calculation loop"""
        try:
            self.logger.info("Indicator calculation loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Calculate indicators for symbols in queue
                    try:
                        symbol = self.indicator_queue.get(timeout=1)
                        self._calculate_symbol_indicators(symbol)
                        self.indicator_queue.task_done()
                        
                    except queue.Empty:
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error in indicator calculation loop: {e}")
                    self.error_queue.put(('indicators', str(e), datetime.now()))
        
        except Exception as e:
            self.logger.critical(f"Fatal error in indicator calculation loop: {e}")
        
        finally:
            self.logger.info("Indicator calculation loop ended")
    
    def _quality_monitoring_loop(self):
        """Data quality monitoring loop"""
        try:
            self.logger.info("Quality monitoring loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Check data quality for all symbols
                    for symbol in self.monitored_symbols:
                        self._check_data_quality(symbol)
                    
                    # Wait before next quality check
                    self.shutdown_event.wait(timeout=600)  # Every 10 minutes
                    
                except Exception as e:
                    self.logger.error(f"Error in quality monitoring loop: {e}")
                    self.error_queue.put(('quality_monitor', str(e), datetime.now()))
        
        except Exception as e:
            self.logger.critical(f"Fatal error in quality monitoring loop: {e}")
        
        finally:
            self.logger.info("Quality monitoring loop ended")
    
    def _error_handling_loop(self):
        """Error handling and recovery loop"""
        try:
            self.logger.info("Error handling loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Process errors from queue
                    try:
                        error_type, error_msg, timestamp = self.error_queue.get(timeout=1)
                        self._handle_pipeline_error(error_type, error_msg, timestamp)
                        self.error_queue.task_done()
                        
                    except queue.Empty:
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error in error handling loop: {e}")
        
        except Exception as e:
            self.logger.critical(f"Fatal error in error handling loop: {e}")
        
        finally:
            self.logger.info("Error handling loop ended")
    
    def _scheduler_loop(self):
        """Periodic task scheduler loop"""
        try:
            self.logger.info("Scheduler loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    schedule.run_pending()
                    self.shutdown_event.wait(timeout=60)  # Check every minute
                    
                except Exception as e:
                    self.logger.error(f"Error in scheduler loop: {e}")
        
        except Exception as e:
            self.logger.critical(f"Fatal error in scheduler loop: {e}")
        
        finally:
            self.logger.info("Scheduler loop ended")
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        # Simple market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _fetch_real_time_data(self):
        """Fetch real-time market data for all symbols"""
        try:
            # Use ThreadPoolExecutor for concurrent requests
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
                future_to_symbol = {
                    executor.submit(self._fetch_symbol_data, symbol): symbol
                    for symbol in self.monitored_symbols
                }
                
                data_batch = {}
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result(timeout=30)
                        if data is not None:
                            data_batch[symbol] = data
                            
                    except Exception as e:
                        self.logger.error(f"Error fetching data for {symbol}: {e}")
                
                # Queue data batch for processing
                if data_batch:
                    try:
                        self.market_data_queue.put_nowait(data_batch)
                    except queue.Full:
                        self.logger.warning("Market data queue is full, dropping batch")
            
        except Exception as e:
            self.logger.error(f"Error fetching real-time data: {e}")
    
    def _fetch_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch data for a single symbol"""
        try:
            # Check cache first
            cached_data = self.cache_manager.get_latest_price(symbol)
            if cached_data and self._is_data_fresh(cached_data):
                self.pipeline_stats['cache_hits'] += 1
                return cached_data
            
            self.pipeline_stats['cache_misses'] += 1
            
            # Fetch from API
            data = self.polygon_client.get_latest_price(symbol)
            
            if data:
                # Cache the data
                self.cache_manager.cache_price_data(symbol, data)
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _is_data_fresh(self, data: Dict[str, Any], max_age_seconds: int = 60) -> bool:
        """Check if data is fresh enough"""
        try:
            if 'timestamp' in data:
                data_time = pd.to_datetime(data['timestamp'])
                age_seconds = (datetime.now() - data_time).total_seconds()
                return age_seconds <= max_age_seconds
            return False
            
        except Exception:
            return False
    
    def _process_data_batch(self, data_batch: Dict[str, Any]):
        """Process a batch of market data"""
        try:
            for symbol, data in data_batch.items():
                # Store in database
                self._store_market_data(symbol, data)
                
                # Queue for indicator calculation
                try:
                    self.indicator_queue.put_nowait(symbol)
                except queue.Full:
                    pass  # Skip if queue is full
                
                # Notify data callbacks
                self._notify_data_callbacks(symbol, data)
            
            self.pipeline_stats['data_points_processed'] += len(data_batch)
            self.pipeline_stats['last_data_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error processing data batch: {e}")
    
    def _store_market_data(self, symbol: str, data: Dict[str, Any]):
        """Store market data in database"""
        try:
            # Convert to DataFrame format expected by database
            df_data = pd.DataFrame([{
                'timestamp': pd.to_datetime(data.get('timestamp', datetime.now())),
                'open': data.get('open', 0.0),
                'high': data.get('high', 0.0),
                'low': data.get('low', 0.0),
                'close': data.get('close', 0.0),
                'volume': data.get('volume', 0)
            }])
            
            df_data.set_index('timestamp', inplace=True)
            
            # Store in database
            self.db_manager.store_price_data(symbol, df_data)
            
        except Exception as e:
            self.logger.error(f"Error storing market data for {symbol}: {e}")
    
    def _calculate_symbol_indicators(self, symbol: str):
        """Calculate technical indicators for a symbol"""
        try:
            # Get recent price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
            
            price_data = self.db_manager.get_price_data(symbol, start_date, end_date)
            
            if price_data is not None and len(price_data) >= 20:
                # Calculate indicators
                indicators = self.ta_calculator.calculate_all_indicators(price_data)
                
                if indicators is not None and not indicators.empty:
                    # Store indicators
                    self.db_manager.store_technical_indicators(symbol, indicators)
                    self.pipeline_stats['indicators_calculated'] += 1
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
    
    def _check_data_quality(self, symbol: str):
        """Check data quality for a symbol"""
        try:
            # Get recent data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            data = self.db_manager.get_price_data(symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                # Calculate quality metrics
                total_records = len(data)
                missing_records = data.isnull().sum().sum()
                duplicate_records = data.duplicated().sum()
                
                # Check for outliers (simplified)
                returns = data['close'].pct_change()
                outlier_threshold = 0.2  # 20% change
                outlier_records = (abs(returns) > outlier_threshold).sum()
                
                # Data completeness
                expected_records = 390  # Approximate for one trading day
                data_completeness = min(total_records / expected_records, 1.0)
                
                # Data freshness
                if len(data) > 0:
                    latest_timestamp = data.index[-1]
                    freshness_seconds = (datetime.now() - latest_timestamp).total_seconds()
                else:
                    freshness_seconds = float('inf')
                
                # Overall quality score
                quality_score = self._calculate_quality_score(
                    data_completeness, missing_records, outlier_records, freshness_seconds
                )
                
                # Create quality metrics
                quality_metrics = DataQualityMetrics(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    total_records=total_records,
                    missing_records=missing_records,
                    duplicate_records=duplicate_records,
                    outlier_records=outlier_records,
                    data_completeness=data_completeness,
                    data_freshness_seconds=freshness_seconds,
                    quality_score=quality_score
                )
                
                self.quality_metrics[symbol] = quality_metrics
                
                # Log quality issues
                if quality_score < 0.7:
                    self.logger.warning(f"Data quality issue for {symbol}: score={quality_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error checking data quality for {symbol}: {e}")
    
    def _calculate_quality_score(self, 
                               completeness: float, 
                               missing_records: int, 
                               outlier_records: int, 
                               freshness_seconds: float) -> float:
        """Calculate overall data quality score"""
        try:
            # Completeness score (0-1)
            completeness_score = completeness
            
            # Missing data penalty
            missing_penalty = min(missing_records / 100, 0.5)
            
            # Outlier penalty
            outlier_penalty = min(outlier_records / 50, 0.3)
            
            # Freshness penalty
            freshness_penalty = min(freshness_seconds / 3600, 0.5)  # 1 hour max penalty
            
            # Combined score
            quality_score = max(0.0, completeness_score - missing_penalty - outlier_penalty - freshness_penalty)
            
            return quality_score
            
        except Exception:
            return 0.0
    
    def _handle_pipeline_error(self, error_type: str, error_msg: str, timestamp: datetime):
        """Handle pipeline errors and implement recovery"""
        try:
            self.pipeline_stats['errors_encountered'] += 1
            
            self.logger.error(f"Pipeline error [{error_type}]: {error_msg}")
            
            # Implement recovery strategies
            if self.config.auto_recovery:
                if error_type == 'market_data':
                    self._recover_market_data()
                elif error_type == 'indicators':
                    self._recover_indicators()
                elif error_type == 'data_processing':
                    self._recover_data_processing()
            
        except Exception as e:
            self.logger.error(f"Error handling pipeline error: {e}")
    
    def _recover_market_data(self):
        """Recover from market data errors"""
        try:
            self.logger.info("Attempting market data recovery...")
            
            # Reconnect to data source
            self.polygon_client = PolygonClient()
            
            # Clear market data queue
            while not self.market_data_queue.empty():
                try:
                    self.market_data_queue.get_nowait()
                    self.market_data_queue.task_done()
                except queue.Empty:
                    break
            
            self.logger.info("Market data recovery completed")
            
        except Exception as e:
            self.logger.error(f"Error in market data recovery: {e}")
    
    def _recover_indicators(self):
        """Recover from indicator calculation errors"""
        try:
            self.logger.info("Attempting indicator recovery...")
            
            # Reinitialize technical indicators calculator
            self.ta_calculator = TechnicalIndicators()
            
            # Clear indicator queue
            while not self.indicator_queue.empty():
                try:
                    self.indicator_queue.get_nowait()
                    self.indicator_queue.task_done()
                except queue.Empty:
                    break
            
            self.logger.info("Indicator recovery completed")
            
        except Exception as e:
            self.logger.error(f"Error in indicator recovery: {e}")
    
    def _recover_data_processing(self):
        """Recover from data processing errors"""
        try:
            self.logger.info("Attempting data processing recovery...")
            
            # Check database connection
            self.db_manager = TradingDatabase()
            
            self.logger.info("Data processing recovery completed")
            
        except Exception as e:
            self.logger.error(f"Error in data processing recovery: {e}")
    
    def _notify_data_callbacks(self, symbol: str, data: Dict[str, Any]):
        """Notify registered data callbacks"""
        try:
            for callback in self.data_callbacks:
                try:
                    callback(symbol, data)
                except Exception as e:
                    self.logger.error(f"Error in data callback: {e}")
        except Exception as e:
            self.logger.error(f"Error notifying data callbacks: {e}")
    
    def _generate_quality_report(self):
        """Generate daily data quality report"""
        try:
            self.logger.info("Generating data quality report...")
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'quality_metrics': {symbol: asdict(metrics) for symbol, metrics in self.quality_metrics.items()},
                'pipeline_stats': self.pipeline_stats.copy()
            }
            
            # Save report
            report_filename = f"data_quality_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(f"./reports/{report_filename}", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Data quality report saved: {report_filename}")
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
    
    def _backfill_historical_data(self):
        """Backfill missing historical data"""
        try:
            self.logger.info("Starting historical data backfill...")
            
            # Check for missing data and backfill
            for symbol in self.monitored_symbols:
                try:
                    self._backfill_symbol_data(symbol)
                except Exception as e:
                    self.logger.error(f"Error backfilling data for {symbol}: {e}")
            
            self.logger.info("Historical data backfill completed")
            
        except Exception as e:
            self.logger.error(f"Error in historical data backfill: {e}")
    
    def _backfill_symbol_data(self, symbol: str):
        """Backfill historical data for a symbol"""
        try:
            # Get existing data range
            existing_data = self.db_manager.get_price_data(symbol, "2020-01-01", datetime.now().strftime('%Y-%m-%d'))
            
            if existing_data is None or existing_data.empty:
                # No existing data, fetch full history
                start_date = "2020-01-01"
            else:
                # Find gaps in existing data
                last_date = existing_data.index[-1]
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            if start_date < end_date:
                # Fetch missing data
                new_data = self.polygon_client.get_historical_data(symbol, start_date, end_date)
                
                if new_data is not None and not new_data.empty:
                    self.db_manager.store_price_data(symbol, new_data)
                    self.logger.info(f"Backfilled {len(new_data)} records for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error backfilling data for {symbol}: {e}")
    
    # Public interface methods
    def add_data_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for real-time data updates"""
        self.data_callbacks.append(callback)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        try:
            uptime = None
            if self.pipeline_stats['uptime_start']:
                uptime = (datetime.now() - self.pipeline_stats['uptime_start']).total_seconds()
            
            return {
                'is_running': self.is_running,
                'uptime_seconds': uptime,
                'pipeline_stats': self.pipeline_stats.copy(),
                'queue_sizes': {
                    'market_data': self.market_data_queue.qsize(),
                    'indicators': self.indicator_queue.qsize(),
                    'errors': self.error_queue.qsize()
                },
                'thread_status': {name: thread.is_alive() for name, thread in self.pipeline_threads.items()},
                'quality_metrics_count': len(self.quality_metrics),
                'monitored_symbols': len(self.monitored_symbols)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get data quality summary"""
        try:
            if not self.quality_metrics:
                return {}
            
            quality_scores = [metrics.quality_score for metrics in self.quality_metrics.values()]
            
            return {
                'total_symbols': len(self.quality_metrics),
                'avg_quality_score': np.mean(quality_scores),
                'min_quality_score': np.min(quality_scores),
                'max_quality_score': np.max(quality_scores),
                'symbols_below_threshold': len([s for s in quality_scores if s < 0.7]),
                'last_updated': max(metrics.timestamp for metrics in self.quality_metrics.values()).isoformat(),
                'quality_by_symbol': {symbol: metrics.quality_score for symbol, metrics in self.quality_metrics.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality summary: {e}")
            return {}
    
    def force_data_refresh(self, symbols: List[str] = None) -> bool:
        """Force refresh of data for specified symbols"""
        try:
            symbols = symbols or self.monitored_symbols
            
            self.logger.info(f"Forcing data refresh for {len(symbols)} symbols")
            
            # Clear relevant caches
            for symbol in symbols:
                self.cache_manager.clear_symbol_cache(symbol)
            
            # Fetch fresh data
            self._fetch_real_time_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error forcing data refresh: {e}")
            return False

# Example usage
if __name__ == "__main__":
    print("--- Running DataPipeline Demo in Live Mode ---")

    # Configure the pipeline for a short, active demonstration
    config = DataPipelineConfig(
        market_data_interval=10,       # Fetch data every 10 seconds
        indicator_update_interval=20,  # Update indicators every 20 seconds
        enable_real_time=True,
        enable_historical_backfill=False, # Disable backfill for this demo
        data_quality_checks=True,
        auto_recovery=True
    )
    
    # Initialize the pipeline
    pipeline = DataPipeline(config)
    
    # Add a callback to see the live data being processed
    def live_data_callback(symbol, data):
        print(f">>> DATA RECEIVED: {symbol} @ {data.get('close', 'N/A')} "
              f"(Time: {datetime.fromtimestamp(data.get('timestamp', 0) / 1000)})")
    
    pipeline.add_data_callback(live_data_callback)
    
    # Start the pipeline
    if pipeline.start_pipeline():
        print("\nData pipeline started successfully.")
        print("It will run for 1 minute, fetching and processing live data.")
        print("Press Ctrl+C to stop the demo early.")
        
        try:
            # Let the pipeline run to demonstrate its functionality
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down pipeline...")
        
        finally:
            # Stop the pipeline
            print("\nStopping data pipeline...")
            pipeline.stop_pipeline()
            
            print("\n--- Final Pipeline Status ---")
            status = pipeline.get_pipeline_status()
            print(f"  - Uptime (s): {status.get('uptime_seconds', 0):.2f}")
            print(f"  - Data Points Processed: {status['pipeline_stats']['data_points_processed']}")
            print(f"  - Indicators Calculated: {status['pipeline_stats']['indicators_calculated']}")
            print(f"  - Errors: {status['pipeline_stats']['errors_encountered']}")

            print("\n--- Final Data Quality Summary ---")
            quality = pipeline.get_data_quality_summary()
            if quality:
                print(f"  - Avg Quality Score: {quality.get('avg_quality_score', 0):.2f}")
                print(f"  - Symbols Below Threshold: {quality.get('symbols_below_threshold', 0)}")
            else:
                print("  No quality metrics were generated.")

    else:
        print("\nFailed to start the data pipeline.")
            
    print("\n--- Demo Finished ---")
