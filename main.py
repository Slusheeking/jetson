#!/usr/bin/env python3
"""
Jetson Trading System - Main Application Entry Point
Production-ready ML4Trading system optimized for NVIDIA Jetson Orin 16GB
"""

import asyncio
import argparse
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Core system imports
from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingParams
from jetson_trading_system.utils.logger import setup_logging, get_logger
from jetson_trading_system.utils.jetson_utils import JetsonMonitor
from jetson_trading_system.utils.database import TradingDatabase

# Data and caching
from jetson_trading_system.data.polygon_client import PolygonDataClient
from jetson_trading_system.data.yahoo_client import YahooFinanceClient
from jetson_trading_system.data.symbol_discovery import SymbolDiscoveryEngine
from jetson_trading_system.data.data_pipeline import DataPipeline
from jetson_trading_system.data.cache_manager import CacheManager

# Feature engineering and ML
from jetson_trading_system.features.feature_engine import FeatureEngine
from jetson_trading_system.models.lightgbm_trainer import LightGBMTrainer
from jetson_trading_system.models.model_predictor import ModelPredictor
from jetson_trading_system.models.model_registry import ModelRegistry

# Risk management and execution
from jetson_trading_system.risk.risk_manager import RiskManager
from jetson_trading_system.risk.position_sizer import PositionSizer
from jetson_trading_system.execution.trading_engine import TradingEngine
from jetson_trading_system.execution.order_manager import OrderManager
from jetson_trading_system.execution.portfolio_tracker import PortfolioTracker

# Backtesting and monitoring
from jetson_trading_system.backtesting.zipline_engine import ZiplineEngine
from jetson_trading_system.backtesting.performance_analyzer import PerformanceAnalyzer
from jetson_trading_system.monitoring.system_monitor import SystemMonitor
from jetson_trading_system.monitoring.trading_monitor import TradingMonitor

class JetsonTradingSystem:
    """
    Main Jetson Trading System orchestrator
    Manages all components and provides unified interface for trading operations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Jetson Trading System
        
        Args:
            config_path: Optional path to configuration file
        """
        # Setup logging first
        setup_logging()
        self.logger = get_logger(__name__)
        
        # System state
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Configuration
        self.jetson_config = JetsonConfig()
        self.trading_params = TradingParams()
        
        # Core components (initialized later)
        self.jetson_monitor = None
        self.database_manager = None
        self.cache_manager = None
        self.data_client = None
        self.yahoo_client = None
        self.symbol_discovery = None
        self.data_pipeline = None
        self.feature_engine = None
        self.model_registry = None
        self.model_trainer = None
        self.model_predictor = None
        self.risk_manager = None
        self.position_sizer = None
        self.order_manager = None
        self.portfolio_tracker = None
        self.trading_engine = None
        self.system_monitor = None
        self.trading_monitor = None
        
        # Backtesting components
        self.zipline_engine = None
        self.performance_analyzer = None
        
        self.logger.info("JetsonTradingSystem initialized")
    
    async def initialize_components(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing Jetson Trading System components...")
            
            # Initialize hardware monitoring
            self.jetson_monitor = JetsonMonitor()
            await self.jetson_monitor.start_monitoring()
            
            # Initialize database
            self.database_manager = TradingDatabase()
            await self.database_manager.initialize()
            
            # Initialize cache manager
            self.cache_manager = CacheManager(
                max_memory_mb=self.jetson_config.CACHE_MEMORY_MB,
                max_disk_mb=self.jetson_config.CACHE_DISK_MB,
                default_ttl_seconds=self.jetson_config.CACHE_TTL_SECONDS
            )
            
            # Initialize data client
            polygon_api_key = self.jetson_config.get_env_var("POLYGON_API_KEY")
            if not polygon_api_key:
                raise ValueError("POLYGON_API_KEY environment variable not set")
            
            self.data_client = PolygonDataClient(
                api_key=polygon_api_key,
                max_requests_per_minute=self.jetson_config.POLYGON_RATE_LIMIT
            )
            
            # Initialize Yahoo Finance client
            self.yahoo_client = YahooFinanceClient()
            
            # Initialize symbol discovery engine
            self.symbol_discovery = SymbolDiscoveryEngine(
                polygon_client=self.data_client,
                yahoo_client=self.yahoo_client,
                database_manager=self.database_manager,
                cache_manager=self.cache_manager
            )
            
            # Initialize data pipeline
            self.data_pipeline = DataPipeline(
                data_client=self.data_client,
                database_manager=self.database_manager,
                cache_manager=self.cache_manager
            )
            
            # Initialize feature engine
            self.feature_engine = FeatureEngine(
                cache_manager=self.cache_manager,
                parallel_workers=self.jetson_config.MAX_PARALLEL_WORKERS
            )
            
            # Initialize model components
            self.model_registry = ModelRegistry(
                base_path="./models",
                max_models_cached=self.jetson_config.MAX_MODELS_CACHED
            )
            
            self.model_trainer = LightGBMTrainer(
                feature_engine=self.feature_engine,
                model_registry=self.model_registry,
                use_gpu=self.jetson_config.USE_GPU_ACCELERATION,
                max_memory_mb=self.jetson_config.MAX_MEMORY_USAGE_MB
            )
            
            self.model_predictor = ModelPredictor(
                model_registry=self.model_registry,
                feature_engine=self.feature_engine,
                cache_manager=self.cache_manager
            )
            
            # Initialize risk management
            self.risk_manager = RiskManager(
                max_portfolio_risk=self.trading_params.MAX_PORTFOLIO_RISK,
                max_position_size=self.trading_params.MAX_POSITION_SIZE_PCT,
                max_sector_exposure=self.trading_params.MAX_SECTOR_EXPOSURE,
                stop_loss_pct=self.trading_params.STOP_LOSS_PCT,
                database_manager=self.database_manager
            )
            
            self.position_sizer = PositionSizer(
                portfolio_value=self.trading_params.INITIAL_CAPITAL,
                risk_manager=self.risk_manager,
                kelly_lookback_days=self.trading_params.KELLY_LOOKBACK_DAYS
            )
            
            # Initialize execution components
            alpaca_api_key = self.jetson_config.get_env_var("ALPACA_API_KEY")
            alpaca_secret_key = self.jetson_config.get_env_var("ALPACA_SECRET_KEY")
            
            if not alpaca_api_key or not alpaca_secret_key:
                self.logger.warning("Alpaca API credentials not found - running in simulation mode")
                paper_trading = True
            else:
                paper_trading = self.trading_params.PAPER_TRADING
            
            self.order_manager = OrderManager(
                alpaca_api_key=alpaca_api_key,
                alpaca_secret_key=alpaca_secret_key,
                paper_trading=paper_trading,
                database_manager=self.database_manager
            )
            
            self.portfolio_tracker = PortfolioTracker(
                order_manager=self.order_manager,
                database_manager=self.database_manager,
                cache_manager=self.cache_manager
            )
            
            # Initialize trading engine with dynamic symbol discovery
            self.trading_engine = TradingEngine(
                data_pipeline=self.data_pipeline,
                feature_engine=self.feature_engine,
                model_predictor=self.model_predictor,
                risk_manager=self.risk_manager,
                position_sizer=self.position_sizer,
                order_manager=self.order_manager,
                portfolio_tracker=self.portfolio_tracker,
                symbol_discovery=self.symbol_discovery,
                target_symbols=self.trading_params.TARGET_SYMBOLS  # Fallback for static symbols
            )
            
            # Initialize monitoring
            self.system_monitor = SystemMonitor(
                jetson_monitor=self.jetson_monitor,
                database_manager=self.database_manager
            )
            
            self.trading_monitor = TradingMonitor(
                portfolio_tracker=self.portfolio_tracker,
                risk_manager=self.risk_manager,
                database_manager=self.database_manager
            )
            
            # Initialize backtesting components
            self.zipline_engine = ZiplineEngine(
                data_pipeline=self.data_pipeline,
                feature_engine=self.feature_engine,
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now(),
                initial_capital=self.trading_params.INITIAL_CAPITAL
            )
            
            self.performance_analyzer = PerformanceAnalyzer()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def run_backtest(self, start_date: datetime, end_date: datetime, symbols: list = None):
        """
        Run a comprehensive backtest
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: List of symbols to backtest (uses default if None)
        """
        try:
            self.logger.info(f"Starting backtest from {start_date} to {end_date}")
            
            # Use default symbols if none provided
            test_symbols = symbols or self.trading_params.TARGET_SYMBOLS
            
            # Configure zipline engine
            self.zipline_engine.start_date = start_date
            self.zipline_engine.end_date = end_date
            
            # Run backtest
            results = await self.zipline_engine.run_backtest(
                symbols=test_symbols,
                model_trainer=self.model_trainer,
                risk_manager=self.risk_manager,
                position_sizer=self.position_sizer
            )
            
            # Analyze performance
            performance_metrics = self.performance_analyzer.analyze_backtest_results(results)
            
            self.logger.info("Backtest completed successfully")
            return {
                'backtest_results': results,
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    async def train_models(self, symbols: list = None, retrain_days: int = 30):
        """
        Train or retrain models for specified symbols
        
        Args:
            symbols: List of symbols to train models for
            retrain_days: Number of days of data to use for training
        """
        try:
            self.logger.info("Starting model training...")
            
            # Use default symbols if none provided
            train_symbols = symbols or self.trading_params.TARGET_SYMBOLS
            
            training_results = {}
            
            for symbol in train_symbols:
                try:
                    self.logger.info(f"Training model for {symbol}")
                    
                    # Get training data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=retrain_days)
                    
                    training_data = await self.data_pipeline.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if training_data is None or training_data.empty:
                        self.logger.warning(f"No training data available for {symbol}")
                        continue
                    
                    # Train model
                    model_performance = await self.model_trainer.train_model(
                        symbol=symbol,
                        training_data=training_data
                    )
                    
                    training_results[symbol] = model_performance
                    self.logger.info(f"Model training completed for {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Error training model for {symbol}: {e}")
                    training_results[symbol] = {'error': str(e)}
            
            self.logger.info("Model training session completed")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    async def test_symbol_discovery(self):
        """
        Test the enhanced symbol discovery system with both Polygon and Yahoo Finance
        """
        try:
            self.logger.info("Testing symbol discovery system...")
            
            # Discover symbols from all configured sources
            discovered_symbols = await self.symbol_discovery.discover_symbols()
            
            # Group symbols by source for reporting
            sources_summary = {}
            for symbol_candidate in discovered_symbols:
                source = symbol_candidate.source.value
                if source not in sources_summary:
                    sources_summary[source] = []
                sources_summary[source].append({
                    'symbol': symbol_candidate.symbol,
                    'score': symbol_candidate.score,
                    'volume': symbol_candidate.volume,
                    'price_change_pct': symbol_candidate.price_change_pct
                })
            
            # Sort each source by score
            for source in sources_summary:
                sources_summary[source].sort(key=lambda x: x['score'], reverse=True)
            
            self.logger.info("Symbol discovery test completed successfully")
            return {
                'total_discovered': len(discovered_symbols),
                'sources_summary': sources_summary,
                'top_candidates': sorted(
                    [{'symbol': sc.symbol, 'score': sc.score, 'source': sc.source.value}
                     for sc in discovered_symbols],
                    key=lambda x: x['score'],
                    reverse=True
                )[:20]  # Top 20 overall
            }
            
        except Exception as e:
            self.logger.error(f"Error testing symbol discovery: {e}")
            raise
    
    async def start_live_trading(self):
        """Start live trading operations"""
        try:
            self.logger.info("Starting live trading...")
            
            # Verify system is ready
            await self._verify_system_ready()
            
            # Start monitoring
            await self.system_monitor.start_monitoring()
            await self.trading_monitor.start_monitoring()
            
            # Start trading engine
            await self.trading_engine.start()
            
            self.is_running = True
            self.logger.info("Live trading started successfully")
            
            # Main trading loop
            while not self.shutdown_event.is_set():
                try:
                    # Check system health
                    system_status = await self.system_monitor.check_system_health()
                    if not system_status['healthy']:
                        self.logger.warning(f"System health issues detected: {system_status}")
                        if system_status['critical']:
                            self.logger.error("Critical system issues - stopping trading")
                            break
                    
                    # Run trading cycle
                    await self.trading_engine.run_trading_cycle()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.trading_params.TRADING_CYCLE_SECONDS)
                    
                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
            
        except Exception as e:
            self.logger.error(f"Error in live trading: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _verify_system_ready(self):
        """Verify all systems are ready for trading"""
        try:
            # Check data connectivity
            await self.data_client.test_connection()
            
            # Check model availability
            available_models = self.model_registry.list_available_models()
            if not available_models:
                self.logger.warning("No trained models available - training default models")
                await self.train_models()
            
            # Check order management connectivity
            if not self.trading_params.PAPER_TRADING:
                await self.order_manager.test_connection()
            
            # Verify risk management
            risk_check = await self.risk_manager.system_health_check()
            if not risk_check['healthy']:
                raise RuntimeError(f"Risk management system not ready: {risk_check}")
            
            self.logger.info("System verification completed - ready for trading")
            
        except Exception as e:
            self.logger.error(f"System verification failed: {e}")
            raise
    
    async def stop(self):
        """Stop all trading operations and shutdown gracefully"""
        try:
            self.logger.info("Shutting down Jetson Trading System...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop trading engine
            if self.trading_engine:
                await self.trading_engine.stop()
            
            # Stop monitoring
            if self.system_monitor:
                await self.system_monitor.stop_monitoring()
            
            if self.trading_monitor:
                await self.trading_monitor.stop_monitoring()
            
            # Close data connections
            if self.data_client:
                await self.data_client.close()
            
            # Shutdown cache manager
            if self.cache_manager:
                self.cache_manager.shutdown()
            
            # Close database connections
            if self.database_manager:
                await self.database_manager.close()
            
            # Stop hardware monitoring
            if self.jetson_monitor:
                await self.jetson_monitor.stop_monitoring()
            
            self.is_running = False
            self.logger.info("Jetson Trading System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'running': self.is_running,
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }
            
            # Check each component
            components = [
                ('jetson_monitor', self.jetson_monitor),
                ('database_manager', self.database_manager),
                ('cache_manager', self.cache_manager),
                ('data_client', self.data_client),
                ('yahoo_client', self.yahoo_client),
                ('symbol_discovery', self.symbol_discovery),
                ('data_pipeline', self.data_pipeline),
                ('feature_engine', self.feature_engine),
                ('model_registry', self.model_registry),
                ('model_trainer', self.model_trainer),
                ('model_predictor', self.model_predictor),
                ('risk_manager', self.risk_manager),
                ('position_sizer', self.position_sizer),
                ('order_manager', self.order_manager),
                ('portfolio_tracker', self.portfolio_tracker),
                ('trading_engine', self.trading_engine),
                ('system_monitor', self.system_monitor),
                ('trading_monitor', self.trading_monitor)
            ]
            
            for name, component in components:
                status['components'][name] = {
                    'initialized': component is not None,
                    'type': type(component).__name__ if component else None
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Jetson Trading System")
    parser.add_argument('--mode', choices=['live', 'backtest', 'train', 'discovery'],
                       default='live', help='Operating mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade/backtest')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--retrain-days', type=int, default=30,
                       help='Days of data for model training')
    
    args = parser.parse_args()
    
    # Initialize system
    system = JetsonTradingSystem(config_path=args.config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize all components
        await system.initialize_components()
        
        if args.mode == 'live':
            print("Starting live trading mode...")
            await system.start_live_trading()
            
        elif args.mode == 'backtest':
            print("Starting backtest mode...")
            
            # Parse dates
            if args.start_date:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            else:
                start_date = datetime.now() - timedelta(days=365)
            
            if args.end_date:
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            else:
                end_date = datetime.now()
            
            results = await system.run_backtest(
                start_date=start_date,
                end_date=end_date,
                symbols=args.symbols
            )
            
            print("\nBacktest Results:")
            print(f"Performance Metrics: {results['performance_metrics']}")
            
        elif args.mode == 'train':
            print("Starting model training mode...")
            
            training_results = await system.train_models(
                symbols=args.symbols,
                retrain_days=args.retrain_days
            )
            
            print("\nTraining Results:")
            for symbol, result in training_results.items():
                print(f"{symbol}: {result}")
        
        elif args.mode == 'discovery':
            print("Starting symbol discovery test mode...")
            
            discovery_results = await system.test_symbol_discovery()
            
            print("\nSymbol Discovery Results:")
            print(f"Total symbols discovered: {discovery_results['total_discovered']}")
            
            print("\nTop 20 candidates overall:")
            for i, candidate in enumerate(discovery_results['top_candidates'][:20], 1):
                print(f"{i:2d}. {candidate['symbol']:6s} - Score: {candidate['score']:.3f} - Source: {candidate['source']}")
            
            print("\nBreakdown by source:")
            for source, symbols in discovery_results['sources_summary'].items():
                print(f"\n{source.upper()} ({len(symbols)} symbols):")
                for i, symbol_data in enumerate(symbols[:10], 1):  # Top 10 per source
                    print(f"  {i:2d}. {symbol_data['symbol']:6s} - Score: {symbol_data['score']:.3f} - "
                          f"Volume: {symbol_data['volume']:,} - Change: {symbol_data.get('price_change_pct', 0):.2f}%")
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await system.stop()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
