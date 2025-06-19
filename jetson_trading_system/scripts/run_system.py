#!/usr/bin/env python3
"""
Jetson Trading System - System Launcher Script
Main system launcher with health checks and monitoring
"""

import sys
import os
import asyncio
import argparse
import signal
import time
from pathlib import Path
from datetime import datetime
import subprocess
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.utils.logger import setup_logging, get_logger

class SystemLauncher:
    """
    System launcher with health checks and monitoring
    """
    
    def __init__(self):
        """Initialize system launcher"""
        setup_logging()
        self.logger = get_logger(__name__)
        self.config = JetsonConfig()
        self.processes = {}
        self.shutdown_requested = False
        
    def check_environment(self):
        """Check environment variables and configuration"""
        self.logger.info("Checking environment configuration...")
        
        required_env_vars = [
            'POLYGON_API_KEY',
            'ALPACA_API_KEY', 
            'ALPACA_SECRET_KEY'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            self.logger.error("Please set these in your .env file or environment")
            return False
        
        self.logger.info("✓ Environment variables configured")
        return True
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        self.logger.info("Checking dependencies...")
        
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'lightgbm', 'talib',
            'alpaca_trade_api', 'polygon', 'psutil', 'asyncio'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {missing_packages}")
            self.logger.error("Run: python3 jetson_trading_system/scripts/install_dependencies.py")
            return False
        
        self.logger.info("✓ All dependencies installed")
        return True
    
    def check_hardware(self):
        """Check hardware requirements"""
        self.logger.info("Checking hardware requirements...")
        
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.total < 8 * 1024 * 1024 * 1024:  # 8GB minimum
                self.logger.warning(f"Low memory: {memory.total / 1024**3:.1f}GB (8GB+ recommended)")
            else:
                self.logger.info(f"✓ Memory: {memory.total / 1024**3:.1f}GB")
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / 1024**3
            if free_gb < 10:
                self.logger.warning(f"Low disk space: {free_gb:.1f}GB free (10GB+ recommended)")
            else:
                self.logger.info(f"✓ Disk space: {free_gb:.1f}GB free")
            
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            self.logger.info(f"✓ CPU cores: {cpu_count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware check failed: {e}")
            return False
    
    def check_directories(self):
        """Create and check required directories"""
        self.logger.info("Setting up directories...")
        
        directories = [
            'data', 'models', 'cache', 'logs', 'backups'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✓ Directory: {directory}")
        
        return True
    
    def run_system_tests(self):
        """Run basic system tests"""
        self.logger.info("Running system tests...")
        
        try:
            # Test database connection
            from jetson_trading_system.utils.database import TradingDatabase
            db_manager = TradingDatabase()
            asyncio.run(db_manager.initialize())
            asyncio.run(db_manager.close())
            self.logger.info("✓ Database connection test passed")
            
            # Test data client
            from jetson_trading_system.data.polygon_client import PolygonDataClient
            api_key = os.getenv('POLYGON_API_KEY')
            data_client = PolygonDataClient(api_key=api_key)
            asyncio.run(data_client.test_connection())
            self.logger.info("✓ Data client test passed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System test failed: {e}")
            return False
    
    def launch_main_system(self, mode, **kwargs):
        """Launch the main trading system"""
        self.logger.info(f"Launching system in {mode} mode...")
        
        # Build command
        cmd = [sys.executable, "main.py", "--mode", mode]
        
        # Add optional arguments
        if kwargs.get('config'):
            cmd.extend(["--config", kwargs['config']])
        if kwargs.get('symbols'):
            cmd.extend(["--symbols"] + kwargs['symbols'])
        if kwargs.get('start_date'):
            cmd.extend(["--start-date", kwargs['start_date']])
        if kwargs.get('end_date'):
            cmd.extend(["--end-date", kwargs['end_date']])
        if kwargs.get('retrain_days'):
            cmd.extend(["--retrain-days", str(kwargs['retrain_days'])])
        
        try:
            # Launch process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes['main'] = process
            self.logger.info(f"System launched with PID: {process.pid}")
            
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to launch system: {e}")
            return None
    
    def monitor_system(self, process):
        """Monitor system process"""
        self.logger.info("Monitoring system...")
        
        try:
            while not self.shutdown_requested:
                # Check if process is still running
                if process.poll() is not None:
                    self.logger.error("System process has terminated unexpectedly")
                    return False
                
                # Read output
                if process.stdout.readable():
                    line = process.stdout.readline()
                    if line:
                        print(line.rstrip())
                
                # Check system health
                self._check_system_health()
                
                time.sleep(1)
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
            self.shutdown_requested = True
            return True
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            return False
    
    def _check_system_health(self):
        """Check system health metrics"""
        try:
            import psutil
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent:.1f}%")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > 95:
                self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                self.logger.warning(f"High disk usage: {disk.percent:.1f}%")
                
        except Exception as e:
            self.logger.debug(f"Health check error: {e}")
    
    def shutdown_system(self):
        """Gracefully shutdown system"""
        self.logger.info("Shutting down system...")
        self.shutdown_requested = True
        
        for name, process in self.processes.items():
            try:
                self.logger.info(f"Terminating {name} process...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {name} process...")
                    process.kill()
                    process.wait()
                
                self.logger.info(f"✓ {name} process stopped")
                
            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")
        
        self.logger.info("System shutdown complete")
    
    def run_diagnostic(self):
        """Run comprehensive system diagnostic"""
        self.logger.info("Running system diagnostic...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.check_environment(),
            'dependencies': self.check_dependencies(),
            'hardware': self.check_hardware(),
            'directories': self.check_directories(),
            'system_tests': self.run_system_tests()
        }
        
        # Save diagnostic results
        diagnostic_file = f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(diagnostic_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("SYSTEM DIAGNOSTIC RESULTS")
        print("="*50)
        
        for check, result in results.items():
            if check == 'timestamp':
                continue
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{check.replace('_', ' ').title()}: {status}")
        
        all_passed = all(v for k, v in results.items() if k != 'timestamp')
        
        if all_passed:
            print("\n✓ All diagnostic checks passed - system ready!")
            return True
        else:
            print("\n✗ Some diagnostic checks failed - please fix issues before proceeding")
            return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Jetson Trading System Launcher")
    parser.add_argument('--mode', choices=['live', 'backtest', 'train', 'diagnostic'], 
                       default='diagnostic', help='Operating mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade/backtest')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--retrain-days', type=int, default=30, 
                       help='Days of data for model training')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip initial system checks')
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = SystemLauncher()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}")
        launcher.shutdown_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("Jetson Trading System Launcher")
        print("=" * 40)
        
        # Run diagnostic mode
        if args.mode == 'diagnostic':
            success = launcher.run_diagnostic()
            sys.exit(0 if success else 1)
        
        # Run system checks unless skipped
        if not args.skip_checks:
            print("Running pre-launch checks...")
            
            checks = [
                launcher.check_environment(),
                launcher.check_dependencies(),
                launcher.check_hardware(),
                launcher.check_directories()
            ]
            
            if not all(checks):
                print("Pre-launch checks failed. Run with --mode diagnostic for details.")
                sys.exit(1)
            
            print("✓ All pre-launch checks passed")
        
        # Launch system
        process = launcher.launch_main_system(
            mode=args.mode,
            config=args.config,
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            retrain_days=args.retrain_days
        )
        
        if not process:
            print("Failed to launch system")
            sys.exit(1)
        
        # Monitor system
        success = launcher.monitor_system(process)
        
        if not success:
            print("System monitoring detected issues")
            sys.exit(1)
        
    except Exception as e:
        print(f"Launcher error: {e}")
        sys.exit(1)
    finally:
        launcher.shutdown_system()

if __name__ == "__main__":
    main()
