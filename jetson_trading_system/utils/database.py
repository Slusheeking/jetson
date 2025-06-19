"""
Database Utilities
SQLite database management optimized for Jetson Orin
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from contextlib import contextmanager
import threading

from jetson_trading_system.config.trading_params import DataConfig
from jetson_trading_system.config.jetson_settings import JetsonConfig

logger = logging.getLogger(__name__)

class TradingDatabase:
    """
    SQLite database manager for trading system
    Optimized for Jetson Orin with efficient ARM64 operations
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DataConfig.DATABASE_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for connections
        self.local = threading.local()
        
        # Initialize database schema
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            # Optimize for performance
            self.local.connection.execute("PRAGMA journal_mode=WAL")
            self.local.connection.execute("PRAGMA synchronous=NORMAL")
            self.local.connection.execute("PRAGMA cache_size=10000")
            self.local.connection.execute("PRAGMA temp_store=MEMORY")
        return self.local.connection
    
    @contextmanager
    def get_cursor(self):
        """Get database cursor with automatic cleanup"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def _init_database(self):
        """Initialize database schema"""
        schema_sql = """
        -- Market data tables
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            vwap REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
        ON market_data(symbol, timestamp);
        
        -- Technical indicators
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            indicator_name TEXT NOT NULL,
            value REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp, indicator_name)
        );
        
        CREATE INDEX IF NOT EXISTS idx_indicators_symbol_time 
        ON technical_indicators(symbol, timestamp);
        
        -- Model predictions
        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            model_name TEXT NOT NULL,
            prediction REAL,
            confidence REAL,
            target_date DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time 
        ON model_predictions(symbol, timestamp);
        
        -- Trading signals
        CREATE TABLE IF NOT EXISTS trading_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            signal_type TEXT NOT NULL,
            signal_strength REAL,
            price REAL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_signals_symbol_time 
        ON trading_signals(symbol, timestamp);
        
        -- Portfolio positions
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            entry_date DATETIME NOT NULL,
            exit_price REAL,
            exit_date DATETIME,
            pnl REAL,
            status TEXT DEFAULT 'open',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_positions_symbol_status 
        ON positions(symbol, status);
        
        -- Trading orders
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            order_type TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            status TEXT DEFAULT 'pending',
            order_id TEXT,
            filled_quantity INTEGER DEFAULT 0,
            filled_price REAL,
            submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            filled_at DATETIME
        );
        
        CREATE INDEX IF NOT EXISTS idx_orders_symbol_status 
        ON orders(symbol, status);
        
        -- Performance metrics
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            total_pnl REAL,
            daily_pnl REAL,
            portfolio_value REAL,
            cash REAL,
            num_positions INTEGER,
            win_rate REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date)
        );
        
        -- System logs
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            level TEXT NOT NULL,
            component TEXT NOT NULL,
            message TEXT NOT NULL,
            metadata TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
        ON system_logs(timestamp);
        """
        
        with self.get_cursor() as cursor:
            cursor.executescript(schema_sql)
    
    def insert_market_data(self, df: pd.DataFrame) -> int:
        """
        Insert market data into database
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0
        
        # Prepare data
        df = df.copy()
        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
            df = df.reset_index()
        
        # Convert to records
        records = df.to_dict('records')
        
        insert_sql = """
        INSERT OR REPLACE INTO market_data
        (symbol, timestamp, open, high, low, close, volume, vwap)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self.get_cursor() as cursor:
                data = []
                for record in records:
                    # Convert timestamp to string if it's not already
                    timestamp = record.get('timestamp')
                    if hasattr(timestamp, 'strftime'):
                        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(timestamp, str):
                        timestamp = timestamp
                    else:
                        timestamp = str(timestamp)
                    
                    # Ensure numeric values are properly converted
                    try:
                        data.append((
                            str(record.get('symbol', '')),
                            timestamp,
                            float(record.get('open', 0.0)) if record.get('open') is not None else 0.0,
                            float(record.get('high', 0.0)) if record.get('high') is not None else 0.0,
                            float(record.get('low', 0.0)) if record.get('low') is not None else 0.0,
                            float(record.get('close', 0.0)) if record.get('close') is not None else 0.0,
                            int(record.get('volume', 0)) if record.get('volume') is not None else 0,
                            float(record.get('vwap', 0.0)) if record.get('vwap') is not None else None
                        ))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping invalid record: {e}")
                        continue
                
                if data:
                    cursor.executemany(insert_sql, data)
                    return cursor.rowcount
                else:
                    return 0
                    
        except Exception as e:
            logger.error(f"Error inserting market data: {e}")
            return 0
    
    def get_market_data(self, 
                       symbol: str = None, 
                       start_date: str = None, 
                       end_date: str = None,
                       limit: int = None) -> pd.DataFrame:
        """
        Retrieve market data from database
        
        Args:
            symbol: Symbol to filter (optional)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of rows
            
        Returns:
            DataFrame with market data
        """
        sql = "SELECT * FROM market_data WHERE 1=1"
        params = []
        
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            sql += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            sql += " AND timestamp <= ?"
            params.append(end_date)
        
        sql += " ORDER BY symbol, timestamp"
        
        if limit:
            sql += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql_query(sql, self._get_connection(), params=params)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Set timestamp as index for compatibility with ZiplineEngine
                df = df.set_index('timestamp')
            return df
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return pd.DataFrame()
    
    def insert_technical_indicators(self, symbol: str, timestamp: str, indicators: Dict[str, float]) -> int:
        """
        Insert technical indicators
        
        Args:
            symbol: Stock symbol
            timestamp: Timestamp
            indicators: Dictionary of indicator name -> value
            
        Returns:
            Number of rows inserted
        """
        insert_sql = """
        INSERT OR REPLACE INTO technical_indicators 
        (symbol, timestamp, indicator_name, value)
        VALUES (?, ?, ?, ?)
        """
        
        with self.get_cursor() as cursor:
            data = [
                (symbol, timestamp, name, value)
                for name, value in indicators.items()
                if not pd.isna(value)
            ]
            cursor.executemany(insert_sql, data)
            return cursor.rowcount
    
    def get_technical_indicators(self,
                               symbol: str = None,
                               start_date: str = None,
                               end_date: str = None,
                               indicator_names: List[str] = None,
                               pivot: bool = True) -> pd.DataFrame:
        """
        Retrieve technical indicators
        
        Args:
            symbol: Symbol to filter
            indicator_names: List of indicator names to retrieve
            start_date: Start date
            end_date: End date
            pivot: If True, pivot data to have indicators as columns (default for ML)
            
        Returns:
            DataFrame with indicators
        """
        sql = "SELECT * FROM technical_indicators WHERE 1=1"
        params = []
        
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        
        if indicator_names:
            placeholders = ','.join(['?' for _ in indicator_names])
            sql += f" AND indicator_name IN ({placeholders})"
            params.extend(indicator_names)
        
        if start_date:
            sql += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            sql += " AND timestamp <= ?"
            params.append(end_date)
        
        sql += " ORDER BY symbol, timestamp, indicator_name"
        
        try:
            df = pd.read_sql_query(sql, self._get_connection(), params=params)
            if df.empty:
                return df
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if pivot and not df.empty:
                # Pivot the data to have indicators as columns
                # This is what ML models expect: timestamp as index, indicators as columns
                pivot_df = df.pivot_table(
                    index='timestamp',
                    columns='indicator_name',
                    values='value',
                    aggfunc='first'  # In case of duplicates, take first
                )
                
                # Reset column names (remove the 'indicator_name' level)
                pivot_df.columns.name = None
                
                # If we filtered by symbol, add it back as a column
                if symbol:
                    pivot_df['symbol'] = symbol
                
                return pivot_df
            else:
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving indicators: {e}")
            return pd.DataFrame()
    
    def insert_model_prediction(self, 
                              symbol: str, 
                              timestamp: str, 
                              model_name: str,
                              prediction: float, 
                              confidence: float = None,
                              target_date: str = None):
        """Insert model prediction"""
        insert_sql = """
        INSERT INTO model_predictions 
        (symbol, timestamp, model_name, prediction, confidence, target_date)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(insert_sql, (symbol, timestamp, model_name, prediction, confidence, target_date))
    
    def insert_trading_signal(self, 
                            symbol: str, 
                            timestamp: str, 
                            signal_type: str,
                            signal_strength: float, 
                            price: float = None,
                            metadata: Dict = None):
        """Insert trading signal"""
        insert_sql = """
        INSERT INTO trading_signals 
        (symbol, timestamp, signal_type, signal_strength, price, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self.get_cursor() as cursor:
            cursor.execute(insert_sql, (symbol, timestamp, signal_type, signal_strength, price, metadata_json))
    
    def insert_position(self, 
                       symbol: str, 
                       quantity: int, 
                       entry_price: float,
                       entry_date: str = None) -> int:
        """
        Insert new position
        
        Returns:
            Position ID
        """
        if entry_date is None:
            entry_date = datetime.now().isoformat()
        
        insert_sql = """
        INSERT INTO positions (symbol, quantity, entry_price, entry_date)
        VALUES (?, ?, ?, ?)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(insert_sql, (symbol, quantity, entry_price, entry_date))
            return cursor.lastrowid
    
    def update_position_exit(self, position_id: int, exit_price: float, exit_date: str = None):
        """Update position with exit information"""
        if exit_date is None:
            exit_date = datetime.now().isoformat()
        
        update_sql = """
        UPDATE positions 
        SET exit_price = ?, exit_date = ?, 
            pnl = (exit_price - entry_price) * quantity,
            status = 'closed'
        WHERE id = ?
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(update_sql, (exit_price, exit_date, position_id))
    
    def get_open_positions(self) -> pd.DataFrame:
        """Get all open positions"""
        sql = "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_date"
        
        try:
            return pd.read_sql_query(sql, self._get_connection())
        except Exception as e:
            logger.error(f"Error retrieving positions: {e}")
            return pd.DataFrame()
    
    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for last N days"""
        cutoff_date = (datetime.now() - timedelta(days=days)).date()
        
        sql = """
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            AVG(pnl) as avg_pnl,
            SUM(pnl) as total_pnl,
            MAX(pnl) as max_win,
            MIN(pnl) as max_loss
        FROM positions 
        WHERE status = 'closed' AND exit_date >= ?
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(sql, (cutoff_date,))
            result = cursor.fetchone()
            
            if result and result[0] > 0:
                total_trades, winning_trades, avg_pnl, total_pnl, max_win, max_loss = result
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                return {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'total_pnl': total_pnl,
                    'max_win': max_win,
                    'max_loss': max_loss,
                    'period_days': days
                }
        
        return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to manage database size"""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).date()
        
        cleanup_sql = [
            "DELETE FROM system_logs WHERE timestamp < ?",
            "DELETE FROM model_predictions WHERE timestamp < ?",
            "DELETE FROM trading_signals WHERE timestamp < ?"
        ]
        
        with self.get_cursor() as cursor:
            for sql in cleanup_sql:
                cursor.execute(sql, (cutoff_date,))
                logger.info(f"Cleaned up {cursor.rowcount} old records")
    
    def vacuum_database(self):
        """Optimize database by running VACUUM"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("VACUUM")
            logger.info("Database vacuum completed")
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
    
    def backup_database(self, backup_path: str = None):
        """Create database backup"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.db_path}.backup_{timestamp}"
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return None
    
    def close(self):
        """Close database connections"""
        if hasattr(self.local, 'connection'):
            self.local.connection.close()

# Global database instance
trading_db = TradingDatabase()


if __name__ == '__main__':
    print("--- Running TradingDatabase Demo ---")
    
    # Use an in-memory database for this demonstration
    db = TradingDatabase(db_path=":memory:")
    
    # 1. Insert Market Data
    print("\n1. Inserting sample market data for AAPL...")
    market_data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-02', '2023-01-03']),
        'symbol': ['AAPL', 'AAPL'],
        'open': [130.28, 130.89],
        'high': [130.90, 131.83],
        'low': [124.17, 129.89],
        'close': [125.07, 130.73],
        'volume': [112117500, 97924700],
        'vwap': [126.41, 130.81]
    })
    rows_inserted = db.insert_market_data(market_data)
    print(f"Inserted {rows_inserted} rows of market data.")
    
    # 2. Retrieve Market Data
    print("\n2. Retrieving market data for AAPL...")
    retrieved_data = db.get_market_data(symbol='AAPL')
    print(retrieved_data)
    
    # 3. Insert Technical Indicators
    print("\n3. Inserting sample technical indicators...")
    indicators = {'RSI': 35.6, 'SMA_20': 132.5}
    rows_inserted = db.insert_technical_indicators('AAPL', '2023-01-03', indicators)
    print(f"Inserted {rows_inserted} indicator rows.")

    # 4. Retrieve Technical Indicators
    print("\n4. Retrieving technical indicators for AAPL...")
    retrieved_indicators = db.get_technical_indicators(symbol='AAPL')
    print(retrieved_indicators)

    # 5. Insert and Update a Position
    print("\n5. Managing a sample position...")
    pos_id = db.insert_position('GOOGL', 10, 91.5)
    print(f"Opened position with ID: {pos_id}")
    db.update_position_exit(pos_id, 95.2)
    print("Closed position.")
    
    # 6. Get Open Positions (should be none)
    print("\n6. Checking for open positions...")
    open_positions = db.get_open_positions()
    print(f"Number of open positions: {len(open_positions)}")

    # 7. Get Performance Summary
    print("\n7. Getting performance summary...")
    summary = db.get_performance_summary()
    print(summary)

    db.close()
    print("\n--- Demo Finished ---")
