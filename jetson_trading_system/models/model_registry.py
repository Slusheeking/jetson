"""
Model Registry for Jetson Trading System
Centralized model management and versioning
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import shutil
import hashlib
import joblib

from ..config.jetson_settings import JetsonConfig
from ..config.trading_params import TradingConfig
from ..utils.logger import get_model_logger

class ModelRegistry:
    """
    Centralized registry for managing trained models
    Handles versioning, metadata, and model lifecycle
    """
    
    def __init__(self, registry_dir: str = "./models", db_path: str = "./models/registry.db"):
        """
        Initialize model registry
        
        Args:
            registry_dir: Directory for storing models
            db_path: SQLite database path for registry
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_model_logger()
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"ModelRegistry initialized at {registry_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for model registry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        metadata_path TEXT,
                        file_hash TEXT,
                        file_size INTEGER,
                        created_date TEXT NOT NULL,
                        training_start_date TEXT,
                        training_end_date TEXT,
                        cv_accuracy REAL,
                        cv_auc REAL,
                        training_samples INTEGER,
                        feature_count INTEGER,
                        is_active BOOLEAN DEFAULT 1,
                        tags TEXT,
                        notes TEXT,
                        UNIQUE(symbol, model_name, version)
                    )
                """)
                
                # Performance tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id INTEGER,
                        date TEXT NOT NULL,
                        predictions_count INTEGER DEFAULT 0,
                        avg_prediction_time REAL,
                        accuracy REAL,
                        precision_score REAL,
                        recall REAL,
                        f1_score REAL,
                        auc_score REAL,
                        total_trades INTEGER DEFAULT 0,
                        profitable_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0.0,
                        FOREIGN KEY (model_id) REFERENCES models (id)
                    )
                """)
                
                # Model deployments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS deployments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id INTEGER,
                        deployed_date TEXT NOT NULL,
                        undeployed_date TEXT,
                        deployment_status TEXT DEFAULT 'active',
                        deployment_notes TEXT,
                        FOREIGN KEY (model_id) REFERENCES models (id)
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def register_model(self, 
                      symbol: str,
                      model_name: str,
                      model_file_path: str,
                      metadata: Dict[str, Any],
                      version: str = None,
                      tags: List[str] = None,
                      notes: str = "") -> int:
        """
        Register a new trained model
        
        Args:
            symbol: Stock symbol
            model_name: Name of the model
            model_file_path: Path to the saved model file
            metadata: Model training metadata
            version: Model version (auto-generated if None)
            tags: List of tags for the model
            notes: Additional notes
            
        Returns:
            Model ID in registry
        """
        try:
            model_path = Path(model_file_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_file_path}")
            
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Calculate file hash and size
            file_hash = self._calculate_file_hash(model_path)
            file_size = model_path.stat().st_size
            
            # Create metadata file path
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            
            # Save metadata to file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Extract training information from metadata
            cv_results = metadata.get('cv_results', {})
            cv_accuracy = cv_results.get('mean_accuracy')
            cv_auc = cv_results.get('mean_auc')
            training_samples = metadata.get('training_samples')
            feature_count = len(metadata.get('features', []))
            
            # Get training date range
            training_start_date = metadata.get('training_start_date')
            training_end_date = metadata.get('training_end_date')
            
            # Convert tags to JSON string
            tags_json = json.dumps(tags) if tags else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Deactivate previous models for this symbol/model_name
                cursor.execute("""
                    UPDATE models 
                    SET is_active = 0 
                    WHERE symbol = ? AND model_name = ?
                """, (symbol, model_name))
                
                # Insert new model
                cursor.execute("""
                    INSERT INTO models (
                        symbol, model_name, version, file_path, metadata_path,
                        file_hash, file_size, created_date, training_start_date,
                        training_end_date, cv_accuracy, cv_auc, training_samples,
                        feature_count, tags, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, model_name, version, str(model_path), str(metadata_path),
                    file_hash, file_size, datetime.now().isoformat(),
                    training_start_date, training_end_date, cv_accuracy, cv_auc,
                    training_samples, feature_count, tags_json, notes
                ))
                
                model_id = cursor.lastrowid
                conn.commit()
            
            self.logger.info(f"Registered model {model_name} v{version} for {symbol} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            raise
    
    def get_active_model(self, symbol: str, model_name: str = "jetson_trading_v1") -> Optional[Dict[str, Any]]:
        """Get the active model for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM models 
                    WHERE symbol = ? AND model_name = ? AND is_active = 1
                    ORDER BY created_date DESC
                    LIMIT 1
                """, (symbol, model_name))
                
                row = cursor.fetchone()
                if row:
                    model_info = dict(row)
                    
                    # Parse tags if they exist
                    if model_info['tags']:
                        model_info['tags'] = json.loads(model_info['tags'])
                    
                    return model_info
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get active model for {symbol}: {e}")
            return None
    
    def get_model_versions(self, symbol: str, model_name: str = "jetson_trading_v1") -> List[Dict[str, Any]]:
        """Get all versions of a model for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM models 
                    WHERE symbol = ? AND model_name = ?
                    ORDER BY created_date DESC
                """, (symbol, model_name))
                
                rows = cursor.fetchall()
                models = []
                
                for row in rows:
                    model_info = dict(row)
                    if model_info['tags']:
                        model_info['tags'] = json.loads(model_info['tags'])
                    models.append(model_info)
                
                return models
                
        except Exception as e:
            self.logger.error(f"Failed to get model versions for {symbol}: {e}")
            return []
    
    def activate_model(self, model_id: int) -> bool:
        """Activate a specific model version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get model info
                cursor.execute("SELECT symbol, model_name FROM models WHERE id = ?", (model_id,))
                row = cursor.fetchone()
                
                if not row:
                    self.logger.error(f"Model ID {model_id} not found")
                    return False
                
                symbol, model_name = row
                
                # Deactivate all models for this symbol/model_name
                cursor.execute("""
                    UPDATE models 
                    SET is_active = 0 
                    WHERE symbol = ? AND model_name = ?
                """, (symbol, model_name))
                
                # Activate the specified model
                cursor.execute("""
                    UPDATE models 
                    SET is_active = 1 
                    WHERE id = ?
                """, (model_id,))
                
                conn.commit()
                
                self.logger.info(f"Activated model ID {model_id} for {symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to activate model {model_id}: {e}")
            return False
    
    def deactivate_model(self, model_id: int) -> bool:
        """Deactivate a specific model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE models 
                    SET is_active = 0 
                    WHERE id = ?
                """, (model_id,))
                
                conn.commit()
                
                self.logger.info(f"Deactivated model ID {model_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to deactivate model {model_id}: {e}")
            return False
    
    def delete_model(self, model_id: int, delete_files: bool = False) -> bool:
        """Delete a model from registry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get model file paths if we need to delete files
                if delete_files:
                    cursor.execute("""
                        SELECT file_path, metadata_path FROM models WHERE id = ?
                    """, (model_id,))
                    row = cursor.fetchone()
                    
                    if row:
                        file_path, metadata_path = row
                        
                        # Delete model file
                        if file_path and Path(file_path).exists():
                            Path(file_path).unlink()
                        
                        # Delete metadata file
                        if metadata_path and Path(metadata_path).exists():
                            Path(metadata_path).unlink()
                
                # Delete from database
                cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
                cursor.execute("DELETE FROM model_performance WHERE model_id = ?", (model_id,))
                cursor.execute("DELETE FROM deployments WHERE model_id = ?", (model_id,))
                
                conn.commit()
                
                self.logger.info(f"Deleted model ID {model_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def record_performance(self, 
                          model_id: int,
                          date: str,
                          performance_metrics: Dict[str, Any]) -> bool:
        """Record performance metrics for a model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if record exists for this date
                cursor.execute("""
                    SELECT id FROM model_performance 
                    WHERE model_id = ? AND date = ?
                """, (model_id, date))
                
                existing_record = cursor.fetchone()
                
                if existing_record:
                    # Update existing record
                    cursor.execute("""
                        UPDATE model_performance SET
                            predictions_count = ?,
                            avg_prediction_time = ?,
                            accuracy = ?,
                            precision_score = ?,
                            recall = ?,
                            f1_score = ?,
                            auc_score = ?,
                            total_trades = ?,
                            profitable_trades = ?,
                            total_pnl = ?
                        WHERE model_id = ? AND date = ?
                    """, (
                        performance_metrics.get('predictions_count', 0),
                        performance_metrics.get('avg_prediction_time'),
                        performance_metrics.get('accuracy'),
                        performance_metrics.get('precision'),
                        performance_metrics.get('recall'),
                        performance_metrics.get('f1_score'),
                        performance_metrics.get('auc_score'),
                        performance_metrics.get('total_trades', 0),
                        performance_metrics.get('profitable_trades', 0),
                        performance_metrics.get('total_pnl', 0.0),
                        model_id, date
                    ))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO model_performance (
                            model_id, date, predictions_count, avg_prediction_time,
                            accuracy, precision_score, recall, f1_score, auc_score,
                            total_trades, profitable_trades, total_pnl
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        model_id, date,
                        performance_metrics.get('predictions_count', 0),
                        performance_metrics.get('avg_prediction_time'),
                        performance_metrics.get('accuracy'),
                        performance_metrics.get('precision'),
                        performance_metrics.get('recall'),
                        performance_metrics.get('f1_score'),
                        performance_metrics.get('auc_score'),
                        performance_metrics.get('total_trades', 0),
                        performance_metrics.get('profitable_trades', 0),
                        performance_metrics.get('total_pnl', 0.0)
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to record performance for model {model_id}: {e}")
            return False
    
    def get_model_performance(self, 
                            model_id: int,
                            start_date: str = None,
                            end_date: str = None) -> List[Dict[str, Any]]:
        """Get performance history for a model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM model_performance WHERE model_id = ?"
                params = [model_id]
                
                if start_date:
                    query += " AND date >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND date <= ?"
                    params.append(end_date)
                
                query += " ORDER BY date DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to get performance for model {model_id}: {e}")
            return []
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of model registry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total models
                cursor.execute("SELECT COUNT(*) FROM models")
                total_models = cursor.fetchone()[0]
                
                # Active models
                cursor.execute("SELECT COUNT(*) FROM models WHERE is_active = 1")
                active_models = cursor.fetchone()[0]
                
                # Models by symbol
                cursor.execute("""
                    SELECT symbol, COUNT(*) as count 
                    FROM models 
                    GROUP BY symbol 
                    ORDER BY count DESC
                """)
                models_by_symbol = dict(cursor.fetchall())
                
                # Latest models
                cursor.execute("""
                    SELECT symbol, model_name, version, created_date, cv_accuracy, cv_auc
                    FROM models 
                    WHERE is_active = 1
                    ORDER BY created_date DESC
                    LIMIT 10
                """)
                latest_models = cursor.fetchall()
                
                return {
                    'total_models': total_models,
                    'active_models': active_models,
                    'models_by_symbol': models_by_symbol,
                    'latest_models': latest_models,
                    'registry_path': str(self.registry_dir),
                    'database_path': str(self.db_path)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get registry summary: {e}")
            return {}
    
    def cleanup_old_models(self, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all symbol/model_name combinations
                cursor.execute("""
                    SELECT DISTINCT symbol, model_name FROM models
                """)
                combinations = cursor.fetchall()
                
                deleted_count = 0
                
                for symbol, model_name in combinations:
                    # Get models for this combination, ordered by creation date
                    cursor.execute("""
                        SELECT id, file_path, metadata_path FROM models
                        WHERE symbol = ? AND model_name = ?
                        ORDER BY created_date DESC
                    """, (symbol, model_name))
                    
                    models = cursor.fetchall()
                    
                    # Delete models beyond the keep_versions limit
                    if len(models) > keep_versions:
                        for model_id, file_path, metadata_path in models[keep_versions:]:
                            # Delete files
                            if file_path and Path(file_path).exists():
                                Path(file_path).unlink()
                            if metadata_path and Path(metadata_path).exists():
                                Path(metadata_path).unlink()
                            
                            # Delete from database
                            cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
                            cursor.execute("DELETE FROM model_performance WHERE model_id = ?", (model_id,))
                            cursor.execute("DELETE FROM deployments WHERE model_id = ?", (model_id,))
                            
                            deleted_count += 1
                
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old models")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old models: {e}")
            return 0
    
    def export_registry(self, export_path: str) -> bool:
        """Export registry database to a file"""
        try:
            export_path = Path(export_path)
            shutil.copy2(self.db_path, export_path)
            
            self.logger.info(f"Registry exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    def validate_model_integrity(self, model_id: int) -> Dict[str, Any]:
        """Validate model file integrity"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT file_path, metadata_path, file_hash, file_size
                    FROM models WHERE id = ?
                """, (model_id,))
                
                row = cursor.fetchone()
                if not row:
                    return {'valid': False, 'error': 'Model not found'}
                
                model_info = dict(row)
                file_path = Path(model_info['file_path'])
                metadata_path = Path(model_info['metadata_path']) if model_info['metadata_path'] else None
                
                # Check if files exist
                if not file_path.exists():
                    return {'valid': False, 'error': 'Model file not found'}
                
                if metadata_path and not metadata_path.exists():
                    return {'valid': False, 'error': 'Metadata file not found'}
                
                # Verify file hash
                current_hash = self._calculate_file_hash(file_path)
                if current_hash != model_info['file_hash']:
                    return {'valid': False, 'error': 'File hash mismatch'}
                
                # Verify file size
                current_size = file_path.stat().st_size
                if current_size != model_info['file_size']:
                    return {'valid': False, 'error': 'File size mismatch'}
                
                # Try to load the model
                try:
                    model = joblib.load(file_path)
                    model_loadable = True
                except Exception as e:
                    model_loadable = False
                    load_error = str(e)
                
                return {
                    'valid': True,
                    'file_exists': True,
                    'metadata_exists': metadata_path.exists() if metadata_path else False,
                    'hash_valid': True,
                    'size_valid': True,
                    'model_loadable': model_loadable,
                    'load_error': load_error if not model_loadable else None
                }
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}

# Example usage
if __name__ == "__main__":
    import os
    import tempfile
    
    print("--- Running ModelRegistry Demo ---")
    
    # Use a temporary directory for the demo to avoid clutter
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_db_path = os.path.join(tmpdir, "registry.db")
        registry = ModelRegistry(registry_dir=tmpdir, db_path=registry_db_path)
        
        print(f"\nUsing temporary registry at: {tmpdir}")

        # --- 1. Register a new model ---
        print("\n--- 1. Registering a new model for AAPL ---")
        
        # Create a dummy model file and metadata
        dummy_model_path = os.path.join(tmpdir, "AAPL_model_v1.pkl")
        with open(dummy_model_path, "wb") as f:
            joblib.dump({"model": "dummy_lgbm"}, f)
        
        metadata = {
            "cv_results": {"mean_accuracy": 0.58, "mean_auc": 0.62},
            "training_samples": 1500,
            "features": ["feature1", "feature2", "feature3"],
            "training_start_date": "2022-01-01",
            "training_end_date": "2023-01-01",
            "training_duration_s": 120
        }
        
        model_id = registry.register_model(
            symbol="AAPL",
            model_name="jetson_trading_v1",
            model_file_path=dummy_model_path,
            metadata=metadata,
            tags=["demo", "lgbm"],
            notes="First model for demo."
        )
        print(f"Registered model with ID: {model_id}")
        assert model_id is not None

        # --- 2. Get active model ---
        print("\n--- 2. Retrieving active model for AAPL ---")
        active_model = registry.get_active_model("AAPL")
        if active_model:
            print(f"Active model version: {active_model['version']}")
            print(f"File path: {active_model['file_path']}")
            assert active_model['is_active'] == 1
        else:
            print("No active model found.")

        # --- 3. Register another version ---
        print("\n--- 3. Registering a second version for AAPL ---")
        dummy_model_path_v2 = os.path.join(tmpdir, "AAPL_model_v2.pkl")
        with open(dummy_model_path_v2, "wb") as f:
            joblib.dump({"model": "dummy_lgbm_v2"}, f)
        metadata_v2 = metadata.copy()
        metadata_v2["cv_results"]["mean_accuracy"] = 0.61 # Improved model
        
        model_id_v2 = registry.register_model("AAPL", "jetson_trading_v1", dummy_model_path_v2, metadata_v2, notes="Improved version.")
        print(f"Registered model v2 with ID: {model_id_v2}")
        
        # --- 4. Get all versions and activate the old one ---
        print("\n--- 4. Listing all versions and reactivating v1 ---")
        versions = registry.get_model_versions("AAPL")
        print(f"Found {len(versions)} versions for AAPL.")
        for v in versions:
            print(f"  - Version: {v['version']}, Active: {v['is_active']}, Accuracy: {v['cv_accuracy']:.2f}")
        
        registry.activate_model(model_id) # Reactivate the first model
        active_model_after_change = registry.get_active_model("AAPL")
        print(f"Newly activated model version: {active_model_after_change['version']}")
        assert active_model_after_change['id'] == model_id

        # --- 5. Record performance for the active model ---
        print("\n--- 5. Recording performance for the active model ---")
        perf_metrics = {
            "predictions_count": 120,
            "avg_prediction_time": 0.05,
            "accuracy": 0.59,
            "total_trades": 15,
            "total_pnl": 2345.67
        }
        today = datetime.now().strftime("%Y-%m-%d")
        registry.record_performance(active_model_after_change['id'], today, perf_metrics)
        
        performance_history = registry.get_model_performance(active_model_after_change['id'])
        print("Performance recorded:")
        print(performance_history[0])
        assert len(performance_history) == 1

        # --- 6. Validate model integrity ---
        print("\n--- 6. Validating model file integrity ---")
        validation = registry.validate_model_integrity(model_id_v2)
        print(f"Model ID {model_id_v2} integrity check valid: {validation['valid']}")
        assert validation['valid']
        
        # --- 7. Get registry summary ---
        print("\n--- 7. Getting registry summary ---")
        summary = registry.get_registry_summary()
        print(json.dumps(summary, indent=2, default=str))

        # --- 8. Clean up old models ---
        print("\n--- 8. Cleaning up old models (keeping 1) ---")
        deleted_count = registry.cleanup_old_models(keep_versions=1)
        print(f"Deleted {deleted_count} old model version(s).")
        versions_after_cleanup = registry.get_model_versions("AAPL")
        print(f"Remaining versions: {len(versions_after_cleanup)}")
        assert len(versions_after_cleanup) == 1
        assert versions_after_cleanup[0]['id'] == model_id_v2 # Should keep latest

    print("\n--- ModelRegistry Demo Complete ---")