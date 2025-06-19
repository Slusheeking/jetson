"""
Jetson Orin 16GB Specific Configuration Settings
Optimized for ARM64 architecture and GPU acceleration
"""

import os
import psutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class JetsonConfig:
    """Configuration settings optimized for Jetson Orin 16GB"""
    
    # Hardware Specifications
    JETSON_MODEL = "Orin 16GB"
    CPU_CORES = 8  # ARM64 Cortex-A78AE
    GPU_CORES = 1024  # NVIDIA Ampere GPU
    TOTAL_MEMORY_GB = 16  # LPDDR5
    AI_PERFORMANCE_TOPS = 100
    
    # Memory Management (Conservative allocation for 16GB)
    GPU_MEMORY_FRACTION = 0.7  # Reserve 70% GPU memory for ML
    SYSTEM_MEMORY_BUFFER_GB = 2  # Keep 2GB free for system
    MAX_MEMORY_USAGE_GB = 12  # Use max 12GB for trading system
    CACHE_SIZE_MB = 2048  # 2GB feature cache
    
    # Processing Configuration
    BATCH_SIZE = 128  # Optimized for 16GB memory
    MODEL_PRECISION = "fp16"  # Half precision for speed
    MAX_CONCURRENT_STOCKS = 500  # Process 500 stocks simultaneously
    PARALLEL_WORKERS = 6  # Leave 2 cores for system
    
    # Performance Targets
    MAX_SIGNAL_LATENCY_MS = 100  # <100ms signal generation
    MAX_POWER_CONSUMPTION_W = 25  # <25W average power
    TARGET_UPTIME_PCT = 99.5  # >99.5% system uptime
    
    # Data Configuration
    MAX_LOOKBACK_DAYS = 252  # 1 year of historical data
    FEATURE_UPDATE_INTERVAL_SEC = 60  # Update features every minute
    MODEL_RETRAIN_INTERVAL_HOURS = 24  # Retrain daily
    
    # Storage Configuration
    DATA_DIR = Path("./data")
    MODEL_DIR = Path("./models")
    LOGS_DIR = Path("./logs")
    CACHE_DIR = Path("./cache")
    
    # Temperature Monitoring (Critical for Jetson)
    MAX_GPU_TEMP_C = 80  # Throttle at 80°C
    MAX_CPU_TEMP_C = 75  # Throttle at 75°C
    TEMP_CHECK_INTERVAL_SEC = 30  # Check every 30 seconds
    
    @classmethod
    def get_available_memory_gb(cls):
        """Get current available system memory"""
        return psutil.virtual_memory().available / (1024**3)
    
    @classmethod
    def get_memory_usage_pct(cls):
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    @classmethod
    def get_cpu_usage_pct(cls):
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    @classmethod
    def is_memory_available(cls, required_gb):
        """Check if required memory is available"""
        available = cls.get_available_memory_gb()
        return available >= required_gb
    
    @classmethod
    def get_optimal_batch_size(cls):
        """Get optimal batch size based on current memory"""
        available_gb = cls.get_available_memory_gb()
        if available_gb < 4:
            return 64  # Conservative
        elif available_gb < 8:
            return 128  # Standard
        else:
            return 256  # Aggressive
    
    @classmethod
    def get_env_var(cls, key, default=None):
        """Get environment variable with optional default"""
        return os.getenv(key, default)
    
    @classmethod
    def get_env_bool(cls, key, default=False):
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @classmethod
    def get_env_int(cls, key, default=0):
        """Get integer environment variable"""
        try:
            return int(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def get_env_float(cls, key, default=0.0):
        """Get float environment variable"""
        try:
            return float(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
    
    # Environment-based configuration overrides
    USE_GPU_ACCELERATION = property(lambda self: self.get_env_bool('USE_GPU_ACCELERATION', True))
    MAX_MEMORY_USAGE_MB = property(lambda self: self.get_env_int('MAX_MEMORY_USAGE_MB', 12000))
    CACHE_MEMORY_MB = property(lambda self: self.get_env_int('CACHE_MEMORY_MB', 1024))
    CACHE_DISK_MB = property(lambda self: self.get_env_int('CACHE_DISK_MB', 4096))
    CACHE_TTL_SECONDS = property(lambda self: self.get_env_int('CACHE_TTL_SECONDS', 3600))
    MAX_PARALLEL_WORKERS = property(lambda self: self.get_env_int('MAX_PARALLEL_WORKERS', 4))
    POLYGON_RATE_LIMIT = property(lambda self: self.get_env_int('POLYGON_RATE_LIMIT', 300))
    ALPACA_RATE_LIMIT = property(lambda self: self.get_env_int('ALPACA_RATE_LIMIT', 200))

# LightGBM Jetson Optimization
LIGHTGBM_JETSON_PARAMS = {
    'boosting': 'gbdt',
    'objective': 'regression',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'min_data_in_leaf': 500,
    'device_type': 'gpu',  # Use Jetson GPU
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_bin': 255,
    'num_threads': JetsonConfig.PARALLEL_WORKERS,
    'verbose': -1
}

# CUDA Configuration for Jetson
CUDA_CONFIG = {
    'enable_gpu': True,
    'gpu_memory_growth': True,
    'mixed_precision': True,  # Use FP16 for speed
    'allow_soft_placement': True,
}

# Environment Variables
ENVIRONMENT_VARS = {
    'CUDA_VISIBLE_DEVICES': '0',
    'TF_GPU_ALLOCATOR': 'cuda_malloc_async',
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
    'NUMBA_CUDA_USE_NVIDIA_BINDING': '1',
}

# Apply environment variables
for key, value in ENVIRONMENT_VARS.items():
    os.environ[key] = value