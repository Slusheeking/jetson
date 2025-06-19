"""
Jetson Orin Hardware Utilities
System monitoring, performance optimization, and hardware management
"""

import psutil
import subprocess
import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class JetsonMonitor:
    """Hardware monitoring and performance optimization for Jetson Orin"""
    
    def __init__(self):
        self.temp_history = []
        self.memory_history = []
        self.gpu_history = []
        
    def get_gpu_stats(self) -> Dict:
        """Get GPU utilization and memory stats"""
        try:
            # Use nvidia-smi to get GPU stats
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                
                def safe_float(val, default=0.0):
                    """Safely convert value to float, return default if fails"""
                    try:
                        # Remove brackets and whitespace, handle N/A values
                        clean_val = val.strip().replace('[', '').replace(']', '')
                        if clean_val.upper() in ['N/A', 'NA', '']:
                            return default
                        return float(clean_val)
                    except (ValueError, AttributeError):
                        return default
                
                return {
                    'gpu_utilization_pct': safe_float(values[0]) if len(values) > 0 else 0.0,
                    'memory_used_mb': safe_float(values[1]) if len(values) > 1 else 0.0,
                    'memory_total_mb': safe_float(values[2], 8192.0) if len(values) > 2 else 8192.0,
                    'temperature_c': safe_float(values[3]) if len(values) > 3 else 0.0
                }
        except Exception as e:
            logger.warning(f"Could not get GPU stats: {e}")
            
        return {
            'gpu_utilization_pct': 0.0,
            'memory_used_mb': 0.0,
            'memory_total_mb': 8192.0,  # Default estimate
            'temperature_c': 0.0
        }
    
    def get_cpu_stats(self) -> Dict:
        """Get CPU utilization and temperature"""
        try:
            # CPU utilization (non-blocking, immediate)
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq()
            cpu_temps = psutil.sensors_temperatures()
            
            # Try to get CPU temperature
            cpu_temp = 0.0
            if 'thermal_zone0' in cpu_temps:
                cpu_temp = cpu_temps['thermal_zone0'][0].current
            elif 'coretemp' in cpu_temps:
                cpu_temp = cpu_temps['coretemp'][0].current
                
            return {
                'cpu_utilization_pct': cpu_percent,
                'cpu_frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'cpu_temperature_c': cpu_temp,
                'cpu_cores': psutil.cpu_count()
            }
        except Exception as e:
            logger.warning(f"Could not get CPU stats: {e}")
            return {
                'cpu_utilization_pct': 0.0,
                'cpu_frequency_mhz': 0.0,
                'cpu_temperature_c': 0.0,
                'cpu_cores': 8
            }
    
    def get_memory_stats(self) -> Dict:
        """Get system memory statistics"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_mb': memory.total // (1024*1024),
                'available_mb': memory.available // (1024*1024),
                'used_mb': memory.used // (1024*1024),
                'usage_pct': memory.percent,
                'swap_used_mb': swap.used // (1024*1024),
                'swap_total_mb': swap.total // (1024*1024)
            }
        except Exception as e:
            logger.warning(f"Could not get memory stats: {e}")
            return {
                'total_mb': 16384,
                'available_mb': 8192,
                'used_mb': 8192,
                'usage_pct': 50.0,
                'swap_used_mb': 0,
                'swap_total_mb': 0
            }
    
    def get_power_stats(self) -> Dict:
        """Get power consumption statistics (if available)"""
        try:
            # Try to read power information from Jetson power monitor
            power_files = [
                '/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input',
                '/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power1_input',
                '/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power2_input'
            ]
            
            total_power_mw = 0
            for pattern in power_files:
                try:
                    files = list(Path().glob(pattern))
                    for file_path in files:
                        with open(file_path, 'r') as f:
                            power_mw = int(f.read().strip())
                            total_power_mw += power_mw
                except:
                    continue
            
            return {
                'total_power_w': total_power_mw / 1000.0,
                'power_available': total_power_mw > 0
            }
        except Exception as e:
            logger.warning(f"Could not get power stats: {e}")
            return {
                'total_power_w': 0.0,
                'power_available': False
            }
    
    def get_disk_stats(self) -> Dict:
        """Get disk usage statistics"""
        try:
            disk_usage = psutil.disk_usage('/')
            return {
                'total_gb': disk_usage.total // (1024**3),
                'used_gb': disk_usage.used // (1024**3),
                'free_gb': disk_usage.free // (1024**3),
                'usage_pct': (disk_usage.used / disk_usage.total) * 100
            }
        except Exception as e:
            logger.warning(f"Could not get disk stats: {e}")
            return {
                'total_gb': 64,
                'used_gb': 32,
                'free_gb': 32,
                'usage_pct': 50.0
            }
    
    def get_complete_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            'timestamp': time.time(),
            'gpu': self.get_gpu_stats(),
            'cpu': self.get_cpu_stats(),
            'memory': self.get_memory_stats(),
            'power': self.get_power_stats(),
            'disk': self.get_disk_stats()
        }
    
    def is_system_healthy(self) -> Tuple[bool, str]:
        """Check if system is operating within safe parameters"""
        stats = self.get_complete_system_stats()
        
        # Temperature checks
        gpu_temp = stats['gpu']['temperature_c']
        cpu_temp = stats['cpu']['cpu_temperature_c']
        
        if gpu_temp > 80:
            return False, f"GPU temperature too high: {gpu_temp}°C"
        if cpu_temp > 75:
            return False, f"CPU temperature too high: {cpu_temp}°C"
        
        # Memory checks
        memory_usage = stats['memory']['usage_pct']
        if memory_usage > 90:
            return False, f"Memory usage too high: {memory_usage}%"
        
        # Disk space checks
        disk_usage = stats['disk']['usage_pct']
        if disk_usage > 95:
            return False, f"Disk usage too high: {disk_usage}%"
        
        return True, "System healthy"
    
    def optimize_for_trading(self):
        """Apply system optimizations for trading workload"""
        try:
            # Set CPU governor to performance mode
            subprocess.run([
                'sudo', 'sh', '-c', 
                'echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
            ], check=False)
            
            # Set GPU to maximum performance
            subprocess.run(['sudo', 'jetson_clocks'], check=False)
            
            # Increase system limits for real-time performance
            subprocess.run([
                'sudo', 'sysctl', '-w', 'kernel.sched_rt_runtime_us=950000'
            ], check=False)
            
            logger.info("Applied Jetson optimizations for trading")
            
        except Exception as e:
            logger.warning(f"Could not apply optimizations: {e}")

class JetsonPerformanceLogger:
    """Log system performance metrics for analysis"""
    
    def __init__(self, log_file: str = "./logs/jetson_performance.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = JetsonMonitor()
        
    def log_performance(self):
        """Log current system performance"""
        stats = self.monitor.get_complete_system_stats()
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(stats) + '\n')
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """Get performance summary for specified hours"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            stats_list = []
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        stats = json.loads(line.strip())
                        if stats['timestamp'] >= cutoff_time:
                            stats_list.append(stats)
                    except:
                        continue
            
            if not stats_list:
                return {}
            
            # Calculate averages and peaks
            gpu_utils = [s['gpu']['gpu_utilization_pct'] for s in stats_list]
            cpu_utils = [s['cpu']['cpu_utilization_pct'] for s in stats_list]
            memory_utils = [s['memory']['usage_pct'] for s in stats_list]
            gpu_temps = [s['gpu']['temperature_c'] for s in stats_list if s['gpu']['temperature_c'] > 0]
            cpu_temps = [s['cpu']['cpu_temperature_c'] for s in stats_list if s['cpu']['cpu_temperature_c'] > 0]
            
            return {
                'period_hours': hours,
                'samples': len(stats_list),
                'gpu_utilization': {
                    'avg': np.mean(gpu_utils),
                    'max': np.max(gpu_utils),
                    'min': np.min(gpu_utils)
                },
                'cpu_utilization': {
                    'avg': np.mean(cpu_utils),
                    'max': np.max(cpu_utils),
                    'min': np.min(cpu_utils)
                },
                'memory_usage': {
                    'avg': np.mean(memory_utils),
                    'max': np.max(memory_utils),
                    'min': np.min(memory_utils)
                },
                'gpu_temperature': {
                    'avg': np.mean(gpu_temps) if gpu_temps else 0,
                    'max': np.max(gpu_temps) if gpu_temps else 0,
                } if gpu_temps else {},
                'cpu_temperature': {
                    'avg': np.mean(cpu_temps) if cpu_temps else 0,
                    'max': np.max(cpu_temps) if cpu_temps else 0,
                } if cpu_temps else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

def get_jetson_info() -> Dict:
    """Get Jetson hardware information"""
    try:
        # Try to get Jetson model information
        result = subprocess.run(['cat', '/proc/device-tree/model'], 
                              capture_output=True, text=True)
        
        model = result.stdout.strip() if result.returncode == 0 else "Unknown Jetson"
        
        # Get L4T version
        l4t_result = subprocess.run(['cat', '/etc/nv_tegra_release'], 
                                   capture_output=True, text=True)
        l4t_version = "Unknown"
        if l4t_result.returncode == 0:
            for line in l4t_result.stdout.split('\n'):
                if 'R' in line and 'REVISION' in line:
                    l4t_version = line.strip()
                    break
        
        return {
            'model': model,
            'l4t_version': l4t_version,
            'cpu_cores': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total // (1024**3)
        }
        
    except Exception as e:
        logger.warning(f"Could not get Jetson info: {e}")
        return {
            'model': 'Jetson Orin 16GB',
            'l4t_version': 'Unknown',
            'cpu_cores': 8,
            'total_memory_gb': 16
        }

# Global monitor instance
jetson_monitor = JetsonMonitor()
performance_logger = JetsonPerformanceLogger()