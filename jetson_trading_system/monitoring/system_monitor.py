"""
System Monitoring for Jetson Trading System
Real-time hardware and system performance monitoring
"""

import psutil
import GPUtil
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import subprocess
import os

from ..config.jetson_settings import JetsonConfig
from ..utils.logger import get_system_logger
from ..utils.database import TradingDatabase

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    cpu_temp: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    gpu_percent: float
    gpu_memory_percent: float
    gpu_temp: float
    power_draw_watts: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    process_count: int
    swap_percent: float

@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    metric_name: str
    warning_level: float
    critical_level: float
    enabled: bool = True
    cooldown_minutes: int = 5

@dataclass
class SystemAlert:
    """System alert"""
    timestamp: datetime
    metric_name: str
    level: str  # 'warning' or 'critical'
    current_value: float
    threshold_value: float
    message: str

class SystemMonitor:
    """
    Comprehensive system monitoring for Jetson Orin
    Tracks hardware performance, resources, and generates alerts
    """
    
    def __init__(self, 
                 monitoring_interval: int = 10,
                 history_size: int = 1000,
                 enable_gpu_monitoring: bool = True):
        """
        Initialize system monitor
        
        Args:
            monitoring_interval: Seconds between measurements
            history_size: Number of historical measurements to keep
            enable_gpu_monitoring: Enable GPU monitoring
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        self.logger = get_system_logger()
        self.db_manager = TradingDatabase()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.shutdown_event = threading.Event()
        
        # Data storage
        self.metrics_history = deque(maxlen=history_size)
        self.alerts_history = deque(maxlen=100)
        
        # Alert system
        self.alert_thresholds = self._setup_default_thresholds()
        self.alert_callbacks = []
        self.last_alert_times = {}  # metric -> timestamp
        
        # Network baseline
        self.network_baseline = self._get_network_baseline()
        
        # Performance tracking
        self.monitoring_stats = {
            'total_measurements': 0,
            'failed_measurements': 0,
            'alerts_generated': 0,
            'uptime_start': datetime.now()
        }
        
        self.logger.info(f"SystemMonitor initialized - Interval: {monitoring_interval}s")
    
    def _setup_default_thresholds(self) -> Dict[str, AlertThreshold]:
        """Setup default alert thresholds for Jetson Orin"""
        return {
            'cpu_percent': AlertThreshold('cpu_percent', 80.0, 95.0),
            'cpu_temp': AlertThreshold('cpu_temp', 75.0, 85.0),
            'memory_percent': AlertThreshold('memory_percent', 80.0, 90.0),
            'disk_percent': AlertThreshold('disk_percent', 85.0, 95.0),
            'gpu_percent': AlertThreshold('gpu_percent', 85.0, 95.0),
            'gpu_temp': AlertThreshold('gpu_temp', 75.0, 85.0),
            'gpu_memory_percent': AlertThreshold('gpu_memory_percent', 80.0, 90.0),
            'power_draw_watts': AlertThreshold('power_draw_watts', 50.0, 60.0),
            'swap_percent': AlertThreshold('swap_percent', 50.0, 75.0)
        }
    
    def start_monitoring(self) -> bool:
        """Start system monitoring"""
        try:
            if self.is_monitoring:
                self.logger.warning("System monitoring is already running")
                return True
            
            self.logger.info("Starting system monitoring...")
            
            # Reset shutdown event
            self.shutdown_event.clear()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.is_monitoring = True
            self.performance_stats['uptime_start'] = datetime.now()
            
            self.logger.info("System monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """Stop system monitoring"""
        try:
            if not self.is_monitoring:
                return True
            
            self.logger.info("Stopping system monitoring...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
            
            self.is_monitoring = False
            
            self.logger.info("System monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping system monitoring: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            self.logger.info("System monitoring loop started")
            
            while not self.shutdown_event.is_set():
                try:
                    # Collect metrics
                    metrics = self._collect_system_metrics()
                    
                    if metrics:
                        # Store metrics
                        self.metrics_history.append(metrics)
                        
                        # Check for alerts
                        self._check_alerts(metrics)
                        
                        # Update stats
                        self.monitoring_stats['total_measurements'] += 1
                    else:
                        self.monitoring_stats['failed_measurements'] += 1
                    
                    # Wait for next interval
                    self.shutdown_event.wait(timeout=self.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    self.monitoring_stats['failed_measurements'] += 1
                    time.sleep(5)  # Brief pause before retrying
        
        except Exception as e:
            self.logger.critical(f"Fatal error in monitoring loop: {e}")
        
        finally:
            self.logger.info("System monitoring loop ended")
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect comprehensive system metrics"""
        try:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_temp = self._get_cpu_temperature()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024 * 1024 * 1024)
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # GPU metrics
            gpu_percent, gpu_memory_percent, gpu_temp = self._get_gpu_metrics()
            
            # Power metrics
            power_draw_watts = self._get_power_consumption()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            network_bytes_sent = network_io.bytes_sent - self.network_baseline.get('bytes_sent', 0)
            network_bytes_recv = network_io.bytes_recv - self.network_baseline.get('bytes_recv', 0)
            
            # System metrics
            load_average = list(psutil.getloadavg())
            process_count = len(psutil.pids())
            
            # Swap metrics
            swap = psutil.swap_memory()
            swap_percent = swap.percent
            
            return SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                cpu_temp=cpu_temp,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                gpu_temp=gpu_temp,
                power_draw_watts=power_draw_watts,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                load_average=load_average,
                process_count=process_count,
                swap_percent=swap_percent
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature for Jetson"""
        try:
            # Try Jetson-specific thermal zones
            thermal_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp',
                '/sys/devices/virtual/thermal/thermal_zone0/temp',
                '/sys/devices/virtual/thermal/thermal_zone1/temp'
            ]
            
            for path in thermal_paths:
                try:
                    with open(path, 'r') as f:
                        temp_millicelsius = int(f.read().strip())
                        return temp_millicelsius / 1000.0
                except:
                    continue
            
            # Fallback to tegrastats if available
            try:
                result = subprocess.run(['tegrastats', '--interval', '100'], 
                                      capture_output=True, text=True, timeout=2)
                # Parse tegrastats output for temperature
                # This is a simplified parser
                if 'Temp' in result.stdout:
                    return 50.0  # Placeholder
            except:
                pass
            
            return 45.0  # Default safe value
            
        except Exception as e:
            self.logger.debug(f"Error getting CPU temperature: {e}")
            return 45.0
    
    def _get_gpu_metrics(self) -> Tuple[float, float, float]:
        """Get GPU utilization, memory, and temperature"""
        try:
            if not self.enable_gpu_monitoring:
                return 0.0, 0.0, 0.0
            
            # Try using GPUtil for NVIDIA GPUs
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    return (
                        gpu.load * 100,  # GPU utilization %
                        gpu.memoryUtil * 100,  # GPU memory %
                        gpu.temperature  # GPU temperature
                    )
            except:
                pass
            
            # Try jetson-specific GPU monitoring
            try:
                # Check for Jetson GPU stats
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=3)
                
                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    gpu_util = float(values[0])
                    memory_used = float(values[1])
                    memory_total = float(values[2])
                    gpu_temp = float(values[3])
                    
                    memory_percent = (memory_used / memory_total) * 100
                    return gpu_util, memory_percent, gpu_temp
            except:
                pass
            
            return 0.0, 0.0, 40.0  # Default values
            
        except Exception as e:
            self.logger.debug(f"Error getting GPU metrics: {e}")
            return 0.0, 0.0, 40.0
    
    def _get_power_consumption(self) -> float:
        """Get power consumption for Jetson"""
        try:
            # Jetson-specific power monitoring
            power_paths = [
                '/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input',
                '/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input',
                '/sys/class/hwmon/hwmon0/power1_input',
                '/sys/class/hwmon/hwmon1/power1_input'
            ]
            
            total_power = 0.0
            
            for path in power_paths:
                try:
                    with open(path, 'r') as f:
                        power_microwatts = int(f.read().strip())
                        total_power += power_microwatts / 1000000.0  # Convert to watts
                except:
                    continue
            
            if total_power > 0:
                return total_power
            
            # Fallback estimation based on CPU/GPU usage
            cpu_percent = psutil.cpu_percent()
            estimated_power = 10.0 + (cpu_percent / 100.0) * 40.0  # 10W idle + up to 40W load
            
            return estimated_power
            
        except Exception as e:
            self.logger.debug(f"Error getting power consumption: {e}")
            return 25.0  # Default estimate
    
    def _get_network_baseline(self) -> Dict[str, int]:
        """Get network baseline for delta calculations"""
        try:
            network_io = psutil.net_io_counters()
            return {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv
            }
        except:
            return {'bytes_sent': 0, 'bytes_recv': 0}
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check metrics against alert thresholds"""
        try:
            current_time = datetime.now()
            
            # Define metric mappings
            metric_values = {
                'cpu_percent': metrics.cpu_percent,
                'cpu_temp': metrics.cpu_temp,
                'memory_percent': metrics.memory_percent,
                'disk_percent': metrics.disk_percent,
                'gpu_percent': metrics.gpu_percent,
                'gpu_temp': metrics.gpu_temp,
                'gpu_memory_percent': metrics.gpu_memory_percent,
                'power_draw_watts': metrics.power_draw_watts,
                'swap_percent': metrics.swap_percent
            }
            
            for metric_name, current_value in metric_values.items():
                if metric_name not in self.alert_thresholds:
                    continue
                
                threshold = self.alert_thresholds[metric_name]
                if not threshold.enabled:
                    continue
                
                # Check cooldown period
                last_alert_time = self.last_alert_times.get(metric_name)
                if last_alert_time:
                    time_since_last = (current_time - last_alert_time).total_seconds() / 60
                    if time_since_last < threshold.cooldown_minutes:
                        continue
                
                # Check thresholds
                alert_level = None
                threshold_value = None
                
                if current_value >= threshold.critical_level:
                    alert_level = 'critical'
                    threshold_value = threshold.critical_level
                elif current_value >= threshold.warning_level:
                    alert_level = 'warning'
                    threshold_value = threshold.warning_level
                
                if alert_level:
                    self._generate_alert(metric_name, alert_level, current_value, threshold_value)
                    self.last_alert_times[metric_name] = current_time
        
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _generate_alert(self, metric_name: str, level: str, current_value: float, threshold_value: float):
        """Generate and handle system alert"""
        try:
            message = f"{metric_name.replace('_', ' ').title()} {level}: {current_value:.1f} (threshold: {threshold_value:.1f})"
            
            alert = SystemAlert(
                timestamp=datetime.now(),
                metric_name=metric_name,
                level=level,
                current_value=current_value,
                threshold_value=threshold_value,
                message=message
            )
            
            # Store alert
            self.alerts_history.append(alert)
            self.monitoring_stats['alerts_generated'] += 1
            
            # Log alert
            if level == 'critical':
                self.logger.critical(f"SYSTEM ALERT: {message}")
            else:
                self.logger.warning(f"SYSTEM ALERT: {message}")
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")
        
        except Exception as e:
            self.logger.error(f"Error generating alert: {e}")
    
    # Public interface methods
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_recent_alerts(self, count: int = 10) -> List[SystemAlert]:
        """Get recent alerts"""
        return list(self.alerts_history)[-count:] if self.alerts_history else []
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system summary with key metrics"""
        try:
            current_metrics = self.get_current_metrics()
            if not current_metrics:
                return {}
            
            uptime = datetime.now() - self.monitoring_stats['uptime_start']
            
            return {
                'status': 'running' if self.is_monitoring else 'stopped',
                'uptime_hours': uptime.total_seconds() / 3600,
                'current_metrics': asdict(current_metrics),
                'recent_alerts': len([a for a in self.alerts_history if 
                                    (datetime.now() - a.timestamp).total_seconds() < 3600]),
                'monitoring_stats': self.monitoring_stats,
                'alert_thresholds': {name: asdict(threshold) for name, threshold in self.alert_thresholds.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system summary: {e}")
            return {}
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def update_alert_threshold(self, metric_name: str, warning_level: float, critical_level: float):
        """Update alert threshold for a metric"""
        if metric_name in self.alert_thresholds:
            self.alert_thresholds[metric_name].warning_level = warning_level
            self.alert_thresholds[metric_name].critical_level = critical_level
            self.logger.info(f"Updated alert threshold for {metric_name}")
    
    def enable_alert(self, metric_name: str, enabled: bool = True):
        """Enable or disable alerts for a metric"""
        if metric_name in self.alert_thresholds:
            self.alert_thresholds[metric_name].enabled = enabled
            self.logger.info(f"Alert for {metric_name} {'enabled' if enabled else 'disabled'}")
    
    def export_metrics(self, filename: str, hours: int = 24) -> bool:
        """Export metrics to JSON file"""
        try:
            metrics = self.get_metrics_history(minutes=hours * 60)
            
            export_data = {
                'export_time': datetime.now().isoformat(),
                'time_period_hours': hours,
                'metrics_count': len(metrics),
                'metrics': [asdict(m) for m in metrics]
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize system monitor
    monitor = SystemMonitor(monitoring_interval=5)
    
    # Add alert callback
    def alert_handler(alert):
        print(f"ALERT: {alert.message}")
    
    monitor.add_alert_callback(alert_handler)
    
    # Start monitoring
    if monitor.start_monitoring():
        print("System monitoring started")
        
        try:
            # Monitor for 30 seconds
            time.sleep(30)
            
            # Get current metrics
            current = monitor.get_current_metrics()
            if current:
                print(f"CPU: {current.cpu_percent:.1f}%")
                print(f"Memory: {current.memory_percent:.1f}%")
                print(f"GPU: {current.gpu_percent:.1f}%")
                print(f"Temperature: {current.cpu_temp:.1f}°C")
            
            # Get summary
            summary = monitor.get_system_summary()
            print(f"Total measurements: {summary.get('monitoring_stats', {}).get('total_measurements', 0)}")
            
        except KeyboardInterrupt:
            print("Stopping...")
        
        finally:
            monitor.stop_monitoring()
    else:
        print("Failed to start system monitoring")
