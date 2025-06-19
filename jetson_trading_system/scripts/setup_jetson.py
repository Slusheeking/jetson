#!/usr/bin/env python3
"""
Jetson Orin Setup Script
Automated setup for the trading system on Jetson Orin 16GB
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, check=True, shell=False):
    """Run a system command with logging"""
    logger.info(f"Running: {command}")
    try:
        if shell:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, capture_output=True, text=True)
        
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if not check:
            return e
        raise

def check_jetson_environment():
    """Check if running on Jetson and get system info"""
    logger.info("🔍 Checking Jetson environment...")
    
    try:
        # Check if this is a Jetson device
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
        logger.info(f"✅ Detected: {model}")
        
        # Check L4T version
        with open('/etc/nv_tegra_release', 'r') as f:
            l4t_info = f.read().strip()
        logger.info(f"L4T Version: {l4t_info}")
        
        return True
    except FileNotFoundError:
        logger.warning("⚠️ Not running on Jetson device")
        return False

def install_system_dependencies():
    """Install system-level dependencies"""
    logger.info("📦 Installing system dependencies...")
    
    # Update package list
    run_command("sudo apt-get update")
    
    # Essential development tools
    deps = [
        "python3-dev", "python3-pip", "python3-venv",
        "build-essential", "cmake", "git",
        "libhdf5-dev", "libhdf5-serial-dev",
        "libxml2-dev", "libxslt1-dev",
        "libblas-dev", "liblapack-dev", "libatlas-base-dev", "gfortran",
        "libfreetype6-dev", "libpng-dev",
        "libssl-dev", "libffi-dev",
        "curl", "wget", "unzip"
    ]
    
    for dep in deps:
        try:
            run_command(f"sudo apt-get install -y {dep}")
        except:
            logger.warning(f"Failed to install {dep}, continuing...")

def install_ta_lib():
    """Install TA-Lib from source (required for ARM64)"""
    logger.info("📊 Installing TA-Lib from source...")
    
    ta_lib_dir = "/tmp/ta-lib"
    
    try:
        # Download TA-Lib source
        run_command("wget -P /tmp http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
        run_command("tar -xzf /tmp/ta-lib-0.4.0-src.tar.gz -C /tmp")
        
        # Compile and install
        os.chdir("/tmp/ta-lib")
        run_command("./configure --prefix=/usr")
        run_command("make")
        run_command("sudo make install")
        
        # Update library path
        run_command("sudo ldconfig")
        
        logger.info("✅ TA-Lib installed successfully")
        
    except Exception as e:
        logger.error(f"Failed to install TA-Lib: {e}")
        logger.info("You may need to install TA-Lib manually")

def setup_python_environment():
    """Set up Python virtual environment"""
    logger.info("🐍 Setting up Python environment...")
    
    venv_path = Path.home() / "jetson_trading_venv"
    
    # Create virtual environment
    run_command(f"python3 -m venv {venv_path}")
    
    # Activate and upgrade pip
    pip_path = venv_path / "bin" / "pip"
    run_command(f"{pip_path} install --upgrade pip setuptools wheel")
    
    return venv_path

def install_python_dependencies(venv_path):
    """Install Python dependencies"""
    logger.info("📚 Installing Python dependencies...")
    
    pip_path = venv_path / "bin" / "pip"
    requirements_path = Path(__file__).parent.parent / "requirements_jetson.txt"
    
    if requirements_path.exists():
        # Install core packages first
        core_packages = [
            "numpy>=1.21.0",
            "pandas>=1.5.0", 
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0"
        ]
        
        for package in core_packages:
            try:
                run_command(f"{pip_path} install {package}")
            except:
                logger.warning(f"Failed to install {package}")
        
        # Install TA-Lib Python wrapper
        try:
            run_command(f"{pip_path} install TA-Lib")
        except:
            logger.warning("Failed to install TA-Lib Python wrapper")
        
        # Install remaining requirements
        try:
            run_command(f"{pip_path} install -r {requirements_path}")
        except:
            logger.warning("Some packages failed to install, check manually")
    else:
        logger.error(f"Requirements file not found: {requirements_path}")

def setup_jetson_optimizations():
    """Apply Jetson-specific optimizations"""
    logger.info("⚡ Applying Jetson optimizations...")
    
    try:
        # Enable maximum performance mode
        run_command("sudo jetson_clocks", check=False)
        
        # Set CPU governor to performance
        run_command("sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'", 
                   shell=True, check=False)
        
        # Increase swap if needed (for 16GB model)
        swap_info = run_command("free -h", check=False)
        if swap_info and "0B" in swap_info.stdout:
            logger.info("Creating swap file for additional memory...")
            commands = [
                "sudo fallocate -l 4G /swapfile",
                "sudo chmod 600 /swapfile", 
                "sudo mkswap /swapfile",
                "sudo swapon /swapfile"
            ]
            for cmd in commands:
                run_command(cmd, check=False)
        
        logger.info("✅ Jetson optimizations applied")
        
    except Exception as e:
        logger.warning(f"Some optimizations failed: {e}")

def create_environment_file():
    """Create .env file template"""
    logger.info("📝 Creating environment file template...")
    
    env_template = """# Jetson Trading System Environment Variables
# Copy this file to .env and fill in your API keys

# Polygon API (required for market data)
POLYGON_API_KEY=your_polygon_api_key_here

# Alpaca API (required for trading)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading initially

# Trading Configuration
STARTING_CAPITAL=25000
MAX_POSITIONS=20
POSITION_SIZE_PCT=5.0

# System Settings
LOG_LEVEL=INFO
ENABLE_GPU=true
CACHE_SIZE_MB=2048

# Performance Monitoring
ENABLE_PERFORMANCE_LOGGING=true
PERFORMANCE_LOG_INTERVAL=60
"""
    
    env_file = Path.cwd() / ".env.template"
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    logger.info(f"✅ Environment template created: {env_file}")
    logger.info("Copy .env.template to .env and add your API keys")

def create_startup_script(venv_path):
    """Create startup script"""
    logger.info("🚀 Creating startup script...")
    
    startup_script = f"""#!/bin/bash
# Jetson Trading System Startup Script

# Activate virtual environment
source {venv_path}/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Navigate to project directory
cd {Path.cwd()}

# Run the trading system
python main.py --polygon-key $POLYGON_API_KEY --demo

echo "Trading system startup complete"
"""
    
    script_path = Path.cwd() / "start_trading_system.sh"
    with open(script_path, 'w') as f:
        f.write(startup_script)
    
    run_command(f"chmod +x {script_path}")
    logger.info(f"✅ Startup script created: {script_path}")

def main():
    """Main setup function"""
    logger.info("🔧 Starting Jetson Orin Trading System Setup")
    
    # Check environment
    is_jetson = check_jetson_environment()
    if not is_jetson:
        logger.warning("Not running on Jetson - some optimizations will be skipped")
    
    try:
        # Install system dependencies
        install_system_dependencies()
        
        # Install TA-Lib from source
        install_ta_lib()
        
        # Setup Python environment
        venv_path = setup_python_environment()
        
        # Install Python dependencies
        install_python_dependencies(venv_path)
        
        # Apply Jetson optimizations
        if is_jetson:
            setup_jetson_optimizations()
        
        # Create configuration files
        create_environment_file()
        create_startup_script(venv_path)
        
        logger.info("🎉 Setup complete!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Copy .env.template to .env and add your API keys")
        logger.info("2. Run: source .env && ./start_trading_system.sh")
        logger.info("3. Or manually: python main.py --polygon-key YOUR_KEY --demo")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()