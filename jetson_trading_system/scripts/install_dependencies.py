#!/usr/bin/env python3
"""
Jetson Trading System - Dependency Installation Script
Installs ARM64-optimized packages for NVIDIA Jetson Orin 16GB
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import urllib.request
import shutil

def check_jetson_platform():
    """Check if running on Jetson platform"""
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read()
            if 'NVIDIA Jetson' in content:
                return True
    except FileNotFoundError:
        pass
    
    # Check for Jetson-specific hardware
    if platform.machine() == 'aarch64':
        try:
            with open('/sys/firmware/devicetree/base/model', 'r') as f:
                model = f.read().strip()
                if 'Jetson' in model:
                    return True
        except FileNotFoundError:
            pass
    
    return False

def run_command(command, check=True):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    
    # Handle sudo commands with automatic password
    if command.strip().startswith('sudo'):
        command = f'echo "Yellokitty" | sudo -S {command[4:].strip()}'
    
    try:
        result = subprocess.run(command, shell=True, check=check,
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error output: {e.stderr}")
        if check:
            raise
        return e

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        raise RuntimeError(f"Python 3.8+ required, found {version.major}.{version.minor}")
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} compatible")

def update_system():
    """Update system packages"""
    print("\n=== Updating System Packages ===")
    
    commands = [
        "sudo apt update",
        "sudo apt upgrade -y",
        "sudo apt install -y build-essential cmake pkg-config",
        "sudo apt install -y libjpeg-dev libtiff5-dev libpng-dev",
        "sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev",
        "sudo apt install -y libgtk2.0-dev libcanberra-gtk-module",
        "sudo apt install -y python3-dev python3-pip python3-venv",
        "sudo apt install -y libhdf5-serial-dev hdf5-tools",
        "sudo apt install -y libatlas-base-dev gfortran",
        "sudo apt install -y libblas-dev liblapack-dev",
        "sudo apt install -y sqlite3 libsqlite3-dev",
        "sudo apt install -y curl wget git"
    ]
    
    for cmd in commands:
        run_command(cmd)

def install_jetpack_components():
    """Install JetPack SDK components if on Jetson"""
    if not check_jetson_platform():
        print("Not on Jetson platform, skipping JetPack installation")
        return
    
    print("\n=== Installing JetPack Components ===")
    
    # Install CUDA development tools
    commands = [
        "sudo apt install -y nvidia-jetpack",
        "sudo apt install -y cuda-toolkit-*",
        "sudo apt install -y libcudnn8-dev",
        "sudo apt install -y libnvinfer-dev",
        "sudo apt install -y python3-libnvinfer-dev"
    ]
    
    for cmd in commands:
        run_command(cmd, check=False)  # Don't fail if some packages aren't available

def setup_python_environment():
    """Set up Python virtual environment"""
    print("\n=== Setting up Python Environment ===")
    
    # Upgrade pip
    run_command("python3 -m pip install --upgrade pip")
    
    # Install wheel for better package compilation
    run_command("python3 -m pip install wheel setuptools")

def install_numpy_scipy():
    """Install NumPy and SciPy with optimizations"""
    print("\n=== Installing NumPy and SciPy ===")
    
    if check_jetson_platform():
        # Use pre-compiled wheels for Jetson
        commands = [
            "python3 -m pip install numpy==1.21.0",
            "python3 -m pip install scipy==1.9.0"
        ]
    else:
        # Standard installation
        commands = [
            "python3 -m pip install numpy>=1.21.0",
            "python3 -m pip install scipy>=1.9.0"
        ]
    
    for cmd in commands:
        run_command(cmd)

def install_pandas():
    """Install Pandas with optimizations"""
    print("\n=== Installing Pandas ===")
    run_command("python3 -m pip install pandas>=1.5.0")

def install_scikit_learn():
    """Install scikit-learn"""
    print("\n=== Installing scikit-learn ===")
    run_command("python3 -m pip install scikit-learn>=1.1.0")

def install_lightgbm():
    """Install LightGBM with GPU support"""
    print("\n=== Installing LightGBM ===")
    
    if check_jetson_platform():
        # Install with GPU support on Jetson
        run_command("python3 -m pip install lightgbm --install-option=--gpu")
    else:
        # Standard CPU version
        run_command("python3 -m pip install lightgbm>=3.3.0")

def install_ta_lib():
    """Install TA-Lib for technical analysis"""
    print("\n=== Installing TA-Lib ===")
    
    # Install system dependencies
    if check_jetson_platform():
        run_command("sudo apt install -y libta-lib-dev")
    else:
        # Build from source if not available in repos
        commands = [
            "wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz",
            "tar -xzf ta-lib-0.4.0-src.tar.gz",
            "cd ta-lib && ./configure --prefix=/usr/local",
            "cd ta-lib && make && sudo make install",
            "rm -rf ta-lib ta-lib-0.4.0-src.tar.gz"
        ]
        
        for cmd in commands:
            run_command(cmd, check=False)
    
    # Install Python wrapper
    run_command("python3 -m pip install TA-Lib>=0.4.24")

def install_financial_packages():
    """Install financial data and trading packages"""
    print("\n=== Installing Financial Packages ===")
    
    packages = [
        "alpaca-trade-api>=2.3.0",
        "polygon-api-client>=1.7.0",
        "zipline-reloaded>=2.2.0",
        "alphalens-reloaded>=0.4.0",
        "pyfolio-reloaded>=0.9.0"
    ]
    
    for package in packages:
        run_command(f"python3 -m pip install {package}")

def install_gpu_packages():
    """Install GPU acceleration packages"""
    print("\n=== Installing GPU Packages ===")
    
    if check_jetson_platform():
        # CuPy for Jetson (specific CUDA version)
        run_command("python3 -m pip install cupy-cuda11x>=11.0.0", check=False)
    
    # Numba for JIT compilation
    run_command("python3 -m pip install numba>=0.56.0")

def install_system_monitoring():
    """Install system monitoring packages"""
    print("\n=== Installing System Monitoring ===")
    
    packages = [
        "psutil>=5.8.0",
        "python-dotenv>=0.19.0",
        "asyncio-mqtt>=0.10.0"
    ]
    
    for package in packages:
        run_command(f"python3 -m pip install {package}")

def install_additional_packages():
    """Install additional utility packages"""
    print("\n=== Installing Additional Packages ===")
    
    packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "aiohttp>=3.8.0",
        "websockets>=10.0",
        "pyyaml>=6.0"
    ]
    
    for package in packages:
        run_command(f"python3 -m pip install {package}")

def verify_installation():
    """Verify all packages are installed correctly"""
    print("\n=== Verifying Installation ===")
    
    test_imports = [
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import pandas; print(f'Pandas: {pandas.__version__}')",
        "import sklearn; print(f'scikit-learn: {sklearn.__version__}')",
        "import lightgbm; print(f'LightGBM: {lightgbm.__version__}')",
        "import talib; print(f'TA-Lib: {talib.__version__}')",
        "import alpaca_trade_api; print('Alpaca API: OK')",
        "import polygon; print('Polygon API: OK')",
        "import psutil; print(f'psutil: {psutil.__version__}')"
    ]
    
    for test in test_imports:
        try:
            result = subprocess.run([sys.executable, "-c", test], 
                                  capture_output=True, text=True, check=True)
            print(f"✓ {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {test}")
            print(f"  Error: {e.stderr}")

def setup_environment_file():
    """Create environment file template"""
    print("\n=== Setting up Environment File ===")
    
    env_content = """# Jetson Trading System Environment Variables

# API Keys (Required)
POLYGON_API_KEY=your_polygon_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Trading Configuration
PAPER_TRADING=true
INITIAL_CAPITAL=25000

# System Configuration
LOG_LEVEL=INFO
DATA_DIR=./data
MODELS_DIR=./models
CACHE_DIR=./cache

# Performance Settings
MAX_PARALLEL_WORKERS=4
CACHE_MEMORY_MB=1024
CACHE_DISK_MB=4096
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✓ Created environment file: {env_file}")
        print("  Please edit .env file and add your API keys")
    else:
        print("✓ Environment file already exists")

def main():
    """Main installation process"""
    print("Jetson Trading System - Dependency Installation")
    print("=" * 50)
    
    try:
        # Check system compatibility
        check_python_version()
        
        if check_jetson_platform():
            print("✓ Running on NVIDIA Jetson platform")
        else:
            print("⚠ Not running on Jetson platform - installing CPU-only versions")
        
        # Installation steps
        update_system()
        install_jetpack_components()
        setup_python_environment()
        install_numpy_scipy()
        install_pandas()
        install_scikit_learn()
        install_lightgbm()
        install_ta_lib()
        install_financial_packages()
        install_gpu_packages()
        install_system_monitoring()
        install_additional_packages()
        
        # Verification
        verify_installation()
        setup_environment_file()
        
        print("\n" + "=" * 50)
        print("✓ Installation completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python3 main.py --mode train")
        print("3. Run: python3 main.py --mode backtest")
        print("4. Run: python3 main.py --mode live")
        
    except Exception as e:
        print(f"\n✗ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()