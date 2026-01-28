#!/usr/bin/env python3
"""
Quantum Wormhole Simulation Framework - Installation Script

This script automates the installation and setup process for the quantum wormhole
simulation framework. It handles dependency installation, environment setup,
component verification, and initial configuration.

Usage:
    python install.py [options]

Options:
    --dev          Install development dependencies
    --minimal      Minimal installation (core components only)
    --gpu          Install GPU acceleration support
    --force        Force reinstallation of all components
    --test         Run verification tests after installation
    --quiet        Suppress verbose output
    --help         Show this help message

Requirements:
    - Python 3.8 or higher
    - pip package manager
    - Internet connection for package downloads
"""

import sys
import os
import subprocess
import platform
import json
import importlib
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('installation.log')
    ]
)
logger = logging.getLogger(__name__)


class InstallationError(Exception):
    """Custom exception for installation errors."""
    pass


class QuantumWormholeInstaller:
    """Main installer class for the quantum wormhole simulation framework."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize installer with command-line arguments."""
        self.args = args
        self.python_executable = sys.executable
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_installed = False
        
        # Installation configuration
        self.config = {
            'min_python_version': (3, 8),
            'recommended_python_version': (3, 10),
            'min_memory_gb': 8,
            'recommended_memory_gb': 16,
            'min_disk_space_gb': 2,
            'test_timeout': 300,  # 5 minutes
        }
        
        # Component verification tests
        self.verification_tests = []
        
    def run_installation(self) -> bool:
        """Run complete installation process."""
        
        try:
            self.print_header()
            self.check_system_requirements()
            self.setup_virtual_environment()
            self.install_dependencies()
            self.setup_project_structure()
            self.configure_project()
            self.verify_installation()
            
            if self.args.test:
                self.run_verification_tests()
            
            self.print_completion_message()
            return True
            
        except InstallationError as e:
            logger.error(f"Installation failed: {e}")
            self.print_error_message(str(e))
            return False
        except KeyboardInterrupt:
            logger.info("Installation cancelled by user")
            self.print_error_message("Installation cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during installation: {e}")
            self.print_error_message(f"Unexpected error: {e}")
            return False
    
    def print_header(self):
        """Print installation header."""
        if not self.args.quiet:
            print("=" * 70)
            print("🌌 Quantum Wormhole Simulation Framework - Installation")
            print("=" * 70)
            print(f"📅 Installation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🐍 Python version: {sys.version}")
            print(f"💻 Platform: {platform.platform()}")
            print(f"📁 Installation directory: {self.project_root}")
            print("-" * 70)
    
    def check_system_requirements(self):
        """Check system requirements and compatibility."""
        
        logger.info("Checking system requirements...")
        if not self.args.quiet:
            print("🔍 Checking system requirements...")
        
        # Python version check
        python_version = sys.version_info[:2]
        min_version = self.config['min_python_version']
        recommended_version = self.config['recommended_python_version']
        
        if python_version < min_version:
            raise InstallationError(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"but {python_version[0]}.{python_version[1]} found"
            )
        
        status = "✓" if python_version >= recommended_version else "⚠"
        if not self.args.quiet:
            print(f"   {status} Python {python_version[0]}.{python_version[1]}")
        
        # Memory check
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb < self.config['min_memory_gb']:
                if not self.args.quiet:
                    print(f"   ⚠ Memory: {memory_gb:.1f} GB (minimum {self.config['min_memory_gb']} GB)")
                    print("     Warning: Low memory may cause performance issues")
            else:
                status = "✓" if memory_gb >= self.config['recommended_memory_gb'] else "⚠"
                if not self.args.quiet:
                    print(f"   {status} Memory: {memory_gb:.1f} GB")
                    
        except ImportError:
            if not self.args.quiet:
                print("   ? Memory: Unable to check (psutil not available)")
        
        # Disk space check
        try:
            disk_usage = psutil.disk_usage(str(self.project_root))
            free_space_gb = disk_usage.free / (1024**3)
            
            if free_space_gb < self.config['min_disk_space_gb']:
                raise InstallationError(
                    f"Insufficient disk space: {free_space_gb:.1f} GB free, "
                    f"minimum {self.config['min_disk_space_gb']} GB required"
                )
            
            if not self.args.quiet:
                print(f"   ✓ Disk space: {free_space_gb:.1f} GB free")
                
        except ImportError:
            if not self.args.quiet:
                print("   ? Disk space: Unable to check")
        
        # Internet connectivity check
        self.check_internet_connection()
        
        # Git availability check
        self.check_git_availability()
        
        logger.info("System requirements check completed")
    
    def check_internet_connection(self):
        """Check internet connectivity for package downloads."""
        
        try:
            urllib.request.urlopen('https://pypi.org', timeout=10)
            if not self.args.quiet:
                print("   ✓ Internet connection")
        except urllib.error.URLError:
            if not self.args.quiet:
                print("   ⚠ Internet connection: Limited or unavailable")
                print("     Warning: Package installation may fail")
    
    def check_git_availability(self):
        """Check if Git is available for potential repository operations."""
        
        try:
            subprocess.run(['git', '--version'], 
                         capture_output=True, check=True, timeout=10)
            if not self.args.quiet:
                print("   ✓ Git available")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            if not self.args.quiet:
                print("   ⚠ Git: Not available")
    
    def setup_virtual_environment(self):
        """Set up Python virtual environment."""
        
        if self.args.dev or not self.args.minimal:
            logger.info("Setting up virtual environment...")
            if not self.args.quiet:
                print("🔧 Setting up virtual environment...")
            
            try:
                # Remove existing virtual environment if force flag is set
                if self.args.force and self.venv_path.exists():
                    if not self.args.quiet:
                        print("   Removing existing virtual environment...")
                    self.remove_directory(self.venv_path)
                
                # Create virtual environment if it doesn't exist
                if not self.venv_path.exists():
                    if not self.args.quiet:
                        print("   Creating virtual environment...")
                    
                    subprocess.run([
                        sys.executable, '-m', 'venv', str(self.venv_path)
                    ], check=True, timeout=120)
                
                # Determine virtual environment activation
                if platform.system() == "Windows":
                    self.venv_python = self.venv_path / "Scripts" / "python.exe"
                    self.venv_pip = self.venv_path / "Scripts" / "pip.exe"
                else:
                    self.venv_python = self.venv_path / "bin" / "python"
                    self.venv_pip = self.venv_path / "bin" / "pip"
                
                # Verify virtual environment
                if not self.venv_python.exists():
                    raise InstallationError("Virtual environment creation failed")
                
                # Update pip in virtual environment
                if not self.args.quiet:
                    print("   Updating pip...")
                
                subprocess.run([
                    str(self.venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'
                ], check=True, timeout=120, capture_output=self.args.quiet)
                
                # Use virtual environment for remaining installation
                self.python_executable = str(self.venv_python)
                
                if not self.args.quiet:
                    print("   ✓ Virtual environment ready")
                
            except subprocess.CalledProcessError as e:
                raise InstallationError(f"Virtual environment setup failed: {e}")
            except subprocess.TimeoutExpired:
                raise InstallationError("Virtual environment setup timed out")
        else:
            if not self.args.quiet:
                print("🔧 Using system Python environment")
            logger.info("Using system Python environment")
    
    def install_dependencies(self):
        """Install Python dependencies."""
        
        logger.info("Installing dependencies...")
        if not self.args.quiet:
            print("📦 Installing dependencies...")
        
        # Define dependency groups
        dependencies = self.get_dependency_groups()
        
        # Select dependencies based on installation type
        if self.args.minimal:
            packages_to_install = dependencies['core']
            install_type = "minimal"
        elif self.args.dev:
            packages_to_install = dependencies['core'] + dependencies['development']
            install_type = "development"
        else:
            packages_to_install = dependencies['core'] + dependencies['optional']
            install_type = "standard"
        
        if self.args.gpu:
            packages_to_install.extend(dependencies['gpu'])
            install_type += " + GPU"
        
        if not self.args.quiet:
            print(f"   Installing {install_type} dependencies...")
            print(f"   Total packages: {len(packages_to_install)}")
        
        # Install packages
        failed_packages = []
        for i, package in enumerate(packages_to_install, 1):
            try:
                if not self.args.quiet:
                    print(f"   [{i:2d}/{len(packages_to_install):2d}] Installing {package}...")
                
                subprocess.run([
                    self.python_executable, '-m', 'pip', 'install', package
                ] + (['--quiet'] if self.args.quiet else []), 
                check=True, timeout=300)
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {package}: {e}")
                failed_packages.append(package)
            except subprocess.TimeoutExpired:
                logger.warning(f"Installation of {package} timed out")
                failed_packages.append(package)
        
        # Handle failed packages
        if failed_packages:
            if len(failed_packages) > len(packages_to_install) * 0.5:
                raise InstallationError(
                    f"Too many package installation failures: {len(failed_packages)}/{len(packages_to_install)}"
                )
            else:
                if not self.args.quiet:
                    print(f"   ⚠ {len(failed_packages)} packages failed to install:")
                    for pkg in failed_packages:
                        print(f"     - {pkg}")
                logger.warning(f"Some packages failed to install: {failed_packages}")
        
        if not self.args.quiet:
            print(f"   ✓ Dependencies installed ({len(packages_to_install) - len(failed_packages)}/{len(packages_to_install)} successful)")
        
        self.requirements_installed = True
        logger.info("Dependencies installation completed")
    
    def get_dependency_groups(self) -> Dict[str, List[str]]:
        """Define dependency groups for different installation types."""
        
        return {
            'core': [
                'numpy>=1.21.0',
                'scipy>=1.7.0',
                'matplotlib>=3.5.0',
                'qutip>=4.6.0',
                'pyyaml>=6.0',
                'click>=8.0.0',
                'psutil>=5.8.0',
            ],
            'optional': [
                'tensorflow>=2.8.0',
                'scikit-learn>=1.0.0',
                'plotly>=5.0.0',
                'dash>=2.0.0',
                'pandas>=1.3.0',
                'seaborn>=0.11.0',
                'sympy>=1.9',
                'jupyterlab>=3.0.0',
            ],
            'development': [
                'pytest>=6.0.0',
                'pytest-cov>=2.12.0',
                'black>=21.0.0',
                'flake8>=4.0.0',
                'mypy>=0.910',
                'sphinx>=4.0.0',
                'pre-commit>=2.15.0',
                'tox>=3.24.0',
            ],
            'gpu': [
                'tensorflow-gpu>=2.8.0',
                'cupy-cuda11x>=9.0.0',  # Adjust CUDA version as needed
                'nvidia-ml-py3>=7.352.0',
            ]
        }
    
    def setup_project_structure(self):
        """Set up project directory structure."""
        
        logger.info("Setting up project structure...")
        if not self.args.quiet:
            print("📁 Setting up project structure...")
        
        # Define required directories
        directories = [
            'data',
            'data/simulations',
            'data/cache',
            'data/results',
            'logs',
            'config/presets',
            'examples/output',
            'tests/unit',
            'tests/integration',
            'docs/generated',
        ]
        
        # Create directories
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            if not self.args.quiet and not dir_path.exists():
                print(f"   Created: {directory}")
        
        # Create essential files
        self.create_essential_files()
        
        if not self.args.quiet:
            print("   ✓ Project structure ready")
        
        logger.info("Project structure setup completed")
    
    def create_essential_files(self):
        """Create essential configuration and setup files."""
        
        # Create .gitignore if it doesn't exist
        gitignore_path = self.project_root / '.gitignore'
        if not gitignore_path.exists():
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Project specific
data/simulations/
data/cache/
data/results/
logs/*.log
examples/output/
*.tmp

# OS
.DS_Store
Thumbs.db
"""
            gitignore_path.write_text(gitignore_content.strip())
        
        # Create environment template
        env_template_path = self.project_root / '.env.template'
        if not env_template_path.exists():
            env_content = """
# Quantum Wormhole Simulation Environment Configuration
# Copy this file to .env and modify as needed

# Computation settings
PYTHONPATH=.
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Memory settings
MEMORY_LIMIT_GB=16

# GPU settings (if available)
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/simulation.log

# Paths
DATA_DIR=data
CONFIG_DIR=config
OUTPUT_DIR=examples/output
"""
            env_template_path.write_text(env_content.strip())
        
        # Create setup.py if it doesn't exist
        setup_py_path = self.project_root / 'setup.py'
        if not setup_py_path.exists():
            setup_content = '''
from setuptools import setup, find_packages

setup(
    name="quantum-wormhole-simulation",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "qutip>=4.6.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
    ],
    python_requires=">=3.8",
    author="Quantum Wormhole Simulation Team",
    description="Advanced quantum wormhole simulation framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
'''
            setup_py_path.write_text(setup_content.strip())
    
    def configure_project(self):
        """Configure project settings and create runtime configuration."""
        
        logger.info("Configuring project...")
        if not self.args.quiet:
            print("⚙️  Configuring project...")
        
        # Create runtime configuration
        runtime_config = {
            'installation': {
                'timestamp': datetime.now().isoformat(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': platform.platform(),
                'installation_type': 'development' if self.args.dev else 'minimal' if self.args.minimal else 'standard',
                'virtual_environment': str(self.venv_path) if hasattr(self, 'venv_path') and self.venv_path.exists() else None,
                'gpu_support': self.args.gpu,
            },
            'paths': {
                'project_root': str(self.project_root),
                'data_dir': 'data',
                'config_dir': 'config',
                'logs_dir': 'logs',
                'output_dir': 'examples/output',
            },
            'compute': {
                'max_threads': os.cpu_count() or 4,
                'memory_limit_gb': self.config['recommended_memory_gb'],
            }
        }
        
        config_path = self.project_root / 'runtime_config.json'
        with open(config_path, 'w') as f:
            json.dump(runtime_config, f, indent=2)
        
        # Set up logging configuration
        self.setup_logging_config()
        
        if not self.args.quiet:
            print("   ✓ Project configured")
        
        logger.info("Project configuration completed")
    
    def setup_logging_config(self):
        """Set up logging configuration."""
        
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'level': 'INFO',
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard'
                },
                'file': {
                    'level': 'DEBUG',
                    'class': 'logging.FileHandler',
                    'filename': 'logs/simulation.log',
                    'formatter': 'detailed',
                    'mode': 'a'
                }
            },
            'loggers': {
                'quantum_wormhole': {
                    'handlers': ['console', 'file'],
                    'level': 'DEBUG',
                    'propagate': False
                }
            },
            'root': {
                'handlers': ['console', 'file'],
                'level': 'INFO'
            }
        }
        
        config_path = self.project_root / 'config' / 'logging_config.json'
        with open(config_path, 'w') as f:
            json.dump(log_config, f, indent=2)
    
    def verify_installation(self):
        """Verify that installation was successful."""
        
        logger.info("Verifying installation...")
        if not self.args.quiet:
            print("✅ Verifying installation...")
        
        verification_results = {}
        
        # Test Python imports
        core_modules = [
            'numpy', 'scipy', 'matplotlib', 'qutip', 'yaml', 'click'
        ]
        
        optional_modules = [
            'tensorflow', 'sklearn', 'plotly', 'dash', 'pandas', 'seaborn'
        ] if not self.args.minimal else []
        
        # Test core modules
        failed_core = []
        for module in core_modules:
            try:
                importlib.import_module(module)
                if not self.args.quiet:
                    print(f"   ✓ {module}")
            except ImportError:
                failed_core.append(module)
                if not self.args.quiet:
                    print(f"   ❌ {module}")
        
        verification_results['core_modules'] = {
            'total': len(core_modules),
            'passed': len(core_modules) - len(failed_core),
            'failed': failed_core
        }
        
        # Test optional modules
        failed_optional = []
        for module in optional_modules:
            try:
                importlib.import_module(module)
                if not self.args.quiet:
                    print(f"   ✓ {module}")
            except ImportError:
                failed_optional.append(module)
                if not self.args.quiet:
                    print(f"   ⚠ {module}")
        
        verification_results['optional_modules'] = {
            'total': len(optional_modules),
            'passed': len(optional_modules) - len(failed_optional),
            'failed': failed_optional
        }
        
        # Test project imports
        project_modules = [
            'src.integration',
            'src.physics.spacetime_metrics',
            'src.physics.exotic_matter',
            'src.quantum.wormhole_circuit',
            'src.ai.stability_predictor'
        ]
        
        failed_project = []
        for module in project_modules:
            try:
                spec = importlib.util.find_spec(module)
                if spec is not None:
                    if not self.args.quiet:
                        print(f"   ✓ {module}")
                else:
                    failed_project.append(module)
                    if not self.args.quiet:
                        print(f"   ⚠ {module} (module file not found)")
            except Exception as e:
                failed_project.append(module)
                if not self.args.quiet:
                    print(f"   ⚠ {module} (import error)")
        
        verification_results['project_modules'] = {
            'total': len(project_modules),
            'passed': len(project_modules) - len(failed_project),
            'failed': failed_project
        }
        
        # Check if core installation is functional
        if failed_core:
            raise InstallationError(
                f"Critical modules failed to import: {', '.join(failed_core)}"
            )
        
        # Save verification results
        results_path = self.project_root / 'installation_verification.json'
        verification_results['timestamp'] = datetime.now().isoformat()
        verification_results['success'] = len(failed_core) == 0
        
        with open(results_path, 'w') as f:
            json.dump(verification_results, f, indent=2)
        
        if not self.args.quiet:
            print("   ✓ Installation verification completed")
        
        logger.info("Installation verification completed")
        return verification_results
    
    def run_verification_tests(self):
        """Run comprehensive verification tests."""
        
        logger.info("Running verification tests...")
        if not self.args.quiet:
            print("🧪 Running verification tests...")
        
        test_results = {}
        
        # Test 1: Basic imports and functionality
        if not self.args.quiet:
            print("   Test 1: Basic functionality...")
        
        try:
            # Test numpy operations
            import numpy as np
            test_array = np.random.random((100, 100))
            result = np.linalg.svd(test_array)
            
            # Test scipy operations
            from scipy import optimize
            result = optimize.minimize(lambda x: x**2, 1.0)
            
            test_results['basic_functionality'] = True
            if not self.args.quiet:
                print("     ✓ Basic functionality test passed")
                
        except Exception as e:
            test_results['basic_functionality'] = False
            logger.error(f"Basic functionality test failed: {e}")
            if not self.args.quiet:
                print(f"     ❌ Basic functionality test failed: {e}")
        
        # Test 2: Quantum mechanics operations
        if not self.args.quiet:
            print("   Test 2: Quantum mechanics...")
        
        try:
            import qutip as qt
            
            # Create simple quantum state
            psi = qt.basis(2, 0)
            sigma_x = qt.sigmax()
            evolved = sigma_x * psi
            
            test_results['quantum_mechanics'] = True
            if not self.args.quiet:
                print("     ✓ Quantum mechanics test passed")
                
        except Exception as e:
            test_results['quantum_mechanics'] = False
            logger.error(f"Quantum mechanics test failed: {e}")
            if not self.args.quiet:
                print(f"     ❌ Quantum mechanics test failed: {e}")
        
        # Test 3: Configuration loading
        if not self.args.quiet:
            print("   Test 3: Configuration system...")
        
        try:
            import yaml
            
            # Test YAML configuration loading
            config_path = self.project_root / 'config' / 'simulation_config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                test_results['configuration'] = True
                if not self.args.quiet:
                    print("     ✓ Configuration system test passed")
            else:
                test_results['configuration'] = False
                if not self.args.quiet:
                    print("     ⚠ Configuration file not found")
                    
        except Exception as e:
            test_results['configuration'] = False
            logger.error(f"Configuration test failed: {e}")
            if not self.args.quiet:
                print(f"     ❌ Configuration test failed: {e}")
        
        # Test 4: Visualization capabilities
        if not self.args.quiet:
            print("   Test 4: Visualization...")
        
        try:
            import matplotlib.pyplot as plt
            
            # Test basic plotting
            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            plt.close(fig)
            
            test_results['visualization'] = True
            if not self.args.quiet:
                print("     ✓ Visualization test passed")
                
        except Exception as e:
            test_results['visualization'] = False
            logger.error(f"Visualization test failed: {e}")
            if not self.args.quiet:
                print(f"     ❌ Visualization test failed: {e}")
        
        # Test 5: File system operations
        if not self.args.quiet:
            print("   Test 5: File system operations...")
        
        try:
            # Test directory creation and file writing
            test_dir = self.project_root / 'data' / 'test'
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / 'test_file.txt'
            test_file.write_text("Installation test")
            
            # Clean up
            test_file.unlink()
            test_dir.rmdir()
            
            test_results['file_system'] = True
            if not self.args.quiet:
                print("     ✓ File system test passed")
                
        except Exception as e:
            test_results['file_system'] = False
            logger.error(f"File system test failed: {e}")
            if not self.args.quiet:
                print(f"     ❌ File system test failed: {e}")
        
        # Calculate overall success rate
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save test results
        results_path = self.project_root / 'verification_test_results.json'
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        if not self.args.quiet:
            print(f"   📊 Test results: {passed_tests}/{total_tests} passed ({success_rate:.1%})")
        
        if success_rate < 0.8:
            logger.warning("Some verification tests failed - installation may not be fully functional")
        
        logger.info(f"Verification tests completed: {passed_tests}/{total_tests} passed")
        return test_results
    
    def print_completion_message(self):
        """Print installation completion message."""
        
        if not self.args.quiet:
            print("\n" + "=" * 70)
            print("🎉 Installation completed successfully!")
            print("=" * 70)
            
            print("\n📋 Next steps:")
            print("   1. Activate virtual environment (if created):")
            
            if hasattr(self, 'venv_path') and self.venv_path.exists():
                if platform.system() == "Windows":
                    print(f"      {self.venv_path}\\Scripts\\activate")
                else:
                    print(f"      source {self.venv_path}/bin/activate")
            
            print("   2. Run your first simulation:")
            print("      python main.py --mode demo")
            print("   3. Try the interactive visualization:")
            print("      python examples/03_interactive_visualization.py")
            print("   4. Read the documentation:")
            print("      docs/user_guide.md")
            
            print("\n📁 Important files:")
            print(f"   - Project root: {self.project_root}")
            print(f"   - Configuration: config/simulation_config.yaml")
            print(f"   - Examples: examples/")
            print(f"   - Documentation: docs/")
            print(f"   - Installation log: installation.log")
            
            print("\n🆘 Support:")
            print("   - Check installation.log for detailed logs")
            print("   - Review docs/user_guide.md for troubleshooting")
            print("   - Run 'python install.py --test' to verify installation")
    
    def print_error_message(self, error_message: str):
        """Print installation error message."""
        
        print("\n" + "=" * 70)
        print("❌ Installation failed!")
        print("=" * 70)
        print(f"Error: {error_message}")
        
        print("\n🔧 Troubleshooting:")
        print("   1. Check installation.log for detailed error information")
        print("   2. Ensure you have Python 3.8+ and internet connectivity")
        print("   3. Try running with --force to reinstall all components")
        print("   4. For minimal installation, use --minimal flag")
        print("   5. Check system requirements in docs/user_guide.md")
        
        print("\n📞 Getting help:")
        print("   - Review the installation log: installation.log")
        print("   - Check the troubleshooting section in docs/user_guide.md")
        print("   - Report issues to the project repository")
    
    def remove_directory(self, path: Path):
        """Safely remove a directory and all its contents."""
        
        if path.exists() and path.is_dir():
            import shutil
            shutil.rmtree(path)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Install Quantum Wormhole Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                    # Standard installation
  python install.py --dev             # Development installation
  python install.py --minimal         # Minimal installation
  python install.py --gpu --test      # GPU installation with tests
  python install.py --force --quiet   # Force reinstall, minimal output
        """
    )
    
    parser.add_argument('--dev', action='store_true',
                       help='Install development dependencies')
    
    parser.add_argument('--minimal', action='store_true',
                       help='Minimal installation (core components only)')
    
    parser.add_argument('--gpu', action='store_true',
                       help='Install GPU acceleration support')
    
    parser.add_argument('--force', action='store_true',
                       help='Force reinstallation of all components')
    
    parser.add_argument('--test', action='store_true',
                       help='Run verification tests after installation')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    return parser


def main():
    """Main installation entry point."""
    
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create and run installer
    installer = QuantumWormholeInstaller(args)
    success = installer.run_installation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()