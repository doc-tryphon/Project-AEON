#!/bin/bash
# Quantum Wormhole Simulation Framework - Unix/Linux/macOS Setup Script
# This script provides an easy way to install the framework on Unix-like systems

set -e  # Exit on error

echo
echo "========================================================="
echo " Quantum Wormhole Simulation Framework - Setup"
echo "========================================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python 3.8+ using your system package manager"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
        echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "  macOS: brew install python"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo "Python $PYTHON_VERSION found. Starting installation..."
else
    echo "ERROR: Python $PYTHON_VERSION found, but version $REQUIRED_VERSION or higher is required"
    exit 1
fi

echo

# Parse command line arguments
INSTALL_TYPE="standard"
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_TYPE="development"
            EXTRA_FLAGS="$EXTRA_FLAGS --dev"
            shift
            ;;
        --minimal)
            INSTALL_TYPE="minimal"
            EXTRA_FLAGS="$EXTRA_FLAGS --minimal"
            shift
            ;;
        --gpu)
            INSTALL_TYPE="$INSTALL_TYPE with GPU"
            EXTRA_FLAGS="$EXTRA_FLAGS --gpu"
            shift
            ;;
        --test)
            EXTRA_FLAGS="$EXTRA_FLAGS --test"
            shift
            ;;
        --force)
            EXTRA_FLAGS="$EXTRA_FLAGS --force"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo
            echo "Options:"
            echo "  --dev      Install development dependencies"
            echo "  --minimal  Minimal installation (core only)"
            echo "  --gpu      Install GPU acceleration support"
            echo "  --test     Run verification tests after install"
            echo "  --force    Force reinstall all components"
            echo "  --help     Show this help message"
            echo
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Installing $INSTALL_TYPE version..."
echo

# Make install script executable
chmod +x install.py 2>/dev/null || true

# Run the Python installer
$PYTHON_CMD install.py $EXTRA_FLAGS

echo
echo "========================================================="
echo " Installation completed successfully!"
echo "========================================================="
echo
echo "To get started:"
echo "  1. Run a demo simulation: $PYTHON_CMD main.py --mode demo"
echo "  2. Try interactive mode: $PYTHON_CMD examples/03_interactive_visualization.py"
echo "  3. Read the user guide: docs/user_guide.md"
echo
echo "For verification: $PYTHON_CMD verify_installation.py"
echo
echo "If you created a virtual environment, activate it with:"
echo "  source venv/bin/activate"
echo