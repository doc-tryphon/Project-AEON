#!/usr/bin/env python3
"""
Installation Verification Script

Simple standalone script to verify that the Quantum Wormhole Simulation
Framework is properly installed and functioning.

Usage:
    python verify_installation.py [--verbose] [--quick]

Options:
    --verbose    Show detailed output
    --quick      Run only essential tests
"""

import sys
import os
import importlib
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        print(f"✓ Python {current_version[0]}.{current_version[1]} (compatible)")
        return True
    else:
        print(f"❌ Python {current_version[0]}.{current_version[1]} (requires {min_version[0]}.{min_version[1]}+)")
        return False


def test_import(module_name, package_name=None, verbose=False):
    """Test if a module can be imported."""
    display_name = package_name or module_name
    
    try:
        importlib.import_module(module_name)
        if verbose:
            print(f"✓ {display_name}")
        return True
    except ImportError as e:
        if verbose:
            print(f"❌ {display_name} - {str(e)}")
        else:
            print(f"❌ {display_name}")
        return False


def test_basic_functionality(verbose=False):
    """Test basic functionality of core libraries."""
    
    if verbose:
        print("\n🧪 Testing basic functionality...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test NumPy
    total_tests += 1
    try:
        import numpy as np
        test_array = np.array([1, 2, 3, 4, 5])
        result = np.sum(test_array)
        assert result == 15
        if verbose:
            print("✓ NumPy mathematical operations")
        tests_passed += 1
    except Exception as e:
        if verbose:
            print(f"❌ NumPy mathematical operations - {e}")
    
    # Test SciPy
    total_tests += 1
    try:
        from scipy import optimize
        result = optimize.minimize(lambda x: x**2, 1.0)
        assert result.success
        if verbose:
            print("✓ SciPy optimization")
        tests_passed += 1
    except Exception as e:
        if verbose:
            print(f"❌ SciPy optimization - {e}")
    
    # Test Matplotlib
    total_tests += 1
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close(fig)
        if verbose:
            print("✓ Matplotlib plotting")
        tests_passed += 1
    except Exception as e:
        if verbose:
            print(f"❌ Matplotlib plotting - {e}")
    
    # Test QuTiP
    total_tests += 1
    try:
        import qutip as qt
        state = qt.basis(2, 0)
        sigma_x = qt.sigmax()
        result = sigma_x * state
        if verbose:
            print("✓ QuTiP quantum operations")
        tests_passed += 1
    except Exception as e:
        if verbose:
            print(f"❌ QuTiP quantum operations - {e}")
    
    return tests_passed, total_tests


def test_project_structure(verbose=False):
    """Test if project structure exists."""
    
    if verbose:
        print("\n📁 Checking project structure...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        'src',
        'config',
        'examples',
        'docs',
        'data'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'config/simulation_config.yaml',
        'examples/01_basic_wormhole.py'
    ]
    
    structure_ok = True
    
    # Check directories
    for dirname in required_dirs:
        dir_path = project_root / dirname
        if dir_path.exists():
            if verbose:
                print(f"✓ Directory: {dirname}")
        else:
            print(f"❌ Missing directory: {dirname}")
            structure_ok = False
    
    # Check files
    for filename in required_files:
        file_path = project_root / filename
        if file_path.exists():
            if verbose:
                print(f"✓ File: {filename}")
        else:
            print(f"❌ Missing file: {filename}")
            structure_ok = False
    
    return structure_ok


def test_project_imports(verbose=False):
    """Test if project modules can be imported."""
    
    if verbose:
        print("\n📦 Testing project imports...")
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    project_modules = [
        ('src.integration', 'Integration Framework'),
        ('src.physics.spacetime_metrics', 'Spacetime Metrics'),
        ('src.physics.exotic_matter', 'Exotic Matter'),
        ('src.quantum.wormhole_circuit', 'Quantum Circuit'),
        ('src.ai.stability_predictor', 'AI Stability Predictor')
    ]
    
    imports_ok = 0
    total_modules = len(project_modules)
    
    for module_name, display_name in project_modules:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                if verbose:
                    print(f"✓ {display_name}")
                imports_ok += 1
            else:
                if verbose:
                    print(f"⚠ {display_name} (module file not found)")
        except Exception as e:
            if verbose:
                print(f"⚠ {display_name} (import error)")
    
    return imports_ok, total_modules


def run_verification(verbose=False, quick=False):
    """Run complete verification process."""
    
    print("🌌 Quantum Wormhole Simulation - Installation Verification")
    print("=" * 60)
    
    overall_success = True
    
    # Check Python version
    if not check_python_version():
        overall_success = False
        return False
    
    # Test core module imports
    print("\n📚 Checking core dependencies...")
    
    core_modules = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'), 
        ('matplotlib', 'Matplotlib'),
        ('qutip', 'QuTiP'),
        ('yaml', 'PyYAML'),
        ('click', 'Click'),
    ]
    
    optional_modules = [
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'scikit-learn'),
        ('plotly', 'Plotly'),
        ('dash', 'Dash'),
        ('pandas', 'Pandas'),
        ('seaborn', 'Seaborn'),
    ] if not quick else []
    
    core_failed = 0
    for module, name in core_modules:
        if not test_import(module, name, verbose):
            core_failed += 1
    
    if core_failed > 0:
        print(f"\n❌ {core_failed} core dependencies missing")
        overall_success = False
    else:
        print("\n✓ All core dependencies available")
    
    # Test optional modules if not quick mode
    if not quick and optional_modules:
        print("\n📦 Checking optional dependencies...")
        optional_failed = 0
        for module, name in optional_modules:
            if not test_import(module, name, verbose):
                optional_failed += 1
        
        if optional_failed > 0:
            print(f"\n⚠ {optional_failed} optional dependencies missing")
        else:
            print("\n✓ All optional dependencies available")
    
    # Test basic functionality
    if not quick:
        func_passed, func_total = test_basic_functionality(verbose)
        if func_passed < func_total:
            print(f"\n⚠ Basic functionality: {func_passed}/{func_total} tests passed")
            if func_passed < func_total * 0.75:
                overall_success = False
        else:
            print(f"\n✓ All basic functionality tests passed ({func_passed}/{func_total})")
    
    # Test project structure
    if not test_project_structure(verbose):
        print("\n⚠ Project structure incomplete")
        # Don't fail for structure issues
    else:
        print("\n✓ Project structure complete")
    
    # Test project imports
    if not quick:
        imports_ok, total_imports = test_project_imports(verbose)
        if imports_ok < total_imports:
            print(f"\n⚠ Project imports: {imports_ok}/{total_imports} modules available")
        else:
            print(f"\n✓ All project modules available ({imports_ok}/{total_imports})")
    
    # Final result
    print("\n" + "=" * 60)
    if overall_success:
        print("🎉 Installation verification PASSED")
        print("\n✅ The framework appears to be correctly installed and ready to use.")
        print("\n📋 Next steps:")
        print("   • Run: python main.py --mode demo")
        print("   • Try: python examples/01_basic_wormhole.py")
        print("   • Read: docs/user_guide.md")
    else:
        print("❌ Installation verification FAILED")
        print("\n🔧 Issues detected with the installation.")
        print("\n💡 Solutions:")
        print("   • Run: python install.py --force")
        print("   • Check: installation.log")
        print("   • Review: docs/user_guide.md (troubleshooting)")
    
    print("=" * 60)
    
    return overall_success


def main():
    """Main verification entry point."""
    
    parser = argparse.ArgumentParser(
        description="Verify Quantum Wormhole Simulation installation"
    )
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--quick', action='store_true',
                       help='Run only essential tests')
    
    args = parser.parse_args()
    
    success = run_verification(verbose=args.verbose, quick=args.quick)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()