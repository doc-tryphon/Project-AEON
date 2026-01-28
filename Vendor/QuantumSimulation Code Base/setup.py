from setuptools import setup, find_packages

setup(
    name="quantum-computing-framework",
    version="1.0.0",
    description="Research-grade quantum computing framework with verified protocols",
    author="Quantum Research Team",
    author_email="quantum@research.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "sympy>=1.8",
        "qutip>=4.7.0",
        "tensorflow>=2.8.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "pytest>=6.2.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)