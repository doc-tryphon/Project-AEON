"""Test imports for verified quantum computing framework."""

def test_imports():
    # Basic Python libraries
    import numpy as np
    import scipy
    import qutip
    print("✓ Basic scientific libraries")

    # Verified quantum modules (Steps 1-5)
    from src.quantum.entanglement import BellStateGenerator
    from src.quantum.teleportation import QuantumTeleportation
    from src.quantum.superdense_coding import SuperdenseCoding
    from src.quantum.decoherence import DensityMatrix, DecoherenceChannel
    from src.quantum.error_correction import BitFlipCode
    from src.quantum.gates import CNOTGate, ToffoliGate
    print("✓ Verified quantum modules (Steps 1-5)")

    # Configuration
    from src.config import QuantumConfig, get_default_config
    print("✓ Configuration module")

    # Verification framework
    from src.verification.symbolic_solver import SymbolicVerifier
    print("✓ SymPy verification framework")

    return True

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    try:
        test_imports()
        print("\n✓ All imports successful!")
    except Exception as e:
        print(f"\n❌ Import error: {str(e)}")
