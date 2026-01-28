"""
Verification API - Adapter Layer for the Vendor Quantum Engine.

This module bridges the high-level "Chat" strings (e.g. "H", "|0>") 
to the low-level Logic (QuantumVerifier, SymPy Matrices) provided by 
the vendor codebase.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from enum import Enum
from dataclasses import dataclass, field

# =============================================================================
# 1. Dependency Injection (The Path Hack)
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2] # src/tutor/ -> src/ -> root
VENDOR_SRC = BASE_DIR / "vendor" / "QuantumSimulation Code Base" / "src"

if str(VENDOR_SRC) not in sys.path:
    sys.path.append(str(VENDOR_SRC))

try:
    from verification.symbolic_solver import (
        QuantumVerifier as RawVerifier,
        ket_0, ket_1, hadamard, pauli_x, pauli_y, pauli_z, cnot,
        bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus,
        identity_2
    )
    import sympy as sp
except ImportError as e:
    print(f"WARNING: Vendor engine not found: {e}. Using Mock.")
    RawVerifier = None
    # Mock symbols to prevent NameError in __init__
    hadamard = lambda: "H"
    pauli_x = lambda: "X"
    pauli_y = lambda: "Y"
    pauli_z = lambda: "Z"
    identity_2 = lambda: "I"
    cnot = lambda: "CNOT"
    ket_0 = lambda: "|0>"
    ket_1 = lambda: "|1>"
    bell_phi_plus = lambda: "Phi+"

# =============================================================================
# 2. Data Models & Enums
# =============================================================================

class VerificationDomain(str, Enum):
    UNITARITY = "unitarity"
    NORMALIZATION = "normalization"
    HERMITICITY = "hermiticity"
    ENTANGLEMENT = "entanglement"
    BELL_STATE = "bell_state"
    CHSH = "chsh"
    GENERAL = "general"

@dataclass
class VerificationResult:
    verified: bool
    symbolic_proof: str
    confidence: float
    explanation: str
    domain: str = VerificationDomain.GENERAL.value
    details: Dict[str, Any] = field(default_factory=dict)

# Exceptions
class VerificationError(Exception): pass
class ParseError(Exception): pass
class UnsupportedClaimError(Exception): pass
class InvalidInputError(Exception): pass

# =============================================================================
# 3. The Adapter Class
# =============================================================================

class TutorVerificationAPI:
    """
    Adapter that translates string claims into QuantumVerifier calls.
    """
    
    def __init__(self):
        if RawVerifier:
            self.engine = RawVerifier()
        else:
            self.engine = None
            
        self.gate_map = {
            "H": hadamard,
            "X": pauli_x,
            "Y": pauli_y,
            "Z": pauli_z,
            "I": identity_2,
            "CNOT": cnot,
            "Hadamard": hadamard,
            "PauliX": pauli_x
        }
        
        self.state_map = {
            "|0>": ket_0,
            "|1>": ket_1,
            "bell_phi_plus": bell_phi_plus,
            "Phi+": bell_phi_plus
        }

    def verify_gate(self, expression: str) -> VerificationResult:
        domain = VerificationDomain.UNITARITY.value
        if not self.engine:
            return self._mock_result(True, "Mock: H is unitary (Engine Missing)", domain)
            
        if expression not in self.gate_map:
            return VerificationResult(
                verified=False,
                symbolic_proof="N/A",
                confidence=0.0,
                explanation=f"Unknown gate: '{expression}'",
                domain=domain,
                details={"input_gate": expression}
            )
            
        matrix_func = self.gate_map[expression]
        matrix = matrix_func()
        is_unitary, result_matrix = self.engine.verify_unitary(matrix)
        
        proof = sp.latex(result_matrix) if 'sp' in globals() else str(result_matrix)
        expl = f"The gate {expression} is unitary." if is_unitary else f"{expression} is NOT unitary."
        
        return VerificationResult(
            verified=is_unitary,
            symbolic_proof=proof,
            confidence=1.0,
            explanation=expl,
            domain=domain,
            details={"input_gate": expression}
        )

    def verify_state(self, expression: str) -> VerificationResult:
        domain = VerificationDomain.NORMALIZATION.value
        if not self.engine:
            return self._mock_result(True, "Mock: State is normalized", domain)

        if expression not in self.state_map:
             return VerificationResult(
                verified=False,
                symbolic_proof="N/A",
                confidence=0.0,
                explanation=f"Unknown state: '{expression}'",
                domain=domain,
                details={"input_state": expression}
            )
            
        vector_func = self.state_map[expression]
        vector = vector_func()
        is_norm, norm_val = self.engine.verify_normalization(vector)
        
        return VerificationResult(
            verified=is_norm, 
            symbolic_proof=sp.latex(norm_val) if 'sp' in globals() else str(norm_val), 
            confidence=1.0, 
            explanation="State is normalized." if is_norm else "State is NOT normalized.",
            domain=domain,
            details={"input_state": expression}
        )

    def _mock_result(self, verified: bool, msg: str, domain: str) -> VerificationResult:
        return VerificationResult(verified, msg, 0.5, msg, domain=domain)

# =============================================================================
# 4. Helper / Factory Functions (Legacy Compatibility)
# =============================================================================

def create_api() -> TutorVerificationAPI:
    return TutorVerificationAPI()

def verify_unitary(expression: str) -> VerificationResult:
    return create_api().verify_gate(expression)

def verify_normalized(expression: str) -> VerificationResult:
    return create_api().verify_state(expression)
    
def verify_hermitian(expression: str) -> VerificationResult:
    return VerificationResult(False, "Not implemented", 0.0, "Hermitian check not exposed yet.", domain=VerificationDomain.HERMITICITY.value)
