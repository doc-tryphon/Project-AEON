"""
Quantum Module.

Provides implementations of quantum protocols and error correction codes.
"""

from .surface_codes import (
    SurfaceCode,
    SurfaceCodeLattice,
    SurfaceCodeDecoder,
    SurfaceCodeResult,
    Stabilizer,
    StabilizerType,
    create_x_stabilizer,
    create_z_stabilizer,
    get_stabilizer_generators,
    apply_error,
    measure_syndrome,
    correct_errors,
)

__all__ = [
    'SurfaceCode',
    'SurfaceCodeLattice',
    'SurfaceCodeDecoder',
    'SurfaceCodeResult',
    'Stabilizer',
    'StabilizerType',
    'create_x_stabilizer',
    'create_z_stabilizer',
    'get_stabilizer_generators',
    'apply_error',
    'measure_syndrome',
    'correct_errors',
]
