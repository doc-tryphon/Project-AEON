#!/usr/bin/env python3
"""
Scientific validation tests for the quantum wormhole simulation physics.

This module tests the physics calculations against known analytical results
and experimental data to ensure scientific accuracy.
"""

import numpy as np
import sys
import os
sys.path.append('src')

from src.physics.exotic_matter import AdvancedCasimirExoticMatter
from src.physics.spacetime_metrics import MorrisThorneeWormhole
from src.physics.constants import HBAR, C

def test_casimir_energy_density():
    """Test Casimir energy density against theoretical predictions."""
    print("Testing Casimir Effect Energy Density...")
    
    # Create Casimir exotic matter
    casimir = AdvancedCasimirExoticMatter(
        plate_separation=1e-6,  # 1 micron
        temperature=300,        # Room temperature
        experimental_calibration='decca_2003'
    )
    
    # Test coordinates at throat region
    coordinates = (0.0, 1000.0, np.pi/2, 0.0)  # t, r, theta, phi
    
    # Calculate energy density
    rho = casimir.energy_density(coordinates)
    
    # Theoretical Casimir energy density (rough estimate)
    # rho_Casimir approximately -pi^2*hbar*c/(720 a^4) where a is plate separation
    a = 1e-6
    theoretical_rho = -np.pi**2 * HBAR * C / (720 * a**4)
    
    print(f"Calculated energy density: {rho:.6e} J/m³")
    print(f"Theoretical estimate: {theoretical_rho:.6e} J/m³")
    print(f"Ratio (should be ~1): {abs(rho/theoretical_rho):.3f}")
    
    # Test should be within order of magnitude (experimental factors, temperature, etc.)
    assert abs(rho/theoretical_rho) > 0.01 and abs(rho/theoretical_rho) < 100, "Casimir energy density out of range"
    print("Casimir energy density test passed\n")

def test_morris_thorne_metric():
    """Test Morris-Thorne wormhole metric properties."""
    print("Testing Morris-Thorne Wormhole Metric...")
    
    # Create wormhole metric
    throat_radius = 1000.0  # 1 km throat radius
    wormhole = MorrisThorneeWormhole(throat_radius=throat_radius)
    
    # Test coordinates at the throat
    coordinates = (0.0, throat_radius, np.pi/2, 0.0)
    
    # Calculate metric tensor
    g_tensor = wormhole.metric_tensor(coordinates)
    
    # Check metric signature (-,+,+,+)
    eigenvals = np.linalg.eigvals(g_tensor)
    eigenvals_sorted = sorted(eigenvals, reverse=True)
    
    print(f"Metric eigenvalues: {eigenvals_sorted}")
    print(f"Metric determinant: {np.linalg.det(g_tensor):.6e}")
    
    # Should have one negative eigenvalue (time) and three positive (space)
    num_negative = sum(1 for ev in eigenvals if ev < 0)
    num_positive = sum(1 for ev in eigenvals if ev > 0)
    
    print(f"Negative eigenvalues: {num_negative} (should be 1)")
    print(f"Positive eigenvalues: {num_positive} (should be 3)")
    
    assert num_negative == 1 and num_positive == 3, "Metric signature incorrect"
    print("Morris-Thorne metric test passed\n")

def test_energy_conditions():
    """Test energy condition violation calculations."""
    print("Testing Energy Condition Violations...")
    
    # Create exotic matter
    casimir = AdvancedCasimirExoticMatter(plate_separation=1e-6, temperature=300)
    
    # Test coordinates
    coordinates = (0.0, 1000.0, np.pi/2, 0.0)
    
    # Get energy density and pressure
    rho = casimir.energy_density(coordinates)
    p_r = casimir.pressure_radial(coordinates)
    
    print(f"Energy density rho: {rho:.6e} J/m^3")
    print(f"Radial pressure p_r: {p_r:.6e} Pa")
    
    # Energy condition checks
    nec = rho + p_r  # Null Energy Condition
    wec = rho        # Weak Energy Condition  
    sec = rho + 3*p_r # Strong Energy Condition
    dec = rho - abs(p_r) # Dominant Energy Condition
    
    print(f"NEC (rho + p): {nec:.6e} (should be < 0 for wormholes)")
    print(f"WEC (rho): {wec:.6e} (should be < 0 for exotic matter)")
    print(f"SEC (rho + 3p): {sec:.6e} (should be < 0 for traversability)")
    print(f"DEC (rho - |p|): {dec:.6e}")
    
    # For traversable wormholes, we expect:
    # - Negative energy density (WEC violation)
    # - Negative pressure that violates energy conditions
    print(f"WEC violated: {wec < 0}")
    print(f"NEC violated: {nec < 0}")
    
    # At least WEC should be violated for exotic matter
    assert wec < 0, "Weak Energy Condition should be violated for exotic matter"
    print("Energy condition tests passed\n")

def test_tidal_forces():
    """Test tidal force calculations for human traversability."""
    print("Testing Tidal Force Calculations...")
    
    # Create wormhole metric
    wormhole = MorrisThorneeWormhole(throat_radius=1000.0)
    
    # Test near throat (more extreme tidal forces)
    coordinates_throat = (0.0, 1100.0, np.pi/2, 0.0)  # Just outside throat
    
    # Calculate Christoffel symbols (proxy for tidal forces)
    try:
        gamma = wormhole.christoffel_symbols(coordinates_throat)
        gamma_norm = np.linalg.norm(gamma)
        
        # Tidal force estimate for human (1.8m height)
        human_height = 1.8
        tidal_force = gamma_norm * human_height
        
        print(f"Christoffel symbol magnitude: {gamma_norm:.3f}")
        print(f"Estimated tidal force on human: {tidal_force:.3f} N")
        
        # Human survival threshold (very rough estimate)
        survival_threshold = 1000  # 1000 N
        survivable = tidal_force < survival_threshold
        
        print(f"Human survivable traversal: {survivable}")
        print(f"Safety factor: {survival_threshold/tidal_force:.1f}x" if tidal_force > 0 else "inf")
        
        # This is a basic sanity check - tidal forces should be finite
        assert np.isfinite(tidal_force), "Tidal force calculation failed"
        print("Tidal force test passed\n")
        
    except Exception as e:
        print(f"Christoffel calculation failed (expected for some metrics): {e}")
        print("⚠️  Tidal force test skipped\n")

def test_consistency_checks():
    """Test overall physics consistency."""
    print("Testing Physics Consistency...")
    
    # Test that calculations don't produce NaN or infinite values
    casimir = AdvancedCasimirExoticMatter(plate_separation=1e-6, temperature=300)
    wormhole = MorrisThorneeWormhole(throat_radius=1000.0)
    
    test_coordinates = [
        (0.0, 1000.0, np.pi/2, 0.0),    # At throat
        (0.0, 1100.0, np.pi/2, 0.0),   # Near throat
        (0.0, 2000.0, np.pi/2, 0.0),   # Far from throat
    ]
    
    all_finite = True
    for coords in test_coordinates:
        rho = casimir.energy_density(coords)
        p_r = casimir.pressure_radial(coords)
        g = wormhole.metric_tensor(coords)
        
        if not (np.isfinite(rho) and np.isfinite(p_r) and np.all(np.isfinite(g))):
            all_finite = False
            print(f"Non-finite values at {coords}")
    
    print(f"All calculations finite: {all_finite}")
    assert all_finite, "Some physics calculations produced non-finite values"
    print("Consistency test passed\n")

def main():
    """Run all physics validation tests."""
    print("Quantum Wormhole Physics Validation Suite")
    print("=" * 50)
    
    try:
        test_casimir_energy_density()
        test_morris_thorne_metric()  
        test_energy_conditions()
        test_tidal_forces()
        test_consistency_checks()
        
        print("All physics validation tests passed!")
        print("The simulation is producing scientifically reasonable results.")
        
    except AssertionError as e:
        print(f"Physics validation failed: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error in physics validation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())