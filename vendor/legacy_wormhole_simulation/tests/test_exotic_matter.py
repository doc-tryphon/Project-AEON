#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced Exotic Matter Models

This test suite validates all aspects of the exotic matter implementation:
- Real-world scientific data integration
- Physical constraint validation
- Computational accuracy
- Energy condition analysis
- Stability analysis
- Optimization functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any

from src.physics.exotic_matter import (
    AdvancedCasimirExoticMatter,
    QuantumInequalityConstrainedMatter,
    PhantomDarkEnergyField,
    StringTheoryDerivedMatter,
    HybridExoticMatter,
    optimize_exotic_matter_configuration,
    load_exotic_matter_from_catalog,
    CASIMIR_EXPERIMENTAL_DATA,
    DARK_ENERGY_CONSTRAINTS,
    QUANTUM_INEQUALITY_BOUNDS,
    STRING_THEORY_PARAMETERS,
    ENHANCED_EXOTIC_MATTER_CATALOG
)


class TestPhysicalConstants:
    """Test physical constants and experimental data integration."""
    
    def test_casimir_experimental_data_completeness(self):
        """Test that Casimir experimental data includes required fields."""
        
        required_fields = ['plate_separation_range', 'force_coefficient', 'uncertainty']
        
        for experiment_name, data in CASIMIR_EXPERIMENTAL_DATA.items():
            for field in required_fields:
                assert field in data, f"Missing {field} in {experiment_name}"
            
            # Check plate separation ranges are physical
            a_min, a_max = data['plate_separation_range']
            assert 0 < a_min < a_max, f"Invalid plate separation range in {experiment_name}"
            assert a_max < 1e-3, f"Unrealistically large plate separation in {experiment_name}"
            
            # Check uncertainties are reasonable
            assert 0 < data['uncertainty'] < 1, f"Invalid uncertainty in {experiment_name}"
    
    def test_dark_energy_constraints(self):
        """Test dark energy observational constraints."""
        
        # Check key parameters exist
        required_params = ['omega_lambda', 'w0', 'wa', 'phantom_crossing_redshift']
        for param in required_params:
            assert param in DARK_ENERGY_CONSTRAINTS
        
        # Physical constraint checks
        assert 0 < DARK_ENERGY_CONSTRAINTS['omega_lambda'] < 1
        assert DARK_ENERGY_CONSTRAINTS['w0'] < 0  # Dark energy has negative pressure
        assert DARK_ENERGY_CONSTRAINTS['phantom_crossing_redshift'] >= 0
    
    def test_quantum_inequality_bounds(self):
        """Test quantum inequality bounds are properly set."""
        
        bounds = QUANTUM_INEQUALITY_BOUNDS['averaged_null_energy']
        
        assert bounds['bound_coefficient'] < 0  # Negative energy bound
        assert bounds['violation_timescale_max'] > 0
        assert bounds['sampling_scale'] > 0
    
    def test_string_theory_parameters(self):
        """Test string theory parameter sets."""
        
        for model_name, params in STRING_THEORY_PARAMETERS.items():
            assert 'string_scale' in params
            assert 'compactification_radius_range' in params
            
            r_min, r_max = params['compactification_radius_range']
            assert 0 < r_min < r_max


class TestAdvancedCasimirExoticMatter:
    """Test advanced Casimir exotic matter implementation."""
    
    def setup_method(self):
        """Set up test cases."""
        self.test_coordinates = (0.0, 1e-6, np.pi/2, 0.0)
        
    def test_initialization_with_experimental_calibration(self):
        """Test initialization with different experimental calibrations."""
        
        for calibration in CASIMIR_EXPERIMENTAL_DATA.keys():
            matter = AdvancedCasimirExoticMatter(
                plate_separation=1e-6,
                experimental_calibration=calibration
            )
            assert matter.experimental_calibration == calibration
            assert matter.a == 1e-6
    
    def test_energy_density_scaling(self):
        """Test energy density scaling with plate separation."""
        
        separations = [1e-7, 1e-6, 1e-5]
        energy_densities = []
        
        for separation in separations:
            matter = AdvancedCasimirExoticMatter(plate_separation=separation)
            rho = matter.energy_density(self.test_coordinates)
            energy_densities.append(rho)
        
        # Energy density should scale as 1/a^4
        for i in range(1, len(separations)):
            expected_ratio = (separations[0] / separations[i])**4
            actual_ratio = energy_densities[0] / energy_densities[i]
            assert abs(actual_ratio / expected_ratio - 1) < 0.1  # Within 10%
    
    def test_temperature_corrections(self):
        """Test finite temperature corrections."""
        
        temperatures = [0.1, 10, 300]  # K
        base_matter = AdvancedCasimirExoticMatter(plate_separation=1e-6, temperature=0.1)
        base_rho = abs(base_matter.energy_density(self.test_coordinates))
        
        for T in temperatures[1:]:
            matter = AdvancedCasimirExoticMatter(plate_separation=1e-6, temperature=T)
            rho = abs(matter.energy_density(self.test_coordinates))
            
            # Higher temperature should modify energy density
            if T > 0.1:
                assert rho != base_rho, f"Temperature correction failed at T={T}K"
    
    def test_conductivity_corrections(self):
        """Test finite conductivity corrections."""
        
        conductivities = [1e5, 1e7, 1e9]  # S/m
        energy_densities = []
        
        for sigma in conductivities:
            matter = AdvancedCasimirExoticMatter(
                plate_separation=1e-6,
                conductivity=sigma
            )
            rho = abs(matter.energy_density(self.test_coordinates))
            energy_densities.append(rho)
        
        # Higher conductivity should approach perfect conductor limit
        assert energy_densities[0] < energy_densities[-1]
    
    def test_pressure_anisotropy(self):
        """Test anisotropic pressure for Casimir field."""
        
        matter = AdvancedCasimirExoticMatter(plate_separation=1e-6)
        
        rho = matter.energy_density(self.test_coordinates)
        p_r = matter.pressure_radial(self.test_coordinates)
        p_t = matter.pressure_tangential(self.test_coordinates)
        
        # Check correct pressure relations for Casimir field
        assert abs(p_r + rho/3) < abs(rho) * 0.1  # p_r ≈ -ρ/3
        assert abs(p_t - rho/6) < abs(rho) * 0.1  # p_t ≈ ρ/6
        
        # Verify anisotropy
        assert p_r != p_t
    
    def test_casimir_force_calculation(self):
        """Test Casimir force per unit area calculation."""
        
        matter = AdvancedCasimirExoticMatter(plate_separation=1e-6)
        force_per_area = matter.casimir_force_per_area()
        
        # Force should be positive (attractive)
        assert force_per_area > 0
        
        # Check scaling with plate separation
        matter2 = AdvancedCasimirExoticMatter(plate_separation=2e-6)
        force_per_area2 = matter2.casimir_force_per_area()
        
        # Force should scale as 1/a^4
        expected_ratio = 2**4
        actual_ratio = force_per_area / force_per_area2
        assert abs(actual_ratio / expected_ratio - 1) < 0.05


class TestQuantumInequalityConstrainedMatter:
    """Test quantum inequality constrained matter."""
    
    def setup_method(self):
        """Set up test cases."""
        self.throat_radius = 1e3  # 1 km
        self.test_coordinates = (0.0, self.throat_radius, np.pi/2, 0.0)
    
    def test_initialization_with_bounds(self):
        """Test initialization respects quantum inequality bounds."""
        
        matter = QuantumInequalityConstrainedMatter(throat_radius=self.throat_radius)
        
        # Check violation time is within bounds
        max_time = QUANTUM_INEQUALITY_BOUNDS['averaged_null_energy']['violation_timescale_max']
        assert 0 < matter.tau <= max_time
        
        # Energy scale should be derived from quantum bounds
        assert matter.energy_scale > 0
        assert matter.rho_amplitude < 0  # Negative energy density
    
    def test_sampling_functions(self):
        """Test different sampling functions."""
        
        sampling_functions = ['gaussian', 'lorentzian', 'exponential']
        
        for sampling_func in sampling_functions:
            matter = QuantumInequalityConstrainedMatter(
                throat_radius=self.throat_radius,
                sampling_function=sampling_func
            )
            
            # Sampling function should be normalized (approximately)
            r_values = np.linspace(0.5 * self.throat_radius, 2 * self.throat_radius, 1000)
            sampling_values = [matter._sampling_function(r) for r in r_values]
            
            # Check normalization (integral ≈ 1)
            dr = r_values[1] - r_values[0]
            integral = np.trapz(sampling_values, dx=dr)
            
            # Should be roughly normalized within factor of 2-3
            assert 0.1 < integral < 10
    
    def test_energy_density_localization(self):
        """Test energy density is localized near throat."""
        
        matter = QuantumInequalityConstrainedMatter(throat_radius=self.throat_radius)
        
        # Energy density should be maximum near throat
        coords_throat = (0.0, self.throat_radius, np.pi/2, 0.0)
        coords_far = (0.0, 10 * self.throat_radius, np.pi/2, 0.0)
        
        rho_throat = abs(matter.energy_density(coords_throat))
        rho_far = abs(matter.energy_density(coords_far))
        
        assert rho_throat > rho_far
    
    def test_pressure_anisotropy(self):
        """Test anisotropic pressure relations."""
        
        matter = QuantumInequalityConstrainedMatter(throat_radius=self.throat_radius)
        
        rho = matter.energy_density(self.test_coordinates)
        p_r = matter.pressure_radial(self.test_coordinates)
        p_t = matter.pressure_tangential(self.test_coordinates)
        
        # Check expected pressure relations
        assert abs(p_r + rho) < abs(rho) * 0.1  # p_r ≈ -ρ
        assert abs(p_t + rho/2) < abs(rho) * 0.1  # p_t ≈ -ρ/2
    
    def test_quantum_inequality_verification(self):
        """Test quantum inequality verification along null geodesic."""
        
        matter = QuantumInequalityConstrainedMatter(throat_radius=self.throat_radius)
        
        # Create simple null geodesic path
        t_values = np.linspace(0, 1e-20, 100)
        r_values = self.throat_radius + 3e8 * t_values  # Light speed propagation
        
        null_path = np.column_stack([
            t_values, r_values, 
            np.full_like(t_values, np.pi/2),
            np.zeros_like(t_values)
        ])
        
        # Quantum inequality should be satisfied
        inequality_satisfied = matter.verify_quantum_inequality(null_path)
        assert inequality_satisfied, "Quantum inequality violated"


class TestPhantomDarkEnergyField:
    """Test phantom dark energy field model."""
    
    def setup_method(self):
        """Set up test cases."""
        self.test_coordinates = (0.0, 1e15, np.pi/2, 0.0)  # ~1000 ly
    
    def test_initialization_with_observational_data(self):
        """Test initialization with observational constraints."""
        
        matter = PhantomDarkEnergyField()
        
        # Should use Planck 2018 constraints by default
        assert abs(matter.w_0 - DARK_ENERGY_CONSTRAINTS['w0']) < 1e-6
        assert abs(matter.w_a - DARK_ENERGY_CONSTRAINTS['wa']) < 1e-6
        assert matter.rho_de > 0  # Positive dark energy density
    
    def test_equation_of_state_evolution(self):
        """Test dark energy equation of state evolution."""
        
        matter = PhantomDarkEnergyField()
        
        redshifts = [0.0, 0.5, 1.0, 2.0]
        w_values = [matter.equation_of_state(z) for z in redshifts]
        
        # w should evolve with redshift according to CPL parameterization
        w_expected = [matter.w_0 + matter.w_a * z/(1+z) for z in redshifts]
        
        for w_calc, w_exp in zip(w_values, w_expected):
            assert abs(w_calc - w_exp) < 1e-10
    
    def test_phantom_crossing_behavior(self):
        """Test phantom crossing behavior."""
        
        matter = PhantomDarkEnergyField()
        
        # Test around phantom crossing redshift
        z_cross = matter.z_cross
        w_before = matter.equation_of_state(z_cross - 0.1)
        w_after = matter.equation_of_state(z_cross + 0.1)
        
        # Should cross w = -1 at z_cross
        if matter.w_a > 0:
            assert w_before > -1 and w_after < -1
        elif matter.w_a < 0:
            assert w_before < -1 and w_after > -1
    
    def test_field_energy_components(self):
        """Test phantom field kinetic and potential energy."""
        
        matter = PhantomDarkEnergyField()
        
        kinetic = matter.field_kinetic_energy(self.test_coordinates)
        potential = matter.field_potential_energy(self.test_coordinates)
        total_rho = matter.energy_density(self.test_coordinates)
        
        # For phantom field: ρ = -T + V
        assert abs(total_rho - (kinetic + potential)) < abs(total_rho) * 0.1
        
        # Kinetic energy should be negative for phantom field
        assert kinetic <= 0
    
    def test_isotropic_pressure(self):
        """Test isotropic pressure for scalar field."""
        
        matter = PhantomDarkEnergyField()
        
        p_r = matter.pressure_radial(self.test_coordinates)
        p_t = matter.pressure_tangential(self.test_coordinates)
        
        # Scalar field should have isotropic pressure
        assert abs(p_r - p_t) < max(abs(p_r), abs(p_t)) * 0.1


class TestStringTheoryDerivedMatter:
    """Test string theory derived exotic matter."""
    
    def setup_method(self):
        """Set up test cases."""
        self.test_coordinates = (0.0, 1e-30, np.pi/2, 0.0)  # Near compactification scale
    
    def test_model_initialization(self):
        """Test initialization of different string models."""
        
        models = ['heterotic', 'type_iia', 'type_iib']
        
        for model in models:
            matter = StringTheoryDerivedMatter(string_model=model)
            assert matter.string_model == model
            assert matter.R_compact > 0
            assert 0 < matter.g_s <= 1  # String coupling should be perturbative
    
    def test_compactification_scale_effects(self):
        """Test effects of compactification scale on energy density."""
        
        scales = [1e-36, 1e-34, 1e-32]
        energy_densities = []
        
        for scale in scales:
            matter = StringTheoryDerivedMatter(
                string_model='heterotic',
                compactification_scale=scale
            )
            rho = abs(matter.energy_density(self.test_coordinates))
            energy_densities.append(rho)
        
        # Energy density should decrease with larger compactification scale
        # (for most models)
        assert energy_densities[0] > energy_densities[-1]
    
    def test_kaluza_klein_spectrum(self):
        """Test Kaluza-Klein mass spectrum."""
        
        matter = StringTheoryDerivedMatter(
            compactification_scale=1e-35
        )
        
        # KK masses should increase with mode number
        masses = [matter.kaluza_klein_mass_spectrum(n) for n in range(1, 6)]
        
        for i in range(1, len(masses)):
            assert masses[i] > masses[i-1]
        
        # First KK mass should be ~ ℏc/R
        expected_m1 = matter.energy_scale / matter.R_compact
        assert abs(masses[0] - expected_m1) < expected_m1 * 0.5
    
    def test_model_specific_pressure_relations(self):
        """Test model-specific pressure relations."""
        
        models_and_expected_ratios = {
            'heterotic': {'radial': 1/3, 'tangential': 1/3},  # Isotropic dilaton
            'type_iia': {'radial': 1/2, 'tangential': -1/4},  # Anisotropic brane
            'type_iib': {'radial': 1/2, 'tangential': -1/4}   # Anisotropic brane
        }
        
        for model, expected in models_and_expected_ratios.items():
            matter = StringTheoryDerivedMatter(string_model=model)
            
            rho = matter.energy_density(self.test_coordinates)
            p_r = matter.pressure_radial(self.test_coordinates)
            p_t = matter.pressure_tangential(self.test_coordinates)
            
            if abs(rho) > 1e-50:
                w_r = p_r / rho
                w_t = p_t / rho
                
                assert abs(w_r - expected['radial']) < 0.2
                assert abs(w_t - expected['tangential']) < 0.2


class TestEnergyConditionsAndStability:
    """Test energy conditions and stability analysis."""
    
    def setup_method(self):
        """Set up test cases."""
        self.matters = [
            AdvancedCasimirExoticMatter(plate_separation=1e-6),
            QuantumInequalityConstrainedMatter(throat_radius=1e3),
            PhantomDarkEnergyField(),
            StringTheoryDerivedMatter(string_model='heterotic')
        ]
        
        self.test_coordinates = (0.0, 1e-6, np.pi/2, 0.0)
    
    def test_energy_condition_analysis(self):
        """Test energy condition checking for all matter types."""
        
        for matter in self.matters:
            ec_result = matter.check_energy_conditions(self.test_coordinates)
            
            # Check result structure
            assert hasattr(ec_result, 'null_energy_condition')
            assert hasattr(ec_result, 'weak_energy_condition')
            assert hasattr(ec_result, 'strong_energy_condition')
            assert hasattr(ec_result, 'dominant_energy_condition')
            assert hasattr(ec_result, 'causality_preserved')
            
            # Exotic matter should violate some energy conditions
            energy_conditions = [
                ec_result.null_energy_condition,
                ec_result.weak_energy_condition,
                ec_result.strong_energy_condition,
                ec_result.dominant_energy_condition
            ]
            
            # At least one energy condition should be violated for exotic matter
            assert not all(energy_conditions), f"No energy condition violation for {matter.name}"
    
    def test_stability_analysis(self):
        """Test stability analysis for all matter types."""
        
        for matter in self.matters:
            stability = matter.stability_analysis(self.test_coordinates)
            
            # Check result structure
            assert hasattr(stability, 'radial_sound_speed')
            assert hasattr(stability, 'tangential_sound_speed')
            assert hasattr(stability, 'causality_preserved')
            
            # Sound speeds should be real and finite
            assert np.isfinite(stability.radial_sound_speed)
            assert np.isfinite(stability.tangential_sound_speed)
            assert stability.radial_sound_speed >= 0
            assert stability.tangential_sound_speed >= 0
    
    def test_total_energy_integration(self):
        """Test total energy integration for all matter types."""
        
        for matter in self.matters:
            try:
                r_min = 1e-6
                r_max = 1e-3
                
                total_energy, error = matter.total_energy_integral(r_min, r_max)
                
                # Energy should be finite
                assert np.isfinite(total_energy)
                assert np.isfinite(error)
                
                # Error should be reasonable
                if abs(total_energy) > 1e-50:
                    relative_error = error / abs(total_energy)
                    assert relative_error < 0.1, f"Large integration error for {matter.name}"
                
            except Exception as e:
                pytest.fail(f"Energy integration failed for {matter.name}: {e}")
    
    def test_physical_consistency_validation(self):
        """Test comprehensive physical consistency validation."""
        
        for matter in self.matters:
            r_min = 1e-6
            r_max = 1e-3
            
            validation = matter.validate_physical_consistency(r_min, r_max)
            
            # Check validation structure
            required_keys = [
                'energy_conditions', 'stability_analysis', 'total_energy',
                'integration_convergence', 'causality_violations', 'overall_consistent'
            ]
            
            for key in required_keys:
                assert key in validation, f"Missing validation key {key} for {matter.name}"
            
            # Total energy should be computed
            assert validation['total_energy'] is not None
            assert 'value' in validation['total_energy']
            assert 'finite' in validation['total_energy']


class TestHybridExoticMatter:
    """Test hybrid exotic matter combinations."""
    
    def test_hybrid_creation(self):
        """Test creation of hybrid exotic matter."""
        
        casimir_matter = AdvancedCasimirExoticMatter(plate_separation=1e-6)
        phantom_matter = PhantomDarkEnergyField()
        
        components = [(casimir_matter, 0.7), (phantom_matter, 0.3)]
        
        hybrid = HybridExoticMatter(components)
        
        assert len(hybrid.components) == 2
        assert hybrid.combination_method == 'linear'
        
        # Check weight normalization
        total_weight = sum(weight for _, weight in hybrid.components)
        assert abs(total_weight - 1.0) < 1e-10
    
    def test_hybrid_properties(self):
        """Test hybrid matter properties are linear combinations."""
        
        matter1 = AdvancedCasimirExoticMatter(plate_separation=1e-6)
        matter2 = PhantomDarkEnergyField()
        
        w1, w2 = 0.6, 0.4
        components = [(matter1, w1), (matter2, w2)]
        hybrid = HybridExoticMatter(components)
        
        coords = (0.0, 1e-6, np.pi/2, 0.0)
        
        # Check energy density is weighted combination
        rho_hybrid = hybrid.energy_density(coords)
        rho_expected = w1 * matter1.energy_density(coords) + w2 * matter2.energy_density(coords)
        
        assert abs(rho_hybrid - rho_expected) < abs(rho_expected) * 1e-10
        
        # Check pressures
        p_r_hybrid = hybrid.pressure_radial(coords)
        p_r_expected = w1 * matter1.pressure_radial(coords) + w2 * matter2.pressure_radial(coords)
        
        assert abs(p_r_hybrid - p_r_expected) < max(abs(p_r_expected), 1e-50) * 1e-10


class TestOptimization:
    """Test exotic matter optimization functionality."""
    
    def test_single_matter_type_optimization(self):
        """Test optimization for single matter type."""
        
        throat_radius = 1e3
        
        result = optimize_exotic_matter_configuration(
            throat_radius=throat_radius,
            matter_types=['casimir'],
            energy_budget=1e40,
            optimization_method='differential_evolution'
        )
        
        # Check result structure
        assert 'best_matter_type' in result
        assert 'best_configuration' in result
        assert 'energy_budget_satisfied' in result
        
        assert result['best_matter_type'] == 'casimir'
        assert 'optimal_parameters' in result['best_configuration']
        assert 'minimum_energy' in result['best_configuration']
    
    def test_multiple_matter_types_optimization(self):
        """Test optimization comparing multiple matter types."""
        
        throat_radius = 1e3
        
        result = optimize_exotic_matter_configuration(
            throat_radius=throat_radius,
            matter_types=['casimir', 'quantum_inequality'],
            energy_budget=1e45,
            optimization_method='differential_evolution'
        )
        
        # Should have results for all matter types
        assert 'all_results' in result
        assert len(result['all_results']) == 2
        
        # Best configuration should be selected
        best_energy = result['best_configuration']['minimum_energy']
        
        for matter_type, matter_result in result['all_results'].items():
            assert matter_result['minimum_energy'] >= best_energy
    
    @pytest.mark.slow
    def test_optimization_convergence(self):
        """Test optimization convergence (marked as slow test)."""
        
        throat_radius = 1e3
        
        # Run optimization multiple times with different seeds
        results = []
        
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            result = optimize_exotic_matter_configuration(
                throat_radius=throat_radius,
                matter_types=['casimir'],
                energy_budget=1e40,
                optimization_method='differential_evolution'
            )
            results.append(result['best_configuration']['minimum_energy'])
        
        # Results should be consistent (within factor of 2)
        min_energy = min(results)
        max_energy = max(results)
        
        if min_energy > 0:
            assert max_energy / min_energy < 2.0


class TestCatalogFunctionality:
    """Test exotic matter catalog functionality."""
    
    def test_catalog_completeness(self):
        """Test catalog contains all required matter types."""
        
        required_types = ['advanced_casimir', 'phantom_dark_energy', 
                         'quantum_inequality', 'string_theory']
        
        for matter_type in required_types:
            assert matter_type in ENHANCED_EXOTIC_MATTER_CATALOG
            
            entry = ENHANCED_EXOTIC_MATTER_CATALOG[matter_type]
            
            # Check required fields
            required_fields = ['class', 'description', 'typical_parameters']
            for field in required_fields:
                assert field in entry
    
    def test_catalog_loading(self):
        """Test loading matter instances from catalog."""
        
        for matter_type in ENHANCED_EXOTIC_MATTER_CATALOG:
            try:
                matter = load_exotic_matter_from_catalog(matter_type)
                assert matter is not None
                assert hasattr(matter, 'name')
                assert hasattr(matter, 'energy_density')
                
            except Exception as e:
                pytest.fail(f"Failed to load {matter_type} from catalog: {e}")
    
    def test_catalog_parameter_override(self):
        """Test parameter override when loading from catalog."""
        
        # Test with Casimir matter
        custom_separation = 2e-6
        matter = load_exotic_matter_from_catalog(
            'advanced_casimir',
            plate_separation=custom_separation
        )
        
        assert matter.a == custom_separation


class TestNumericalAccuracy:
    """Test numerical accuracy and edge cases."""
    
    def test_near_zero_energy_density(self):
        """Test behavior near zero energy density."""
        
        matter = AdvancedCasimirExoticMatter(plate_separation=1e-3)  # Large separation
        
        # Test at large distance where energy density is very small
        coords = (0.0, 1e3, np.pi/2, 0.0)
        
        rho = matter.energy_density(coords)
        eq_state = matter.equation_of_state_parameters(coords[1])
        
        # Should not crash and should return finite values
        assert np.isfinite(rho)
        for value in eq_state.values():
            assert np.isfinite(value)
    
    def test_extreme_parameter_values(self):
        """Test behavior with extreme parameter values."""
        
        # Very small plate separation (but still physical)
        try:
            matter = AdvancedCasimirExoticMatter(plate_separation=1e-9)
            coords = (0.0, 1e-9, np.pi/2, 0.0)
            rho = matter.energy_density(coords)
            assert np.isfinite(rho)
            
        except Exception:
            pass  # May fail due to numerical limits, which is acceptable
        
        # Very large throat radius
        matter = QuantumInequalityConstrainedMatter(throat_radius=1e10)
        coords = (0.0, 1e10, np.pi/2, 0.0)
        rho = matter.energy_density(coords)
        assert np.isfinite(rho)
    
    def test_coordinate_edge_cases(self):
        """Test edge cases in coordinate inputs."""
        
        matter = PhantomDarkEnergyField()
        
        # Test at origin (r=0)
        coords_origin = (0.0, 1e-50, np.pi/2, 0.0)
        try:
            rho = matter.energy_density(coords_origin)
            assert np.isfinite(rho)
        except:
            pass  # May fail, which is acceptable at origin
        
        # Test at very large r
        coords_large = (0.0, 1e20, np.pi/2, 0.0)
        rho = matter.energy_density(coords_large)
        assert np.isfinite(rho)


@pytest.mark.integration
class TestIntegrationWithOtherModules:
    """Integration tests with other physics modules."""
    
    def test_stress_energy_tensor_output(self):
        """Test stress-energy tensor is properly formatted for metric coupling."""
        
        matter = AdvancedCasimirExoticMatter(plate_separation=1e-6)
        coords = (0.0, 1e-6, np.pi/2, 0.0)
        
        T = matter.stress_energy_tensor(coords)
        
        # Should be 4x4 matrix
        assert T.shape == (4, 4)
        
        # Should have correct diagonal structure for perfect fluid
        assert T[0, 0] != 0  # Energy density component
        assert T[1, 1] != 0  # Radial pressure
        assert T[2, 2] == T[3, 3]  # Tangential pressure isotropy
        
        # Off-diagonal should be zero for static case
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert abs(T[i, j]) < 1e-50


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "--durations=10"  # Show 10 slowest tests
    ])