"""
Example demonstrating how to use the StabilityPredictor for wormhole stability analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.ai.stability_predictor import StabilityPredictor, PhysicsFeatures

def main():
    # 1. Create an instance of the StabilityPredictor
    predictor = StabilityPredictor()
    
    # 2. Create some example physics features for a potentially stable wormhole
    stable_features = PhysicsFeatures(
        # Stress-energy tensor components
        energy_density=-0.1,            # Small negative energy density
        radial_pressure=-0.05,          # Small negative radial pressure
        tangential_pressure=0.02,       # Small positive tangential pressure
        stress_anisotropy=0.07,         # Low stress anisotropy
        
        # Curvature scalars
        ricci_scalar=0.01,              # Small curvature
        kretschmann_scalar=0.02,        # Small tidal forces
        weyl_scalar=0.01,               # Low conformal curvature
        
        # Exotic matter properties
        exotic_energy_density=-0.1,     # Minimal exotic matter
        negative_pressure_ratio=0.5,    # Balanced pressures
        energy_condition_violations=[False, False, True],  # Only violates one condition
        
        # Geometric parameters
        throat_radius=1.0,              # Large throat
        shape_function_derivative=0.1,   # Gentle shape function
        flare_out_parameter=0.05,       # Good flare-out
        
        # Quantum corrections
        quantum_energy_correction=0.01,  # Small quantum effects
        vacuum_polarization=0.02,       # Low vacuum polarization
        hawking_temperature=0.01,       # Cool temperature
        
        # Stability indicators
        tidal_forces=0.02,              # Low tidal forces
        geodesic_deviation=0.01,        # Small geodesic deviation
        perturbation_growth_rate=0.01   # Slow perturbation growth
    )
    
    # 3. Create example features for a potentially unstable wormhole
    unstable_features = PhysicsFeatures(
        # Stress-energy tensor components
        energy_density=-0.5,            # Large negative energy density
        radial_pressure=-0.4,           # Large negative radial pressure
        tangential_pressure=0.3,        # High tangential pressure
        stress_anisotropy=0.7,          # High stress anisotropy
        
        # Curvature scalars
        ricci_scalar=0.5,               # High curvature
        kretschmann_scalar=0.8,         # Strong tidal forces
        weyl_scalar=0.4,                # High conformal curvature
        
        # Exotic matter properties
        exotic_energy_density=-0.8,     # Large exotic matter requirement
        negative_pressure_ratio=0.9,    # Highly negative pressure
        energy_condition_violations=[True, True, True],  # Violates all conditions
        
        # Geometric parameters
        throat_radius=0.1,              # Very small throat
        shape_function_derivative=0.8,   # Steep shape function
        flare_out_parameter=0.5,        # Poor flare-out
        
        # Quantum corrections
        quantum_energy_correction=0.2,   # Large quantum effects
        vacuum_polarization=0.3,        # High vacuum polarization
        hawking_temperature=0.5,        # Hot temperature
        
        # Stability indicators
        tidal_forces=0.8,               # High tidal forces
        geodesic_deviation=0.6,         # Large geodesic deviation
        perturbation_growth_rate=0.7    # Fast perturbation growth
    )
    
    # 4. Generate some training data
    n_samples = 1000
    n_features = 19  # Number of features in PhysicsFeatures
    X = np.random.randn(n_samples, n_features)  # Random features
    
    # Simple rule for demo purposes - stable if:
    # - throat radius is large enough (X[:, 10] > 0.5)
    # - energy density is not too negative (X[:, 0] > -0.3)
    # - tidal forces are small (X[:, 17] < 0.1)
    y = (X[:, 10] > 0.5) & (X[:, 0] > -0.3) & (X[:, 14] < 0.1)
    
    # 5. Train the model
    print("Training stability predictor...")
    history = predictor.train(X, y, epochs=10)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("\nTraining history plot saved to models/training_history.png")
    
    # 6. Make predictions and visualize them
    print("\nPredicting stability for example configurations:")
    
    # Create feature importance plot
    feature_names = [
        'Energy Density', 'Radial Pressure', 'Tangential Pressure', 'Stress Anisotropy',
        'Ricci Scalar', 'Kretschmann Scalar', 'Weyl Scalar',
        'Exotic Energy', 'Negative Pressure Ratio', 'Energy Violations',
        'Throat Radius', 'Shape Function', 'Flare-out',
        'Quantum Energy', 'Vacuum Polarization', 'Hawking Temperature',
        'Tidal Forces', 'Geodesic Dev.', 'Perturbation Growth'
    ]
    
    # Get feature importance by analyzing model weights
    weights = predictor.model.get_layer(index=1).get_weights()[0]
    importance = np.abs(weights).mean(axis=1)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, importance)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance in Stability Prediction')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    print("Feature importance plot saved to models/feature_importance.png")
    
    # Predict and visualize example configurations
    stable_features_array = predictor.extract_features(stable_features)
    unstable_features_array = predictor.extract_features(unstable_features)
    
    # Predict stable configuration
    is_stable, confidence = predictor.predict_stability(stable_features)
    print(f"\nStable configuration:")
    print(f"Is stable: {is_stable}")
    print(f"Confidence: {confidence:.2f}")
    
    # Predict unstable configuration
    is_stable, confidence = predictor.predict_stability(unstable_features)
    print(f"\nUnstable configuration:")
    print(f"Is stable: {is_stable}")
    print(f"Confidence: {confidence:.2f}")
    
    # Visualize the differences between stable and unstable configurations
    plt.figure(figsize=(12, 6))
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, stable_features_array[0], width, label='Stable')
    plt.bar(x + width/2, unstable_features_array[0], width, label='Unstable')
    
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.title('Feature Comparison: Stable vs Unstable Configurations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/configuration_comparison.png')
    print("Configuration comparison plot saved to models/configuration_comparison.png")
    
    # 7. Save the trained model
    predictor.save_model("models/stability_predictor.weights.h5")
    print("\nModel saved to models/stability_predictor_weights")

if __name__ == "__main__":
    main()
