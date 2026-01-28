"""
Deep learning neural networks for wormhole stability prediction.

This module implements sophisticated neural networks that predict wormhole stability
from physics parameters including stress-energy tensor components, curvature scalars,
exotic matter density, and quantum corrections.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


@dataclass
class PhysicsFeatures:
    """Features for stability prediction."""
    throat_radius: float
    mass: float
    tension: float
    energy_density: float
    pressure_radial: float
    pressure_tangential: float
    quantum_fluctuations: float
    entanglement_entropy: float


class StabilityPredictor:
    """Predicts wormhole stability using deep learning."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize stability predictor.
        
        Args:
            model_path: Path to saved model weights
        """
        self.model = self._build_model()
        
        if model_path:
            self.model.load_weights(model_path)
            
    def _build_model(self) -> Model:
        """Build neural network model architecture."""
        # Input layer
        inputs = layers.Input(shape=(19,))  # 19 physics features
        
        # Hidden layers with dropout for regularization
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def extract_features(self, features: PhysicsFeatures) -> np.ndarray:
        """Convert physics features to model input array."""
        # Calculate single value for energy condition violations
        energy_violations = sum(1 for v in features.energy_condition_violations if v) / len(features.energy_condition_violations)
        
        return np.array([[
            # Stress-energy tensor components
            features.energy_density,
            features.radial_pressure,
            features.tangential_pressure,
            features.stress_anisotropy,
            
            # Curvature scalars
            features.ricci_scalar,
            features.kretschmann_scalar,
            features.weyl_scalar,
            
            # Exotic matter properties
            features.exotic_energy_density,
            features.negative_pressure_ratio,
            energy_violations,  # Fraction of energy conditions violated
            
            # Geometric parameters
            features.throat_radius,
            features.shape_function_derivative,
            features.flare_out_parameter,
            
            # Quantum corrections
            features.quantum_energy_correction,
            features.vacuum_polarization,
            features.hawking_temperature,
            
            # Stability indicators
            features.tidal_forces,
            features.geodesic_deviation,
            features.perturbation_growth_rate
        ]])
        
    def predict_stability(self, features: PhysicsFeatures) -> Tuple[bool, float]:
        """Predict if wormhole configuration is stable.
        
        Args:
            features: Physical features of wormhole
            
        Returns:
            Tuple of (is_stable, confidence)
        """
        # Convert features to input array
        X = self.extract_features(features)
        
        # Get prediction
        pred = self.model.predict(X, verbose=0)[0,0]
        
        is_stable = bool(pred > 0.5)
        confidence = float(abs(pred - 0.5) * 2)
        
        return is_stable, confidence
        
    def train(self,
             X: np.ndarray,
             y: np.ndarray,
             validation_split: float = 0.2,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """Train stability prediction model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history.history
        
    def save_model(self, path: str):
        """Save trained model weights."""
        self.model.save_weights(path)
        
    def load_model(self, path: str):
        """Load trained model weights."""
        self.model.load_weights(path)
from tensorflow import keras
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.physics.spacetime_metrics import SpacetimeMetric
from src.physics.stress_energy_tensor import StressEnergyTensor
from src.physics.exotic_matter import ExoticMatter
from src.quantum.quantum_gravity import QuantumGravityCorrection


@dataclass
class PhysicsFeatures:
    """Container for physics-based features used in stability prediction."""
    
    # Stress-energy tensor components
    energy_density: float
    radial_pressure: float
    tangential_pressure: float
    stress_anisotropy: float
    
    # Curvature scalars
    ricci_scalar: float
    kretschmann_scalar: float
    weyl_scalar: float
    
    # Exotic matter properties
    exotic_energy_density: float
    negative_pressure_ratio: float
    energy_condition_violations: List[bool]
    
    # Geometric parameters
    throat_radius: float
    shape_function_derivative: float
    flare_out_parameter: float
    
    # Quantum corrections
    quantum_energy_correction: float
    vacuum_polarization: float
    hawking_temperature: float
    
    # Stability indicators
    tidal_forces: float
    geodesic_deviation: float
    perturbation_growth_rate: float


class StabilityDataset:
    """Generate and manage datasets for wormhole stability prediction."""
    
    def __init__(self, num_samples: int = 10000, random_seed: int = 42):
        """Initialize dataset generator.
        
        Args:
            num_samples: Number of samples to generate
            random_seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_synthetic_data(self, 
                              parameter_ranges: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic wormhole physics data.
        
        Args:
            parameter_ranges: Dictionary with parameter ranges for generation
        
        Returns:
            Feature matrix X and stability labels y
        """
        # Default parameter ranges if not provided
        default_ranges = {
            'throat_radius': (1e3, 1e6),  # meters
            'exotic_energy_scale': (-1e15, -1e10),  # J/m³
            'quantum_correction': (0.0, 1.0),
            'shape_parameter': (0.1, 10.0),
            'temperature': (1e-10, 1e-5)  # Kelvin
        }
        
        ranges = {**default_ranges, **parameter_ranges}
        
        # Generate random parameters
        samples = []
        labels = []
        
        for i in range(self.num_samples):
            # Random parameter sampling
            throat_r = np.random.uniform(*ranges['throat_radius'])
            exotic_rho = np.random.uniform(*ranges['exotic_energy_scale'])
            quantum_corr = np.random.uniform(*ranges['quantum_correction'])
            shape_param = np.random.uniform(*ranges['shape_parameter'])
            temp = np.random.uniform(*ranges['temperature'])
            
            # Compute physics features
            features = self._compute_physics_features(
                throat_r, exotic_rho, quantum_corr, shape_param, temp
            )
            
            # Stability criterion based on physics
            stability = self._compute_stability_label(features)
            
            samples.append(self._features_to_array(features))
            labels.append(stability)
        
        X = np.array(samples)
        y = np.array(labels, dtype=int)
        
        # Store feature names
        self.feature_names = self._get_feature_names()
        
        return X, y
    
    def _compute_physics_features(self, throat_radius: float, exotic_density: float,
                                quantum_correction: float, shape_param: float,
                                temperature: float) -> PhysicsFeatures:
        """Compute physics features from basic parameters."""
        
        # Stress-energy components (simplified models)
        energy_density = abs(exotic_density)
        radial_pressure = -exotic_density * (1 + quantum_correction)
        tangential_pressure = -exotic_density * 0.5
        stress_anisotropy = abs(radial_pressure - tangential_pressure) / energy_density
        
        # Curvature scalars (geometric estimates)
        ricci_scalar = -6 * shape_param / throat_radius**2
        kretschmann_scalar = 48 * shape_param**2 / throat_radius**4
        weyl_scalar = ricci_scalar / 3
        
        # Exotic matter properties
        negative_pressure_ratio = min(radial_pressure, tangential_pressure) / energy_density
        
        # Energy condition violations
        null_violation = energy_density + radial_pressure < 0
        weak_violation = energy_density < 0
        strong_violation = energy_density + radial_pressure + 2*tangential_pressure < 0
        
        # Geometric parameters
        shape_derivative = -shape_param / throat_radius
        flare_out = max(0, 1 - shape_param / throat_radius)
        
        # Quantum corrections
        quantum_energy = quantum_correction * energy_density
        vacuum_pol = quantum_correction * 1e-10  # Small vacuum polarization
        hawking_temp = temperature
        
        # Stability indicators (simplified)
        tidal_forces = kretschmann_scalar * throat_radius**2
        geodesic_dev = abs(ricci_scalar) * throat_radius
        perturbation_rate = np.sqrt(abs(ricci_scalar))
        
        return PhysicsFeatures(
            energy_density=energy_density,
            radial_pressure=radial_pressure,
            tangential_pressure=tangential_pressure,
            stress_anisotropy=stress_anisotropy,
            ricci_scalar=ricci_scalar,
            kretschmann_scalar=kretschmann_scalar,
            weyl_scalar=weyl_scalar,
            exotic_energy_density=exotic_density,
            negative_pressure_ratio=negative_pressure_ratio,
            energy_condition_violations=[null_violation, weak_violation, strong_violation],
            throat_radius=throat_radius,
            shape_function_derivative=shape_derivative,
            flare_out_parameter=flare_out,
            quantum_energy_correction=quantum_energy,
            vacuum_polarization=vacuum_pol,
            hawking_temperature=hawking_temp,
            tidal_forces=tidal_forces,
            geodesic_deviation=geodesic_dev,
            perturbation_growth_rate=perturbation_rate
        )
    
    def _compute_stability_label(self, features: PhysicsFeatures) -> int:
        """Compute stability label based on physics features."""
        
        stability_score = 0
        
        # Positive factors for stability
        if features.flare_out_parameter > 0.1:
            stability_score += 2
        
        if abs(features.stress_anisotropy) < 1.0:
            stability_score += 1
        
        if features.tidal_forces < 1e10:
            stability_score += 2
        
        if features.perturbation_growth_rate < 1e5:
            stability_score += 1
        
        # Negative factors (instability)
        if any(features.energy_condition_violations):
            stability_score -= 1
        
        if abs(features.kretschmann_scalar) > 1e20:
            stability_score -= 3
        
        if features.geodesic_deviation > features.throat_radius * 10:
            stability_score -= 2
        
        if abs(features.quantum_energy_correction) > features.energy_density * 0.5:
            stability_score -= 1
        
        # Binary classification: stable (1) or unstable (0)
        return 1 if stability_score > 0 else 0
    
    def _features_to_array(self, features: PhysicsFeatures) -> np.ndarray:
        """Convert PhysicsFeatures to numpy array."""
        return np.array([
            features.energy_density,
            features.radial_pressure,
            features.tangential_pressure,
            features.stress_anisotropy,
            features.ricci_scalar,
            features.kretschmann_scalar,
            features.weyl_scalar,
            features.exotic_energy_density,
            features.negative_pressure_ratio,
            sum(features.energy_condition_violations),  # Count violations
            features.throat_radius,
            features.shape_function_derivative,
            features.flare_out_parameter,
            features.quantum_energy_correction,
            features.vacuum_polarization,
            features.hawking_temperature,
            features.tidal_forces,
            features.geodesic_deviation,
            features.perturbation_growth_rate
        ])
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'energy_density', 'radial_pressure', 'tangential_pressure',
            'stress_anisotropy', 'ricci_scalar', 'kretschmann_scalar',
            'weyl_scalar', 'exotic_energy_density', 'negative_pressure_ratio',
            'energy_condition_violations', 'throat_radius', 'shape_function_derivative',
            'flare_out_parameter', 'quantum_energy_correction', 'vacuum_polarization',
            'hawking_temperature', 'tidal_forces', 'geodesic_deviation',
            'perturbation_growth_rate'
        ]


class BaseStabilityPredictor(ABC):
    """Abstract base class for stability prediction models."""
    
    def __init__(self, model_name: str):
        """Initialize base predictor.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = {}
        self.feature_importance = None
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> Model:
        """Build the neural network model."""
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Train the model."""
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance."""
        predictions = (self.predict_proba(X_test) > 0.5).astype(int).flatten()
        probabilities = self.predict_proba(X_test).flatten()
        
        # Metrics
        accuracy = np.mean(predictions == y_test)
        auc_score = roc_auc_score(y_test, probabilities)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'model_name': self.model_name
        }
        
        # Save using different methods based on model type
        if hasattr(self.model, 'save'):  # Keras model
            self.model.save(f"{filepath}_model.h5")
            with open(f"{filepath}_metadata.pkl", 'wb') as f:
                pickle.dump({k: v for k, v in model_data.items() if k != 'model'}, f)
        else:  # Sklearn-like model
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        try:
            # Try Keras model first
            self.model = keras.models.load_model(f"{filepath}_model.h5")
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.scaler = metadata['scaler']
            self.feature_importance = metadata['feature_importance']
            self.training_history = metadata['training_history']
            self.model_name = metadata['model_name']
            self.is_trained = True
            
        except:
            # Fall back to pickle
            with open(f"{filepath}.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data['feature_importance']
            self.training_history = model_data['training_history']
            self.model_name = model_data['model_name']
            self.is_trained = True


class DeepStabilityNet(BaseStabilityPredictor):
    """Deep neural network for wormhole stability prediction."""
    
    def __init__(self, architecture: str = 'standard'):
        """Initialize deep stability network.
        
        Args:
            architecture: Network architecture type ('standard', 'residual', 'attention')
        """
        super().__init__(f"DeepStabilityNet_{architecture}")
        self.architecture = architecture
    
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> Model:
        """Build deep neural network model."""
        
        # Hyperparameters
        hidden_units = kwargs.get('hidden_units', [256, 128, 64, 32])
        dropout_rate = kwargs.get('dropout_rate', 0.3)
        activation = kwargs.get('activation', 'relu')
        l2_reg = kwargs.get('l2_regularization', 1e-4)
        
        inputs = keras.Input(shape=input_shape, name='physics_features')
        
        if self.architecture == 'standard':
            x = self._build_standard_layers(inputs, hidden_units, dropout_rate, 
                                          activation, l2_reg)
        elif self.architecture == 'residual':
            x = self._build_residual_layers(inputs, hidden_units, dropout_rate, 
                                          activation, l2_reg)
        elif self.architecture == 'attention':
            x = self._build_attention_layers(inputs, hidden_units, dropout_rate, 
                                           activation, l2_reg)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='stability_probability')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        
        # Compile model
        optimizer = kwargs.get('optimizer', 'adam')
        learning_rate = kwargs.get('learning_rate', 1e-3)
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'adamw':
            opt = keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        return model
    
    def _build_standard_layers(self, inputs, hidden_units, dropout_rate, 
                             activation, l2_reg):
        """Build standard dense layers."""
        x = inputs
        
        for i, units in enumerate(hidden_units):
            x = layers.Dense(
                units, 
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        return x
    
    def _build_residual_layers(self, inputs, hidden_units, dropout_rate, 
                             activation, l2_reg):
        """Build residual connection layers."""
        x = inputs
        
        for i, units in enumerate(hidden_units):
            # Main path
            residual = x
            
            x = layers.Dense(
                units, 
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'res_dense_{i+1}_a'
            )(x)
            x = layers.BatchNormalization(name=f'res_bn_{i+1}_a')(x)
            
            x = layers.Dense(
                units, 
                activation=None,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'res_dense_{i+1}_b'
            )(x)
            x = layers.BatchNormalization(name=f'res_bn_{i+1}_b')(x)
            
            # Residual connection with projection if needed
            if residual.shape[-1] != units:
                residual = layers.Dense(units, activation=None, 
                                      name=f'res_projection_{i+1}')(residual)
            
            x = layers.Add(name=f'res_add_{i+1}')([x, residual])
            x = layers.Activation(activation, name=f'res_activation_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'res_dropout_{i+1}')(x)
        
        return x
    
    def _build_attention_layers(self, inputs, hidden_units, dropout_rate, 
                              activation, l2_reg):
        """Build attention mechanism layers."""
        x = inputs
        
        # Feature attention mechanism
        attention_weights = layers.Dense(
            inputs.shape[-1], 
            activation='softmax',
            name='feature_attention'
        )(x)
        
        x = layers.Multiply(name='attended_features')([x, attention_weights])
        
        # Standard dense layers with attention
        for i, units in enumerate(hidden_units):
            x = layers.Dense(
                units, 
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'att_dense_{i+1}'
            )(x)
            
            # Self-attention within layer
            if i < len(hidden_units) - 1:  # Not on last layer
                att = layers.Dense(units, activation='tanh', 
                                 name=f'self_att_tanh_{i+1}')(x)
                att = layers.Dense(units, activation='softmax', 
                                 name=f'self_att_softmax_{i+1}')(att)
                x = layers.Multiply(name=f'self_att_mult_{i+1}')([x, att])
            
            x = layers.BatchNormalization(name=f'att_bn_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'att_dropout_{i+1}')(x)
        
        return x
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Train the deep stability network."""
        
        # Data preprocessing
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/validation split
        test_size = kwargs.get('validation_split', 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Build model
        self.model = self.build_model((X.shape[1],), **kwargs)
        
        # Callbacks
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=kwargs.get('early_stopping_patience', 15),
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=kwargs.get('lr_patience', 10),
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(lr_scheduler)
        
        # Model checkpoint
        if kwargs.get('save_best_model', True):
            checkpoint = callbacks.ModelCheckpoint(
                'best_stability_model.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks_list.append(checkpoint)
        
        # Training
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 32)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=kwargs.get('verbose', 1)
        )
        
        self.training_history = history.history
        self.is_trained = True
        
        # Feature importance (using permutation importance)
        self.feature_importance = self._compute_feature_importance(X_val, y_val)
        
        # Training summary
        best_epoch = np.argmax(history.history['val_auc'])
        training_summary = {
            'best_epoch': best_epoch,
            'best_val_loss': min(history.history['val_loss']),
            'best_val_accuracy': max(history.history['val_accuracy']),
            'best_val_auc': max(history.history['val_auc']),
            'training_time_epochs': len(history.history['loss']),
            'final_learning_rate': float(self.model.optimizer.learning_rate.numpy())
        }
        
        return training_summary
    
    def _compute_feature_importance(self, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Compute feature importance using permutation method."""
        baseline_score = self.model.evaluate(X_val, y_val, verbose=0)[4]  # AUC
        
        importances = []
        
        for i in range(X_val.shape[1]):
            # Permute feature i
            X_permuted = X_val.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Compute score with permuted feature
            permuted_score = self.model.evaluate(X_permuted, y_val, verbose=0)[4]
            
            # Importance = drop in performance
            importance = baseline_score - permuted_score
            importances.append(importance)
        
        return np.array(importances)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        if not self.training_history:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['loss'], label='Training')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.training_history['accuracy'], label='Training')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # AUC
        axes[1, 0].plot(self.training_history['auc'], label='Training')
        axes[1, 0].plot(self.training_history['val_auc'], label='Validation')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Feature importance
        if self.feature_importance is not None:
            feature_names = [f'Feature_{i}' for i in range(len(self.feature_importance))]
            axes[1, 1].barh(feature_names[-10:], self.feature_importance[-10:])  # Top 10
            axes[1, 1].set_title('Top 10 Feature Importances')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class EnsembleStabilityPredictor:
    """Ensemble of multiple stability prediction models."""
    
    def __init__(self, models: List[BaseStabilityPredictor]):
        """Initialize ensemble predictor.
        
        Args:
            models: List of individual prediction models
        """
        self.models = models
        self.weights = None
        self.is_trained = False
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, 
                      ensemble_method: str = 'voting', **kwargs) -> Dict:
        """Train ensemble of models.
        
        Args:
            X: Training features
            y: Training labels
            ensemble_method: 'voting', 'stacking', or 'weighted'
            **kwargs: Additional training parameters
        """
        
        # Train individual models
        training_results = []
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            
            # Create model-specific parameters
            model_kwargs = kwargs.copy()
            
            result = model.train(X, y, **model_kwargs)
            training_results.append(result)
        
        # Ensemble-specific training
        if ensemble_method == 'weighted':
            self._train_weighted_ensemble(X, y)
        elif ensemble_method == 'stacking':
            self._train_stacking_ensemble(X, y)
        else:
            # Simple voting - equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        self.is_trained = True
        
        return {
            'individual_results': training_results,
            'ensemble_method': ensemble_method,
            'ensemble_weights': self.weights.tolist() if self.weights is not None else None
        }
    
    def _train_weighted_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train weighted ensemble based on individual model performance."""
        
        # Cross-validation to get model weights
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model_scores = []
        
        for model in self.models:
            # Evaluate on validation set
            eval_results = model.evaluate(X_val, y_val)
            model_scores.append(eval_results['auc_score'])
        
        # Convert to weights (higher score = higher weight)
        scores = np.array(model_scores)
        self.weights = scores / np.sum(scores)
    
    def _train_stacking_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train stacking ensemble with meta-learner."""
        
        # Generate meta-features using cross-validation
        from sklearn.model_selection import StratifiedKFold
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            
            for i, model in enumerate(self.models):
                # Train on fold
                model.train(X_fold_train, y_fold_train, epochs=50, verbose=0)
                
                # Predict on validation fold
                pred_proba = model.predict_proba(X_fold_val)
                meta_features[val_idx, i] = pred_proba.flatten()
        
        # Train meta-learner
        from sklearn.linear_model import LogisticRegression
        
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(meta_features, y)
        
        # Weights from meta-model coefficients
        self.weights = np.abs(self.meta_model.coef_[0])
        self.weights = self.weights / np.sum(self.weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Get predictions from all models
        predictions = []
        
        for model in self.models:
            pred = model.predict_proba(X).flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=1, weights=self.weights)
        
        return ensemble_pred
    
    def evaluate_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate ensemble performance."""
        
        ensemble_pred = self.predict(X_test)
        ensemble_binary = (ensemble_pred > 0.5).astype(int)
        
        # Individual model performance
        individual_scores = []
        for model in self.models:
            scores = model.evaluate(X_test, y_test)
            individual_scores.append(scores['auc_score'])
        
        # Ensemble performance
        ensemble_accuracy = np.mean(ensemble_binary == y_test)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        
        return {
            'ensemble_accuracy': ensemble_accuracy,
            'ensemble_auc': ensemble_auc,
            'individual_aucs': individual_scores,
            'improvement_over_best': ensemble_auc - max(individual_scores),
            'model_weights': self.weights.tolist() if self.weights is not None else None
        }


def hyperparameter_optimization(X: np.ndarray, y: np.ndarray, 
                               model_class: type = DeepStabilityNet,
                               n_trials: int = 50) -> Dict:
    """Optimize hyperparameters using Optuna or GridSearch.
    
    Args:
        X: Training features
        y: Training labels  
        model_class: Model class to optimize
        n_trials: Number of optimization trials
    
    Returns:
        Best hyperparameters and performance
    """
    
    # Define parameter space
    param_space = {
        'hidden_units': [
            [128, 64, 32],
            [256, 128, 64, 32],
            [512, 256, 128, 64],
            [256, 256, 128, 64, 32]
        ],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'batch_size': [16, 32, 64, 128],
        'l2_regularization': [1e-5, 1e-4, 1e-3, 1e-2]
    }
    
    best_score = -np.inf
    best_params = {}
    
    # Random search over parameter space
    for trial in range(n_trials):
        # Sample parameters
        params = {}
        for key, values in param_space.items():
            params[key] = np.random.choice(values)
        
        try:
            # Create and train model
            model = model_class()
            
            # Cross-validation
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kfold.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.train(X_train, y_train, epochs=30, verbose=0, **params)
                result = model.evaluate(X_val, y_val)
                scores.append(result['auc_score'])
            
            # Average score
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()
                
            print(f"Trial {trial+1}/{n_trials}: AUC = {avg_score:.4f}")
            
        except Exception as e:
            print(f"Trial {trial+1} failed: {e}")
            continue
    
    return {
        'best_parameters': best_params,
        'best_score': best_score,
        'optimization_trials': n_trials
    }