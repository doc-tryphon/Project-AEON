"""
Unsupervised learning for anomaly detection in wormhole simulations.

This module implements various unsupervised learning techniques to detect dangerous
instabilities, singularity formation, vacuum decay events, and other anomalous
behavior in real-time during wormhole simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Detects anomalies in wormhole simulation data using isolation forests."""
    
    def __init__(self, 
                feature_names: List[str],
                contamination: float = 0.1):
        """Initialize anomaly detector.
        
        Args:
            feature_names: Names of features to monitor
            contamination: Expected proportion of anomalies
        """
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Initialize monitoring
        self.alert_history = []
        self.baseline_stats = {}
        
    def fit(self, data: Dict[str, np.ndarray]):
        """Train anomaly detector on normal data.
        
        Args:
            data: Dictionary of feature arrays
        """
        # Prepare training data
        X = np.column_stack([
            data[feature] for feature in self.feature_names
        ])
        
        # Compute baseline statistics
        self.baseline_stats = {
            feature: {
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std()),
                'min': float(data[feature].min()),
                'max': float(data[feature].max())
            }
            for feature in self.feature_names
        }
        
        # Fit preprocessor and model
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.detector.fit(X_scaled)
        
    def predict(self, 
               data: Dict[str, np.ndarray],
               threshold: float = -0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in new data.
        
        Args:
            data: Dictionary of feature arrays
            threshold: Detection threshold (-1 to 1)
            
        Returns:
            Tuple of (anomaly_labels, anomaly_scores)
        """
        # Prepare input data
        X = np.column_stack([
            data[feature] for feature in self.feature_names
        ])
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.detector.score_samples(X_scaled)
        labels = scores < threshold
        
        # Record significant anomalies
        if labels.any():
            self._record_anomaly_alert(data, labels, scores)
            
        return labels, scores
        
    def _record_anomaly_alert(self,
                            data: Dict[str, np.ndarray],
                            labels: np.ndarray,
                            scores: np.ndarray):
        """Record details of detected anomalies."""
        for idx in np.where(labels)[0]:
            alert = {
                'timestamp': pd.Timestamp.now(),
                'anomaly_score': float(scores[idx]),
                'feature_values': {
                    feature: float(data[feature][idx])
                    for feature in self.feature_names
                },
                'deviations': self._calculate_deviations(
                    {feature: float(data[feature][idx])
                     for feature in self.feature_names}
                )
            }
            self.alert_history.append(alert)
            
    def _calculate_deviations(self, values: Dict[str, float]) -> Dict[str, float]:
        """Calculate deviations from baseline for each feature."""
        deviations = {}
        for feature, value in values.items():
            baseline = self.baseline_stats[feature]
            deviation = (value - baseline['mean']) / baseline['std']
            deviations[feature] = float(deviation)
        return deviations
        
    def get_anomaly_summary(self) -> Dict:
        """Get summary statistics of detected anomalies."""
        if not self.alert_history:
            return {'total_anomalies': 0}
            
        alerts_df = pd.DataFrame(self.alert_history)
        
        return {
            'total_anomalies': len(self.alert_history),
            'first_detected': alerts_df['timestamp'].min(),
            'last_detected': alerts_df['timestamp'].max(),
            'mean_score': float(alerts_df['anomaly_score'].mean()),
            'max_score': float(alerts_df['anomaly_score'].min()),
            'feature_frequencies': self._count_anomalous_features()
        }
        
    def _count_anomalous_features(self) -> Dict[str, int]:
        """Count how often each feature contributes to anomalies."""
        frequencies = {feature: 0 for feature in self.feature_names}
        
        for alert in self.alert_history:
            deviations = alert['deviations']
            # Count features with large deviations
            for feature, deviation in deviations.items():
                if abs(deviation) > 2.0:  # 2 standard deviations
                    frequencies[feature] += 1
                    
        return frequencies
        
    def get_alert_history(self) -> List[Dict]:
        """Get full history of anomaly alerts."""
        return self.alert_history
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Time series analysis
from scipy import stats
from scipy.signal import find_peaks, periodogram
from scipy.stats import entropy

import warnings
warnings.filterwarnings('ignore')


@dataclass
class AnomalyAlert:
    """Container for anomaly detection alerts."""
    
    timestamp: float
    anomaly_type: str
    severity_level: int  # 1-5 scale
    confidence_score: float
    affected_parameters: List[str]
    description: str
    recommended_actions: List[str]
    physics_context: Dict[str, float]


@dataclass
class SimulationState:
    """Container for current simulation state data."""
    
    # Physics parameters
    energy_density: float
    stress_tensor_components: np.ndarray
    curvature_scalars: Dict[str, float]
    metric_components: np.ndarray
    
    # Quantum parameters
    entanglement_entropy: float
    quantum_fluctuations: float
    vacuum_energy: float
    
    # Stability indicators
    perturbation_modes: np.ndarray
    lyapunov_exponents: List[float]
    geodesic_deviation: float
    
    # System health
    numerical_errors: float
    convergence_metrics: Dict[str, float]
    computational_resources: Dict[str, float]


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detection methods."""
    
    def __init__(self, detector_name: str):
        """Initialize base anomaly detector.
        
        Args:
            detector_name: Name of the detection method
        """
        self.detector_name = detector_name
        self.is_trained = False
        self.scaler = None
        self.anomaly_threshold = None
        self.feature_names = []
        
    @abstractmethod
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> None:
        """Train the anomaly detector on normal data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        pass
    
    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for samples."""
        pass
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess data for anomaly detection."""
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def detect_realtime_anomaly(self, current_state: SimulationState) -> Optional[AnomalyAlert]:
        """Detect anomalies in real-time simulation data."""
        if not self.is_trained:
            return None
        
        # Convert simulation state to feature vector
        features = self._state_to_features(current_state)
        features_scaled = self.preprocess_data(features.reshape(1, -1))
        
        # Get anomaly prediction and score
        prediction = self.predict(features_scaled)[0]
        anomaly_score = self.score_samples(features_scaled)[0]
        
        if prediction == -1:  # Anomaly detected
            return self._create_anomaly_alert(current_state, anomaly_score)
        
        return None
    
    def _state_to_features(self, state: SimulationState) -> np.ndarray:
        """Convert simulation state to feature vector."""
        features = []
        
        # Physics features
        features.append(state.energy_density)
        features.extend(state.stress_tensor_components.flatten())
        features.extend(list(state.curvature_scalars.values()))
        features.extend(state.metric_components.flatten())
        
        # Quantum features
        features.extend([
            state.entanglement_entropy,
            state.quantum_fluctuations,
            state.vacuum_energy
        ])
        
        # Stability features
        features.extend(state.perturbation_modes.flatten())
        features.extend(state.lyapunov_exponents)
        features.append(state.geodesic_deviation)
        
        # System health features
        features.append(state.numerical_errors)
        features.extend(list(state.convergence_metrics.values()))
        features.extend(list(state.computational_resources.values()))
        
        return np.array(features)
    
    def _create_anomaly_alert(self, state: SimulationState, 
                            anomaly_score: float) -> AnomalyAlert:
        """Create anomaly alert from detected anomaly."""
        
        # Determine anomaly type based on which parameters are most anomalous
        anomaly_type = self._classify_anomaly_type(state)
        
        # Determine severity based on anomaly score
        severity = self._compute_severity(anomaly_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomaly_type, state)
        
        return AnomalyAlert(
            timestamp=0.0,  # Would be set by simulation framework
            anomaly_type=anomaly_type,
            severity_level=severity,
            confidence_score=abs(anomaly_score),
            affected_parameters=self._identify_affected_parameters(state),
            description=f"{anomaly_type} detected with confidence {abs(anomaly_score):.3f}",
            recommended_actions=recommendations,
            physics_context=self._extract_physics_context(state)
        )
    
    def _classify_anomaly_type(self, state: SimulationState) -> str:
        """Classify the type of anomaly based on simulation state."""
        
        # Check for different types of anomalies
        if abs(state.energy_density) > 1e20:
            return "Energy Density Divergence"
        elif any(abs(lya) > 100 for lya in state.lyapunov_exponents):
            return "Chaotic Instability"
        elif state.numerical_errors > 1e-6:
            return "Numerical Instability"
        elif any(np.abs(state.curvature_scalars[k]) > 1e15 for k in state.curvature_scalars):
            return "Curvature Singularity"
        elif state.vacuum_energy < -1e25:
            return "Vacuum Decay Event"
        elif state.geodesic_deviation > 1e10:
            return "Spacetime Distortion"
        else:
            return "Unknown Anomaly"
    
    def _compute_severity(self, anomaly_score: float) -> int:
        """Compute severity level from anomaly score."""
        abs_score = abs(anomaly_score)
        
        if abs_score > 10:
            return 5  # Critical
        elif abs_score > 5:
            return 4  # High
        elif abs_score > 2:
            return 3  # Medium
        elif abs_score > 1:
            return 2  # Low
        else:
            return 1  # Minimal
    
    def _generate_recommendations(self, anomaly_type: str, state: SimulationState) -> List[str]:
        """Generate recommended actions based on anomaly type."""
        
        recommendations = {
            "Energy Density Divergence": [
                "Reduce exotic matter density",
                "Increase regularization parameters",
                "Check energy condition violations"
            ],
            "Chaotic Instability": [
                "Decrease time step size",
                "Add numerical damping",
                "Check initial conditions"
            ],
            "Numerical Instability": [
                "Reduce time step",
                "Increase numerical precision",
                "Check boundary conditions"
            ],
            "Curvature Singularity": [
                "Apply coordinate transformation",
                "Increase regularization",
                "Check for naked singularities"
            ],
            "Vacuum Decay Event": [
                "Halt simulation immediately",
                "Save current state",
                "Check vacuum stability"
            ],
            "Spacetime Distortion": [
                "Check metric components",
                "Verify coordinate system",
                "Examine tidal forces"
            ]
        }
        
        return recommendations.get(anomaly_type, ["Monitor closely", "Consider parameter adjustment"])
    
    def _identify_affected_parameters(self, state: SimulationState) -> List[str]:
        """Identify which parameters are most affected by the anomaly."""
        
        affected = []
        
        if abs(state.energy_density) > 1e15:
            affected.append("energy_density")
        
        if any(abs(lya) > 10 for lya in state.lyapunov_exponents):
            affected.append("lyapunov_exponents")
        
        if state.numerical_errors > 1e-8:
            affected.append("numerical_errors")
        
        if abs(state.vacuum_energy) > 1e20:
            affected.append("vacuum_energy")
        
        return affected
    
    def _extract_physics_context(self, state: SimulationState) -> Dict[str, float]:
        """Extract relevant physics context for the alert."""
        
        return {
            'energy_density_magnitude': abs(state.energy_density),
            'max_lyapunov_exponent': max(state.lyapunov_exponents) if state.lyapunov_exponents else 0.0,
            'numerical_error_level': state.numerical_errors,
            'vacuum_energy_scale': abs(state.vacuum_energy),
            'geodesic_deviation_scale': abs(state.geodesic_deviation)
        }


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
        """
        super().__init__("Isolation Forest")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> None:
        """Train Isolation Forest on normal data."""
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Preprocess data
        X_scaled = self.preprocess_data(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        
        self.model.fit(X_scaled)
        
        # Set anomaly threshold
        scores = self.model.decision_function(X_scaled)
        self.anomaly_threshold = np.percentile(scores, self.contamination * 100)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using Isolation Forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.preprocess_data(X)
        return self.model.decision_function(X_scaled)


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: str = 'scale'):
        """Initialize One-Class SVM detector.
        
        Args:
            nu: Upper bound on fraction of anomalies
            kernel: Kernel type
            gamma: Kernel coefficient
        """
        super().__init__("One-Class SVM")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> None:
        """Train One-Class SVM."""
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Preprocess data
        X_scaled = self.preprocess_data(X)
        
        # Train One-Class SVM
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        
        self.model.fit(X_scaled)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using One-Class SVM."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.preprocess_data(X)
        return self.model.score_samples(X_scaled)


class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, encoding_dim: int = 10, hidden_layers: List[int] = None):
        """Initialize autoencoder detector.
        
        Args:
            encoding_dim: Dimension of encoded representation
            hidden_layers: List of hidden layer sizes
        """
        super().__init__("Autoencoder")
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.autoencoder = None
        self.encoder = None
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None,
            epochs: int = 100, validation_split: float = 0.2) -> None:
        """Train autoencoder on normal data."""
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Preprocess data
        X_scaled = self.preprocess_data(X)
        input_dim = X_scaled.shape[1]
        
        # Build autoencoder
        self.autoencoder, self.encoder = self._build_autoencoder(input_dim)
        
        # Compile model
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train autoencoder
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Set anomaly threshold based on reconstruction error
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.anomaly_threshold = np.percentile(mse, 95)  # 95th percentile
        
        self.is_trained = True
    
    def _build_autoencoder(self, input_dim: int) -> Tuple[Model, Model]:
        """Build autoencoder architecture."""
        
        # Input layer
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for hidden_size in self.hidden_layers:
            encoded = layers.Dense(hidden_size, activation='relu')(encoded)
            encoded = layers.BatchNormalization()(encoded)
            encoded = layers.Dropout(0.1)(encoded)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='linear', name='encoded')(encoded)
        
        # Decoder
        decoded = encoded
        for hidden_size in reversed(self.hidden_layers):
            decoded = layers.Dense(hidden_size, activation='relu')(decoded)
            decoded = layers.BatchNormalization()(decoded)
            decoded = layers.Dropout(0.1)(decoded)
        
        # Output layer
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Create models
        autoencoder = Model(input_layer, decoded, name='autoencoder')
        encoder = Model(input_layer, encoded, name='encoder')
        
        return autoencoder, encoder
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.preprocess_data(X)
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Return -1 for anomalies, 1 for normal
        return np.where(mse > self.anomaly_threshold, -1, 1)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly score."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.preprocess_data(X)
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Return negative MSE (higher error = more negative score)
        return -mse


class VariationalAutoencoder(BaseAnomalyDetector):
    """Variational Autoencoder for anomaly detection."""
    
    def __init__(self, latent_dim: int = 10, hidden_layers: List[int] = None):
        """Initialize VAE detector.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_layers: Hidden layer sizes
        """
        super().__init__("Variational Autoencoder")
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.vae = None
        self.encoder = None
        self.decoder = None
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None,
            epochs: int = 100, validation_split: float = 0.2) -> None:
        """Train VAE on normal data."""
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Preprocess data
        X_scaled = self.preprocess_data(X)
        input_dim = X_scaled.shape[1]
        
        # Build VAE
        self.vae, self.encoder, self.decoder = self._build_vae(input_dim)
        
        # Compile VAE
        self.vae.compile(optimizer='adam')
        
        # Train VAE
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=7)
        ]
        
        history = self.vae.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Set threshold based on reconstruction + KL loss
        reconstructions, z_mean, z_log_var = self.encoder.predict(X_scaled)
        reconstruction_loss = np.mean(np.square(X_scaled - reconstructions), axis=1)
        kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)
        total_loss = reconstruction_loss + kl_loss
        
        self.anomaly_threshold = np.percentile(total_loss, 95)
        
        self.is_trained = True
    
    def _build_vae(self, input_dim: int) -> Tuple[Model, Model, Model]:
        """Build VAE architecture."""
        
        # Encoder
        encoder_inputs = keras.Input(shape=(input_dim,))
        x = encoder_inputs
        
        for hidden_size in self.hidden_layers:
            x = layers.Dense(hidden_size, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = latent_inputs
        
        for hidden_size in reversed(self.hidden_layers):
            x = layers.Dense(hidden_size, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        decoder_outputs = layers.Dense(input_dim, activation='linear')(x)
        decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = Model(encoder_inputs, outputs, name='vae')
        
        # Add VAE loss
        reconstruction_loss = tf.reduce_mean(tf.square(encoder_inputs - outputs))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        vae_loss = reconstruction_loss + kl_loss
        vae.add_loss(vae_loss)
        
        return vae, encoder, decoder
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using VAE loss."""
        scores = self.score_samples(X)
        return np.where(scores < -self.anomaly_threshold, -1, 1)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return VAE loss as anomaly score."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.preprocess_data(X)
        z_mean, z_log_var, _ = self.encoder.predict(X_scaled)
        reconstructions = self.vae.predict(X_scaled)
        
        reconstruction_loss = np.mean(np.square(X_scaled - reconstructions), axis=1)
        kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)
        total_loss = reconstruction_loss + kl_loss
        
        return -total_loss  # Negative loss (higher loss = more negative score)


class TimeSeriesAnomalyDetector:
    """Specialized detector for time series anomalies in simulations."""
    
    def __init__(self, window_size: int = 100, overlap: float = 0.5):
        """Initialize time series anomaly detector.
        
        Args:
            window_size: Size of sliding window
            overlap: Overlap between windows (0-1)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
        # Multiple detectors for different aspects
        self.statistical_detector = None
        self.frequency_detector = None
        self.pattern_detector = None
        
    def fit(self, time_series_data: np.ndarray, feature_names: List[str] = None):
        """Train on normal time series data.
        
        Args:
            time_series_data: Shape (n_timesteps, n_features)
            feature_names: Names of features
        """
        
        # Create windows
        windows = self._create_windows(time_series_data)
        
        # Statistical features
        statistical_features = self._extract_statistical_features(windows)
        
        # Frequency features
        frequency_features = self._extract_frequency_features(windows)
        
        # Train detectors
        self.statistical_detector = IsolationForestDetector(contamination=0.05)
        self.statistical_detector.fit(statistical_features)
        
        self.frequency_detector = OneClassSVMDetector(nu=0.05)
        self.frequency_detector.fit(frequency_features)
        
    def detect_online(self, current_window: np.ndarray) -> Dict[str, Union[bool, float]]:
        """Detect anomalies in current window online.
        
        Args:
            current_window: Current time window data
        
        Returns:
            Anomaly detection results
        """
        
        if current_window.shape[0] < self.window_size:
            return {'anomaly': False, 'confidence': 0.0, 'type': 'insufficient_data'}
        
        # Extract features
        stat_features = self._extract_statistical_features(current_window.reshape(1, *current_window.shape))
        freq_features = self._extract_frequency_features(current_window.reshape(1, *current_window.shape))
        
        # Get predictions
        stat_pred = self.statistical_detector.predict(stat_features)[0]
        freq_pred = self.frequency_detector.predict(freq_features)[0]
        
        # Get scores
        stat_score = self.statistical_detector.score_samples(stat_features)[0]
        freq_score = self.frequency_detector.score_samples(freq_features)[0]
        
        # Combined decision
        anomaly_detected = (stat_pred == -1) or (freq_pred == -1)
        combined_score = (abs(stat_score) + abs(freq_score)) / 2
        
        # Determine anomaly type
        anomaly_type = self._classify_time_series_anomaly(current_window, stat_pred, freq_pred)
        
        return {
            'anomaly': anomaly_detected,
            'confidence': combined_score,
            'type': anomaly_type,
            'statistical_score': stat_score,
            'frequency_score': freq_score
        }
    
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows from time series."""
        
        n_windows = (len(data) - self.window_size) // self.step_size + 1
        windows = []
        
        for i in range(n_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            windows.append(data[start_idx:end_idx])
        
        return np.array(windows)
    
    def _extract_statistical_features(self, windows: np.ndarray) -> np.ndarray:
        """Extract statistical features from windows."""
        
        features = []
        
        for window in windows:
            # Basic statistics
            mean_vals = np.mean(window, axis=0)
            std_vals = np.std(window, axis=0)
            min_vals = np.min(window, axis=0)
            max_vals = np.max(window, axis=0)
            
            # Higher order moments
            skew_vals = stats.skew(window, axis=0)
            kurt_vals = stats.kurtosis(window, axis=0)
            
            # Trend and change features
            if len(window) > 1:
                slopes = []
                for feature_idx in range(window.shape[1]):
                    x = np.arange(len(window))
                    y = window[:, feature_idx]
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.append(slope)
                slopes = np.array(slopes)
            else:
                slopes = np.zeros(window.shape[1])
            
            # Volatility
            if len(window) > 1:
                volatility = np.std(np.diff(window, axis=0), axis=0)
            else:
                volatility = np.zeros(window.shape[1])
            
            # Combine all features
            window_features = np.concatenate([
                mean_vals, std_vals, min_vals, max_vals,
                skew_vals, kurt_vals, slopes, volatility
            ])
            
            features.append(window_features)
        
        return np.array(features)
    
    def _extract_frequency_features(self, windows: np.ndarray) -> np.ndarray:
        """Extract frequency domain features from windows."""
        
        features = []
        
        for window in windows:
            freq_features = []
            
            # For each feature dimension
            for feature_idx in range(window.shape[1]):
                signal = window[:, feature_idx]
                
                # Power spectral density
                freqs, psd = periodogram(signal)
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(psd)
                dominant_freq = freqs[dominant_freq_idx]
                dominant_power = psd[dominant_freq_idx]
                
                # Spectral entropy
                psd_normalized = psd / np.sum(psd)
                spectral_entropy = entropy(psd_normalized)
                
                # Spectral centroid
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                
                # Spectral bandwidth
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
                
                freq_features.extend([
                    dominant_freq, dominant_power, spectral_entropy,
                    spectral_centroid, spectral_bandwidth
                ])
            
            features.append(freq_features)
        
        return np.array(features)
    
    def _classify_time_series_anomaly(self, window: np.ndarray, 
                                    stat_pred: int, freq_pred: int) -> str:
        """Classify type of time series anomaly."""
        
        if stat_pred == -1 and freq_pred == -1:
            return "Statistical and Frequency Anomaly"
        elif stat_pred == -1:
            return "Statistical Anomaly"
        elif freq_pred == -1:
            return "Frequency Anomaly"
        else:
            return "Normal"


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detectors for robust detection."""
    
    def __init__(self, detectors: List[BaseAnomalyDetector] = None):
        """Initialize ensemble detector.
        
        Args:
            detectors: List of individual detectors
        """
        self.detectors = detectors or [
            IsolationForestDetector(contamination=0.1),
            OneClassSVMDetector(nu=0.1),
            AutoencoderDetector(encoding_dim=10)
        ]
        
        self.voting_weights = None
        self.is_trained = False
    
    def fit(self, X: np.ndarray, feature_names: List[str] = None):
        """Train ensemble of detectors."""
        
        print("Training ensemble detectors...")
        
        # Train individual detectors
        for i, detector in enumerate(self.detectors):
            print(f"Training {detector.detector_name}...")
            try:
                detector.fit(X, feature_names)
            except Exception as e:
                print(f"Error training {detector.detector_name}: {e}")
                continue
        
        # Compute voting weights based on cross-validation performance
        self.voting_weights = self._compute_voting_weights(X)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction using weighted voting."""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        predictions = []
        
        for detector in self.detectors:
            if detector.is_trained:
                try:
                    pred = detector.predict(X)
                    predictions.append(pred)
                except Exception:
                    # Skip detector if prediction fails
                    continue
        
        if not predictions:
            return np.ones(len(X))  # Default to normal if no predictions
        
        predictions = np.array(predictions)
        
        # Weighted majority voting
        weighted_votes = np.zeros(len(X))
        
        for i, pred in enumerate(predictions):
            weight = self.voting_weights[i] if i < len(self.voting_weights) else 1.0
            weighted_votes += weight * pred
        
        # Return -1 if weighted vote is negative, 1 otherwise
        return np.where(weighted_votes < 0, -1, 1)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Ensemble anomaly scoring."""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        scores = []
        
        for detector in self.detectors:
            if detector.is_trained:
                try:
                    score = detector.score_samples(X)
                    scores.append(score)
                except Exception:
                    continue
        
        if not scores:
            return np.zeros(len(X))
        
        scores = np.array(scores)
        
        # Weighted average of scores
        weighted_scores = np.zeros(len(X))
        
        for i, score in enumerate(scores):
            weight = self.voting_weights[i] if i < len(self.voting_weights) else 1.0
            weighted_scores += weight * score
        
        return weighted_scores / np.sum(self.voting_weights)
    
    def _compute_voting_weights(self, X: np.ndarray) -> np.ndarray:
        """Compute voting weights based on detector performance."""
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import make_scorer
        
        weights = []
        
        for detector in self.detectors:
            if detector.is_trained:
                try:
                    # Use internal cross-validation to estimate performance
                    # This is a simplified approach - in practice you'd want held-out validation data
                    scores = detector.score_samples(X)
                    
                    # Use variance of scores as a proxy for discriminative power
                    weight = np.var(scores) if np.var(scores) > 0 else 1.0
                    weights.append(weight)
                    
                except Exception:
                    weights.append(1.0)  # Default weight
            else:
                weights.append(0.0)  # No weight for untrained detectors
        
        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights
    
    def get_detector_consensus(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all individual detectors."""
        
        consensus = {}
        
        for detector in self.detectors:
            if detector.is_trained:
                try:
                    pred = detector.predict(X)
                    score = detector.score_samples(X)
                    
                    consensus[detector.detector_name] = {
                        'predictions': pred,
                        'scores': score
                    }
                except Exception as e:
                    print(f"Error getting consensus from {detector.detector_name}: {e}")
        
        return consensus


def create_synthetic_anomaly_data(n_samples: int = 1000, 
                                n_features: int = 20,
                                anomaly_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for testing anomaly detectors."""
    
    np.random.seed(42)
    
    # Normal data - multivariate normal distribution
    normal_samples = int(n_samples * (1 - anomaly_fraction))
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=normal_samples
    )
    
    # Anomalous data - different distributions
    anomaly_samples = n_samples - normal_samples
    
    anomaly_types = []
    
    # Type 1: Outliers (shifted mean)
    type1_samples = anomaly_samples // 3
    type1_data = np.random.multivariate_normal(
        mean=5 * np.ones(n_features),
        cov=np.eye(n_features),
        size=type1_samples
    )
    anomaly_types.extend(['outlier'] * type1_samples)
    
    # Type 2: Different covariance
    type2_samples = anomaly_samples // 3
    type2_cov = np.eye(n_features) * 0.1  # Much smaller variance
    type2_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=type2_cov,
        size=type2_samples
    )
    anomaly_types.extend(['covariance_shift'] * type2_samples)
    
    # Type 3: Correlated features
    remaining_samples = anomaly_samples - type1_samples - type2_samples
    type3_cov = np.full((n_features, n_features), 0.8)
    np.fill_diagonal(type3_cov, 1.0)
    type3_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=type3_cov,
        size=remaining_samples
    )
    anomaly_types.extend(['correlation_shift'] * remaining_samples)
    
    # Combine data
    if anomaly_samples > 0:
        anomaly_data = np.vstack([type1_data, type2_data, type3_data])
        X = np.vstack([normal_data, anomaly_data])
        y = np.hstack([np.ones(normal_samples), -np.ones(anomaly_samples)])
    else:
        X = normal_data
        y = np.ones(normal_samples)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def evaluate_anomaly_detector(detector: BaseAnomalyDetector,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate anomaly detector performance."""
    
    predictions = detector.predict(X_test)
    scores = detector.score_samples(X_test)
    
    # Convert to binary (normal=0, anomaly=1)
    y_binary = (y_test == -1).astype(int)
    pred_binary = (predictions == -1).astype(int)
    
    # Compute metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    precision = precision_score(y_binary, pred_binary)
    recall = recall_score(y_binary, pred_binary)
    f1 = f1_score(y_binary, pred_binary)
    
    # ROC AUC (using scores)
    try:
        auc = roc_auc_score(y_binary, -scores)  # Negative scores for anomalies
    except ValueError:
        auc = 0.5  # Default if all one class
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc,
        'accuracy': np.mean(y_test == predictions)
    }