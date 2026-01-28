"""
Reinforcement learning agents for adaptive wormhole control.

This module implements RL agents that learn to adaptively control wormhole
parameters to maintain stability and enable safe traversal, using deep Q-learning,
policy gradient methods, and actor-critic algorithms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class DQNetwork(nn.Module):
    """Deep Q-Network for wormhole control."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WormholeRLAgent:
    """Reinforcement learning agent for wormhole control."""
    
    def __init__(self,
                state_dim: int,
                action_dim: int,
                learning_rate: float = 0.001,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                memory_size: int = 10000):
        """Initialize RL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
            memory_size: Size of replay memory
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimization
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 64
        
        # Learning parameters
        self.gamma = gamma
        
        # Action space
        self.action_dim = action_dim
        self.actions = self._setup_action_space()
        
    def _setup_action_space(self) -> List[Dict[str, float]]:
        """Define possible actions for wormhole control."""
        actions = []
        
        # Throat radius adjustments
        radius_changes = [-0.1, -0.05, 0.0, 0.05, 0.1]
        
        # Energy density adjustments
        energy_changes = [-0.1, -0.05, 0.0, 0.05, 0.1]
        
        # Combine all possible actions
        for dr in radius_changes:
            for de in energy_changes:
                actions.append({
                    'radius_change': dr,
                    'energy_change': de
                })
                
        return actions
        
    def select_action(self, state: np.ndarray) -> Dict[str, float]:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            
        Returns:
            Selected action as dictionary of parameter changes
        """
        if random.random() < self.epsilon:
            action_idx = random.randrange(len(self.actions))
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = int(q_values.argmax().item())
                
        return self.actions[action_idx]
            
    def store_transition(self, 
                       state: np.ndarray,
                       action: Dict[str, float],
                       reward: float,
                       next_state: np.ndarray,
                       done: bool):
        """Store transition in replay memory."""
        # Convert action dict to index
        action_idx = self.actions.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))
        
    def update(self) -> Optional[float]:
        """Update policy network using replay memory."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Update policy network
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        if random.random() < 0.001:
            target_state_dict = self.target_net.state_dict()
            policy_state_dict = self.policy_net.state_dict()
            for key in policy_state_dict:
                target_state_dict[key] = 0.95 * target_state_dict[key] + \
                                       0.05 * policy_state_dict[key]
            self.target_net.load_state_dict(target_state_dict)
            
        # Decay exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return float(loss.item())
        
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }, path)
        
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = deque(checkpoint['memory'], maxlen=self.memory.maxlen)
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
import random
import pickle

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_probability as tfp

# Utilities
import matplotlib.pyplot as plt
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

from src.ai.stability_predictor import BaseStabilityPredictor
from src.ai.anomaly_detector import BaseAnomalyDetector


@dataclass
class WormholeState:
    """State representation for wormhole RL environment."""
    
    # Physical parameters
    throat_radius: float
    exotic_energy_density: float
    shape_parameter: float
    quantum_correction: float
    
    # Derived physics quantities
    energy_density: float
    stress_tensor_trace: float
    ricci_scalar: float
    stability_score: float
    
    # Environment status
    time_step: int
    is_stable: bool
    is_traversable: bool
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array."""
        return np.array([
            self.throat_radius,
            self.exotic_energy_density,
            self.shape_parameter,
            self.quantum_correction,
            self.energy_density,
            self.stress_tensor_trace,
            self.ricci_scalar,
            self.stability_score,
            float(self.time_step),
            float(self.is_stable),
            float(self.is_traversable)
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'WormholeState':
        """Create state from numpy array."""
        return cls(
            throat_radius=array[0],
            exotic_energy_density=array[1],
            shape_parameter=array[2],
            quantum_correction=array[3],
            energy_density=array[4],
            stress_tensor_trace=array[5],
            ricci_scalar=array[6],
            stability_score=array[7],
            time_step=int(array[8]),
            is_stable=bool(array[9]),
            is_traversable=bool(array[10])
        )


@dataclass
class WormholeAction:
    """Action representation for wormhole control."""
    
    # Parameter adjustments (relative changes)
    delta_throat_radius: float
    delta_exotic_density: float
    delta_shape_parameter: float
    delta_quantum_correction: float
    
    # Control actions
    stabilization_mode: int  # 0=none, 1=active, 2=emergency
    traversal_mode: int      # 0=maintenance, 1=prepare, 2=traverse
    
    def to_array(self) -> np.ndarray:
        """Convert action to numpy array."""
        return np.array([
            self.delta_throat_radius,
            self.delta_exotic_density,
            self.delta_shape_parameter,
            self.delta_quantum_correction,
            float(self.stabilization_mode),
            float(self.traversal_mode)
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'WormholeAction':
        """Create action from numpy array."""
        return cls(
            delta_throat_radius=array[0],
            delta_exotic_density=array[1],
            delta_shape_parameter=array[2],
            delta_quantum_correction=array[3],
            stabilization_mode=int(array[4]),
            traversal_mode=int(array[5])
        )


class WormholeEnvironment:
    """Reinforcement learning environment for wormhole control."""
    
    def __init__(self, stability_predictor: Optional[BaseStabilityPredictor] = None,
                 anomaly_detector: Optional[BaseAnomalyDetector] = None):
        """Initialize wormhole environment.
        
        Args:
            stability_predictor: Trained stability prediction model
            anomaly_detector: Trained anomaly detection model
        """
        self.stability_predictor = stability_predictor
        self.anomaly_detector = anomaly_detector
        
        # Environment parameters
        self.max_steps = 1000
        self.current_step = 0
        self.current_state = None
        
        # Parameter bounds
        self.parameter_bounds = {
            'throat_radius': (1e3, 1e6),
            'exotic_energy_density': (-1e20, -1e10),
            'shape_parameter': (0.1, 10.0),
            'quantum_correction': (0.0, 1.0)
        }
        
        # Action bounds (relative changes)
        self.action_bounds = {
            'delta_throat_radius': (-0.1, 0.1),      # ±10% change
            'delta_exotic_density': (-0.2, 0.2),     # ±20% change
            'delta_shape_parameter': (-0.1, 0.1),    # ±10% change
            'delta_quantum_correction': (-0.05, 0.05), # ±5% change
            'stabilization_mode': (0, 2),
            'traversal_mode': (0, 2)
        }
        
        # Reward shaping parameters
        self.reward_weights = {
            'stability': 10.0,
            'traversability': 5.0,
            'energy_efficiency': 2.0,
            'smoothness': 1.0,
            'anomaly_penalty': -20.0,
            'instability_penalty': -50.0
        }
        
        # History tracking
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        
        self.current_step = 0
        
        # Sample initial parameters within bounds
        initial_params = {}
        for param, (min_val, max_val) in self.parameter_bounds.items():
            initial_params[param] = np.random.uniform(min_val, max_val)
        
        # Compute initial physics quantities
        physics_quantities = self._compute_physics_quantities(initial_params)
        
        # Create initial state
        self.current_state = WormholeState(
            throat_radius=initial_params['throat_radius'],
            exotic_energy_density=initial_params['exotic_energy_density'],
            shape_parameter=initial_params['shape_parameter'],
            quantum_correction=initial_params['quantum_correction'],
            energy_density=physics_quantities['energy_density'],
            stress_tensor_trace=physics_quantities['stress_tensor_trace'],
            ricci_scalar=physics_quantities['ricci_scalar'],
            stability_score=physics_quantities['stability_score'],
            time_step=self.current_step,
            is_stable=physics_quantities['is_stable'],
            is_traversable=physics_quantities['is_traversable']
        )
        
        # Clear history
        self.state_history = [self.current_state]
        self.action_history = []
        self.reward_history = []
        
        return self.current_state.to_array()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info."""
        
        if self.current_state is None:
            raise ValueError("Environment must be reset before taking actions")
        
        self.current_step += 1
        
        # Convert action array to WormholeAction
        action_obj = WormholeAction.from_array(action)
        
        # Apply action to current state
        new_params = self._apply_action(self.current_state, action_obj)
        
        # Compute new physics quantities
        physics_quantities = self._compute_physics_quantities(new_params)
        
        # Create new state
        new_state = WormholeState(
            throat_radius=new_params['throat_radius'],
            exotic_energy_density=new_params['exotic_energy_density'],
            shape_parameter=new_params['shape_parameter'],
            quantum_correction=new_params['quantum_correction'],
            energy_density=physics_quantities['energy_density'],
            stress_tensor_trace=physics_quantities['stress_tensor_trace'],
            ricci_scalar=physics_quantities['ricci_scalar'],
            stability_score=physics_quantities['stability_score'],
            time_step=self.current_step,
            is_stable=physics_quantities['is_stable'],
            is_traversable=physics_quantities['is_traversable']
        )
        
        # Compute reward
        reward = self._compute_reward(self.current_state, action_obj, new_state)
        
        # Check if episode is done
        done = self._is_done(new_state)
        
        # Update state
        self.current_state = new_state
        
        # Update history
        self.state_history.append(new_state)
        self.action_history.append(action_obj)
        self.reward_history.append(reward)
        
        # Additional info
        info = {
            'stability_score': physics_quantities['stability_score'],
            'is_stable': physics_quantities['is_stable'],
            'is_traversable': physics_quantities['is_traversable'],
            'anomaly_detected': physics_quantities.get('anomaly_detected', False)
        }
        
        return new_state.to_array(), reward, done, info
    
    def _apply_action(self, state: WormholeState, action: WormholeAction) -> Dict[str, float]:
        """Apply action to current state to get new parameters."""
        
        new_params = {
            'throat_radius': state.throat_radius * (1 + action.delta_throat_radius),
            'exotic_energy_density': state.exotic_energy_density * (1 + action.delta_exotic_density),
            'shape_parameter': state.shape_parameter * (1 + action.delta_shape_parameter),
            'quantum_correction': state.quantum_correction + action.delta_quantum_correction
        }
        
        # Enforce bounds
        for param, value in new_params.items():
            min_val, max_val = self.parameter_bounds[param]
            new_params[param] = np.clip(value, min_val, max_val)
        
        return new_params
    
    def _compute_physics_quantities(self, params: Dict[str, float]) -> Dict[str, Union[float, bool]]:
        """Compute physics quantities from parameters."""
        
        # Extract parameters
        throat_r = params['throat_radius']
        exotic_rho = params['exotic_energy_density']
        shape_param = params['shape_parameter']
        quantum_corr = params['quantum_correction']
        
        # Compute derived quantities (simplified physics)
        energy_density = abs(exotic_rho) * (1 + quantum_corr)
        stress_tensor_trace = -3 * exotic_rho * (1 + quantum_corr * 0.5)
        ricci_scalar = -6 * shape_param / throat_r**2
        
        # Stability assessment
        stability_score = self._assess_stability(params)
        
        # Stability and traversability flags
        is_stable = stability_score > 0.5
        is_traversable = (stability_score > 0.3 and 
                         throat_r > 1e4 and 
                         abs(ricci_scalar) < 1e10)
        
        # Anomaly detection
        anomaly_detected = False
        if self.anomaly_detector and self.anomaly_detector.is_trained:
            try:
                features = self._params_to_features(params)
                prediction = self.anomaly_detector.predict(features.reshape(1, -1))[0]
                anomaly_detected = (prediction == -1)
            except:
                anomaly_detected = False
        
        return {
            'energy_density': energy_density,
            'stress_tensor_trace': stress_tensor_trace,
            'ricci_scalar': ricci_scalar,
            'stability_score': stability_score,
            'is_stable': is_stable,
            'is_traversable': is_traversable,
            'anomaly_detected': anomaly_detected
        }
    
    def _assess_stability(self, params: Dict[str, float]) -> float:
        """Assess stability score using physics-based heuristics or ML model."""
        
        if self.stability_predictor and self.stability_predictor.is_trained:
            try:
                features = self._params_to_features(params)
                stability_prob = self.stability_predictor.predict_proba(features.reshape(1, -1))[0]
                return stability_prob[0] if len(stability_prob) > 1 else stability_prob
            except:
                pass
        
        # Fallback: physics-based heuristics
        throat_r = params['throat_radius']
        exotic_rho = params['exotic_energy_density']
        shape_param = params['shape_parameter']
        quantum_corr = params['quantum_correction']
        
        # Stability factors
        flare_out_factor = max(0, 1 - shape_param / throat_r * 1e-3)
        energy_factor = min(1, abs(exotic_rho) / 1e15)
        quantum_factor = max(0, 1 - abs(quantum_corr - 0.1) / 0.9)
        
        stability_score = (flare_out_factor + energy_factor + quantum_factor) / 3
        return stability_score
    
    def _params_to_features(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters to feature vector for ML models."""
        
        # This should match the feature format expected by the ML models
        throat_r = params['throat_radius']
        exotic_rho = params['exotic_energy_density']
        shape_param = params['shape_parameter']
        quantum_corr = params['quantum_correction']
        
        # Compute derived features (simplified)
        energy_density = abs(exotic_rho)
        radial_pressure = -exotic_rho * (1 + quantum_corr)
        tangential_pressure = -exotic_rho * 0.5
        stress_anisotropy = abs(radial_pressure - tangential_pressure) / energy_density
        
        ricci_scalar = -6 * shape_param / throat_r**2
        kretschmann_scalar = 48 * shape_param**2 / throat_r**4
        
        flare_out = max(0, 1 - shape_param / throat_r)
        tidal_forces = kretschmann_scalar * throat_r**2
        
        features = np.array([
            energy_density, radial_pressure, tangential_pressure,
            stress_anisotropy, ricci_scalar, kretschmann_scalar,
            ricci_scalar/3, exotic_rho, radial_pressure/energy_density,
            1.0, throat_r, -shape_param/throat_r,
            flare_out, quantum_corr * energy_density, quantum_corr * 1e-10,
            1e-8, tidal_forces, abs(ricci_scalar) * throat_r,
            np.sqrt(abs(ricci_scalar))
        ])
        
        return features
    
    def _compute_reward(self, old_state: WormholeState, action: WormholeAction, 
                       new_state: WormholeState) -> float:
        """Compute reward for the transition."""
        
        reward = 0.0
        
        # Stability reward
        stability_reward = self.reward_weights['stability'] * new_state.stability_score
        reward += stability_reward
        
        # Traversability reward
        if new_state.is_traversable:
            traversability_reward = self.reward_weights['traversability']
            reward += traversability_reward
        
        # Energy efficiency (prefer lower energy requirements)
        energy_efficiency = 1.0 / (1.0 + abs(new_state.exotic_energy_density) / 1e15)
        energy_reward = self.reward_weights['energy_efficiency'] * energy_efficiency
        reward += energy_reward
        
        # Smoothness (prefer smaller parameter changes)
        parameter_changes = [
            abs(action.delta_throat_radius),
            abs(action.delta_exotic_density),
            abs(action.delta_shape_parameter),
            abs(action.delta_quantum_correction)
        ]
        smoothness = 1.0 / (1.0 + np.sum(parameter_changes))
        smoothness_reward = self.reward_weights['smoothness'] * smoothness
        reward += smoothness_reward
        
        # Penalty for anomalies
        if hasattr(new_state, 'anomaly_detected') and new_state.anomaly_detected:
            anomaly_penalty = self.reward_weights['anomaly_penalty']
            reward += anomaly_penalty
        
        # Penalty for instability
        if not new_state.is_stable:
            instability_penalty = self.reward_weights['instability_penalty']
            reward += instability_penalty
        
        # Bonus for maintaining stability over time
        if len(self.state_history) > 10:
            recent_stability = np.mean([s.is_stable for s in self.state_history[-10:]])
            if recent_stability > 0.8:
                reward += 5.0  # Consistency bonus
        
        return reward
    
    def _is_done(self, state: WormholeState) -> bool:
        """Check if episode should terminate."""
        
        # Terminate if max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Terminate if severe instability
        if state.stability_score < 0.1:
            return True
        
        # Terminate if anomaly detected
        if hasattr(state, 'anomaly_detected') and state.anomaly_detected:
            return True
        
        # Terminate if physics becomes unphysical
        if (abs(state.ricci_scalar) > 1e15 or 
            abs(state.exotic_energy_density) > 1e25 or
            state.throat_radius < 100):
            return True
        
        return False
    
    def get_observation_space_size(self) -> int:
        """Get size of observation space."""
        dummy_state = WormholeState(
            throat_radius=1e4, exotic_energy_density=-1e15,
            shape_parameter=1.0, quantum_correction=0.1,
            energy_density=1e15, stress_tensor_trace=-3e15,
            ricci_scalar=-6e-8, stability_score=0.5,
            time_step=0, is_stable=True, is_traversable=True
        )
        return len(dummy_state.to_array())
    
    def get_action_space_size(self) -> int:
        """Get size of action space."""
        dummy_action = WormholeAction(
            delta_throat_radius=0.0, delta_exotic_density=0.0,
            delta_shape_parameter=0.0, delta_quantum_correction=0.0,
            stabilization_mode=0, traversal_mode=0
        )
        return len(dummy_action.to_array())


class BaseRLAgent(ABC):
    """Abstract base class for RL agents."""
    
    def __init__(self, agent_name: str, state_size: int, action_size: int):
        """Initialize base RL agent.
        
        Args:
            agent_name: Name of the agent
            state_size: Size of state space
            action_size: Size of action space
        """
        self.agent_name = agent_name
        self.state_size = state_size
        self.action_size = action_size
        self.training_history = []
    
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action given current state."""
        pass
    
    @abstractmethod
    def train(self, state: np.ndarray, action: np.ndarray, reward: float,
              next_state: np.ndarray, done: bool) -> Dict:
        """Train agent on experience tuple."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        pass


class DQNAgent(BaseRLAgent):
    """Deep Q-Network agent for discrete action spaces."""
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 1e-3, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 memory_size: int = 10000, batch_size: int = 32):
        """Initialize DQN agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate for neural network
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for exploration
            epsilon_min: Minimum exploration rate
            memory_size: Size of experience replay buffer
            batch_size: Batch size for training
        """
        super().__init__("DQN", state_size, action_size)
        
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.target_update_frequency = 100
        self.training_step = 0
    
    def _build_model(self) -> Model:
        """Build Q-network."""
        
        inputs = keras.Input(shape=(self.state_size,))
        
        # Hidden layers
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer (Q-values for each action)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        return model
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        
        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            action_indices = np.random.choice(self.action_size, size=self.action_size, replace=False)
            # Convert discrete actions to continuous action space
            return self._discrete_to_continuous_action(action_indices[0])
        else:
            # Greedy action (exploitation)
            state_batch = state.reshape(1, -1)
            q_values = self.q_network.predict(state_batch, verbose=0)[0]
            best_action_index = np.argmax(q_values)
            return self._discrete_to_continuous_action(best_action_index)
    
    def _discrete_to_continuous_action(self, action_index: int) -> np.ndarray:
        """Convert discrete action index to continuous action vector."""
        
        # Define discrete action space
        # For simplicity, we define a limited set of actions
        
        actions = [
            # [delta_throat, delta_exotic, delta_shape, delta_quantum, stab_mode, trav_mode]
            [0.0, 0.0, 0.0, 0.0, 0, 0],        # No action
            [0.05, 0.0, 0.0, 0.0, 1, 0],       # Increase throat radius
            [-0.05, 0.0, 0.0, 0.0, 1, 0],      # Decrease throat radius
            [0.0, 0.1, 0.0, 0.0, 1, 0],        # Increase exotic energy
            [0.0, -0.1, 0.0, 0.0, 1, 0],       # Decrease exotic energy
            [0.0, 0.0, 0.05, 0.0, 1, 0],       # Increase shape parameter
            [0.0, 0.0, -0.05, 0.0, 1, 0],      # Decrease shape parameter
            [0.0, 0.0, 0.0, 0.02, 1, 0],       # Increase quantum correction
            [0.0, 0.0, 0.0, -0.02, 1, 0],      # Decrease quantum correction
            [0.0, 0.0, 0.0, 0.0, 2, 0],        # Emergency stabilization
            [0.0, 0.0, 0.0, 0.0, 0, 1],        # Prepare traversal
            [0.0, 0.0, 0.0, 0.0, 0, 2],        # Execute traversal
        ]
        
        action_index = min(action_index, len(actions) - 1)
        return np.array(actions[action_index], dtype=np.float32)
    
    def train(self, state: np.ndarray, action: np.ndarray, reward: float,
              next_state: np.ndarray, done: bool) -> Dict:
        """Train DQN using experience replay."""
        
        # Convert continuous action back to discrete index (simplified)
        action_index = self._continuous_to_discrete_action(action)
        
        # Store experience in replay buffer
        self.memory.append((state, action_index, reward, next_state, done))
        
        training_info = {'loss': 0.0, 'q_value': 0.0}
        
        # Train if we have enough experiences
        if len(self.memory) >= self.batch_size:
            loss, avg_q = self._replay_train()
            training_info['loss'] = loss
            training_info['q_value'] = avg_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.update_target_network()
        
        return training_info
    
    def _continuous_to_discrete_action(self, action: np.ndarray) -> int:
        """Convert continuous action to discrete index (simplified)."""
        # This is a simplified reverse mapping - in practice would use proper discretization
        return 0  # Default action
    
    def _replay_train(self) -> Tuple[float, float]:
        """Train network on batch of experiences."""
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Compute target Q-values
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train network
        history = self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Average Q-value
        avg_q = np.mean(current_q_values)
        
        return loss, avg_q
    
    def update_target_network(self):
        """Update target network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath: str) -> None:
        """Save DQN model."""
        self.q_network.save(f"{filepath}_q_network.h5")
        
        # Save agent parameters
        agent_data = {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'memory': list(self.memory),
            'training_history': self.training_history
        }
        
        with open(f"{filepath}_agent_data.pkl", 'wb') as f:
            pickle.dump(agent_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load DQN model."""
        self.q_network = keras.models.load_model(f"{filepath}_q_network.h5")
        self.target_network = self._build_model()
        self.update_target_network()
        
        try:
            with open(f"{filepath}_agent_data.pkl", 'rb') as f:
                agent_data = pickle.load(f)
            
            self.epsilon = agent_data.get('epsilon', self.epsilon)
            self.training_step = agent_data.get('training_step', 0)
            self.memory = deque(agent_data.get('memory', []), maxlen=10000)
            self.training_history = agent_data.get('training_history', [])
        except FileNotFoundError:
            print("Agent data file not found, using default parameters")


class PPOAgent(BaseRLAgent):
    """Proximal Policy Optimization agent for continuous control."""
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        """Initialize PPO agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        super().__init__("PPO", state_size, action_size)
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Networks
        self.actor_critic = self._build_actor_critic()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def _build_actor_critic(self) -> Model:
        """Build actor-critic network."""
        
        inputs = keras.Input(shape=(self.state_size,))
        
        # Shared hidden layers
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Actor head (policy)
        actor_hidden = layers.Dense(64, activation='relu')(x)
        
        # Action means
        action_means = layers.Dense(self.action_size, activation='tanh')(actor_hidden)
        
        # Action log standard deviations
        action_log_stds = layers.Dense(self.action_size, activation='linear')(actor_hidden)
        
        # Critic head (value function)
        critic_hidden = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1, activation='linear')(critic_hidden)
        
        model = Model(inputs=inputs, outputs=[action_means, action_log_stds, value])
        
        return model
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy."""
        
        state_batch = state.reshape(1, -1)
        action_means, action_log_stds, value = self.actor_critic(state_batch)
        
        # Sample from Gaussian policy
        action_stds = tf.exp(action_log_stds)
        action_dist = tfp.distributions.Normal(action_means, action_stds)
        
        if training:
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Store for training
            self.states.append(state)
            self.actions.append(action.numpy()[0])
            self.values.append(value.numpy()[0, 0])
            self.log_probs.append(tf.reduce_sum(log_prob).numpy())
        else:
            action = action_means  # Use mean for evaluation
        
        # Clip action to valid range
        action = tf.clip_by_value(action, -1.0, 1.0)
        
        return action.numpy()[0]
    
    def train(self, state: np.ndarray, action: np.ndarray, reward: float,
              next_state: np.ndarray, done: bool) -> Dict:
        """Store experience and train when buffer is full."""
        
        # Store reward and done flag
        self.rewards.append(reward)
        self.dones.append(done)
        
        # Train when episode ends or buffer is full
        if done or len(self.rewards) >= 64:  # Batch size
            training_info = self._train_ppo()
            self._clear_buffer()
            return training_info
        
        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
    
    def _train_ppo(self) -> Dict:
        """Train using PPO algorithm."""
        
        if len(self.states) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Convert to tensors
        states = tf.constant(np.array(self.states), dtype=tf.float32)
        actions = tf.constant(np.array(self.actions), dtype=tf.float32)
        old_log_probs = tf.constant(np.array(self.log_probs), dtype=tf.float32)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = tf.constant(advantages, dtype=tf.float32)
        returns = tf.constant(returns, dtype=tf.float32)
        
        # Normalize advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        # PPO training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(10):  # PPO epochs
            
            with tf.GradientTape() as tape:
                # Forward pass
                action_means, action_log_stds, current_values = self.actor_critic(states)
                action_stds = tf.exp(action_log_stds)
                
                # Policy distribution
                action_dist = tfp.distributions.Normal(action_means, action_stds)
                new_log_probs = tf.reduce_sum(action_dist.log_prob(actions), axis=1)
                
                # Importance sampling ratio
                ratio = tf.exp(new_log_probs - old_log_probs)
                
                # Clipped objective
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                policy_loss_1 = ratio * advantages
                policy_loss_2 = clipped_ratio * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(policy_loss_1, policy_loss_2))
                
                # Value loss
                current_values = tf.squeeze(current_values)
                value_loss = tf.reduce_mean(tf.square(returns - current_values))
                
                # Entropy bonus
                entropy = tf.reduce_mean(action_dist.entropy())
                
                # Total loss
                total_loss = (policy_loss + 
                            self.value_coef * value_loss - 
                            self.entropy_coef * entropy)
            
            # Compute gradients and update
            gradients = tape.gradient(total_loss, self.actor_critic.trainable_variables)
            gradients = [tf.clip_by_norm(g, 0.5) for g in gradients]  # Gradient clipping
            self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))
            
            total_policy_loss += policy_loss.numpy()
            total_value_loss += value_loss.numpy()
            total_entropy += entropy.numpy()
        
        return {
            'policy_loss': total_policy_loss / 10,
            'value_loss': total_value_loss / 10,
            'entropy': total_entropy / 10
        }
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                    dones: np.ndarray, lambda_gae: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * lambda_gae * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def _clear_buffer(self):
        """Clear experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save_model(self, filepath: str) -> None:
        """Save PPO model."""
        self.actor_critic.save(f"{filepath}_actor_critic.h5")
    
    def load_model(self, filepath: str) -> None:
        """Load PPO model."""
        self.actor_critic = keras.models.load_model(f"{filepath}_actor_critic.h5")


def train_rl_agent(agent: BaseRLAgent, environment: WormholeEnvironment,
                  num_episodes: int = 1000, max_steps_per_episode: int = 500,
                  save_frequency: int = 100) -> Dict:
    """Train RL agent on wormhole control task.
    
    Args:
        agent: RL agent to train
        environment: Wormhole environment
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_frequency: Frequency of model saving
    
    Returns:
        Training statistics
    """
    
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    
    best_reward = -np.inf
    
    for episode in range(num_episodes):
        
        state = environment.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(max_steps_per_episode):
            
            # Select action
            action = agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = environment.step(action)
            
            # Train agent
            training_info = agent.train(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if 'loss' in training_info:
            training_losses.append(training_info['loss'])
        elif 'policy_loss' in training_info:
            training_losses.append(training_info['policy_loss'])
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(f"best_{agent.agent_name.lower()}_model")
        
        # Periodic saving
        if (episode + 1) % save_frequency == 0:
            agent.save_model(f"{agent.agent_name.lower()}_model_episode_{episode+1}")
        
        # Progress reporting
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.1f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_losses': training_losses,
        'best_reward': best_reward
    }


def evaluate_rl_agent(agent: BaseRLAgent, environment: WormholeEnvironment,
                     num_episodes: int = 100) -> Dict:
    """Evaluate trained RL agent."""
    
    episode_rewards = []
    success_rates = []
    stability_scores = []
    
    for episode in range(num_episodes):
        
        state = environment.reset()
        episode_reward = 0
        stability_sum = 0
        step_count = 0
        
        for step in range(500):  # Max evaluation steps
            
            # Select action (no exploration)
            action = agent.act(state, training=False)
            
            # Execute action
            next_state, reward, done, info = environment.step(action)
            
            episode_reward += reward
            stability_sum += info.get('stability_score', 0)
            step_count += 1
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        success_rates.append(1 if episode_reward > 0 else 0)
        stability_scores.append(stability_sum / step_count if step_count > 0 else 0)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean(success_rates),
        'mean_stability': np.mean(stability_scores),
        'episode_rewards': episode_rewards
    }


def visualize_training_results(training_results: Dict, save_path: Optional[str] = None):
    """Visualize RL training results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    rewards = training_results['episode_rewards']
    axes[0, 0].plot(rewards, alpha=0.6)
    
    # Moving average
    window = min(50, len(rewards) // 10)
    if window > 1:
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        axes[0, 0].plot(moving_avg, color='red', linewidth=2)
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    lengths = training_results['episode_lengths']
    axes[0, 1].plot(lengths, alpha=0.6, color='green')
    
    if window > 1:
        moving_avg_lengths = pd.Series(lengths).rolling(window=window).mean()
        axes[0, 1].plot(moving_avg_lengths, color='darkgreen', linewidth=2)
    
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Training losses
    if training_results['training_losses']:
        losses = [loss for loss in training_results['training_losses'] if loss > 0]
        axes[1, 0].plot(losses, alpha=0.6, color='orange')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
    
    # Reward distribution
    axes[1, 1].hist(rewards, bins=30, alpha=0.7, color='purple')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# Factory function for creating RL agents
def create_rl_agent(agent_type: str, state_size: int, action_size: int, **kwargs) -> BaseRLAgent:
    """Factory function for creating RL agents.
    
    Args:
        agent_type: Type of agent ('DQN', 'PPO')
        state_size: Size of observation space
        action_size: Size of action space
        **kwargs: Agent-specific parameters
    
    Returns:
        Initialized RL agent
    """
    
    if agent_type.upper() == 'DQN':
        return DQNAgent(state_size, action_size, **kwargs)
    elif agent_type.upper() == 'PPO':
        return PPOAgent(state_size, action_size, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")