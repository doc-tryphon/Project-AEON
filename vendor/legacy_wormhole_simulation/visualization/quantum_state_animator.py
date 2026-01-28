"""
Real-time quantum evolution animations showing entanglement dynamics, state vector evolution, and quantum teleportation through wormholes.

This module provides advanced animation capabilities for quantum states, including:
- Bloch sphere evolution
- Entanglement dynamics visualization  
- Quantum teleportation animations
- Multi-particle quantum systems
- Wavefunction probability distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
import qutip as qt
from scipy.special import sph_harm
from scipy.integrate import solve_ivp

from src.quantum.wormhole_circuit import WormholeQuantumCircuit
from src.quantum.entanglement_dynamics import EntanglementDynamics
from src.physics.constants import HBAR, C


@dataclass
class AnimationConfig:
    """Configuration for quantum state animations."""
    
    # Time parameters
    total_time: float = 10.0
    time_steps: int = 200
    frame_duration: float = 50  # milliseconds
    
    # Visual parameters
    sphere_resolution: int = 20
    colormap: str = 'viridis'
    opacity: float = 0.8
    line_width: float = 2.0
    marker_size: float = 8.0
    
    # Animation parameters
    loop_animation: bool = True
    save_frames: bool = False
    output_format: str = 'html'
    
    # Quantum system parameters
    max_particles: int = 8
    hilbert_space_dim: int = 1024
    entanglement_threshold: float = 0.1


class QuantumStateAnimator:
    """Advanced quantum state animation and visualization."""
    
    def __init__(self, config: AnimationConfig = None):
        """Initialize quantum state animator.
        
        Args:
            config: Animation configuration
        """
        self.config = config or AnimationConfig()
        
        # Time evolution parameters
        self.time_points = np.linspace(0, self.config.total_time, self.config.time_steps)
        self.dt = self.time_points[1] - self.time_points[0]
        
        # Quantum system components
        self.wormhole_circuit = None
        self.entanglement_dynamics = None
        
        # Animation state
        self.current_frame = 0
        self.animation_data = {}
        
    def setup_wormhole_system(self, num_qubits: int = 4, 
                            traversal_probability: float = 0.8):
        """Setup wormhole quantum system for animation.
        
        Args:
            num_qubits: Number of qubits in the system
            traversal_probability: Probability of successful wormhole traversal
        """
        self.wormhole_circuit = WormholeQuantumCircuit(
            num_qubits=num_qubits,
            traversal_probability=traversal_probability
        )
        
        self.entanglement_dynamics = EntanglementDynamics(
            num_particles=num_qubits//2,
            dimension=2**num_qubits
        )
        
    def animate_bloch_sphere_evolution(self, initial_state: qt.Qobj = None,
                                     hamiltonian: qt.Qobj = None) -> go.Figure:
        """Animate quantum state evolution on Bloch sphere.
        
        Args:
            initial_state: Initial quantum state
            hamiltonian: System Hamiltonian
        
        Returns:
            Animated Plotly figure
        """
        
        # Default initial state
        if initial_state is None:
            initial_state = qt.basis(2, 0)  # |0⟩ state
        
        # Default Hamiltonian (random for demonstration)
        if hamiltonian is None:
            hamiltonian = qt.rand_herm(2)
        
        # Compute time evolution
        states = []
        bloch_vectors = []
        
        for t in self.time_points:
            # Time evolution operator
            U = (-1j * hamiltonian * t).expm()
            evolved_state = U * initial_state
            states.append(evolved_state)
            
            # Compute Bloch vector
            bloch_vec = qt.expect([qt.sigmax(), qt.sigmay(), qt.sigmaz()], evolved_state)
            bloch_vectors.append(bloch_vec)
        
        bloch_vectors = np.array(bloch_vectors)
        
        # Create Bloch sphere
        u = np.linspace(0, 2 * np.pi, self.config.sphere_resolution)
        v = np.linspace(0, np.pi, self.config.sphere_resolution)
        u_mesh, v_mesh = np.meshgrid(u, v)
        
        x_sphere = np.cos(u_mesh) * np.sin(v_mesh)
        y_sphere = np.sin(u_mesh) * np.sin(v_mesh)
        z_sphere = np.cos(v_mesh)
        
        # Create frames for animation
        frames = []
        
        for i, t in enumerate(self.time_points):
            frame_data = []
            
            # Bloch sphere surface
            frame_data.append(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.3,
                colorscale='Blues',
                showscale=False,
                name='Bloch Sphere'
            ))
            
            # State trajectory up to current time
            if i > 0:
                frame_data.append(go.Scatter3d(
                    x=bloch_vectors[:i+1, 0],
                    y=bloch_vectors[:i+1, 1], 
                    z=bloch_vectors[:i+1, 2],
                    mode='lines',
                    line=dict(width=4, color='red'),
                    name='State Trajectory'
                ))
            
            # Current state vector
            frame_data.append(go.Scatter3d(
                x=[0, bloch_vectors[i, 0]],
                y=[0, bloch_vectors[i, 1]],
                z=[0, bloch_vectors[i, 2]],
                mode='lines+markers',
                line=dict(width=6, color='yellow'),
                marker=dict(size=8, color='yellow'),
                name='State Vector'
            ))
            
            # Coordinate axes
            frame_data.extend([
                go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0],
                           mode='lines', line=dict(color='black', width=2),
                           showlegend=False),
                go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0],
                           mode='lines', line=dict(color='black', width=2),
                           showlegend=False),
                go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2],
                           mode='lines', line=dict(color='black', width=2),
                           showlegend=False)
            ])
            
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        # Create figure with initial frame
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        # Add animation controls
        fig.update_layout(
            title='Quantum State Evolution on Bloch Sphere',
            scene=dict(
                xaxis_title='σₓ',
                yaxis_title='σᵧ',
                zaxis_title='σᵤ',
                aspectmode='cube',
                xaxis=dict(range=[-1.2, 1.2]),
                yaxis=dict(range=[-1.2, 1.2]),
                zaxis=dict(range=[-1.2, 1.2])
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": self.config.frame_duration, 
                                              "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate"}])
                ]
            )],
            width=800,
            height=600
        )
        
        return fig
    
    def animate_entanglement_dynamics(self, initial_entanglement: float = 0.0,
                                    interaction_strength: float = 1.0) -> go.Figure:
        """Animate entanglement dynamics between quantum systems.
        
        Args:
            initial_entanglement: Initial entanglement measure
            interaction_strength: Strength of system interaction
        
        Returns:
            Animated figure showing entanglement evolution
        """
        
        if self.entanglement_dynamics is None:
            self.setup_wormhole_system()
        
        # Generate entanglement evolution data
        entanglement_measures = []
        quantum_states = []
        
        for t in self.time_points:
            # Simulate entanglement growth/decay
            concurrence = initial_entanglement + interaction_strength * np.sin(t) * np.exp(-0.1*t)
            concurrence = np.clip(concurrence, 0, 1)
            
            # Negativity measure
            negativity = 0.5 * (np.sqrt(1 + concurrence**2) - 1)
            
            # Von Neumann entropy
            entropy = -concurrence * np.log2(concurrence + 1e-10) if concurrence > 0 else 0
            
            entanglement_measures.append({
                'time': t,
                'concurrence': concurrence,
                'negativity': negativity,
                'entropy': entropy
            })
            
            # Generate quantum state visualization data
            # Simplified density matrix representation
            rho = np.array([[0.5 + 0.3*concurrence, 0.2*np.exp(1j*t)],
                           [0.2*np.exp(-1j*t), 0.5 - 0.3*concurrence]])
            
            quantum_states.append(rho)
        
        # Create animated subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Entanglement Measures', 'Quantum State Real Part',
                          'Quantum State Imaginary Part', 'Bloch Vectors'],
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'scatter3d'}]]
        )
        
        frames = []
        
        for i, t in enumerate(self.time_points):
            frame_data = []
            
            # Entanglement measures time series
            times_so_far = [m['time'] for m in entanglement_measures[:i+1]]
            concurrences = [m['concurrence'] for m in entanglement_measures[:i+1]]
            negativities = [m['negativity'] for m in entanglement_measures[:i+1]]
            entropies = [m['entropy'] for m in entanglement_measures[:i+1]]
            
            frame_data.append(go.Scatter(
                x=times_so_far, y=concurrences,
                mode='lines', name='Concurrence',
                line=dict(color='red', width=3)
            ))
            frame_data.append(go.Scatter(
                x=times_so_far, y=negativities,
                mode='lines', name='Negativity',
                line=dict(color='blue', width=3)
            ))
            frame_data.append(go.Scatter(
                x=times_so_far, y=entropies,
                mode='lines', name='Entropy',
                line=dict(color='green', width=3)
            ))
            
            # Quantum state density matrix (real part)
            rho_real = np.real(quantum_states[i])
            frame_data.append(go.Heatmap(
                z=rho_real,
                colorscale='RdBu',
                zmid=0,
                showscale=False
            ))
            
            # Quantum state density matrix (imaginary part)  
            rho_imag = np.imag(quantum_states[i])
            frame_data.append(go.Heatmap(
                z=rho_imag,
                colorscale='RdBu',
                zmid=0,
                showscale=False
            ))
            
            # Bloch vectors for entangled qubits
            # Generate correlated Bloch vectors
            phi = 2 * np.pi * t / self.config.total_time
            bloch1 = [np.cos(phi), np.sin(phi), concurrences[i]]
            bloch2 = [-np.cos(phi), -np.sin(phi), -concurrences[i]]
            
            frame_data.append(go.Scatter3d(
                x=[0, bloch1[0]], y=[0, bloch1[1]], z=[0, bloch1[2]],
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=6, color='red'),
                name='Qubit 1'
            ))
            frame_data.append(go.Scatter3d(
                x=[0, bloch2[0]], y=[0, bloch2[1]], z=[0, bloch2[2]],
                mode='lines+markers',
                line=dict(color='blue', width=4),
                marker=dict(size=6, color='blue'),
                name='Qubit 2'
            ))
            
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        fig.update_layout(
            title='Quantum Entanglement Dynamics',
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate", 
                         args=[None, {"frame": {"duration": self.config.frame_duration,
                                              "redraw": True}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": True}}])
                ]
            )],
            height=800,
            width=1000
        )
        
        return fig
    
    def animate_quantum_teleportation(self, teleported_state: qt.Qobj = None) -> go.Figure:
        """Animate quantum teleportation through wormhole.
        
        Args:
            teleported_state: Quantum state to teleport
        
        Returns:
            Animation showing teleportation process
        """
        
        if teleported_state is None:
            # Create interesting superposition state
            teleported_state = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
        
        if self.wormhole_circuit is None:
            self.setup_wormhole_system()
        
        # Teleportation protocol phases
        phases = [
            'Initial State Preparation',
            'Entanglement Creation', 
            'Bell State Measurement',
            'Classical Communication',
            'Unitary Correction',
            'Teleportation Complete'
        ]
        
        frames = []
        
        for i, phase in enumerate(phases):
            frame_data = []
            
            # Progress indicator
            progress = (i + 1) / len(phases)
            
            # Wormhole visualization
            theta = np.linspace(0, 2*np.pi, 50)
            r_wormhole = 2 + 0.5 * np.sin(4*theta + progress*10)  # Dynamic wormhole
            
            x_wormhole = r_wormhole * np.cos(theta)
            y_wormhole = r_wormhole * np.sin(theta)
            z_wormhole = np.zeros_like(theta)
            
            frame_data.append(go.Scatter3d(
                x=x_wormhole, y=y_wormhole, z=z_wormhole,
                mode='lines',
                line=dict(color='purple', width=8),
                name='Wormhole Throat'
            ))
            
            # Alice's side (sender)
            alice_x, alice_y, alice_z = -4, 0, 0
            alice_color = 'red' if i < 3 else 'orange'
            
            frame_data.append(go.Scatter3d(
                x=[alice_x], y=[alice_y], z=[alice_z],
                mode='markers+text',
                marker=dict(size=15, color=alice_color),
                text=['Alice'], textposition='top center',
                name='Alice (Sender)'
            ))
            
            # Bob's side (receiver)
            bob_x, bob_y, bob_z = 4, 0, 0
            bob_color = 'blue' if i < 5 else 'green'
            
            frame_data.append(go.Scatter3d(
                x=[bob_x], y=[bob_y], z=[bob_z],
                mode='markers+text',
                marker=dict(size=15, color=bob_color),
                text=['Bob'], textposition='top center',
                name='Bob (Receiver)'
            ))
            
            # Quantum state visualization
            if i == 0:  # Initial state
                # Show original state at Alice
                frame_data.append(go.Scatter3d(
                    x=[alice_x], y=[alice_y + 0.5], z=[alice_z + 0.5],
                    mode='markers',
                    marker=dict(size=10, color='gold', symbol='diamond'),
                    name='Quantum State'
                ))
                
            elif i == 1:  # Entanglement creation
                # Show entangled pair
                frame_data.extend([
                    go.Scatter3d(
                        x=[alice_x + 0.5], y=[alice_y], z=[alice_z + 0.5],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='circle'),
                        name='Entangled Qubit A'
                    ),
                    go.Scatter3d(
                        x=[bob_x - 0.5], y=[bob_y], z=[bob_z + 0.5],
                        mode='markers',
                        marker=dict(size=8, color='blue', symbol='circle'),
                        name='Entangled Qubit B'
                    )
                ])
                
                # Entanglement connection
                frame_data.append(go.Scatter3d(
                    x=[alice_x + 0.5, bob_x - 0.5],
                    y=[alice_y, bob_y],
                    z=[alice_z + 0.5, bob_z + 0.5],
                    mode='lines',
                    line=dict(color='magenta', width=4, dash='dash'),
                    name='Entanglement'
                ))
                
            elif i >= 2:  # Measurement and beyond
                # Show classical information transfer
                if i >= 3:
                    # Classical bits traveling through wormhole
                    travel_progress = (i - 3) / 3
                    bit_x = alice_x + travel_progress * (bob_x - alice_x)
                    
                    frame_data.append(go.Scatter3d(
                        x=[bit_x], y=[0], z=[1],
                        mode='markers+text',
                        marker=dict(size=8, color='yellow', symbol='square'),
                        text=['11'], textposition='top center',
                        name='Classical Bits'
                    ))
                
                # Show state reconstruction at Bob's side
                if i >= 4:
                    reconstruction_progress = (i - 4) / 2
                    state_alpha = min(1.0, reconstruction_progress)
                    
                    frame_data.append(go.Scatter3d(
                        x=[bob_x], y=[bob_y + 0.5], z=[bob_z + 0.5],
                        mode='markers',
                        marker=dict(size=10, color='gold', symbol='diamond', 
                                  opacity=state_alpha),
                        name='Reconstructed State'
                    ))
            
            # Phase indicator
            frame_data.append(go.Scatter3d(
                x=[0], y=[0], z=[3],
                mode='markers+text',
                marker=dict(size=1, color='white'),
                text=[phase], textposition='middle center',
                showlegend=False
            ))
            
            frames.append(go.Frame(data=frame_data, name=phase))
        
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        fig.update_layout(
            title='Quantum Teleportation Through Wormhole',
            scene=dict(
                xaxis_title='Spatial Dimension',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube',
                xaxis=dict(range=[-5, 5]),
                yaxis=dict(range=[-2, 2]),
                zaxis=dict(range=[-1, 4])
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 2000, "redraw": True}}]),
                    dict(label="Pause",
                         method="animate", 
                         args=[[None], {"frame": {"duration": 0, "redraw": True}}])
                ]
            )],
            width=900,
            height=700
        )
        
        return fig
    
    def animate_wavefunction_evolution(self, potential: Callable[[float], float] = None,
                                     initial_wavefunction: Callable[[float], complex] = None) -> go.Figure:
        """Animate wavefunction evolution in wormhole potential.
        
        Args:
            potential: Wormhole potential function V(x)
            initial_wavefunction: Initial wavefunction ψ(x,0)
        
        Returns:
            Animation of wavefunction evolution
        """
        
        # Spatial grid
        x_min, x_max = -10, 10
        nx = 500
        x = np.linspace(x_min, x_max, nx)
        dx = x[1] - x[0]
        
        # Default wormhole-like potential
        if potential is None:
            def potential(x):
                return -50 * np.exp(-(x**2)/2) + x**4/100  # Attractive well with barrier
        
        V = np.array([potential(xi) for xi in x])
        
        # Default initial Gaussian wavepacket
        if initial_wavefunction is None:
            def initial_wavefunction(x):
                return np.exp(-(x + 3)**2/2) * np.exp(1j * 2 * x)
        
        psi_0 = np.array([initial_wavefunction(xi) for xi in x])
        psi_0 = psi_0 / np.sqrt(np.trapz(np.abs(psi_0)**2, x))  # Normalize
        
        # Time evolution using split-operator method
        def evolve_wavefunction(psi, dt):
            """Evolve wavefunction by time step dt using split-operator method."""
            
            # Kinetic energy evolution in momentum space
            psi_k = np.fft.fft(psi)
            k = np.fft.fftfreq(nx, dx) * 2 * np.pi
            kinetic_phase = np.exp(-1j * HBAR * k**2 * dt / (4 * 9.1e-31))  # m_e
            psi_k *= kinetic_phase
            psi = np.fft.ifft(psi_k)
            
            # Potential energy evolution in position space
            potential_phase = np.exp(-1j * V * dt / (2 * HBAR))
            psi *= potential_phase
            
            # Second kinetic evolution
            psi_k = np.fft.fft(psi)
            psi_k *= kinetic_phase
            psi = np.fft.ifft(psi_k)
            
            return psi
        
        # Compute time evolution
        wavefunctions = []
        psi = psi_0.copy()
        
        for t in self.time_points:
            wavefunctions.append(psi.copy())
            if t < self.time_points[-1]:
                psi = evolve_wavefunction(psi, self.dt)
        
        # Create animation frames
        frames = []
        
        for i, t in enumerate(self.time_points):
            psi_t = wavefunctions[i]
            prob_density = np.abs(psi_t)**2
            real_part = np.real(psi_t)
            imag_part = np.imag(psi_t)
            
            frame_data = []
            
            # Probability density
            frame_data.append(go.Scatter(
                x=x, y=prob_density,
                mode='lines',
                fill='tonexty',
                line=dict(color='blue', width=2),
                name='|ψ|²'
            ))
            
            # Real part
            frame_data.append(go.Scatter(
                x=x, y=real_part,
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Re(ψ)'
            ))
            
            # Imaginary part
            frame_data.append(go.Scatter(
                x=x, y=imag_part,
                mode='lines',
                line=dict(color='green', width=2, dash='dot'),
                name='Im(ψ)'
            ))
            
            # Potential (scaled for visibility)
            V_scaled = V / np.max(np.abs(V)) * np.max(prob_density) * 0.5
            frame_data.append(go.Scatter(
                x=x, y=V_scaled,
                mode='lines',
                line=dict(color='black', width=3),
                name='V(x) (scaled)'
            ))
            
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        fig.update_layout(
            title='Quantum Wavefunction Evolution in Wormhole Potential',
            xaxis_title='Position x',
            yaxis_title='Amplitude',
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": self.config.frame_duration,
                                              "redraw": True}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": True}}])
                ]
            )],
            width=800,
            height=600
        )
        
        return fig
    
    def animate_multi_particle_system(self, num_particles: int = 3) -> go.Figure:
        """Animate multi-particle quantum system evolution.
        
        Args:
            num_particles: Number of particles in the system
        
        Returns:
            Multi-particle animation figure
        """
        
        if num_particles > self.config.max_particles:
            num_particles = self.config.max_particles
        
        # Generate particle trajectories in configuration space
        particle_trajectories = []
        
        for p in range(num_particles):
            # Each particle follows a different path
            phase_offset = 2 * np.pi * p / num_particles
            
            x_traj = 2 * np.cos(self.time_points + phase_offset) * np.exp(-0.1 * self.time_points)
            y_traj = 2 * np.sin(self.time_points + phase_offset) * np.exp(-0.1 * self.time_points)
            z_traj = np.sin(2 * self.time_points + phase_offset)
            
            particle_trajectories.append((x_traj, y_traj, z_traj))
        
        # Create frames
        frames = []
        colors = px.colors.qualitative.Set1[:num_particles]
        
        for i, t in enumerate(self.time_points):
            frame_data = []
            
            for p in range(num_particles):
                x_traj, y_traj, z_traj = particle_trajectories[p]
                
                # Current position
                frame_data.append(go.Scatter3d(
                    x=[x_traj[i]], y=[y_traj[i]], z=[z_traj[i]],
                    mode='markers',
                    marker=dict(size=12, color=colors[p]),
                    name=f'Particle {p+1}'
                ))
                
                # Trajectory trail
                if i > 0:
                    trail_length = min(i, 20)  # Show last 20 points
                    start_idx = max(0, i - trail_length)
                    
                    frame_data.append(go.Scatter3d(
                        x=x_traj[start_idx:i+1],
                        y=y_traj[start_idx:i+1], 
                        z=z_traj[start_idx:i+1],
                        mode='lines',
                        line=dict(color=colors[p], width=4, 
                                alpha=np.linspace(0.1, 1.0, i+1-start_idx)),
                        showlegend=False
                    ))
            
            # Add quantum correlations visualization
            if num_particles >= 2:
                # Draw entanglement connections
                for p1 in range(num_particles):
                    for p2 in range(p1+1, num_particles):
                        x1, y1, z1 = (particle_trajectories[p1][j][i] for j in range(3))
                        x2, y2, z2 = (particle_trajectories[p2][j][i] for j in range(3))
                        
                        # Correlation strength based on distance
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                        correlation = np.exp(-distance/5)  # Exponential decay
                        
                        if correlation > 0.1:  # Only show strong correlations
                            frame_data.append(go.Scatter3d(
                                x=[x1, x2], y=[y1, y2], z=[z1, z2],
                                mode='lines',
                                line=dict(color='purple', width=2*correlation, dash='dash'),
                                opacity=correlation,
                                showlegend=False
                            ))
            
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        fig = go.Figure(data=frames[0].data, frames=frames)
        
        fig.update_layout(
            title=f'Multi-Particle Quantum System ({num_particles} particles)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": self.config.frame_duration,
                                              "redraw": True}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": True}}])
                ]
            )],
            width=900,
            height=700
        )
        
        return fig
    
    def create_comprehensive_quantum_animation(self) -> go.Figure:
        """Create comprehensive multi-panel quantum animation.
        
        Returns:
            Multi-panel figure with various quantum visualizations
        """
        
        # Setup quantum system
        self.setup_wormhole_system()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Bloch Sphere Evolution', 'Entanglement Dynamics',
                          'Wavefunction Evolution', 'Multi-Particle System'],
            specs=[[{'type': 'scene'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scene'}]]
        )
        
        # This would need significant work to combine multiple animations
        # For now, return a placeholder that shows the capability
        
        fig.update_layout(
            title='Comprehensive Quantum State Animation Suite',
            height=800,
            width=1200
        )
        
        return fig
    
    def export_animation(self, figure: go.Figure, filename: str,
                        format_type: str = None) -> None:
        """Export animation to file.
        
        Args:
            figure: Plotly figure to export
            filename: Output filename
            format_type: Export format ('html', 'gif', 'mp4')
        """
        
        if format_type is None:
            format_type = self.config.output_format
        
        if format_type == 'html':
            figure.write_html(filename)
        elif format_type == 'gif':
            # Would require additional setup for gif export
            print(f"GIF export not implemented. Saving as HTML: {filename}.html")
            figure.write_html(filename + '.html')
        elif format_type == 'mp4':
            # Would require additional setup for video export  
            print(f"MP4 export not implemented. Saving as HTML: {filename}.html")
            figure.write_html(filename + '.html')
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def save_animation_data(self, data: Dict[str, Any], filename: str) -> None:
        """Save animation data for later analysis.
        
        Args:
            data: Animation data dictionary
            filename: Output filename
        """
        np.save(filename, data)


def create_quantum_animation_suite(config: AnimationConfig = None) -> Dict[str, go.Figure]:
    """Create complete suite of quantum animations.
    
    Args:
        config: Animation configuration
    
    Returns:
        Dictionary of animation figures
    """
    
    animator = QuantumStateAnimator(config)
    animator.setup_wormhole_system()
    
    animations = {
        'bloch_evolution': animator.animate_bloch_sphere_evolution(),
        'entanglement_dynamics': animator.animate_entanglement_dynamics(),
        'quantum_teleportation': animator.animate_quantum_teleportation(),
        'wavefunction_evolution': animator.animate_wavefunction_evolution(),
        'multi_particle': animator.animate_multi_particle_system()
    }
    
    return animations