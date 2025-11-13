import numpy as np

#!/usr/bin/env python3
"""
Generate a single-qubit quantum state from Bloch-sphere angles and optionally plot it.

Usage: run the file. Adjust theta, phi in the main() example or call bloch_state / plot_bloch from your code.
"""
import matplotlib
matplotlib.use('Agg')  # Add this before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting backend)


def bloch_stateS(theta: float, phi: float) -> np.ndarray:
    """
    Return the state vector |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    theta: polar angle in radians (0..pi)
    phi: azimuthal angle in radians (0..2pi)
    Returns a 2-element complex numpy array normalized to 1.
    """
    a = np.cos(theta / 2.0)
    b = np.exp(1j * phi) * np.sin(theta / 2.0)
    state = np.array([a, b], dtype=complex)
    # numerical normalization guard
    state = state / np.linalg.norm(state)
    return state


def bloch_vector(theta: float, phi: float) -> np.ndarray:
    """
    Convert Bloch angles to the 3D Bloch vector (x, y, z).
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def plot_bloch(theta: float, phi: float, ax=None):
    """
    Plot the Bloch sphere and the state vector for given angles.
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = None

    # sphere
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=(0.9, 0.9, 0.95), alpha=0.35, linewidth=0)

    # axes
    ax.quiver(0, 0, 0, 1.0, 0, 0, color="r", length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 1.0, 0, color="g", length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, 0, 0, 1.0, color="b", length=1.0, arrow_length_ratio=0.08)

    # state vector
    v = bloch_vector(theta, phi)
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color="k", linewidth=2.5, arrow_length_ratio=0.12)

    # aesthetics
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Bloch vector (θ={theta:.2f}, φ={phi:.2f})")
    if fig:
        plt.show()


def state_to_bloch_angles(state: np.ndarray) -> tuple:
    """
    Convert a normalized 2-element state vector to Bloch angles (theta, phi).
    Returns theta in [0, pi], phi in (-pi, pi].
    """
    state = np.asarray(state, dtype=complex).reshape(2)
    # global phase irrelevant: fix global phase so that a is real and >= 0 if possible
    a, b = state
    # normalize
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Zero vector provided")
    a, b = state / norm

    theta = 2 * np.arccos(np.clip(np.abs(a), 0.0, 1.0))
    # determine phi from phase(b) - phase(a)
    phase_a = np.angle(a)
    phase_b = np.angle(b)
    phi = (phase_b - phase_a)
    # wrap phi into (-pi, pi]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    return float(theta), float(phi)


if __name__ == "__main__":
    # example: set angles here  
    theta = np.pi / 3  # polar angle
    phi = np.pi / 4    # azimuthal angle

    psi = bloch_state(theta, phi)
    print("State vector |psi> =", psi)
    print("Bloch vector (x,y,z) =", bloch_vector(theta, phi))

    # recover angles from state
    th_rec, ph_rec = state_to_bloch_angles(psi)
    print(f"Recovered angles: theta={th_rec:.6f}, phi={ph_rec:.6f}")

    # plot
    plot_bloch(theta, phi)  