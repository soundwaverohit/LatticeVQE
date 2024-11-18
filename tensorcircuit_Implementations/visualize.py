import numpy as np
import tensorflow as tf
import tensorcircuit as tc
import cotengra
from functools import partial
import random
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.spatial import cKDTree

# Set up TensorCircuit with custom contractor
optc = cotengra.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel="ray",
    minimize="combo",
    max_time=30,
    max_repeats=1024,
    progbar=True,
)
tc.set_contractor("custom", optimizer=optc, preprocessing=True)

K = tc.set_backend("tensorflow")
tc.set_dtype("complex128")

# Fix the random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define Hamiltonian parameters
g = 2.0  # Gauge coupling constant
t = 1.0  # Hopping parameter
m = 0.5  # Mass
lattice_size = 2  # Define lattice size for triamond lattice
num_colors = 2  # SU(2) has 2 colors
a = 1.0  # Lattice constant

# SU(2) Pauli matrices (generators)
su2_generators = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

# Helper functions for triamond lattice
def generate_triamond_lattice(lattice_size, a=1.0):
    """
    Generate site positions and neighbor list for the triamond lattice.
    """
    # FCC lattice vectors
    a1 = np.array([0.5, 0.5, 0.0]) * a
    a2 = np.array([0.5, 0.0, 0.5]) * a
    a3 = np.array([0.0, 0.5, 0.5]) * a

    # Basis atoms for triamond lattice
    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.25, 0.25, 0.25]) * a,
        np.array([0.5, 0.5, 0.5]) * a,
        np.array([0.75, 0.75, 0.75]) * a,
    ]

    # Generate lattice positions
    site_positions = []
    for i in range(lattice_size):
        for j in range(lattice_size):
            for k in range(lattice_size):
                lattice_point = i * a1 + j * a2 + k * a3
                for b in basis:
                    pos = lattice_point + b
                    pos_mod = np.mod(pos, lattice_size * a)
                    site_positions.append(pos_mod)

    site_positions = np.array(site_positions)

    # Create neighbor list
    neighbor_list = {}
    tree = cKDTree(site_positions)
    for idx, site in enumerate(site_positions):
        # Find neighbors within a certain distance (less than sqrt(3)/2 * a)
        indices = tree.query_ball_point(site, r=(np.sqrt(3) / 2 + 0.1) * a)
        indices.remove(idx)  # Remove self
        neighbor_list[idx] = indices

    return site_positions, neighbor_list

# Generate the triamond lattice
site_positions, neighbor_list = generate_triamond_lattice(lattice_size, a)
num_sites = len(site_positions)

# Update flatten_index function
def flatten_index(idx, color, num_colors):
    """
    Map a site index and color to a linear index.
    """
    return idx * num_colors + color

# Initialize gauge fields for each unique link
link_indices = []
gauge_field = {}
for idx, neighbors in neighbor_list.items():
    for neighbor_idx in neighbors:
        link = tuple(sorted([idx, neighbor_idx]))
        if link not in link_indices:
            link_indices.append(link)
            U = expm(1j * sum(random.uniform(0, 1) * G for G in su2_generators))
            gauge_field[link] = U

num_links = len(link_indices)

# Initialize the fermionic Hamiltonian terms
fermionic_terms = {}
site_masses = {}
link_coefficients = {}
for idx, site in enumerate(site_positions):
    # Mass term with staggered phase
    x, y, z = site / a  # Normalize to lattice units
    staggered_phase = (-1) ** (int(np.floor(x) + np.floor(y) + np.floor(z)) % 2)
    mass = m * staggered_phase
    site_masses[idx] = mass
    for alpha in range(num_colors):
        idx_flat = flatten_index(idx, alpha, num_colors)
        fermionic_terms[f"+_{idx_flat} -_{idx_flat}"] = mass

    # Hopping terms with gauge field interaction
    for neighbor_idx in neighbor_list[idx]:
        if neighbor_idx > idx:  # Avoid double counting
            link = (idx, neighbor_idx)
            U = gauge_field[link]
            coeff = -t * np.real(np.trace(U)) / 2
            link_coefficients[link] = coeff
            for alpha in range(num_colors):
                idx1 = flatten_index(idx, alpha, num_colors)
                idx2 = flatten_index(neighbor_idx, alpha, num_colors)
                fermionic_terms[f"+_{idx1} -_{idx2}"] = coeff
                fermionic_terms[f"+_{idx2} -_{idx1}"] = coeff

# Fermionic Hamiltonian from fermionic terms
fermionic_hamiltonian = FermionicOp(
    fermionic_terms, num_spin_orbitals=num_sites * num_colors
)
mapper = JordanWignerMapper()
fermionic_qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

# Total qubits in the combined system
total_qubits = num_sites * num_colors + num_links

# Pad fermionic operators to include gauge qubits
fermionic_qubit_hamiltonian = SparsePauliOp.from_list(
    [
        (pauli + "I" * num_links, coeff)
        for pauli, coeff in fermionic_qubit_hamiltonian.to_list()
    ]
)

# Create and pad gauge field Hamiltonian
gauge_field_ops = []
for i, link in enumerate(link_indices):
    # Pauli Z operator acting on the gauge field qubits
    pauli_str = (
        "I" * (num_sites * num_colors)
        + "I" * i
        + "Z"
        + "I" * (num_links - i - 1)
    )
    gauge_field_ops.append(SparsePauliOp.from_list([(pauli_str, g)]))

# Sum the gauge field operators into a single SparsePauliOp
gauge_field_hamiltonian = sum(gauge_field_ops)

# Combine fermionic and gauge field Hamiltonians
total_hamiltonian = fermionic_qubit_hamiltonian + gauge_field_hamiltonian

print("Number of sites:", num_sites)
print("Number of links:", num_links)
print("Total number of qubits:", total_qubits)

# Function to visualize the triamond lattice
def visualize_triamond_lattice(site_positions, link_indices, site_masses, link_coefficients):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get mass values and normalize
    masses = np.array([site_masses[idx] for idx in range(len(site_positions))])
    mass_min = masses.min()
    mass_max = masses.max()
    norm_masses = (masses - mass_min) / (mass_max - mass_min) if mass_max != mass_min else np.zeros_like(masses)
    
    # Use a colormap for site masses
    cmap = plt.cm.coolwarm
    
    # Plot sites and labels
    for idx, (site, norm_mass) in enumerate(zip(site_positions, norm_masses)):
        x, y, z = site
        color = cmap(norm_mass)
        ax.scatter(x, y, z, c=[color], s=50)
        ax.text(x, y, z, f'{idx}', color='black', fontsize=8)
    
    # Get link coefficients and normalize
    coeffs = np.array([link_coefficients[link] for link in link_indices if link in link_coefficients])
    coeff_min = coeffs.min()
    coeff_max = coeffs.max()
    norm_coeffs = (coeffs - coeff_min) / (coeff_max - coeff_min) if coeff_max != coeff_min else np.zeros_like(coeffs)
    
    # Use a colormap for links
    link_cmap = plt.cm.viridis
    
    # Plot links
    for link, norm_coeff in zip(link_indices, norm_coeffs):
        idx1, idx2 = link
        x0, y0, z0 = site_positions[idx1]
        x1, y1, z1 = site_positions[idx2]
        color = link_cmap(norm_coeff)
        ax.plot([x0, x1], [y0, y1], [z0, z1], c=color, linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Triiamond Lattice')
    
    # Add colorbars for site masses and link coefficients
    mappable_site = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=mass_min, vmax=mass_max))
    mappable_site.set_array([])
    cbar_site = plt.colorbar(mappable_site, ax=ax, shrink=0.5, pad=0.1)
    cbar_site.set_label('Mass Term at Sites')
    
    mappable_link = plt.cm.ScalarMappable(cmap=link_cmap, norm=plt.Normalize(vmin=coeff_min, vmax=coeff_max))
    mappable_link.set_array([])
    cbar_link = plt.colorbar(mappable_link, ax=ax, shrink=0.5, pad=0.05)
    cbar_link.set_label('Hopping Coefficient on Links')
    
    plt.show()

# Visualize the triamond lattice
visualize_triamond_lattice(site_positions, link_indices, site_masses, link_coefficients)
