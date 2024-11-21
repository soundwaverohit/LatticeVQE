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
num_colors = 1  # SU(2) has 2 colors, but for simplicity we use 1 color here
a = 1.0  # Lattice constant

# SU(2) Pauli matrices (generators)
su2_generators = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

# Helper functions for triamond lattice
def generate_triamond_lattice():
    """
    Generate site positions and neighbor list for the triamond lattice unit cell.
    """
    # Define the positions of the 8 sites in the unit cell
    site_positions = [
        np.array([0, 0, 0]),
        np.array([a, 0, 0]),
        np.array([0, a, 0]),
        np.array([0, 0, a]),
        np.array([a, a, 0]),
        np.array([a, 0, a]),
        np.array([0, a, a]),
        np.array([a, a, a]),
    ]

    # Define the neighbor list (12 links)
    neighbor_list = {
        0: [1, 2, 3],
        1: [0, 4, 5],
        2: [0, 4, 6],
        3: [0, 5, 6],
        4: [1, 2, 7],
        5: [1, 3, 7],
        6: [2, 3, 7],
        7: [4, 5, 6],
    }

    return site_positions, neighbor_list

# Generate the triamond lattice
site_positions, neighbor_list = generate_triamond_lattice()
num_sites = len(site_positions)

# Function to flatten index
def flatten_index(site_idx, color, num_colors):
    """
    Map a site index and color to a linear index.
    """
    return site_idx * num_colors + color

# Initialize gauge fields for each unique link
link_indices = []
gauge_field = {}
for site_idx, neighbors in neighbor_list.items():
    for neighbor_idx in neighbors:
        link = tuple(sorted([site_idx, neighbor_idx]))
        if link not in link_indices:
            link_indices.append(link)
            # Initialize gauge fields as identity for simplicity
            U = np.eye(2, dtype=complex)
            gauge_field[link] = U

num_links = len(link_indices)

# Initialize the fermionic Hamiltonian terms
fermionic_terms = {}
site_masses = {}
link_coefficients = {}
for site_idx in range(num_sites):
    # Mass term with staggered phase
    x, y, z = site_positions[site_idx] / a  # Normalize to lattice units
    staggered_phase = (-1) ** (int(x + y + z) % 2)
    mass = m * staggered_phase
    site_masses[site_idx] = mass
    for alpha in range(num_colors):
        idx_flat = flatten_index(site_idx, alpha, num_colors)
        fermionic_terms[f"+_{idx_flat} -_{idx_flat}"] = mass

    # Hopping terms with gauge field interaction
    for neighbor_idx in neighbor_list[site_idx]:
        if neighbor_idx > site_idx:  # Avoid double counting
            link = (site_idx, neighbor_idx)
            U = gauge_field[link]
            coeff = -t * np.real(np.trace(U)) / 2
            link_coefficients[link] = coeff
            for alpha in range(num_colors):
                idx1 = flatten_index(site_idx, alpha, num_colors)
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
    fig = plt.figure(figsize=(8, 6))
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
        ax.scatter(x, y, z, c=[color], s=100)
        ax.text(x, y, z, f'{idx}', color='black', fontsize=10)

    # Get link coefficients and normalize
    coeffs = np.array([link_coefficients[link] for link in link_indices])
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
        ax.plot([x0, x1], [y0, y1], [z0, z1], c=color, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Triamond Lattice')

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

# Extract Hamiltonian terms
hamiltonian_terms = total_hamiltonian.to_list()

# Define Pauli multiplication rules
def multiply_single_paulis(p1, p2):
    if p1 == 'I':
        return p2, 1
    if p2 == 'I':
        return p1, 1
    if p1 == p2:
        return 'I', 1
    if p1 == 'X' and p2 == 'Y':
        return 'Z', 1j
    if p1 == 'Y' and p2 == 'Z':
        return 'X', 1j
    if p1 == 'Z' and p2 == 'X':
        return 'Y', 1j
    if p1 == 'Y' and p2 == 'X':
        return 'Z', -1j
    if p1 == 'Z' and p2 == 'Y':
        return 'X', -1j
    if p1 == 'X' and p2 == 'Z':
        return 'Y', -1j
    else:
        raise ValueError(f"Invalid Pauli operators: {p1}, {p2}")

def multiply_pauli_strings(pauli1, pauli2):
    assert len(pauli1) == len(pauli2)
    result_pauli = ''
    phase = 1
    for p1, p2 in zip(pauli1, pauli2):
        res_p, ph = multiply_single_paulis(p1, p2)
        result_pauli += res_p
        phase *= ph
    return result_pauli, phase

# Define the basis operators σ_i
basis_operators = []
for q in range(total_qubits):
    basis_operators.append(('X', q))
    basis_operators.append(('Y', q))
    basis_operators.append(('Z', q))

# Initialize the quantum state |ψ⟩ as |0...0⟩ with random rotations to avoid starting from an eigenstate
circuit = tc.Circuit(total_qubits)
for qubit in range(total_qubits):
    circuit.rx(qubit, theta=np.random.uniform(0, 2 * np.pi))
    circuit.ry(qubit, theta=np.random.uniform(0, 2 * np.pi))
    circuit.rz(qubit, theta=np.random.uniform(0, 2 * np.pi))

# Function to compute expectation value of a Pauli string
def expectation_pauli_string(c: tc.Circuit, pauli_string):
    operators = []
    for idx, pauli in enumerate(pauli_string):
        if pauli != "I":
            if pauli == "X":
                operators.append((tc.gates.x(), [idx]))
            elif pauli == "Y":
                operators.append((tc.gates.y(), [idx]))
            elif pauli == "Z":
                operators.append((tc.gates.z(), [idx]))
            else:
                raise ValueError(f"Unknown Pauli operator: {pauli}")
    if operators:
        return c.expectation(*operators)
    else:
        return 1.0 + 0.0j

# Function to compute energy ⟨ψ| H |ψ⟩
def compute_energy(c: tc.Circuit):
    e = 0.0 + 0.0j  # Initialize as complex128
    for pauli_string, coeff in hamiltonian_terms:
        expval = expectation_pauli_string(c, pauli_string)
        e += coeff * expval
    return e  # Return complex128

# QITE parameters
delta_tau = 10  # Imaginary time step
num_steps = 100  # Number of QITE steps

energies = []

# QITE loop
for step in range(num_steps):
    print(f"\n--- Step {step} ---")
    # Compute current energy E = ⟨ψ| H |ψ⟩
    E_complex = compute_energy(circuit)
    E = np.real(E_complex)
    energies.append(E)
    print(f"Energy E: {E}")

    # Prepare b vector and S matrix
    N = len(basis_operators)
    b = np.zeros(N, dtype=np.float64)
    S = np.zeros((N, N), dtype=np.float64)

    # Compute ⟨ψ| σ_i |ψ⟩ for all σ_i
    sigma_expectations = []
    for idx, (pauli, qubit) in enumerate(basis_operators):
        pauli_string = ['I'] * total_qubits
        pauli_string[qubit] = pauli
        expval = expectation_pauli_string(circuit, pauli_string)
        sigma_expectations.append(np.real(expval))

    # Compute b_i
    for i, (pauli_i, qubit_i) in enumerate(basis_operators):
        # Compute ⟨ψ| H σ_i |ψ⟩
        H_sigma_i = 0.0 + 0.0j
        for pauli_H, coeff_H in hamiltonian_terms:
            # Multiply pauli_H and σ_i
            pauli_string_i = ['I'] * total_qubits
            pauli_string_i[qubit_i] = pauli_i
            new_pauli, phase = multiply_pauli_strings(pauli_H, ''.join(pauli_string_i))
            expval = expectation_pauli_string(circuit, new_pauli)
            H_sigma_i += coeff_H * phase * expval

        b_i = -np.real(H_sigma_i) + E * sigma_expectations[i]
        b[i] = b_i

    # Compute S_ij
    for i, (pauli_i, qubit_i) in enumerate(basis_operators):
        for j, (pauli_j, qubit_j) in enumerate(basis_operators):
            # Compute ⟨ψ| σ_i σ_j |ψ⟩
            pauli_string_i = ['I'] * total_qubits
            pauli_string_i[qubit_i] = pauli_i
            pauli_string_j = ['I'] * total_qubits
            pauli_string_j[qubit_j] = pauli_j
            # Multiply σ_i and σ_j
            new_pauli, phase = multiply_pauli_strings(''.join(pauli_string_i), ''.join(pauli_string_j))
            expval = expectation_pauli_string(circuit, new_pauli)
            S_ij = np.real(phase * expval) - sigma_expectations[i] * sigma_expectations[j]
            S[i, j] = S_ij

    # Solve S x = b
    try:
        x = np.linalg.solve(S + 1e-6 * np.eye(N), b)
    except np.linalg.LinAlgError:
        print("Singular matrix encountered at step", step)
        break

    # Update the state: Apply U(δτ) = exp(-i δτ ∑ x_i σ_i)
    # For small δτ, we can approximate U(δτ) ≈ ∏ exp(-i δτ x_i σ_i)
    for i, (pauli_i, qubit_i) in enumerate(basis_operators):
        theta = 2 * delta_tau * x[i]  # Factor of 2 because rotations are defined as exp(-i θ σ/2)
        if pauli_i == 'X':
            circuit.rx(qubit_i, theta=theta)
        elif pauli_i == 'Y':
            circuit.ry(qubit_i, theta=theta)
        elif pauli_i == 'Z':
            circuit.rz(qubit_i, theta=theta)

# After QITE, compute the final energy
final_energy = compute_energy(circuit)
print("Final Energy from QITE:", np.real(final_energy))

# Use a classical sparse eigenvalue solver to find the minimum eigenvalue
sparse_matrix = total_hamiltonian.to_matrix(sparse=True)
min_eigenvalue, _ = eigsh(sparse_matrix, k=1, which="SA")

print("Minimum Eigenvalue from classical solver:", min_eigenvalue[0])

# Plot the energy convergence
plt.figure(figsize=(8, 6))
plt.plot(range(len(energies)), energies, label="QITE Energy")
plt.axhline(y=min_eigenvalue[0], color="r", linestyle="--", label="Classical Minimum Energy")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.title("Energy Convergence During QITE")
plt.legend()
plt.show()
