# Import necessary libraries
import numpy as np
import tensorflow as tf
import tensorcircuit as tc
import cotengra
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
num_colors = 2  # Using one color (one fermionic mode per site)
a = 1.0  # Lattice constant

# SU(2) generators (Pauli matrices divided by 2)
su2_generators = [
    0.5 * np.array([[0, 1], [1, 0]], dtype=complex),     # sigma_x / 2
    0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex),  # sigma_y / 2
    0.5 * np.array([[1, 0], [0, -1]], dtype=complex),    # sigma_z / 2
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
def flatten_index(site_idx):
    """
    Map a site index to a linear index.
    """
    return site_idx  # Since num_colors = 1, the index is just the site index

# Initialize gauge fields for each unique link
link_indices = []
gauge_field = {}
for site_idx, neighbors in neighbor_list.items():
    for neighbor_idx in neighbors:
        link = tuple(sorted([site_idx, neighbor_idx]))
        if link not in gauge_field:
            # Initialize SU(2) gauge field as a unitary 2x2 matrix
            theta = np.random.uniform(0, 2 * np.pi)
            n = np.random.normal(size=3)
            n /= np.linalg.norm(n)
            U = expm(1j * theta * sum(n_i * G_i for n_i, G_i in zip(n, su2_generators)))
            gauge_field[link] = U
            link_indices.append(link)

num_links = len(link_indices)

# Total number of fermionic modes
num_fermionic_modes = num_sites  # Since num_colors = 1

# Gauge field Hilbert space per link
# For SU(2), we'll represent the gauge field using 1 qubit per link
qubits_per_link = 1
num_gauge_qubits = num_links * qubits_per_link

# Total qubits in the combined system
total_qubits = num_fermionic_modes + num_gauge_qubits

# Initialize the fermionic Hamiltonian terms
fermionic_terms = {}
site_masses = {}
for site_idx in range(num_sites):
    # Mass term with staggered phase
    x, y, z = site_positions[site_idx] / a  # Normalize to lattice units
    staggered_phase = (-1) ** (int(x + y + z) % 2)
    mass = m * staggered_phase
    site_masses[site_idx] = mass
    idx_flat = flatten_index(site_idx)
    fermionic_terms[f"+_{idx_flat} -_{idx_flat}"] = mass

# Hopping terms with full SU(2) gauge field matrices
for link_idx, link in enumerate(link_indices):
    site_i, site_j = link
    U_ij = gauge_field[link]
    idx_i = flatten_index(site_i)
    idx_j = flatten_index(site_j)
    # Hopping term with gauge field U_ij
    term1 = f"+_{idx_i} -_{idx_j}"
    # The gauge field operator will be included later
    coeff1 = -t * U_ij[0, 0]  # Since num_colors = 1, indices are 0
    fermionic_terms[term1] = fermionic_terms.get(term1, 0) + coeff1
    # Hermitian conjugate term
    term2 = f"+_{idx_j} -_{idx_i}"
    coeff2 = -t * np.conj(U_ij[0, 0])
    fermionic_terms[term2] = fermionic_terms.get(term2, 0) + coeff2

# Create Fermionic Hamiltonian
fermionic_hamiltonian = FermionicOp(
    fermionic_terms, num_spin_orbitals=num_fermionic_modes
)
mapper = JordanWignerMapper()
fermionic_qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

# Pad fermionic operators to include gauge qubits
fermionic_qubit_hamiltonian = fermionic_qubit_hamiltonian.tensor(
    SparsePauliOp.from_list([('I' * num_gauge_qubits, 1)])
)

# Function to map SU(2) matrices to qubit operators acting on 1 qubit
def su2_to_qubit_operator(U):
    """
    Map a 2x2 SU(2) matrix to a qubit operator acting on 1 qubit.
    """
    # Decompose U into Pauli matrices
    pauli_labels = ['I', 'X', 'Y', 'Z']
    paulis = {
        'I': np.eye(2, dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex),
    }
    
    pauli_strings = {}
    for label in pauli_labels:
        pauli_op = paulis[label]
        coeff = np.trace(U @ pauli_op.conj().T) / 2  # Normalize
        if np.abs(coeff) > 1e-8:
            pauli_strings[label] = coeff
    
    # Create the SparsePauliOp from list
    pauli_list = [(label, coeff) for label, coeff in pauli_strings.items()]
    qubit_op = SparsePauliOp.from_list(pauli_list)
    
    return qubit_op

# Construct gauge field Hamiltonian
gauge_field_hamiltonian = SparsePauliOp.from_list([], num_qubits=total_qubits)
for idx, link in enumerate(link_indices):
    U_op = su2_to_qubit_operator(gauge_field[link])
    # Shift the qubit indices to match the gauge field qubits
    shift = num_fermionic_modes + idx * qubits_per_link
    # Adjust Pauli strings to match the total qubit space
    adjusted_pauli_list = []
    for label, coeff in U_op.to_list():
        full_label = 'I' * shift + label + 'I' * (total_qubits - shift - qubits_per_link)
        adjusted_pauli_list.append((full_label, coeff))
    # Create the adjusted operator
    padded_U_op = SparsePauliOp.from_list(adjusted_pauli_list)
    # Add to gauge field Hamiltonian
    gauge_field_hamiltonian += g * padded_U_op

# Combine fermionic and gauge field Hamiltonians
total_hamiltonian = fermionic_qubit_hamiltonian + gauge_field_hamiltonian

print("Number of sites:", num_sites)
print("Number of links:", num_links)
print("Total number of qubits:", total_qubits)

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
delta_tau = 0.1  # Imaginary time step
num_steps = 20  # Number of QITE steps

energies = []

# QITE loop
for step in range(num_steps):
    print(f"\n--- Step {step} ---")
    # Compute current energy E = ⟨ψ| H |ψ⟩
    E = compute_energy(circuit)
    E_real = tf.math.real(E).numpy()
    energies.append(E_real)
    print(f"Energy E: {E_real}")
    
    # Prepare b vector and S matrix
    N = len(basis_operators)
    b = np.zeros(N, dtype=np.float64)
    S = np.zeros((N, N), dtype=np.float64)
    
    # Compute ⟨ψ| σ_i |ψ⟩ for all σ_i
    sigma_expectations = []
    for idx, (pauli, qubit) in enumerate(basis_operators):
        pauli_string = ['I'] * total_qubits
        pauli_string[qubit] = pauli
        expval = expectation_pauli_string(circuit, ''.join(pauli_string))
        expval_real = tf.math.real(expval).numpy()
        sigma_expectations.append(expval_real)
    
    # Compute b_i
    for i, (pauli_i, qubit_i) in enumerate(basis_operators):
        # Compute ⟨ψ| [H, σ_i] |ψ⟩
        commutator = 0.0 + 0.0j
        for pauli_H, coeff_H in hamiltonian_terms:
            # Multiply H term with σ_i and subtract σ_i with H term
            # [H, σ_i] = Hσ_i - σ_i H
            sigma_i_string = ['I'] * total_qubits
            sigma_i_string[qubit_i] = pauli_i
            sigma_i_string = ''.join(sigma_i_string)
            new_pauli1, phase1 = multiply_pauli_strings(pauli_H, sigma_i_string)
            expval1 = expectation_pauli_string(circuit, new_pauli1)
            new_pauli2, phase2 = multiply_pauli_strings(sigma_i_string, pauli_H)
            expval2 = expectation_pauli_string(circuit, new_pauli2)
            commutator += coeff_H * (phase1 * expval1 - phase2 * expval2)
        b[i] = -delta_tau * tf.math.imag(commutator).numpy()  # Only the imaginary part contributes
    
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
            expval_real = tf.math.real(expval).numpy()
            S_ij = expval_real - sigma_expectations[i] * sigma_expectations[j]
            S[i, j] = S_ij
    
    # Regularize S matrix
    S += 1e-6 * np.eye(N)
    
    # Solve S x = b
    try:
        x = np.linalg.solve(S, b)
    except np.linalg.LinAlgError:
        print("Singular matrix encountered at step", step)
        break
    
    # Update the state: Apply U(δτ) = exp(-i δτ ∑ x_i σ_i)
    # For small δτ, we can approximate U(δτ) ≈ ∏ exp(-i δτ x_i σ_i)
    for i, (pauli_i, qubit_i) in enumerate(basis_operators):
        theta = 2 * x[i]  # Factor of 2 because rotations are defined as exp(-i θ σ/2)
        if pauli_i == 'X':
            circuit.rx(qubit_i, theta=theta)
        elif pauli_i == 'Y':
            circuit.ry(qubit_i, theta=theta)
        elif pauli_i == 'Z':
            circuit.rz(qubit_i, theta=theta)

# After QITE, compute the final energy
final_energy = compute_energy(circuit)
final_energy_real = tf.math.real(final_energy).numpy()
print("Final Energy from QITE:", final_energy_real)

# Use a classical sparse eigenvalue solver to find the minimum eigenvalue
# Note: For large systems, this may not be feasible due to memory constraints
try:
    sparse_matrix = total_hamiltonian.to_matrix(sparse=True)
    min_eigenvalue, _ = eigsh(sparse_matrix, k=1, which="SA")
    print("Minimum Eigenvalue from classical solver:", min_eigenvalue[0])
except MemoryError:
    print("MemoryError: Unable to compute minimum eigenvalue due to memory constraints.")

# Plot the energy convergence
plt.figure(figsize=(8, 6))
plt.plot(range(len(energies)), energies, label="QITE Energy")
if 'min_eigenvalue' in locals():
    plt.axhline(y=min_eigenvalue[0], color="r", linestyle="--", label="Classical Minimum Energy")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.title("Energy Convergence During QITE")
plt.legend()
plt.show()
