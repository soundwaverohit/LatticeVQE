import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from scipy.linalg import expm
# Parameters
g = 2.0  # Coupling constant
t = 1.0  # Hopping term
m = 0.5  # Fermion mass
lattice_dim = 2  # 2D lattice
lattice_size = 3  # 3x3 lattice for simplicity
num_sites = lattice_size * lattice_size  # Total number of lattice sites
num_colors = 3  # SU(3) has 3 color indices

# SU(3) Gell-Mann matrices (SU(3) generators)
su3_generators = [
    np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
]

# Initialize gauge field as random SU(3) matrices for each link in the 2D lattice
gauge_field = [expm(1j * random.random() * su3_generators[random.randint(0, 7)]) for _ in range(2 * num_sites)]

# Helper function to flatten site and color indices into a single index
def flatten_index(site, color, num_colors):
    return site * num_colors + color

# Hopping terms for SU(3) (fermionic hopping terms)
fermionic_terms = {}
for n in range(num_sites - 1):
    U = gauge_field[n]
    for alpha in range(num_colors):
        flattened_idx_1 = flatten_index(n, alpha, num_colors)
        flattened_idx_2 = flatten_index(n + 1, alpha, num_colors)
        coeff = -t * np.trace(U @ U.T.conj()) / 2
        
        # Add both the forward and reverse hopping terms to ensure Hermiticity
        fermionic_terms[f"+_{flattened_idx_1} -_{flattened_idx_2}"] = coeff
        fermionic_terms[f"+_{flattened_idx_2} -_{flattened_idx_1}"] = np.conj(coeff)

# On-site mass terms (fermionic creation/annihilation for each color)
for n in range(num_sites):
    for alpha in range(num_colors):
        flattened_idx = flatten_index(n, alpha, num_colors)
        fermionic_terms[f"+_{flattened_idx} -_{flattened_idx}"] = np.real(m)

# Plaquette terms for SU(3) (magnetic terms in 2D lattice)
plaquette_energy = 0
for i in range(lattice_size - 1):
    for j in range(lattice_size - 1):
        # Find the 4 links around a plaquette (Uij, Ujk, Ukl, Uli)
        U_ij = gauge_field[i * lattice_size + j]  # Link from site (i, j) to (i, j+1)
        U_jk = gauge_field[(i + 1) * lattice_size + j]  # Link from (i, j+1) to (i+1, j+1)
        U_kl = gauge_field[(i + 1) * lattice_size + j + 1]  # Link from (i+1, j+1) to (i+1, j)
        U_li = gauge_field[i * lattice_size + j + 1]  # Link from (i+1, j) to (i, j)

        # Plaquette term for this square
        plaquette = np.trace(U_ij @ U_jk @ U_kl @ U_li)
        
        # Symmetrize the plaquette term to force it to be real (remove any residual imaginary part)
        plaquette = (plaquette + np.conj(plaquette)) / 2.0
        
        # Add to total plaquette energy
        plaquette_energy += g**2 * plaquette / 2

# Create the FermionicOp with the dictionary of fermionic hopping and mass terms
fermionic_hamiltonian = FermionicOp(fermionic_terms, num_spin_orbitals=num_sites * num_colors)

# **Mapping to Qubits**
mapper = JordanWignerMapper()

# Map to qubit operator
qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

# **Custom Ansatz Circuit**
# Define the number of qubits
num_qubits = num_sites * num_colors  # 3 colors per site

# Create the quantum circuit (ansatz)
ansatz = QuantumCircuit(num_qubits)

# Create parameters for the ansatz
parameters = []
for i in range(num_qubits):
    theta = Parameter(f"theta_{i}")
    parameters.append(theta)
    ansatz.ry(theta, i)

# Add entangling gates
for i in range(0, num_qubits - 1, 2):
    ansatz.cx(i, i + 1)
for i in range(1, num_qubits - 1, 2):
    ansatz.cx(i, i + 1)
# Optionally, add entanglement between the last and first qubits
# ansatz.cx(num_qubits - 1, 0)

# **Print the custom ansatz circuit**
print("Custom ansatz circuit:")
print(ansatz.draw())

print("Qubit count: ", num_qubits)

# Optimizer
optimizer = COBYLA(maxiter=1000)

# Create an Estimator
estimator = Estimator()

# Create the VQE instance with the custom ansatz
vqe = VQE(estimator, ansatz, optimizer)

# Compute the ground state energy (without plaquette terms for now)
result = vqe.compute_minimum_eigenvalue(qubit_hamiltonian)

# Total energy includes plaquette energy (calculated separately)
total_energy = result.eigenvalue.real + plaquette_energy

# Print the results
print("Ground state energy without plaquette:", result.eigenvalue.real)
print("Plaquette energy contribution:", plaquette_energy)
print("Total ground state energy:", total_energy)
