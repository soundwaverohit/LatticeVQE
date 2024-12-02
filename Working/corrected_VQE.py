# Import necessary libraries
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
lattice_size = 2  # Define lattice size for 3D triamond lattice
num_colors = 2  # Using one color (one fermionic mode per site)

# SU(2) generators (Pauli matrices divided by 2)
su2_generators = [
    0.5 * np.array([[0, 1], [1, 0]], dtype=complex),     # sigma_x / 2
    0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex),  # sigma_y / 2
    0.5 * np.array([[1, 0], [0, -1]], dtype=complex),    # sigma_z / 2
]

# Helper functions for triamond lattice
def generate_triamond_lattice(lattice_size):
    """
    Generate site positions and neighbor list for the triamond lattice.
    """
    site_positions = []
    neighbor_list = {}

    for x in range(lattice_size):
        for y in range(lattice_size):
            for z in range(lattice_size):
                site = (x, y, z)
                site_positions.append(site)
                neighbors = []

                # Define the neighbor offsets for the triamond lattice
                neighbor_offsets = [
                    (1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    # Additional offsets to ensure three neighbors per site
                    (-1, 1, 0),
                    (0, -1, 1),
                    (1, 0, -1),
                ]

                for offset in neighbor_offsets:
                    nx = (x + offset[0]) % lattice_size
                    ny = (y + offset[1]) % lattice_size
                    nz = (z + offset[2]) % lattice_size
                    neighbor = (nx, ny, nz)
                    if neighbor != site:
                        neighbors.append(neighbor)

                # Ensure only three neighbors per site
                neighbors = neighbors[:3]
                neighbor_list[site] = neighbors

    return site_positions, neighbor_list

def flatten_index(site, lattice_size):
    """
    Map a site to a linear index.
    """
    x, y, z = site
    site_index = x + y * lattice_size + z * lattice_size ** 2
    return site_index

# Generate the triamond lattice
site_positions, neighbor_list = generate_triamond_lattice(lattice_size)
num_sites = len(site_positions)

# Map sites to indices
site_to_index = {site: idx for idx, site in enumerate(site_positions)}

# Initialize gauge fields for each unique link
link_indices = []
gauge_field = {}
for site in site_positions:
    for neighbor in neighbor_list[site]:
        link = tuple(sorted([site_to_index[site], site_to_index[neighbor]]))
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
qubits_per_link = 1
num_gauge_qubits = num_links * qubits_per_link

# Total qubits in the combined system
total_qubits = num_fermionic_modes + num_gauge_qubits

# Initialize the fermionic Hamiltonian terms
fermionic_terms = {}
site_masses = {}
for site in site_positions:
    # Mass term with staggered phase
    x, y, z = site
    staggered_phase = (-1) ** ((x + y + z) % 2)
    mass = m * staggered_phase
    site_idx = site_to_index[site]
    site_masses[site_idx] = mass
    idx_flat = flatten_index(site, lattice_size)
    fermionic_terms[f"+_{idx_flat} -_{idx_flat}"] = mass

# Hopping terms with full SU(2) gauge field matrices
for link_idx, link in enumerate(link_indices):
    site_i_idx, site_j_idx = link
    U_ij = gauge_field[link]
    site_i = site_positions[site_i_idx]
    site_j = site_positions[site_j_idx]
    idx_i = flatten_index(site_i, lattice_size)
    idx_j = flatten_index(site_j, lattice_size)
    # Hopping term with gauge field U_ij
    term1 = f"+_{idx_i} -_{idx_j}"
    coeff1 = -t * U_ij[0, 0]  # Since num_colors = 1
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

# Define the energy function
def energy(c: tc.Circuit):
    e = 0.0 + 0.0j
    for pauli_string, coeff in hamiltonian_terms:
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
            expval = c.expectation(*operators)
            e += coeff * expval
        else:
            e += coeff
    return tf.math.real(e)

# Adjusted MERA function to handle arbitrary n
def MERA(inp, n, d=2, layers=3, energy_flag=False):
    params = K.cast(inp["params"], dtype="float32")
    params = K.cast(params, "complex128")
    params = K.reshape(params, [-1])
    c = tc.Circuit(n)
    idx = 0

    # Initial single-qubit rotations
    for i in range(n):
        c.rx(i, theta=params[idx])
        c.rz(i, theta=params[idx + 1])
        c.rx(i, theta=params[idx + 2])
        idx += 3

    # Multi-layer entanglement blocks
    for layer in range(layers):
        for depth in range(d):
            for i in range(n - 1):
                c.cnot(i, i + 1)
                c.ry(i + 1, theta=params[idx])
                idx += 1

            for i in range(0, n - 1, 2):
                if i + 1 < n:
                    c.cu(i, i + 1, theta=params[idx], phi=params[idx + 1], lbd=0)
                    idx += 2

        for i in range(n):
            c.rx(i, theta=params[idx])
            c.rz(i, theta=params[idx + 1])
            idx += 2

    if energy_flag:
        return energy(c)
    else:
        return c, idx

# QuantumLayer for Keras
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, circuit_fn, **kwargs):
        super().__init__(**kwargs)
        self.circuit_fn = circuit_fn

    def call(self, inputs):
        return self.circuit_fn(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

# Create neural network
def create_NN_MERA(n, d, NN_shape, stddev):
    input = tf.keras.layers.Input(shape=[1])
    x = tf.keras.layers.Dense(
        units=NN_shape,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="relu",
    )(input)
    x = tf.keras.layers.Dropout(0.05)(x)

    dummy_params = np.zeros(5000)
    _, idx = MERA({"params": dummy_params}, n, d, energy_flag=False)

    params = tf.keras.layers.Dense(
        units=idx,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="sigmoid",
    )(x)

    qlayer = QuantumLayer(partial(MERA, n=n, d=d, energy_flag=True))
    output = qlayer({"params": 6.3 * params})
    m = tf.keras.Model(inputs=input, outputs=output)
    return m

# Training function
def train(n, d, NN_shape, maxiter=1000, lr=0.005, stddev=1.0):
    exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=1000, decay_rate=0.7
    )
    opt = tf.keras.optimizers.Adam(exp_lr)
    m = create_NN_MERA(n, d, NN_shape, stddev)
    energies = []

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            e = m(K.reshape([0.0], [1]))
        grads = tape.gradient(e, m.trainable_variables)
        opt.apply_gradients(zip(grads, m.trainable_variables))
        return e

    for i in range(maxiter):
        total_e = train_step()
        energies.append(total_e.numpy())
        if i % 100 == 0:
            print("epoch", i, ":", total_e.numpy())

    m.save_weights("NN-VQE.weights.h5")
    return energies, m

# Set number of qubits
n = total_qubits
d = 2
NN_shape = 50
maxiter = 2500
lr = 0.005
stddev = 0.1

# Train and plot
with tf.device("/cpu:0"):
    energies, m = train(n, d, NN_shape=NN_shape, maxiter=maxiter, lr=lr, stddev=stddev)

# Use a classical sparse eigenvalue solver to find the minimum eigenvalue
try:
    sparse_matrix = total_hamiltonian.to_matrix(sparse=True)
    min_eigenvalue, _ = eigsh(sparse_matrix, k=1, which="SA")
    print("Minimum Eigenvalue from classical solver:", min_eigenvalue[0])
except MemoryError:
    print("MemoryError: Unable to compute minimum eigenvalue due to memory constraints.")
    min_eigenvalue = [None]

print("Minimum Energy from VQE:", energies[-1])

# Plot the energy convergence
plt.figure(figsize=(8, 6))
plt.plot(range(len(energies)), energies, label="VQE Energy")
if min_eigenvalue[0] is not None:
    plt.axhline(y=min_eigenvalue[0], color="r", linestyle="--", label="Classical Minimum Energy")
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("Energy Convergence During Training")
plt.legend()
plt.show()
