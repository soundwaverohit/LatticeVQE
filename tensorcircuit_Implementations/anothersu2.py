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
from qiskit.visualization import circuit_drawer
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
num_colors = 2  # SU(2) has 2 colors

# SU(2) Pauli matrices (generators)
su2_generators = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
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

                # Define the three neighbor directions specific to triamond lattice
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

def flatten_index(site, color, num_colors, lattice_size):
    """
    Map a site and color to a linear index.
    """
    x, y, z = site
    site_index = x + y * lattice_size + z * lattice_size ** 2
    return site_index * num_colors + color

# Generate the triamond lattice
site_positions, neighbor_list = generate_triamond_lattice(lattice_size)
num_sites = len(site_positions)

# Initialize gauge fields for each unique link
link_indices = []
gauge_field = {}
for site in site_positions:
    for neighbor in neighbor_list[site]:
        link = tuple(sorted([site, neighbor]))
        if link not in link_indices:
            link_indices.append(link)
            U = expm(1j * sum(random.uniform(0, 1) * G for G in su2_generators))
            gauge_field[link] = U

num_links = len(link_indices)

# Initialize the fermionic Hamiltonian terms
fermionic_terms = {}
for site in site_positions:
    # Mass term with staggered phase
    x, y, z = site
    staggered_phase = (-1) ** ((x + y + z) % 2)
    for alpha in range(num_colors):
        idx = flatten_index(site, alpha, num_colors, lattice_size)
        fermionic_terms[f"+_{idx} -_{idx}"] = m * staggered_phase

    # Hopping terms with gauge field interaction
    for neighbor in neighbor_list[site]:
        link = tuple(sorted([site, neighbor]))
        U = gauge_field[link]
        for alpha in range(num_colors):
            idx1 = flatten_index(site, alpha, num_colors, lattice_size)
            idx2 = flatten_index(neighbor, alpha, num_colors, lattice_size)
            coeff = -t * np.real(np.trace(U)) / 2
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

# Extract Hamiltonian terms
hamiltonian_terms = total_hamiltonian.to_list()

# Define the energy function
def energy(c: tc.Circuit):
    e = 0.0
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
            e += coeff * c.expectation(*operators)
        else:
            e += coeff
    return K.real(e)

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
maxiter = 1000
lr = 0.005
stddev = 0.1

# Train and plot
with tf.device("/cpu:0"):
    energies, m = train(n, d, NN_shape=NN_shape, maxiter=maxiter, lr=lr, stddev=stddev)

# Use a classical sparse eigenvalue solver to find the minimum eigenvalue
sparse_matrix = total_hamiltonian.to_matrix(sparse=True)
min_eigenvalue, _ = eigsh(sparse_matrix, k=1, which="SA")

print("Minimum Eigenvalue from classical solver:", min_eigenvalue[0])
print("Minimum Energy from VQE:", energies[-1])

# Plot the energy convergence
plt.plot(range(len(energies)), energies, label="VQE Energy")
plt.axhline(y=min_eigenvalue[0], color="r", linestyle="--", label="Classical Minimum Energy")
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("Energy Convergence During Training")
plt.legend()
plt.show()
