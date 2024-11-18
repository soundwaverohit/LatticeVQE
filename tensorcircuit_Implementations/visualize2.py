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
num_colors = 1  # SU(2) has 2 colors
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
            U = expm(1j * sum(random.uniform(0, 1) * G for G in su2_generators))
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

# Proceed to training and plotting
with tf.device("/cpu:0"):
    energies, m = train(n, d, NN_shape=NN_shape, maxiter=maxiter, lr=lr, stddev=stddev)

# Use a classical sparse eigenvalue solver to find the minimum eigenvalue
sparse_matrix = total_hamiltonian.to_matrix(sparse=True)
min_eigenvalue, _ = eigsh(sparse_matrix, k=1, which="SA")

print("Minimum Eigenvalue from classical solver:", min_eigenvalue[0])
print("Minimum Energy from VQE:", energies[-1])

# Plot the energy convergence
plt.figure(figsize=(8, 6))
plt.plot(range(len(energies)), energies, label="VQE Energy")
plt.axhline(y=min_eigenvalue[0], color="r", linestyle="--", label="Classical Minimum Energy")
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("Energy Convergence During Training")
plt.legend()
plt.show()
