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
lattice_size = 1  # Define lattice size for 3D triamond lattice
num_sites = lattice_size ** 3  # Total number of lattice sites in 3D
num_colors = 2  # SU(2) has 2 colors

# SU(2) Pauli matrices
su2_generators = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

# Initialize gauge field as random SU(2) matrices for each unique link
num_links = int(num_sites * 1.5)  # Approximate number of unique links in the triamond lattice
gauge_field = [expm(1j * sum(random.uniform(0, 1) * G for G in su2_generators)) for _ in range(num_links)]

# Helper functions
def coord_to_index(x, y, z, lattice_size):
    return x + y * lattice_size + z * lattice_size**2

def flatten_index(site, color, num_colors):
    return site * num_colors + color

# Initialize the fermionic Hamiltonian terms
fermionic_terms = {}
link_count = 0
for z in range(lattice_size):
    for y in range(lattice_size):
        for x in range(lattice_size):
            site = coord_to_index(x, y, z, lattice_size)
            for alpha in range(num_colors):
                idx = flatten_index(site, alpha, num_colors)
                staggered_phase = (-1) ** ((x + y + z) % 2)
                fermionic_terms[f"+_{idx} -_{idx}"] = m * staggered_phase

            for direction, neighbor_offset in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
                nx, ny, nz = x + neighbor_offset[0], y + neighbor_offset[1], z + neighbor_offset[2]
                if 0 <= nx < lattice_size and 0 <= ny < lattice_size and 0 <= nz < lattice_size:
                    neighbor_site = coord_to_index(nx, ny, nz, lattice_size)
                    U = gauge_field[link_count]
                    link_count += 1
                    for alpha in range(num_colors):
                        idx1 = flatten_index(site, alpha, num_colors)
                        idx2 = flatten_index(neighbor_site, alpha, num_colors)
                        coeff = -t * np.real(np.trace(U)) / 2
                        fermionic_terms[f"+_{idx1} -_{idx2}"] = coeff
                        fermionic_terms[f"+_{idx2} -_{idx1}"] = coeff

# Fermionic Hamiltonian from fermionic terms
fermionic_hamiltonian = FermionicOp(fermionic_terms, num_spin_orbitals=num_sites * num_colors)
mapper = JordanWignerMapper()
fermionic_qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

# Total qubits in the combined system
total_qubits = num_sites * num_colors + num_links

# Expand fermionic terms to the full qubit space
fermionic_qubit_hamiltonian = SparsePauliOp.from_list(
    [(pauli + "I" * num_links, coeff) for pauli, coeff in fermionic_qubit_hamiltonian.to_list()]
)

# Create and pad gauge field Hamiltonian
gauge_field_ops = []
for i in range(num_links):
    pauli_str = "I" * (num_sites * num_colors + i) + "Z" + "I" * (num_links - i - 1)
    gauge_field_ops.append(SparsePauliOp.from_list([(pauli_str, g)]))

# Sum the gauge field operators into a single SparsePauliOp
gauge_field_hamiltonian = sum(gauge_field_ops)

gauge_qubit_hamiltonian = SparsePauliOp.from_list(
    [(pauli + "I" * num_links, coeff) for pauli, coeff in gauge_field_hamiltonian.to_list()]
)


# Combine fermionic and gauge field Hamiltonians
total_hamiltonian = fermionic_qubit_hamiltonian + gauge_field_hamiltonian


print("Fermionic qubits: ",fermionic_qubit_hamiltonian.num_qubits)
print("Gauge qubits: ", gauge_qubit_hamiltonian.num_qubits)
print("Number of qubits in total Hamiltonian:", total_hamiltonian.num_qubits)

# Extract Hamiltonian terms
hamiltonian_terms = total_hamiltonian.to_list()

# Define the energy function
def energy(c: tc.Circuit):
    e = 0.0
    for pauli_string, coeff in hamiltonian_terms:
        operators = []
        for idx, pauli in enumerate(pauli_string):
            if pauli != 'I':
                if pauli == 'X':
                    operators.append((tc.gates.x(), [idx]))
                elif pauli == 'Y':
                    operators.append((tc.gates.y(), [idx]))
                elif pauli == 'Z':
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
NN_shape = 30
maxiter = 1000
lr = 0.009
stddev = 0.1

# Train and plot
with tf.device("/cpu:0"):
    energies, m = train(n, d, NN_shape=NN_shape, maxiter=maxiter, lr=lr, stddev=stddev)

from qiskit.quantum_info import SparsePauliOp

# Map Fermionic Hamiltonian to Qubit Operator
jw_mapper = JordanWignerMapper()
qubit_hamiltonian = jw_mapper.map(fermionic_hamiltonian)

# Convert Qubit Operator to Sparse Matrix directly
sparse_matrix = qubit_hamiltonian.to_matrix(sparse=True)  # Directly convert to matrix

# Use a classical sparse eigenvalue solver to find the minimum eigenvalue
from scipy.sparse.linalg import eigsh
min_eigenvalue, _ = eigsh(sparse_matrix, k=1, which='SA')  # 'SA' finds smallest algebraic eigenvalue

print("Minimum Eigenvalue:", min_eigenvalue[0])



# Plot the energy convergence
plt.plot(range(len(energies)), energies)
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("Energy Convergence During Training")
plt.show()
