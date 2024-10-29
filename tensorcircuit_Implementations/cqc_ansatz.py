import numpy as np
import tensorflow as tf
import tensorcircuit as tc
import cotengra
from functools import partial
import random
from scipy.linalg import expm
import matplotlib.pyplot as plt

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

# Import Qiskit and related libraries
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

# Fix the random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define Hamiltonian parameters
g = 2.0
t = 1.0
m = 0.5
lattice_size = 2
num_sites = lattice_size * lattice_size
num_colors = 3

# SU(3) Gell-Mann matrices
su3_generators = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
    np.array([[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(3), 0], [0, 0, -2/np.sqrt(3)]], dtype=complex),
]

# Random SU(3) matrix generator
def random_su3():
    random_combination = sum(random.uniform(0, 1) * G for G in su3_generators)
    exp_matrix = expm(1j * random_combination)
    U, _, Vh = np.linalg.svd(exp_matrix)
    return U @ Vh

gauge_field = [random_su3() for _ in range(2 * num_sites)]

# Helper for flattening site and color indices
def flatten_index(site, color, num_colors):
    return site * num_colors + color

# Fermionic terms based on Hamiltonian structure
fermionic_terms = {}
for n in range(num_sites - 1):
    U = gauge_field[n]
    for alpha in range(num_colors):
        idx1 = flatten_index(n, alpha, num_colors)
        idx2 = flatten_index(n + 1, alpha, num_colors)
        coeff = -t * np.real(np.trace(U @ U.conj().T)) / 2
        fermionic_terms[f"+_{idx1} -_{idx2}"] = coeff
        fermionic_terms[f"+_{idx2} -_{idx1}"] = coeff

for n in range(num_sites):
    for alpha in range(num_colors):
        idx = flatten_index(n, alpha, num_colors)
        fermionic_terms[f"+_{idx} -_{idx}"] = m

fermionic_hamiltonian = FermionicOp(fermionic_terms, num_spin_orbitals=num_sites * num_colors)
mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
hamiltonian_terms = qubit_hamiltonian.to_list()

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
        if operators:
            e += coeff * c.expectation(*operators)
        else:
            e += coeff
    return K.real(e)

# CQC Ansatz with convolutional and global entanglement layers
def CQC_ansatz(inp, n, conv_depth=2, layers=5, energy_flag=False):
    params = K.cast(inp["params"], dtype="float32")
    params = K.cast(params, "complex128")
    params = K.reshape(params, [-1])
    c = tc.Circuit(n)
    idx = 0

    # Initial single-qubit rotations
    for i in range(n):
        c.rx(i, theta=params[idx])
        c.ry(i, theta=params[idx + 1])
        c.rz(i, theta=params[idx + 2])
        idx += 3

    # Convolutional layers with local entanglement
    for _ in range(conv_depth):
        for i in range(n - 1):
            c.cry(i, i + 1, theta=params[idx])
            c.cz(i + 1, i)
            idx += 1
        for i in range(n):
            c.rz(i, theta=params[idx])
            c.rx(i, theta=params[idx + 1])
            idx += 2

    # Global entanglement layers
    for _ in range(layers):
        for i in range(0, n - 1, 2):
            if i + 1 < n:
                c.cu(i, i + 1, theta=params[idx], phi=params[idx + 1])
                idx += 3
            if i + 2 < n:
                c.cu(i, i + 2, theta=params[idx], phi=params[idx + 1])
                idx += 3
        for i in range(n):
            c.rx(i, theta=params[idx])
            c.ry(i, theta=params[idx + 1])
            c.rz(i, theta=params[idx + 2])
            idx += 3

    if energy_flag:
        return energy(c)
    else:
        return c, idx

# QuantumLayer
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, circuit_fn, **kwargs):
        super().__init__(**kwargs)
        self.circuit_fn = circuit_fn

    def call(self, inputs):
        return self.circuit_fn(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

# Model creation function
def create_NN_CQC(n, conv_depth, layers, NN_shape, stddev):
    input = tf.keras.layers.Input(shape=[1])
    x = tf.keras.layers.Dense(
        units=NN_shape,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="relu",
    )(input)
    x = tf.keras.layers.Dropout(0.05)(x)

    dummy_params = np.zeros(3000)
    _, idx = CQC_ansatz({"params": dummy_params}, n, conv_depth, layers, energy_flag=False)

    params = tf.keras.layers.Dense(
        units=idx,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="sigmoid",
    )(x)

    qlayer = QuantumLayer(partial(CQC_ansatz, n=n, conv_depth=conv_depth, layers=layers, energy_flag=True))
    output = qlayer({"params": 6.3 * params})
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

# Training function with energy recording
def train(n, conv_depth, layers, NN_shape, maxiter=1000, lr=0.005, stddev=1.0):
    exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=1000, decay_rate=0.7
    )
    opt = tf.keras.optimizers.Adam(exp_lr)
    model = create_NN_CQC(n, conv_depth, layers, NN_shape, stddev)
    energies = []

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            e = model(K.reshape([0.0], [1]))
        grads = tape.gradient(e, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return e

    for i in range(maxiter):
        total_e = train_step()
        energies.append(total_e.numpy())
        if i % 100 == 0:
            print("epoch", i, ":", total_e.numpy())

    model.save_weights("CQC-VQE.weights.h5")
    return energies, model

# Set parameters and run training
n = num_sites * num_colors
conv_depth = 2
layers = 5
NN_shape = 30
maxiter = 10000
lr = 0.009
stddev = 0.1

with tf.device("/cpu:0"):
    energies, model = train(n, conv_depth, layers, NN_shape, maxiter, lr, stddev)





from qiskit.quantum_info import SparsePauliOp

# Step 1: Map Fermionic Hamiltonian to Qubit Operator
jw_mapper = JordanWignerMapper()
qubit_hamiltonian = jw_mapper.map(fermionic_hamiltonian)

# Step 2: Convert Qubit Operator to Sparse Matrix directly
sparse_matrix = qubit_hamiltonian.to_matrix(sparse=True)  # Directly convert to matrix

# Step 3: Use a classical sparse eigenvalue solver to find the minimum eigenvalue
from scipy.sparse.linalg import eigsh
min_eigenvalue, _ = eigsh(sparse_matrix, k=1, which='SA')  # 'SA' finds smallest algebraic eigenvalue

print("Minimum Eigenvalue:", min_eigenvalue[0])

# Visualize energy convergence
plt.plot(range(len(energies)), energies)
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("Energy Convergence During Training")
plt.show()
