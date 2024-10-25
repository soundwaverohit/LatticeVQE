import numpy as np
import tensorflow as tf
import tensorcircuit as tc
import cotengra
from functools import partial
import random
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
import scipy.sparse.linalg

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

# Define your Hamiltonian parameters and generate the Hamiltonian
g = 2.0
t = 1.0
m = 0.5
lattice_size = 3
num_sites = lattice_size * lattice_size
num_colors = 3

# SU(3) Gell-Mann matrices (simplified representation)
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

# Initialize gauge field as random SU(3) matrices for each link in the 2D lattice
def random_su3():
    random_combination = sum(random.uniform(0, 1) * G for G in su3_generators)
    exp_matrix = expm(1j * random_combination)
    # Normalize to ensure unitary
    U, _, Vh = np.linalg.svd(exp_matrix)
    return U @ Vh

gauge_field = [random_su3() for _ in range(2 * num_sites)]  # For each link

# Helper function to flatten site and color indices into a single index
def flatten_index(site, color, num_colors):
    return site * num_colors + color

# Hopping terms for SU(3) (fermionic hopping terms)
fermionic_terms = {}
for n in range(num_sites - 1):
    U = gauge_field[n]
    for alpha in range(num_colors):
        idx1 = flatten_index(n, alpha, num_colors)
        idx2 = flatten_index(n + 1, alpha, num_colors)
        coeff = -t * np.real(np.trace(U @ U.conj().T)) / 2
        # Add both the forward and reverse hopping terms to ensure Hermiticity
        fermionic_terms[f"+_{idx1} -_{idx2}"] = coeff
        fermionic_terms[f"+_{idx2} -_{idx1}"] = coeff

# On-site mass terms (fermionic creation/annihilation for each color)
for n in range(num_sites):
    for alpha in range(num_colors):
        idx = flatten_index(n, alpha, num_colors)
        fermionic_terms[f"+_{idx} -_{idx}"] = m

# Plaquette terms for SU(3) (not included here due to complexity)

# Create the FermionicOp with the dictionary of fermionic hopping and mass terms
fermionic_hamiltonian = FermionicOp(fermionic_terms, num_spin_orbitals=num_sites * num_colors)

# Map to qubit Hamiltonian
mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(fermionic_hamiltonian)

# Extract Hamiltonian terms
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
                else:
                    raise ValueError(f"Unknown Pauli operator: {pauli}")
        if operators:
            e += coeff * c.expectation(*operators)
        else:
            # Identity operator
            e += coeff
    return K.real(e)

# Adjusted MERA function to handle arbitrary n
def MERA(inp, n, d=1, energy_flag=False):
    params = K.reshape(K.cast(inp["params"], "complex128"), [-1])
    c = tc.Circuit(n)

    idx = 0

    # Initial layer of single-qubit rotations
    for i in range(n):
        c.rx(i, theta=params[3 * i])
        c.rz(i, theta=params[3 * i + 1])
        c.rx(i, theta=params[3 * i + 2])
    idx += 3 * n

    max_layers = int(np.ceil(np.log2(n)))  # Adjusted here
    for n_layer in range(1, max_layers + 1):
        n_qubit = 2 ** n_layer
        step = max(1, n // n_qubit)

        for _ in range(d):
            for i in range(step, n - step, 2 * step):
                if i + step < n:
                    c.rxx(i, i + step, theta=params[idx])
                    c.rzz(i, i + step, theta=params[idx + 1])
                    idx += 2

            for i in range(0, n - step, 2 * step):
                if i + step < n:
                    c.rxx(i, i + step, theta=params[idx])
                    c.rzz(i, i + step, theta=params[idx + 1])
                    idx += 2

            for i in range(0, n):
                c.rx(i, theta=params[idx])
                c.rz(i, theta=params[idx + 1])
                idx += 2

    if energy_flag:
        return energy(c)
    else:
        return c, idx

# Adjusted QuantumLayer
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, circuit_fn, **kwargs):
        super().__init__(**kwargs)
        self.circuit_fn = circuit_fn

    def call(self, inputs):
        return self.circuit_fn(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([1])

# Adjusted model creation function
def create_NN_MERA(n, d, NN_shape, stddev):
    input = tf.keras.layers.Input(shape=[1])
    x = tf.keras.layers.Dense(
        units=NN_shape,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="relu",
    )(input)
    x = tf.keras.layers.Dropout(0.05)(x)

    _, idx = MERA({"params": np.zeros(3000)}, n, d, energy_flag=False)
    params = tf.keras.layers.Dense(
        units=idx,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="sigmoid",
    )(x)

    qlayer = QuantumLayer(partial(MERA, n=n, d=d, energy_flag=True))
    output = qlayer({"params": 6.3 * params})
    m = tf.keras.Model(inputs=input, outputs=output)
    return m

# Adjusted training function
def train(n, d, NN_shape, maxiter=10000, lr=0.005, stddev=1.0):
    exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=1000, decay_rate=0.7
    )
    opt = tf.keras.optimizers.Adam(exp_lr)
    m = create_NN_MERA(n, d, NN_shape, stddev)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            e = m(K.reshape([0.0], [1]))
        grads = tape.gradient(e, m.trainable_variables)
        opt.apply_gradients(zip(grads, m.trainable_variables))
        return e

    for i in range(maxiter):
        total_e = train_step()
        if i % 500 == 0:
            print("epoch", i, ":", total_e.numpy())

    m.save_weights("NN-VQE.weights.h5")

# Set the number of qubits
n = num_sites * num_colors
d = 2
NN_shape = 20
maxiter = 2500
lr = 0.009
stddev = 0.1

# Run the training
with tf.device("/cpu:0"):
    train(n, d, NN_shape=NN_shape, maxiter=maxiter, lr=lr, stddev=stddev)

# After training, you can test the model or compute observables as needed.



# Load the trained model
m = create_NN_MERA(n, d, NN_shape, stddev)
m.load_weights("NN-VQE.weights.h5")

# Compute the energy predicted by the model
with tf.device("/cpu:0"):
    predicted_energy = m.predict(K.reshape([0.0], [1]))[0][0].numpy()
print(f"Predicted Ground State Energy: {predicted_energy}")

# Compute the exact energy (if feasible)
if n <= 12:  # Adjust this based on your computational resources
    pauli_op = SparsePauliOp.from_list(hamiltonian_terms)
    hamiltonian_matrix = pauli_op.to_spmatrix()
    eigenvalues, _ = scipy.sparse.linalg.eigsh(hamiltonian_matrix, k=1, which='SA')
    exact_energy = eigenvalues[0].real
    print(f"Exact Ground State Energy: {exact_energy}")
    print(f"Absolute Error: {abs(predicted_energy - exact_energy)}")
    print(f"Relative Error: {abs(predicted_energy - exact_energy) / abs(exact_energy)}")
else:
    print("System size too large for exact diagonalization.")

# Plot energy convergence (if energies were recorded during training)
if 'energies' in locals():
    plt.plot(range(len(energies)), energies)
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title("Energy Convergence During Training")
    plt.show()