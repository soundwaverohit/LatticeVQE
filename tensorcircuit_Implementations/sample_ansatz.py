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

# Define your Hamiltonian parameters and generate the Hamiltonian
g = 2.0
t = 1.0
m = 0.5
lattice_size = 2
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
# Improved MERA function with more complex entangling blocks for better Hamiltonian representation
def MERA(inp, n, d=3, layers=4, energy_flag=False):
    """
    Builds an advanced MERA-inspired ansatz with alternating entanglement patterns and parameter-rich gates.
    - n: Number of qubits
    - d: Depth of each entangling block
    - layers: Number of layers to increase the circuit depth and expressivity
    - energy_flag: If True, returns the energy expectation value; otherwise, returns the circuit.
    """
    params = K.cast(inp["params"], dtype="float32")  # Ensure real-valued input
    params = K.cast(params, "complex128")  # Cast to complex
    params = K.reshape(params, [-1])  # Flatten parameters
    c = tc.Circuit(n)
    idx = 0

    # Apply initial layer of single-qubit rotations for each qubit
    for i in range(n):
        c.rx(i, theta=params[idx])
        c.ry(i, theta=params[idx + 1])
        c.rz(i, theta=params[idx + 2])
        idx += 3

    # Multi-layer entangling blocks with parameter-rich gates
    for layer in range(layers):
        for depth in range(d):
            # Nearest-neighbor entanglement with CRY + CNOT + RX blocks
            for i in range(n - 1):
                c.cry(i, i + 1, theta=params[idx])
                idx += 1
                c.cnot(i + 1, i)
                c.rx(i, theta=params[idx])
                idx += 1

            # Next-nearest-neighbor entanglement with CU3 gates
            for i in range(0, n - 2, 2):
                if i + 2 < n:
                    c.cu(i, i + 2, theta=params[idx], phi=params[idx + 1])
                    idx += 3

            # Single-qubit rotations after entanglement for flexibility
            for i in range(n):
                c.rz(i, theta=params[idx])
                c.rx(i, theta=params[idx + 1])
                c.ry(i, theta=params[idx + 2])
                idx += 3

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

    # Get the number of parameters required for the MERA circuit
    dummy_params = np.zeros(3000)
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

# Adjusted training function with energy recording
def train(n, d, NN_shape, maxiter=10000, lr=0.005, stddev=1.0):
    exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=1000, decay_rate=0.7
    )
    opt = tf.keras.optimizers.Adam(exp_lr)
    m = create_NN_MERA(n, d, NN_shape, stddev)
    energies = []  # List to store energies

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
        if i % 500 == 0:
            print("epoch", i, ":", total_e.numpy())

    m.save_weights("NN-VQE.weights.h5")
    return energies, m  # Return the list of energies and the trained model

# Set the number of qubits
n = num_sites * num_colors
d = 2
NN_shape = 30
maxiter = 10000  
lr = 0.009
stddev = 0.1

# Run the training and record energies
with tf.device("/cpu:0"):
    energies, m = train(n, d, NN_shape=NN_shape, maxiter=maxiter, lr=lr, stddev=stddev)



# --- Code to print and save the quantum circuit ---


# Extract the parameters from the trained model
params_model = tf.keras.Model(inputs=m.input, outputs=m.layers[-2].output)
input_val = np.array([[0.0]])
params = params_model.predict(input_val)[0]

# Build the circuit with the trained parameters
c, idx = MERA({"params": params}, n, d, energy_flag=False)

# Convert the TensorCircuit circuit to a Qiskit QuantumCircuit
qiskit_circuit = c.to_qiskit()

# Draw and save the circuit using Qiskit
from qiskit.visualization import circuit_drawer

# Save the circuit diagram as a PNG image
circuit_drawer(qiskit_circuit, output='mpl', filename='quantum_circuit.png')

print("Quantum circuit saved as 'quantum_circuit.png'.")


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


# Plot the energy convergence
plt.plot(range(len(energies)), energies)
plt.xlabel("Epoch")
plt.ylabel("Energy")
plt.title("Energy Convergence During Training")
plt.show()

