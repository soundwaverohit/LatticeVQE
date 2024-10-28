import numpy as np
import tensorflow as tf
import tensorcircuit as tc
from functools import partial

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

def energy(c: tc.Circuit, j: float = 1.0, hx: float = 1.0):
    e = 0.0
    n = c._nqubits
    # <Z_i Z_{i+1}>
    for i in range(n - 1):
        e += j * c.expectation((tc.gates.z(), [i]), (tc.gates.z(), [i + 1]))
    # <X_i>
    for i in range(n):
        e -= hx * c.expectation((tc.gates.x(), [i]))
    return tc.backend.real(e)

def build_MERA_circuit(params, n):
    params = tc.backend.cast(params, "complex128")
    c = tc.Circuit(n)

    idx = 0  # index of params

    for i in range(n):
        c.rx(i, theta=params[2 * i])
        c.rz(i, theta=params[2 * i + 1])
    idx += 2 * n

    for n_layer in range(1, int(np.log2(n)) + 1):
        n_qubit = 2 ** n_layer  # number of qubits involving
        step = int(n / n_qubit)

        # even
        for i in range(step, n - step, 2 * step):
            c.exp1(i, i + step, theta=params[idx], unitary=tc.gates._xx_matrix)
            c.exp1(i, i + step, theta=params[idx + 1], unitary=tc.gates._zz_matrix)
            idx += 2

        # odd
        for i in range(0, n - step, 2 * step):
            c.exp1(i, i + step, theta=params[idx], unitary=tc.gates._xx_matrix)
            c.exp1(i, i + step, theta=params[idx + 1], unitary=tc.gates._zz_matrix)
            idx += 2

        # single qubit rotations
        for i in range(0, n, step):
            c.rx(i, theta=params[idx])
            c.rz(i, theta=params[idx + 1])
            idx += 2

    return c, idx

def MERA(params):
    c, _ = build_MERA_circuit(params, n)
    e = energy(c)
    return e

n = 8
params = np.zeros(1000)
cirq, idx = build_MERA_circuit(params, n)
print("The number of parameters is", idx)
# cirq.draw()  # Uncomment if you want to draw the circuit

# Fix 'n' using partial to allow vectorization over 'params'
MERA_partial = partial(MERA)
MERA_tfim_vvag = tc.backend.jit(tc.backend.vectorized_value_and_grad(MERA_partial))

def batched_train(idx, batch=10, maxiter=10000, lr=0.005):
    params = tf.Variable(
        initial_value=tf.random.normal(
            shape=[batch, idx], stddev=1, dtype=getattr(tf, tc.rdtypestr)
        )
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    lowest_energy = 1e5
    for i in range(maxiter):
        e, grad = MERA_tfim_vvag(params)
        opt.apply_gradients([(grad, params)])
        current_lowest = tf.reduce_min(e)
        if current_lowest < lowest_energy:
            lowest_energy = current_lowest
        if i % 200 == 0:
            print(f"Iteration {i}, Energy: {e.numpy()}")
    return lowest_energy

lowest_energy = batched_train(idx, batch=5, maxiter=2000, lr=0.007)

# DMRG comparison
try:
    # Option A: Try importing from quimb.tensor.tensor_gen
    #from quimb.tensor.tensor_gen import MPO_ham_ising
    h = MPO_ham_ising(n, j=4.0, bx=2.0, S=0.5, cyclic=False)
except ImportError:
    try:
        # Option B: Try importing from quimb.tensor.tensor_1d
        from quimb.tensor.tensor_1d import MPO_ham_ising
        h = MPO_ham_ising(n, j=4.0, bx=2.0, S=0.5, cyclic=False)
    except ImportError:
        try:
            # Option C: Try importing from quimb.tensor
            from quimb.tensor import MPO_ham_ising
            h = MPO_ham_ising(n, j=4.0, bx=2.0, S=0.5, cyclic=False)
        except ImportError:
            try:
                # Option D: Try importing ham_mpo_ising from quimb
                from quimb import ham_mpo_ising
                h = ham_mpo_ising(n, j=4.0, h=2.0, cyclic=False)
            except ImportError:
                # Option E: Use an alternative method
                import quimb
                import quimb.linalg as qla
                H = quimb.ham_1d_ising(n, j=4.0, bx=2.0, sparse=True)
                eigs = qla.eigh(H, k=1, which='SA')
                energy_DMRG = eigs[0]
                # Compare
                print("Exact solution:", energy_DMRG)
                print("MERA solution:", lowest_energy.numpy())
                exit()

from quimb.tensor.tensor_dmrg import DMRG2
dmrg = DMRG2(h, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-13)
dmrg.solve(tol=1e-9, verbosity=0)
energy_DMRG = dmrg.energy

# Compare
print("DMRG solution:", energy_DMRG)
print("MERA solution:", lowest_energy.numpy())
