import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorcircuit as tc
import cotengra
import quimb
from tqdm import tqdm
from functools import partial

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

def energy(c: tc.Circuit, lamb: float = 1.0, delta: float = 1.0):
    e = 0.0
    n = c._nqubits
    for i in range(n):
        e += lamb * c.expectation((tc.gates.z(), [i]))  # <Z_i>
    for i in range(n):
        e += c.expectation(
            (tc.gates.x(), [i]), (tc.gates.x(), [(i + 1) % n])
        )  # <X_i X_{i+1}>
        e += c.expectation(
            (tc.gates.y(), [i]), (tc.gates.y(), [(i + 1) % n])
        )  # <Y_i Y_{i+1}>
        e += delta * c.expectation(
            (tc.gates.z(), [i]), (tc.gates.z(), [(i + 1) % n])
        )  # <Z_i Z_{i+1}>
    return K.real(e)

def MERA(inp, n, d=1, lamb=1.0, energy_flag=False):
    # Remove the batch dimension
    params = K.reshape(K.cast(inp["params"], "complex128"), [-1])
    delta = K.reshape(K.cast(inp["delta"], "complex128"), [])
    c = tc.Circuit(n)

    idx = 0

    for i in range(n):
        c.rx(i, theta=params[3 * i])
        c.rz(i, theta=params[3 * i + 1])
        c.rx(i, theta=params[3 * i + 2])
    idx += 3 * n

    for n_layer in range(1, int(np.log2(n)) + 1):
        n_qubit = 2**n_layer
        step = int(n / n_qubit)

        for _ in range(d):
            for i in range(step, n - step, 2 * step):
                c.rxx(i, i + step, theta=params[idx])
                c.rzz(i, i + step, theta=params[idx + 1])
                idx += 2

            for i in range(0, n, 2 * step):
                c.rxx(i, i + step, theta=params[idx])
                c.rzz(i, i + step, theta=params[idx + 1])
                idx += 2

            for i in range(0, n, step):
                c.rx(i, theta=params[idx])
                c.rz(i, theta=params[idx + 1])
                idx += 2

    if energy_flag:
        return energy(c, lamb, delta)
    else:
        return c, idx


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, circuit_fn, **kwargs):
        super().__init__(**kwargs)
        self.circuit_fn = circuit_fn

    def call(self, inputs):
        return self.circuit_fn(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape['delta'][0], 1])

def create_NN_MERA(n, d, lamb, NN_shape, stddev):
    input = tf.keras.layers.Input(shape=[1])

    x = tf.keras.layers.Dense(
        units=NN_shape,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="relu",
    )(input)

    x = tf.keras.layers.Dropout(0.05)(x)

    _, idx = MERA({"params": np.zeros(3000), "delta": 0.0}, n, d, 1.0, energy_flag=False)
    params = tf.keras.layers.Dense(
        units=idx,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=stddev),
        activation="sigmoid",
    )(x)

    qlayer = QuantumLayer(partial(MERA, n=n, d=d, lamb=lamb, energy_flag=True))

    output = qlayer({"params": 6.3 * params, "delta": input})

    m = tf.keras.Model(inputs=input, outputs=output)

    return m

def train(n, d, lamb, delta, NN_shape, maxiter=10000, lr=0.005, stddev=1.0):
    exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr, decay_steps=1000, decay_rate=0.7
    )
    opt = tf.keras.optimizers.Adam(exp_lr)

    m = create_NN_MERA(n, d, lamb, NN_shape, stddev)

    @tf.function
    def train_step(de):
        with tf.GradientTape() as tape:
            e = m(K.reshape(de, [1]))
        grads = tape.gradient(e, m.trainable_variables)
        opt.apply_gradients(zip(grads, m.trainable_variables))
        return e

    for i in range(maxiter):
        total_e = tf.zeros([1], dtype=tf.float64)
        for de in delta:
            total_e += train_step(de)
        if i % 500 == 0:
            print("epoch", i, ":", total_e)

    m.save_weights("NN-VQE.weights.h5")

n = 8
d = 2
lamb = 0.75
delta = np.linspace(-3.0, 3.0, 20, dtype="complex128")
NN_shape = 20
maxiter = 2500
lr = 0.009
stddev = 0.1

with tf.device("/cpu:0"):
    train(n, d, lamb, delta, NN_shape=NN_shape, maxiter=maxiter, lr=lr, stddev=stddev)

test_delta = np.linspace(-4.0, 4.0, 201)
test_energies = tf.zeros_like(test_delta).numpy()
m = create_NN_MERA(n, d, lamb, NN_shape, stddev)
m.load_weights("NN-VQE.weights.h5")
for i, de in tqdm(enumerate(test_delta)):
    test_energies[i] = m(K.reshape(de, [1]))

analytical_energies = []
for i in test_delta:
    h = quimb.tensor.tensor_builder.MPO_ham_XXZ(
        n, i * 4, jxy=4.0, bz=2.0 * 0.75, S=0.5, cyclic=True
    )
    h = h.to_dense()
    analytical_energies.append(np.min(quimb.eigvalsh(h)))

plt.plot(
    test_delta,
    (test_energies - analytical_energies) / np.abs(analytical_energies),
    "-",
    color="b",
)
plt.xlabel("Delta", fontsize=14)
plt.ylabel("GS Relative Error", fontsize=14)
plt.axvspan(-3.0, 3.0, color="darkgrey", alpha=0.5)
plt.savefig("plot.png")
plt.show()