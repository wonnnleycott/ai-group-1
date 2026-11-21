"""
neural_network.py  (Part 2)
A multi-layer neural network built from scratch with numpy (no PyTorch / TensorFlow).

Based on the Full_NN class from Computer Practical 5 (week 5):
  - weights for every layer, a feed-forward method (FF), back propagation (BP)
    and a gradient-descent weight update, with a sigmoid activation.
Changes we made (and why):
  1. BIASES: each layer also has a bias, like the "1.0" extra input in the
     back propagation equations sheet. Without biases a neuron's output is
     fixed at 0.5 whenever its inputs are 0, which limits what it can learn.
  2. WHOLE BATCHES AT ONCE: inputs are matrices (rows = examples), so one
     matrix multiplication handles every example (vectorised linear algebra).
  3. SMALL RANDOM STARTING WEIGHTS between -0.5 and 0.5 (the practical uses 0 to 1).
     Mixing positive and negative weights stops every neuron starting out
     pushed the same way, which helps training get going.

The maths (Practical 5):
    x[i+1] = y[i] . W[i] + b[i]               (weighted sum for the next layer)
    y[i+1] = S(x[i+1])                        (sigmoid activation)
    S'(x)  = S(x) (1 - S(x))                  (sigmoid derivative, from its output)
    dE/dW[i] = y[i]^T . [ (t - y) * S'(x[i+1]) ]   (back propagated layer by layer)
"""
import numpy as np


class Full_NN:
    """Multi-layer neural network: X inputs -> hidden layers HL -> Y outputs."""

    def __init__(self, X=2, HL=(16,), Y=24, seed=0):
        self.X, self.HL, self.Y = X, list(HL), Y
        self.L = [X] + self.HL + [Y]               # neurons in every layer, e.g. [2, 16, 24]
        rng = np.random.default_rng(seed)           # fixed seed = same starting weights every run

        self.W, self.b = [], []                     # weights and biases for each layer
        for i in range(len(self.L) - 1):
            self.W.append(rng.uniform(-0.5, 0.5, (self.L[i], self.L[i + 1])))
            self.b.append(np.zeros(self.L[i + 1]))

        self.Der = [np.zeros_like(w) for w in self.W]   # weight changes from back propagation
        self.dB = [np.zeros_like(b) for b in self.b]    # bias changes from back propagation
        self.out = [None] * len(self.L)                 # output of every layer (kept for BP)

    # ----- activation function (Practical 5) -----
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_der(s):
        """Derivative of the sigmoid, given the sigmoid's OUTPUT s: S'(x) = s (1 - s)."""
        return s * (1.0 - s)

    # ----- forward pass -----
    def FF(self, x):
        """Run the network forward. x = (examples x X). Returns (examples x Y)."""
        out = np.atleast_2d(x)
        self.out[0] = out                           # the input layer's output is the input
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            out = self.sigmoid(out @ w + b)         # weighted sum + bias, then sigmoid
            self.out[i + 1] = out                   # keep it for back propagation
        return out

    # ----- back propagation -----
    def BP(self, Er):
        """Er = target - output (examples x Y). Works backwards through the layers,
        storing how much each weight and bias should change in Der and dB."""
        n = Er.shape[0]                             # number of examples in this batch
        for i in reversed(range(len(self.W))):
            delta = Er * self.sigmoid_der(self.out[i + 1])   # error x slope of the sigmoid
            self.Der[i] = self.out[i].T @ delta / n           # average over the batch
            self.dB[i] = delta.mean(axis=0)
            Er = delta @ self.W[i].T                          # pass the error back one layer

    # ----- gradient descent update -----
    def GD(self, lr):
        """Move every weight and bias a small step (learning rate lr) to reduce the error."""
        for i in range(len(self.W)):
            self.W[i] += lr * self.Der[i]
            self.b[i] += lr * self.dB[i]

    # ----- loss -----
    @staticmethod
    def mse(target, output):
        """Mean squared error: average of (target - output)^2 over every number."""
        return float(np.mean((target - output) ** 2))

    # ----- save / load the trained model -----
    def save(self, path):
        arrays = {f"W{i}": w for i, w in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, layers=np.array(self.L), **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        L = [int(v) for v in data["layers"]]
        nn = cls(X=L[0], HL=L[1:-1], Y=L[-1])
        nn.W = [data[f"W{i}"] for i in range(len(L) - 1)]
        nn.b = [data[f"b{i}"] for i in range(len(L) - 1)]
        return nn