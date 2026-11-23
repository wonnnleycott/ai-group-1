"""Checks that the from-scratch network and its back propagation are correct."""
import numpy as np
from neural_network import Full_NN

rng = np.random.default_rng(1)

# ---- Test 1: shapes ----
nn = Full_NN(X=2, HL=[16], Y=24)
out = nn.FF(rng.uniform(-1, 1, (5, 2)))
print("Layers:", nn.L, " output shape for 5 examples:", out.shape)
print("Outputs between 0 and 1:", bool(out.min() > 0 and out.max() < 1))

# ---- Test 2: gradient check ----
# Back propagation should give the same weight changes as nudging each weight
# by a tiny amount and measuring how the error changes (a numerical check).
nn = Full_NN(X=2, HL=[4], Y=3, seed=2)
x = rng.uniform(-1, 1, (6, 2))
t = rng.uniform(0, 1, (6, 3))


def error(net):
    return 0.5 * np.sum((t - net.FF(x)) ** 2) / x.shape[0]


nn.BP(t - nn.FF(x))
bp_change = nn.Der[0][0, 0]                  # BP's answer for one weight
eps = 1e-6
nn.W[0][0, 0] += eps;     e_up = error(nn)
nn.W[0][0, 0] -= 2 * eps; e_down = error(nn)
nn.W[0][0, 0] += eps
numeric_change = -(e_up - e_down) / (2 * eps)
print("Gradient check - back propagation:", round(bp_change, 8), " numerical:", round(numeric_change, 8))
print("Back propagation correct:", bool(np.isclose(bp_change, numeric_change, rtol=1e-5)))

# ---- Test 3: does it learn? ----
nn = Full_NN(X=2, HL=[4], Y=3, seed=2)
before = nn.mse(t, nn.FF(x))
for _ in range(2000):
    nn.BP(t - nn.FF(x))
    nn.GD(lr=1.0)
after = nn.mse(t, nn.FF(x))
print("Loss before training:", round(before, 4), " after 2000 updates:", round(after, 4))