"""Quick check that spider_kinematics.py gives the expected foot positions."""
import numpy as np
from spider_kinematics import get_foot_positions

np.set_printoptions(precision=3, suppress=True)   # show numbers nicely (3 decimals)

# Test 1: every angle is 0
print("Test 1 - all angles 0 (each row = one foot: x, y, z):")
print(get_foot_positions(np.zeros(24)))

# Test 2: bend L1's femur DOWN
pose = np.zeros(24)
pose[1] = -0.8               # index 1 = L1 femur
print("\nTest 2 - L1 femur = -0.8, L1 foot:", get_foot_positions(pose)[0])

# Test 3: bend L1's femur UP
pose = np.zeros(24)
pose[1] = 0.8
print("Test 3 - L1 femur = +0.8, L1 foot:", get_foot_positions(pose)[0])