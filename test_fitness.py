import numpy as np
from fitness import get_gait_feet, ground_z

np.set_printoptions(precision=3, suppress=True)

pose = np.zeros(24)
pose[1::3] = -1.0     
pose[2::3] = -0.5     
standing = np.tile(pose, (300, 1))   # repeat that pose 300 times 

feet = get_gait_feet(standing)
print("Shape:", feet.shape)
print("Foot heights in frame 0:", feet[0, :, 2])
print("Ground is at:", ground_z)