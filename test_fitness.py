import numpy as np
from fitness import get_gait_feet, score_ground_contact

def still_walk(femur, tibia):
# Test walk: spider holds one pose for all 300 frames.
    pose = np.zeros(24)
    pose[1::3] = femur        # every femur
    pose[2::3] = tibia        # every tibia
    return np.tile(pose, (300, 1))


tests = {
    "standing on ground (femur -1.0, tibia -0.5)": still_walk(-1.0, -0.5),
    "all zeros (feet in the air)":                 still_walk(0.0, 0.0),
    "legs pushed too deep (femur -1.57)":          still_walk(-1.57, -0.5),
}

for name, gait in tests.items():
    feet = get_gait_feet(gait)
    print(f"{name}: ground score = {score_ground_contact(feet):.3f}")