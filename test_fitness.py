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

from fitness import score_stability


def legs_down_walk(down_legs):
#Test walk: legs in down_legs stand on the ground, the rest are lifted.
#Leg numbers: 0=L1 1=L2 2=L3 3=L4 4=R4 5=R3 6=R2 7=R1
    pose = np.zeros(24)                  # feet up in the air
    for leg in down_legs:
        pose[3*leg + 1] = -1.0           # femur down
        pose[3*leg + 2] = -0.5           # tibia down
    return np.tile(pose, (300, 1))


print("\nStability:")
print("all 8 feet down     :", score_stability(get_gait_feet(legs_down_walk(range(8)))))
print("all feet up         :", score_stability(get_gait_feet(legs_down_walk([]))))
print("only left feet down :", score_stability(get_gait_feet(legs_down_walk([0, 1, 2, 3]))))
print("L1 L3 R4 R2 down    :", score_stability(get_gait_feet(legs_down_walk([0, 2, 4, 6]))))

half = np.vstack([legs_down_walk(range(8))[:150], legs_down_walk([0, 1, 2, 3])[:150]])
print("stable half the time:", score_stability(get_gait_feet(half)))