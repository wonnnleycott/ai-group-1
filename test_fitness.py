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
# Test walk: legs in down_legs stand on the ground, the rest are lifted.
# Leg numbers: 0=L1 1=L2 2=L3 3=L4 4=R4 5=R3 6=R2 7=R1
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


from fitness import score_forward

def simple_walk(direction=1, lift=True, n=300):
    t = np.arange(n) / n
    gait = np.zeros((n, 24))
    group_a = [0, 2, 4, 6]
    for leg in range(8):
        side = 1 if leg < 4 else -1                      # left and right coxa turn opposite ways
        phase = 0 if leg in group_a else np.pi           # the 2 groups are half a cycle apart
        swing_pos = np.sin(2*np.pi*t + phase)           
        moving_fwd = np.cos(2*np.pi*t + phase) < 0       
        gait[:, 3*leg] = direction * side * 0.3 * swing_pos         
        lifted = moving_fwd if lift else np.zeros(n, bool)
        gait[:, 3*leg + 1] = np.where(lifted, -0.5, -1.0)          
        gait[:, 3*leg + 2] = -0.5                                   
    return gait

print("\nMoving forward:")
for name, gait in {
    "standing still   ": still_walk(-1.0, -0.5),
    "walking forwards ": simple_walk(direction=1),
    "walking backwards": simple_walk(direction=-1),
    "sliding, no lift ": simple_walk(lift=False),
}.items():
    print(name, "forward score =", round(score_forward(get_gait_feet(gait)), 3))


from fitness import penalty_energy, penalty_smoothness

t = np.arange(300) / 300
smooth_wave = still_walk(-1.0, -0.5)
smooth_wave[:, 0::3] += 0.3 * np.sin(2 * np.pi * t)[:, None]   # add swing to every coxa

rng = np.random.default_rng(0)                                 # fixed seed = same "random" every run
random_poses = rng.uniform(-2.0, 2.0, (300, 24))

print("\nEnergy and smoothness (lower = better):")
for name, gait in {
    "standing still": still_walk(-1.0, -0.5),
    "smooth wave   ": smooth_wave,
    "simple walk   ": simple_walk(),
    "random poses  ": random_poses,
}.items():
    print(name, " energy =", round(penalty_energy(gait), 3),
          "  smoothness =", round(penalty_smoothness(gait), 3))


from fitness import penalty_joint_limits

print("\nJoint limits penalty (0 = all allowed):")
print("standing still       :", penalty_joint_limits(still_walk(-1.0, -0.5)))
print("simple walk          :", penalty_joint_limits(simple_walk()))
print("tibias at 1.5 (max 1):", round(penalty_joint_limits(still_walk(-1.0, 1.5)), 4))
print("random -2 to 2 poses :", round(penalty_joint_limits(random_poses), 4))


from fitness import fitness

print("\nFull Fitness: ")
results = {
    "walking forwards ": fitness(simple_walk(direction=1)),
    "standing still   ": fitness(still_walk(-1.0, -0.5)),
    "sliding, no lift ": fitness(simple_walk(lift=False)),
    "walking backwards": fitness(simple_walk(direction=-1)),
    "all zeros        ": fitness(still_walk(0.0, 0.0)),
    "random poses     ": fitness(random_poses),
}
# print them from best to worst
for name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
    print(f"{name}  {score:8.3f}")