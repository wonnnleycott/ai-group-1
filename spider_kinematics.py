import numpy as np

segment_lengths = [1.2, 0.7, 1.0]          # length of coxa, femur, tibia
body_a, body_b = 1.5, 1.0                  
base_angles = np.deg2rad([45, 75, 105, 135, -135, -105, -75, -45])  # where each leg joins the body
leg_names = ["L1", "L2", "L3", "L4", "R4", "R3", "R2", "R1"]


def axis_angle_rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)      
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c],
    ])

def forward_leg_kinematics(base_pos, base_angle, joint_angles, segment_lengths=segment_lengths):
    """Follows one leg from the body to the foot.
    Returns j1 (hip), j2 (knee), j3 (ankle), j4 (foot), each as [x, y, z]."""
    theta1, theta2, theta3 = joint_angles    # coxa, femur, tibia angles
    L1, L2, L3 = segment_lengths             # coxa, femur, tibia lengths
    up = np.array([0.0, 0.0, 1.0])           # vertical direction

    # where the leg joins the body
    j1 = np.asarray(base_pos, dtype=float)

    # points outwards (base_angle + theta1), tilted up by a fixed 30 degrees
    coxa_horiz = np.array([np.cos(base_angle + theta1), np.sin(base_angle + theta1), 0.0])
    R = axis_angle_rotation_matrix(np.cross(coxa_horiz, up), np.deg2rad(30))
    coxa_dir = R @ coxa_horiz
    j2 = j1 + L1 * coxa_dir

    # turn the coxa direction up/down by theta2
    femur_axis = np.cross(coxa_dir, up)
    femur_dir = axis_angle_rotation_matrix(femur_axis, theta2) @ coxa_dir
    j3 = j2 + L2 * femur_dir

    # turn the femur direction up/down by theta3
    tibia_axis = np.cross(femur_dir, up)
    tibia_dir = axis_angle_rotation_matrix(tibia_axis, theta3) @ femur_dir
    j4 = j3 + L3 * tibia_dir                 # j4 is the foot

    return j1, j2, j3, j4


def get_foot_positions(pose):
    """for one pose (24 angles), returns an 8x3 array: the [x, y, z] of each foot.
    row order matches the legs: L1 L2 L3 L4 R4 R3 R2 R1."""
    pose = np.asarray(pose, dtype=float)
    feet = np.zeros((8, 3))                  # 8 legs x (x, y, z)
    for i in range(8):
        ang = base_angles[i]
        base = [body_a * np.cos(ang), body_b * np.sin(ang), 0.0]   # where this leg joins the body
        leg_angles = pose[3*i : 3*i + 3]     # this leg's 3 angles
        feet[i] = forward_leg_kinematics(base, ang, leg_angles)[3]  # [3] = j4, the foot
    return feet