# Scores how good a walk is. Higher score = better walk.
# A gait is 300 poses x 24 joint angles, in the leg order
# L1 L2 L3 L4 R4 R3 R2 R1

import numpy as np
from spider_kinematics import get_foot_positions

ground_z = -0.5            # where the ground is (body is at z = 0)
contact_tolerance = 0.1    # a foot within 0.1 of the ground counts as touching it


# Works out every foot position for every frame of a walk.
# Input:  gait  = 300 x 24 angles
# Output: feet  = 300 x 8 x 3  (frame, leg, [x y z])
def get_gait_feet(gait):
    gait = np.asarray(gait, dtype=float)
    n_frames = gait.shape[0]                 # how many poses 
    feet = np.zeros((n_frames, 8, 3))       
    for f in range(n_frames):
        feet[f] = get_foot_positions(gait[f])   # all feet for this pose
    return feet



def score_ground_contact(feet):
    heights = feet[:, :, 2]             

    # does each foot reach the ground at its lowest point
    lowest = heights.min(axis=0)             
    gap = np.abs(lowest - ground_z)      
    touch = np.clip(1 - gap / 0.5, 0, 1)     # gap 0 -> 1, gap 0.5+ -> 0
    touch_score = touch.mean()               # average over the 8 legs

    # don't let feet sink 
    depth = np.clip((ground_z - contact_tolerance) - heights, 0, None)
    sink_penalty = depth.mean()

    return touch_score - sink_penalty
