import numpy as np
from spider_kinematics import get_foot_positions
from joint_limits import lower, upper

ground_z = -0.5            # where the ground is (body is at z = 0)
contact_tolerance = 0.1    

def get_gait_feet(gait):
    gait = np.asarray(gait, dtype=float)
    n_frames = gait.shape[0]                 # how many poses 
    feet = np.zeros((n_frames, 8, 3))       
    for f in range(n_frames):
        feet[f] = get_foot_positions(gait[f])   # all feet for this pose
    return feet

def score_ground_contact(feet):
    heights = feet[:, :, 2]             
    lowest = heights.min(axis=0)             
    gap = np.abs(lowest - ground_z)      
    touch = np.clip(1 - gap / 0.5, 0, 1)     # gap 0 -> 1, gap 0.5+ -> 0
    touch_score = touch.mean()               # average over the 8 legs

    # don't let feet sink 
    depth = np.clip((ground_z - contact_tolerance) - heights, 0, None)
    sink_penalty = depth.mean()

    return touch_score - sink_penalty

def score_stability(feet):
    heights = feet[:, :, 2]                                  
    on_ground = heights <= ground_z + contact_tolerance      

    # Leg order is L1 L2 L3 L4 R4 R3 R2 R1
    left_down = on_ground[:, 0:4].sum(axis=1)     # left feet down
    right_down = on_ground[:, 4:8].sum(axis=1)    # right feet down

    stable = (left_down >= 2) & (right_down >= 2)           
    return stable.mean()                          # fraction of stable frames


def score_forward(feet):
    heights = feet[:, :, 2]                              
    xs = feet[:, :, 0]                                      
    on_ground = heights <= ground_z + contact_tolerance

    next_x = np.roll(xs, -1, axis=0)
    next_on = np.roll(on_ground, -1, axis=0)
    dx = next_x - xs                                       

    planted = on_ground & next_on                            
    push = np.where(planted, -dx, 0.0)                   

    feet_down = planted.sum(axis=1)                       
    body_move = np.where(feet_down > 0,
                         push.sum(axis=1) / np.maximum(feet_down, 1),
                         0.0)                                # average push per frame
    return body_move.sum()                                   # total distance over the walk


def penalty_energy(gait):
    gait = np.asarray(gait, dtype=float)
    step = np.roll(gait, -1, axis=0) - gait          # movement from each frame to the next
    return np.abs(step).sum(axis=0).mean()           # total per joint, then average of 24 joints


def penalty_smoothness(gait):
    gait = np.asarray(gait, dtype=float)
    step = np.roll(gait, -1, axis=0) - gait          # speed of each joint each frame
    change = np.roll(step, -1, axis=0) - step        # how much that speed changes
    return np.abs(change).mean() * 100            



def penalty_joint_limits(gait):
    gait = np.asarray(gait, dtype=float)
    too_low = np.clip(lower - gait, 0, None)     # how far below the lower limit 
    too_high = np.clip(gait - upper, 0, None)    # how far above the upper limit 
    return (too_low + too_high).mean()