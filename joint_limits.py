import numpy as np

# (lowest, highest) for each joint type
coxa_limits  = (-1.5708, 1.5708)   
femur_limits = (-1.5708, 1.5708)   
tibia_limits = (-2.0, 1.0)        

lower = np.array([coxa_limits[0], femur_limits[0], tibia_limits[0]] * 8)
upper = np.array([coxa_limits[1], femur_limits[1], tibia_limits[1]] * 8)