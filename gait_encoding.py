import random
import numpy as np
from joint_limits import lower, upper

n_genes = 56
frames = 300                     
max_amplitude = 0.6              

joint_phase_shift = np.tile([0.0, np.pi / 2, np.pi / 2], 8)

gene_lower = np.concatenate([lower,                        
                             np.full(24, -max_amplitude),   
                             np.zeros(8)])                
gene_upper = np.concatenate([upper,
                             np.full(24, max_amplitude),
                             np.full(8, 2 * np.pi)])

start_offset_low = np.array([-0.3, -1.3, -1.0] * 8)     # coxa, femur, tibia
start_offset_high = np.array([0.3, -0.7, 0.0] * 8)

def create_gait_chromosome(rng=random):
    """One random chromosome (56 genes)."""
    offsets = [rng.uniform(start_offset_low[i], start_offset_high[i]) for i in range(24)]
    amplitudes = [rng.uniform(-max_amplitude, max_amplitude) for _ in range(24)]
    leg_phases = [rng.uniform(0.0, 2 * np.pi) for _ in range(8)]
    return [float(g) for g in offsets + amplitudes + leg_phases]

def create_gait_population(size, rng=random):
    """A list of `size` random chromosomes."""
    return [create_gait_chromosome(rng) for _ in range(size)]

def decode_chromosome(chromosome, frames=frames):
    """Turn a chromosome (genotype) into a walk (phenotype): frames x 24 angles."""
    c = np.asarray(chromosome, dtype=float)
    offsets = c[0:24]
    amplitudes = c[24:48]
    leg_phases = c[48:56]

    joint_phases = np.repeat(leg_phases, 3) + joint_phase_shift   # 8 -> 24 timings
    t = np.arange(frames).reshape(-1, 1) / frames                 
    return offsets + amplitudes * np.sin(2 * np.pi * t + joint_phases)