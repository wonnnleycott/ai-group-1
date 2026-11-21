import random
import numpy as np
from gait_encoding import (create_gait_chromosome, create_gait_population,
                           decode_chromosome, gene_lower, gene_upper, n_genes)

rng = random.Random(1)                     # fixed seed = same results every run
chrom = create_gait_chromosome(rng)
gait = decode_chromosome(chrom)

print("Genes in chromosome:", len(chrom))
print("Walk shape:", gait.shape)
print("All genes inside their limits:",
      bool(np.all((np.array(chrom) >= gene_lower) & (np.array(chrom) <= gene_upper))))
print("Frame 0 differs from frame 75:", not np.allclose(gait[0], gait[75]))
print("Population size:", len(create_gait_population(20, rng)))