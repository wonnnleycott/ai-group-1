import random
from joint_limits import lower, upper


def create_chromosome():

    chromosome = []

    for i in range(24):
        angle = random.uniform(float(lower[i]), float(upper[i]))
        chromosome.append(angle)

    return chromosome


def create_gait():

    gait = []

    for frame in range(300):
        chromosome = create_chromosome()
        gait.append(chromosome)

    return gait


def create_population(population_size):

    population = []

    for i in range(population_size):
        chromosome = create_chromosome()
        population.append(chromosome)

    return population


if __name__ == "__main__":
    population = create_population(50)
    print("First chromosome in population:")
    print(population[0])