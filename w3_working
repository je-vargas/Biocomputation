import random
import matplotlib.pyplot as plt
import copy
import numpy as np


N = 10
P = 1000
MUTATION_RATE = 0.5  # 0.04  # 0.05
MUTATION_STEP = 1.0  # 0.85  # 0.80
MIN = -5.12  # 0.0
MAX = 5.12  # 1.0

class Individual:
    def __init__(self):
        self.gene = [0] * N
        self.fitness = 0


def fitness_function(ind):
    fitness = 0

    for i in range(N):
        fitness += ind.gene[i]

    return fitness


def generate_genes():
    population = []
    # Create random population of genes
    for i in range(0, P):
        temp_gene = []
        for j in range(0, N):
            temp_gene.append(random.uniform(MIN, MAX))
        new_ind = Individual()
        new_ind.gene = temp_gene.copy()
        new_ind.fitness = minimisation(new_ind)
        population.append(new_ind)

    return population


def generate_fitness(population):
    for ind in population:
        # ind.fitness = fitness_function(ind)
        ind.fitness = minimisation(ind)
    return population


# TODO create max/min selection functions
def selection_min(population):
    offspring = []
    # SELECTION (Select Parents - Tournament Selection)
    for i in range(0, P):
        parent_1 = random.randint(0, P - 1)
        off_1 = population[parent_1]
        parent_2 = random.randint(0, P - 1)
        off_2 = population[parent_2]
        if off_1.fitness > off_2.fitness:  # Changed to less than
            offspring.append(off_2)
        else:
            offspring.append(off_1)

    return offspring


def selection_max(population):
    offspring = []
    # SELECTION (Select Parents - Tournament Selection)
    for i in range(0, P):
        parent_1 = random.randint(0, P - 1)
        off_1 = population[parent_1]
        parent_2 = random.randint(0, P - 1)
        off_2 = population[parent_2]
        if off_1.fitness > off_2.fitness:  # Changed to less than
            offspring.append(off_1)
        else:
            offspring.append(off_2)

    return offspring


def crossover(offspring):
    temp_individual = Individual()
    for i in range(0, P, 2):
        for j in range(0, N):
            temp_individual.gene[j] = offspring[i].gene[j]
        crossover_point = random.randint(0, N - 1)
        for j in range(crossover_point, N):
            offspring[i].gene[j] = offspring[i + 1].gene[j]
            offspring[i + 1].gene[j] = temp_individual.gene[j]

    return offspring


def crossover_test(offspring):
    for i in range(0, P, 2):
        off1 = copy.deepcopy(offspring[i])
        off2 = copy.deepcopy(offspring[i + 1])
        temp = copy.deepcopy(offspring[i])
        crossover_point = random.randint(1, N)
        for j in range(crossover_point, N):
            off1.gene[j] = off2.gene[j]
            off2.gene[j] = temp.gene[j]
        off1.fitness = minimisation(off1)
        off2.fitness = minimisation(off2)
        offspring[i] = copy.deepcopy(off1)
        offspring[i + 1] = copy.deepcopy(off2)
    return offspring


def mutation(offspring):
    for i in range(0, P):
        new_individual = Individual()
        new_individual.gene = []
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutation_probability = random.uniform(MIN, MAX)

            if mutation_probability < MUTATION_RATE:
                alter = random.uniform(0, MUTATION_STEP)
                if random.choice([0, 1]) == 1:
                    offspring[i].gene[j] = offspring[i].gene[j] + alter
                    if offspring[i].gene[j] > MAX:
                        offspring[i].gene[j] = MAX
                    else:
                        offspring[i].gene[j] = offspring[i].gene[j] - alter
                        if offspring[i].gene[j] < MIN:
                            offspring[i].gene[j] = MIN

            new_individual.gene.append(gene)
        new_individual.fitness = minimisation(new_individual)  # new
        offspring[i] = new_individual
    return offspring


def fitness_total(population):
    offspring_total_fitness_local = 0
    for individual in population:
        individual.fitness = fitness_function(individual)
        offspring_total_fitness_local += individual.fitness
    return offspring_total_fitness_local


def minimisation(individual) -> float:
    fitness = 10 * N
    for i in range(N):
        fitness += (individual.gene[i] ** 2 - 10 * np.cos(2 * np.pi * individual.gene[i]))
    return fitness


def best_individual(population):
    best = 0
    bestIndividual = None
    for bestInd in population:
        if bestInd.fitness > best:
            best = bestInd.fitness
            bestIndividual = bestInd
    return bestIndividual


def best_population_fitness(population):
    best_fitness = []
    best_fitness.append(max(individual.fitness for individual in population))
    return best_fitness


def worst_individual(population):
    worst = 1000
    worstIndividual = None
    for worstInd in population:
        if worstInd.fitness < worst:
            worst = worstInd.fitness
            worstIndividual = worstInd
    return worstIndividual


def test(population, offspring):
    population.sort(key=lambda individual: individual.fitness, reverse=True)
    bestIndividual = population[-1]  # Changed to -1 (worst) from 0 (best)

    new_population = copy.deepcopy(offspring)

    new_population.sort(key=lambda individual: individual.fitness, reverse=True)
    new_population[0] = bestIndividual  # Changed to 0 (best) from -1 (worst)
    offspring.clear()

    return new_population


# population = generate_fitness()
def iterate(population):
    generations = 30
    best_fitness = []
    mean_fitness_plot = []
    for i in range(generations):
        offspring = selection_min(population)
        offspring_crossover = crossover(offspring)
        offspring_mutated = mutation(offspring_crossover)
        population = test(population, offspring_mutated)
        fitness = []
        for individual in population:
            fitness.append(individual.fitness)
        min_fitness = min(fitness)

        # best_fitness.append(best_population_fitness(population))
        mean_fitness = (sum(fitness) / P)

        best_fitness.append(min_fitness)
        mean_fitness_plot.append(mean_fitness)

    return best_fitness, mean_fitness_plot



best_fitness_data, mean_fitness_data = iterate(generate_genes())
plt.plot(mean_fitness_data)
plt.plot(best_fitness_data)

plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.show()