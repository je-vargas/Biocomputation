import matplotlib.pyplot as plt
import statistics
import numpy as np
import random
import copy
import math

class individual:
    def __init__(self):
        self.gene = []
        self.fitness = 0
        self.relativeFitness = 0

    def __str__ (self):
        return f"Genes:\n{self.gene}\nFitness: {self.fitness}\t| RelativeFitness: {self.relativeFitness}\n"

P = 600
N = 20
G = 450
GMIN = -32
GMAX = 32

# STEP = 1 #best
# MUTATION = 0.03 #best

# MUTATION = 0.03
# STEP = 3


# --------- FITNESS FUNCTIONS


def ackleys_fitness_seeding(gene):
    fitness = 0
    firstSum = 0
    secondSum = 0

    for j in range(N):
        firstSum += gene[j]**2
        secondSum += np.cos(2*np.pi*gene[j])

    sect_1 = -20 * np.exp(-0.2 * np.sqrt((1/N) * firstSum))
    sect_2 = np.exp((1/N)*secondSum)
    fitness = sect_1 - sect_2
        
    return fitness

def ackleys_fitness_function(population):
    for i in range(0, len(population)):
        fitness = 0
        firstSum = 0
        secondSum = 0

        for j in range(0, N):
            firstSum += population[i].gene[j]**2
            secondSum += np.cos(2*np.pi*population[i].gene[j])
        
        sect_1 = -20 * np.exp(-0.2 * np.sqrt((1/N) * firstSum))
        sect_2 = np.exp((1/N)*secondSum)
        fitness = sect_1 - sect_2
    
        population[i].fitness = copy.deepcopy(fitness)
    return population

# --------- GA METHODS
def seed_pop():
    population = []
    for x in range (0, P):
        tempgene=[]
        for y in range (0, N):
            tempgene.append(random.uniform(GMIN, GMAX))
        newind = individual()
        newind.gene = copy.deepcopy(tempgene)
        newind.fitness = ackleys_fitness_seeding(newind.gene)
        population.append(newind)
    return population

def selection(population):
    # --- SELECTION made to offspring array
    offspring = []
    for i in range (0, P):
        parent1 = random.randint( 0, P-1 )
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint( 0, P-1 )
        off2 = copy.deepcopy(population[parent2])

        if off1.fitness < off2.fitness: #! FOR MAXIMATATION CHANGE TO >
            offspring.append( off1 )
        else:
            offspring.append( off2 )
    return offspring

def arithmetic_recombination(offspring):
    # --- Remember random weight of 0.5 will prouce twins 
    # Child1 = α.x + (1-α).y
    # Child2 = α.x + (1-α).y

    tempoff1 = individual()
    tempoff2 = individual()
    for i in range( 0, P, 2 ):
        tempoff1 = copy.deepcopy(offspring[i])
        tempoff2 = copy.deepcopy(offspring[i+1])

        random_weight = np.random.rand()

        for j in range (0, N):

            tempoff1.gene[j] = random_weight * tempoff1.gene[j] + (1-random_weight) * tempoff2.gene[j]
            tempoff2.gene[j] = random_weight * tempoff2.gene[j] + (1-random_weight) * tempoff1.gene[j]

        offspring[i] = copy.deepcopy(tempoff1)
        offspring[i+1] = copy.deepcopy(tempoff2)
    return offspring

def gaussian_mutation(offspring, mut, step):
    # --- MUTATION
    gauss_mean = [offspring[i].fitness for i in range(P)]
    mu = statistics.mean(gauss_mean)
    # sigma = STEP/(GMAX-GMIN)

    for i in range(0, P):
        newind = individual()
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < mut :
                alter = random.gauss(0, step)
                gene = gene + alter
            newind.gene.append(gene)
        offspring[i] = copy.deepcopy(newind)
    return offspring


def utility(population, offspring):
    # --- SORT POPULATION / OFFSPRING --> AND PERSIST BEST INDIVIDUAL
    population.sort(key=lambda ind: ind.fitness, reverse = True)
    popBest = population[-1] #! MAXIMATATION = 0, MINIMATATION = -1

    newPopulation = copy.deepcopy(offspring)
    newPopulation.sort(key=lambda ind: ind.fitness)
    newPopulation[-1] = popBest #! MAXIMATATION = 0, MINIMASATION = -1
    
    return newPopulation

def run_gau_arithmetic(population, mut, step):
    plotPopulationMean = []
    plotBest = []

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = arithmetic_recombination(offspring)
        off_mutation = gaussian_mutation(off_combined, mut, step)
        off_mutation = copy.deepcopy(ackleys_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness = [ind.fitness for ind in population]
        minFitness = min(pop_fitness)
        meanFitness = (sum(pop_fitness) / P)

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

    # ---------- Plot ----------
#? -------------------- GA WITH AVERAGE --------------------

_5_iterations_best_plot = [] 
_5_iteration_popMean_plot = []
iteration_average = []

for i in range(5):
    popBest, popMean = run_gau_arithmetic(seed_pop(), 0.03, 1)

    _5_iterations_best_plot.append(popBest)
    _5_iteration_popMean_plot.append(popMean)
    iteration_average.append(popBest[-1])

    print(f"{popBest[-6:]}")

_5_iteration_best_ind_average = statistics.mean(iteration_average)

# plots against the best mean returned from 5 runs 
popMean_sum = [sum(beastMean) for beastMean in _5_iteration_popMean_plot]
beast_popMean  = min(popMean_sum)
_10_iteration_lowest_popMean_index = popMean_sum.index(beast_popMean)

print(f"AVERAGE : {_5_iteration_best_ind_average}")

plt.title("Akleys Mut: 0.03 & Step: 1 & Pop: 600")
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.text(150, -15,"Average: {0}".format(_5_iteration_best_ind_average), color="b")
# plt.plot(popMean, label = "popAverage")
plt.plot(_5_iterations_best_plot[0], label = "bestIndividual_r1")
plt.plot(_5_iterations_best_plot[1], label = "bestIndividual_r2")
plt.plot(_5_iterations_best_plot[2], label = "bestIndividual_r3")
plt.plot(_5_iterations_best_plot[3], label = "bestIndividual_r4")
plt.plot(_5_iterations_best_plot[4], label = "bestIndividual_r5")
plt.legend(loc="upper right")
plt.show()

#? -------------------- SINGLE GA No Average --------------------

# popBest, popMean = run_gau_arithmetic(seed_pop())

# plt.xlabel('generations')
# plt.ylabel('fitness')
# plt.plot(popMean, label = "popAverage")
# plt.plot(popBest, label = "bestIndividual")
# plt.legend(loc="upper right")
# plt.show()

#     #? -------------------- TABLE PLOT --------------------

# table_range = [
#     [0.0175, 0.02, 0.0225, 0.0255, 0.0275, 0.03, 0.0325, 0.0355, 0.04],
#     [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
# ]

# for mut in range(0, len(table_range[0])):
#     print("!! ------------- MUT CHANGE ------------- !!\n")
#     for step in range(0, len(table_range[1])):
        
#         #? indent to here to generate table
#         iteration_average = []

#         for i in range(5):

#             popBest, popMean = run_gau_arithmetic(seed_pop(), table_range[0][mut], table_range[1][step])
#             print(f"{popBest[-6:]}")

#             iteration_average.append(popBest[-1])
            
#         _5_iteration_best_ind_average = statistics.mean(iteration_average)
#         iteration_average.clear()

#         print("RUN USING: \t|MUT: {0} \t|STEP: {1}".format(table_range[0][mut], table_range[1][step]))
#         print(f"5 RUN AVERAGE: {_5_iteration_best_ind_average}\n")

# #? --------------------  --------------------