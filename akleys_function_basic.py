import matplotlib.pyplot as plt
import numpy as np
import random          
import copy
import statistics

class individual:
    def __init__(self):
        self.gene = []
        self.fitness = 0
        self.relativeFitness = 0

    def __str__ (self):
        return f"Genes:\n{self.gene}\nFitness: {self.fitness}\t| RelativeFitness: {self.relativeFitness}\n"

P = 200 #200 #600
N = 20
G = 200

# CrossOver Rate ? might be useful to
GMIN = 32
GMAX = -32
# GMIN = 100
# GMAX = -100
STEP = 5.5
# LOW_MUTATION = 0.0015
# HIGH_MUTATION = 0.003
# LOW_MUTATION = 0.00275
# HIGH_MUTATION = 0.004

# MUTATION = 0.046
MUTATION = 0.0025

# MUTATION = 0.0015
# MUTATION = 0.02
# MUTATION = 0.0010
# MUTATION = 0.00275
# MUTATION = 0.0025 #^
# MUTATION = 0.0010 #^
# MUTATION = 0.0040 # ^


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
        newind.fitness = copy.deepcopy(ackleys_fitness_seeding(newind.gene))
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

def recombination(offspring):
    # --- RECOMBINATION  (crossover)
    tempoff1 = individual()
    tempoff2 = individual()
    temp = individual()
    for i in range( 0, P, 2 ):
        tempoff1 = copy.deepcopy(offspring[i])
        tempoff2 = copy.deepcopy(offspring[i+1])
        temp = copy.deepcopy(offspring[i])
        crosspoint = random.randint(1,N)

        for j in range (crosspoint, N):
            tempoff1.gene[j] = tempoff2.gene[j]
            tempoff2.gene[j] = temp.gene[j]

        offspring[i] = copy.deepcopy(tempoff1)
        offspring[i+1] = copy.deepcopy(tempoff2)
    return offspring

def mutation(offspring, mut, step):
    # --- MUTATION
    for i in range(0, P):
        newind = individual()
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < mut :
                alter = random.uniform(0, step)
                if random.randint(0, 1) :
                    gene = gene + alter
                    if gene > GMAX: gene = GMAX
                else :
                    gene = gene - alter
                    if gene < GMIN : gene = GMIN
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

def population_fitness(population):
    pop_fitness = []
    for ind in population:
        pop_fitness.append(ind.fitness)
    meanFitness = (sum(pop_fitness) / P)
    return pop_fitness, meanFitness
    

def run(population, mut, step):
    plotPopulationMean = []
    plotBest = []

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = recombination(offspring)
        off_mutation = mutation(off_combined, mut, step)
        off_mutation = copy.deepcopy(ackleys_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness, meanFitness = copy.deepcopy(population_fitness(population))
        minFitness = min(pop_fitness)
        

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

#! calculating best solution 
best = [0 for i in range(20)]
# best2 = [1 for i in range(20)]
print("Akle: {0}".format(ackleys_fitness_seeding(best)))

    #? ---------- Plot ----------

# run_5_best = [] 
# run_5_mean = []
# for i in range(5):
#     popBest, popMean = run(seed_pop(), MUTATION, STEP)

#     run_5_mean.append(popBest[-1])
#     run_5_best.append(popBest)
#     print(f"{popBest[-6:]}")

# average = statistics.mean(run_5_mean)
# print(f"AVERAGE : {average}")

# plt.title("Akleys - Mut: 0.0025 & Step: 5.5\nSingle PcrossOver, Uniform random mutation")
# plt.xlabel('Generations')
# plt.ylabel('Fitness')
# plt.plot(popMean, label = "popAverage")
# plt.plot(run_5_best[0], label = "bestIndividual_r1")
# plt.plot(run_5_best[1], label = "bestIndividual_r2")
# plt.plot(run_5_best[2], label = "bestIndividual_r3")
# plt.plot(run_5_best[3], label = "bestIndividual_r4")
# plt.plot(run_5_best[4], label = "bestIndividual_r5")
# plt.legend(loc="upper right")
# plt.show()

# #? -------------------- TABLE PLOT --------------------

# table_range = [
#     [0.0025, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3],
#     [10, 20, 30]
# ]

# for mut in range(0, len(table_range[0])):
#     print("!! ------------- MUT CHANGE ------------- !!\n")
#     for step in range(0, len(table_range[1])):
        
#         #? indent to here to generate table
#         iteration_average = []

#         for i in range(5):

#             popBest, popMean = run(seed_pop(), table_range[0][mut], table_range[1][step])
#             print(f"{popBest[-6:]}")

#             iteration_average.append(popBest[-1])
            
#         _5_iteration_best_ind_average = statistics.mean(iteration_average)
#         iteration_average.clear()

#         print("RUN USING: \t|MUT: {0} \t|STEP: {1}".format(table_range[0][mut], table_range[1][step]))
#         print(f"5 RUN AVERAGE: {_5_iteration_best_ind_average}\n")