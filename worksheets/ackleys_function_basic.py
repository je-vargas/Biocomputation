import matplotlib.pyplot as plt
import statistics
import numpy as np
import random
import copy

class individual:
    def __init__(self):
        self.gene = []
        self.fitness = 0
        self.relativeFitness = 0

    def __str__ (self):
        return f"Genes:\n{self.gene}\nFitness: {self.fitness}\t| RelativeFitness: {self.relativeFitness}\n"

P = 200
N = 20
G = 200

MUTATION = 0.0025
GMIN = -32
GMAX = 32
STEP = 5.5

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

def mutation(offspring):
    # --- MUTATION
    for i in range(0, P):
        newind = individual()
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTATION :
                alter = random.uniform(0, STEP)
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
    
def run(population):
    plotPopulationMean = []
    plotBest = []

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = recombination(offspring)
        off_mutation = mutation(off_combined)
        off_mutation = copy.deepcopy(ackleys_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness, meanFitness = copy.deepcopy(population_fitness(population))
        minFitness = min(pop_fitness)
        
        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

# #? -------------------- TABLE PLOT --------------------

# table_range = [
#     # [0.0075, 0.0085, 0.009, 0.01, 0.015, 0.025],
#     [0.035],
#     [1, 2, 3, 4, 5, 6, 7, 8, 9]
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


#? -------------------- AVERAGE RUN WITH PLOTS --------------------
_5_iterations_best_plot = [] 
_5_iteration_popMean_plot = []
iteration_average = []
for i in range(5):
    # popBest, popMean = run(seed_pop(), MUTATION, STEP)
    popBest, popMean = run(seed_pop())

    _5_iterations_best_plot.append(popBest)
    _5_iteration_popMean_plot.append(popMean)
    iteration_average.append(popBest[-1])

    print(f"{popBest[-6:]}")

_5_iteration_best_ind_average = statistics.mean(iteration_average)
print(f"AVERAGE : {_5_iteration_best_ind_average}")

# #? Works out best mean from all 5 runs to plot against
# # plots against the best mean returned from 5 runs 
# # popMean_sum = [sum(beastMean) for beastMean in _5_iteration_popMean_plot]
# # beast_popMean  = min(popMean_sum)
# # _10_iteration_lowest_popMean_index = popMean_sum.index(beast_popMean)



plt.title("Akleys")
plt.xlabel('Generations')
plt.ylabel('Fitness')
# plt.plot(popMean, label = "popAverage")
plt.title("Akleys Mut:0.0025 & Step: 5.5")
plt.plot(_5_iterations_best_plot[0], label = "bestIndividual_r1")
plt.plot(_5_iterations_best_plot[1], label = "bestIndividual_r2")
plt.plot(_5_iterations_best_plot[2], label = "bestIndividual_r3")
plt.plot(_5_iterations_best_plot[3], label = "bestIndividual_r4")
plt.plot(_5_iterations_best_plot[4], label = "bestIndividual_r5")
plt.legend(loc="upper right")
plt.show()


#? -------------------- AVERAGE RUN WITH PLOTS --------------------
# # popBest, popMean = run(seed_pop())

# plt.xlabel('generations')
# plt.ylabel('fitness')
# plt.plot(popMean, label = "popAverage")
# plt.plot(popBest, label = "bestIndividual")
# plt.legend(loc="upper right")
# plt.show()


#! calculating best solution 
# best = [0 for i in range(20)]
# best2 = [1 for i in range(20)]
# print(len(best))
# print("Akle: {0}".format(ackleys_fitness_seeding(best)))
# exit()