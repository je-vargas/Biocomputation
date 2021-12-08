import matplotlib.pyplot as plt
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

P = 200 #600
N = 20
G = 150
# G = 1000 #chinesGuy

# CrossOver Rate ? might be useful to
GMIN = 100
GMAX = -100
STEP = 10
MUTATION = 0.0015
# MUTATION = 0.00275
# MUTATION = 0.0015
# MUTATION = 0.02
# MUTATION = 0.0010
# MUTATION = 0.00275
# MUTATION = 0.0025 #^
# MUTATION = 0.0010 #^
# MUTATION = 0.0040 # ^


# --------- FITNESS FUNCTIONS

def rosenbrock_seeding_fitness(gene):
    fitness = 0
    for j in range(N-1):
        fitness += 100 * pow(gene[j + 1] - gene[j] ** 2, 2) + pow(1 - gene[j], 2)
    return fitness
    

def rosenbrock_fitness_function(population):
    '''assignment function 1'''
    for i in range(0, len(population)):
        fitness = 0
        for j in range(N-1):
            fitness += 100 * pow(population[i].gene[j + 1] - population[i].gene[j] ** 2, 2) + pow(1 - population[i].gene[j], 2)

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
        newind.fitness = copy.deepcopy(rosenbrock_seeding_fitness(newind.gene)) #TODO UPDATE DEPENDING ON FITNESS FUNCT USED -> line 82
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
        # newind.fitness = fit.rastrigin_fitness_function(newind.gene) #TODO UPDATE DEPENDING ON FITNESS FUNCT USED
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

def run(population, mut, step):
    plotPopulationMean = []
    plotBest = []

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = recombination(offspring)
        off_mutation = mutation(off_combined, mut, step)
        off_mutation = copy.deepcopy(rosenbrock_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness = []
        for ind in population:
            pop_fitness.append(ind.fitness)
        minFitness = min(pop_fitness)
        meanFitness = (sum(pop_fitness) / P)

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

    # ---------- Plot ----------

table_range = [
    [0.003, 0.005, 0.0010, 0.0015, 0.0020, 0.0025, 0.00275, 0.003],
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
]

for mut in range(0, len(table_range[0])):
    print("!! ------------- MUT CHANGE ------------- !!\n")
    for step in range(0, len(table_range[1])):

        # run_10_plot = []
        table_mean = []
        
        for i in range(10):

            popBest, popMean = run(seed_pop(), table_range[0][mut], table_range[1][step])

            print(f"{popBest[-6:]}")

            table_mean.append(popBest[-1])
            # run_10_plot.append(popBest)
            
        average = sum(table_mean)/10
        table_mean.clear()
        print("RUN USING: \t|MUT: {0} \t|STEP: {1}".format(table_range[0][mut], table_range[1][step]))
        print(f"AVERAGE: {average}\n")
        
plt.xlabel('generations')
plt.ylabel('fitness')
# plt.plot(popMean, label = "popAverage")
plt.plot(run_5_best[0], label = "bestIndividual_r1")
plt.plot(run_5_best[1], label = "bestIndividual_r2")
plt.plot(run_5_best[2], label = "bestIndividual_r3")
plt.plot(run_5_best[3], label = "bestIndividual_r4")
plt.plot(run_5_best[4], label = "bestIndividual_r5")
plt.legend(loc="upper right")
plt.show()