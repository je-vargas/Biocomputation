from sympy.ntheory.factor_ import totient
import matplotlib.pyplot as plt
import numpy as np
import statistics
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

P = 200 #600
N = 20
G = 300
# G = 1000 #chinesGuy

CROSSOVER = 1
GMIN = -100
GMAX = 100
# STEP = 30
STEP = 4
# RECOM_STEP = 0.5
# MUTATION = 0.0015
# MUTATION = 0.0
MUTATION = 0.046
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
        newind.fitness = copy.deepcopy(rosenbrock_seeding_fitness(newind.gene))
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
        crosspoint = random.randint(0,N-1)

        for j in range (crosspoint, N):
            tempoff1.gene[j] = tempoff2.gene[j]
            tempoff2.gene[j] = temp.gene[j]

        offspring[i] = copy.deepcopy(tempoff1)
        offspring[i+1] = copy.deepcopy(tempoff2)
    return offspring

def arithmetic_recombination(offspring):
    # --- ARITHMETIC RECOMBINATION  (crossover)
    # --- Remember alpha of 0.5 will prouce twins 
    # Child1 = α.x + (1-α).y
    # Child2 = α.x + (1-α).y

    tempoff1 = individual()
    tempoff2 = individual()
    for i in range( 0, P, 2 ):
        tempoff1 = copy.deepcopy(offspring[i])
        tempoff2 = copy.deepcopy(offspring[i+1])

        # crosspoint = random.randint(0,N-1)
        random_weight = np.random.rand()

        for j in range (0, N):

            # tempoff1.gene[j] = RECOM_STEP * tempoff1.gene[j] + (1-RECOM_STEP) * tempoff2.gene[j]
            # tempoff2.gene[j] = RECOM_STEP * tempoff2.gene[j] + (1-RECOM_STEP) * tempoff1.gene[j]

            tempoff1.gene[j] = random_weight * tempoff1.gene[j] + (1-random_weight) * tempoff2.gene[j]
            tempoff2.gene[j] = random_weight * tempoff2.gene[j] + (1-random_weight) * tempoff1.gene[j]

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

def gaussian_mutation_adaptive(offspring, mut, step, tot_mut, succ_mut):
    # sigma = STEP/(GMAX-GMIN)

    for i in range(0, P):
        newind = individual()
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < mut :
                alter = random.gauss(0, step)
                if alter != 0: tot_mut += 1
                gene = gene + alter
            newind.gene.append(gene)

        off_mut_fitness = rosenbrock_seeding_fitness(newind.gene)
        off_fitness = rosenbrock_seeding_fitness(offspring[i].gene)

        if off_mut_fitness > off_fitness: succ_mut+= 1

        offspring[i] = copy.deepcopy(newind)
    return offspring, tot_mut, succ_mut

def calculate_mutation(muteRate, K):
    g = 0.85   # Test
    # number of generations between changes of mutation rate
    if K < (1/5):
        muteRate = muteRate * g
        print(f"{K} is LESS than {1/5}\tMut: {muteRate}")
        K = 1
    elif K > (1/5):
        muteRate = muteRate / g
        print(f"{K} is GREATER than {1/5}\tMut: {muteRate}")
        K = 1
    elif K == (1/5):
        K += K
        print(f"{K} is EQUAL to {1/5}\tMut: {muteRate}")
    return K, muteRate


def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


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
        off_combined = arithmetic_recombination(offspring)
        off_mutation = mutation(off_combined, mut, step)
        off_mutation = copy.deepcopy(rosenbrock_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness = [ind.fitness for ind in population]
        minFitness = min(pop_fitness)
        meanFitness = (sum(pop_fitness) / P)

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

def run_gaussian(population, mut, step):
    plotPopulationMean = []
    plotBest = []

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = arithmetic_recombination(offspring)
        off_mutation = gaussian_mutation(off_combined, mut, step)
        off_mutation = copy.deepcopy(rosenbrock_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness = [ind.fitness for ind in population]
        minFitness = min(pop_fitness)
        meanFitness = (sum(pop_fitness) / P)

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

def run_gaussian_adaptive(population, mut, step):
    plotPopulationMean = []
    plotBest = []
    k = 0 #1 should start K at 1 based on book
    tot_mut = 0
    succ_mut = 0

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = arithmetic_recombination(offspring)
        off_mutation, tot_mut, succ_mut = gaussian_mutation_adaptive(offspring, mut, step, tot_mut, succ_mut)

        off_mutation = copy.deepcopy(rosenbrock_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness = [ind.fitness for ind in population]
        minFitness = min(pop_fitness)
        meanFitness = (sum(pop_fitness) / P)

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)

        k = succ_mut / tot_mut
        print(f"succ_mut: {succ_mut}, tot_mut: {tot_mut}")
        k, mut = calculate_mutation(mut, k)
        
        tot_mut = 0
        succ_mut = 0
    
    return plotBest, plotPopulationMean


#? ------------ TEST RUN START FROM HERE

iteration_plot_best_individual = []
iteration_plot_popMean = []
iteration_average = []

for i in range(2):

    popBest, popMean = run_gaussian_adaptive(seed_pop(), MUTATION, STEP)
    # popBest, popMean = run(seed_pop())
    print(f"{popBest[-6:]}")

    iteration_plot_best_individual.append(popBest)
    iteration_plot_popMean.append(popMean)
    iteration_average.append(popBest[-1])
    
_10_iteration_best_ind_average = statistics.mean(iteration_average)

popMean_sum = [sum(beastMean) for beastMean in iteration_plot_popMean]
beast_popMean  = min(popMean_sum)
_10_iteration_lowest_popMean_index = popMean_sum.index(beast_popMean)

#? ------------ PLOTTING CODE
        
plt.xlabel('generations')
plt.ylabel('fitness')
plt.plot(iteration_plot_popMean[_10_iteration_lowest_popMean_index], label = "BEAST ITERATION AVERAGE")
plt.plot(iteration_plot_best_individual[0], label = "bestIndividual_r1")
# plt.plot(iteration_plot_best_individual[1], label = "bestIndividual_r2")
# plt.plot(iteration_plot_best_individual[2], label = "bestIndividual_r3")
# plt.plot(iteration_plot_best_individual[3], label = "bestIndividual_r4")
# plt.plot(iteration_plot_best_individual[4], label = "bestIndividual_r5")
# plt.plot(iteration_plot_best_individual[5], label = "bestIndividual_r1")
# plt.plot(iteration_plot_best_individual[6], label = "bestIndividual_r2")
# plt.plot(iteration_plot_best_individual[7], label = "bestIndividual_r3")
# plt.plot(iteration_plot_best_individual[8], label = "bestIndividual_r4")
# plt.plot(iteration_plot_best_individual[9], label = "bestIndividual_r5")
plt.legend(loc="upper right")
plt.show()