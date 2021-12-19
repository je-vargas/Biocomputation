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

MUTATION = 0.025
GMIN = -32
GMAX = 32
STEP = 2.5

# --------- FITNESS FUNCTIONS


def ackleys_fitness_seeding(gene):
    fitness = 0
    firstSum = 0
    secondSum = 0
    for j in range(N):
        firstSum += gene[j]**2
        secondSum += np.cos(2*np.pi*gene[j])
        fitness += -20.0*np.exp((-0.2*np.sqrt(firstSum/N)) - np.exp(secondSum/N))
    return fitness

def ackleys_fitness_function(population):


    for i in range(0, len(population)):
        fitness = 0
        firstSum = 0
        secondSum = 0
        for j in range(N):
            firstSum += population[i].gene[j]**2
            secondSum += np.cos(2*np.pi*population[i].gene[j])
            fitness += -20.0 * np.exp(-0.2 * np.sqrt(firstSum/N)) - np.exp(secondSum/N)
    
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

def arithmetic_recombination(offspring):
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


def mutation(offspring):
    # --- MUTATION
    for i in range(0, P):
        newind = individual()
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTATION :
                alter = random.uniform(0, STEP)
                # print(f"gene alter by: {alter}\n")
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


def utility(population, offspring):
    # --- SORT POPULATION / OFFSPRING --> AND PERSIST BEST INDIVIDUAL
    population.sort(key=lambda ind: ind.fitness, reverse = True)
    popBest = population[-1] #! MAXIMATATION = 0, MINIMATATION = -1

    newPopulation = copy.deepcopy(offspring)
    newPopulation.sort(key=lambda ind: ind.fitness)
    newPopulation[-1] = popBest #! MAXIMATATION = 0, MINIMASATION = -1
    
    return newPopulation

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

        pop_fitness = []
        for ind in population:
            pop_fitness.append(ind.fitness)
        minFitness = min(pop_fitness)
        meanFitness = (sum(pop_fitness) / P)

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

def run_gau_arithmetic(population):
    plotPopulationMean = []
    plotBest = []

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = arithmetic_recombination(offspring)
        off_mutation = gaussian_mutation(off_combined, MUTATION, STEP)
        off_mutation = copy.deepcopy(ackleys_fitness_function(off_mutation))
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

# popBest, popMean = run(seed_pop())
popBest, popMean = run_gau_arithmetic(seed_pop())

plt.xlabel('generations')
plt.ylabel('fitness')
plt.plot(popMean, label = "popAverage")
plt.plot(popBest, label = "bestIndividual")
plt.legend(loc="upper right")
plt.show()