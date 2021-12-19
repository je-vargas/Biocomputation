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

P = 400 #200 #600
N = 20
G = 120
# G = 1000 #chinesGuy

# CrossOver Rate ? might be useful to
GMIN = 100
GMAX = -100
STEP = 4
# LOW_MUTATION = 0.0015
# HIGH_MUTATION = 0.003
# LOW_MUTATION = 0.00275
# HIGH_MUTATION = 0.004
MUTATION = 0.046
# MUTATION = 0.0015
# MUTATION = 0.02
# MUTATION = 0.0010
# MUTATION = 0.00275
# MUTATION = 0.0025 #^
# MUTATION = 0.0010 #^
# MUTATION = 0.0040 # ^


# --------- FITNESS FUNCTIONS

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
        # newind.fitness = fit.rastrigin_fitness_function(newind.gene) #TODO UPDATE DEPENDING ON FITNESS FUNCT USED -> line 82
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

def mutation_adaptive(offspring):
    gene = []
    #!don't need offspringFitness as this is list woth all fitness
    offspringFitness, meanFitness = population_fitness(offspring) 
    
    # --- MUTATION
    for i in range(0, P):
        newind = individual()
        for j in range(0, N):
            gene.append(offspring[i].gene[j])

        if offspring[i].fitness > meanFitness:
            gene = copy.deepcopy(high_mutation(gene))
        else:
            gene = copy.deepcopy(low_mutation(gene))
            
        newind.gene = copy.deepcopy(gene)
        gene.clear()
        offspring[i] = copy.deepcopy(newind)

    return offspring

def high_mutation(gene):
    for i in range(len(gene)): 
        mutprob = random.random()
        if mutprob < HIGH_MUTATION :
            alter = random.uniform(0, STEP)
            if random.randint(0, 1) :
                gene[i] = gene[i] + alter
                if gene[i] > GMAX: gene[i] = GMAX
            else :
                gene[i] = gene[i] - alter
                if gene[i] < GMIN : gene[i] = GMIN
    return gene

def low_mutation(gene):
    for i in range(len(gene)): 
        mutprob = random.random()
        if mutprob < LOW_MUTATION :
            alter = random.uniform(0, STEP)
            if random.randint(0, 1) :
                gene[i] = gene[i] + alter
                if gene[i] > GMAX: gene[i] = GMAX
            else :
                gene[i] = gene[i] - alter
                if gene[i] < GMIN : gene[i] = GMIN
    return gene

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
        off_mutation = copy.deepcopy(rosenbrock_fitness_function(off_mutation))
        population = utility(population, off_mutation)
        
        offspring.clear()

        pop_fitness, meanFitness = copy.deepcopy(population_fitness(population))
        minFitness = min(pop_fitness)
        

        plotBest.append(minFitness)
        plotPopulationMean.append(meanFitness)
    
    return plotBest, plotPopulationMean

    # ---------- Plot ----------

run_5_best = [] 
run_5_mean = []
for i in range(10):
    population = seed_pop()
    population = copy.deepcopy(rosenbrock_fitness_function(population))
    popBest, popMean = run(population)
    population.clear()

    run_5_mean.append(popBest[-1])
    print(f"{popBest[-1]}")
    run_5_best.append(popBest)

average = sum(run_5_mean)/10
print(f"AVERAGE : {average}")

plt.title("Rosenbrock - Mut: 0.046 & Step: 4\nSingle PcrossOver, Urandom mutation")
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.plot(popMean, label = "popAverage")
plt.plot(run_5_best[0], label = "bestIndividual_r1")
plt.plot(run_5_best[1], label = "bestIndividual_r2")
plt.plot(run_5_best[2], label = "bestIndividual_r3")
plt.plot(run_5_best[3], label = "bestIndividual_r4")
plt.plot(run_5_best[4], label = "bestIndividual_r5")
plt.plot(run_5_best[5], label = "bestIndividual_r6")
plt.plot(run_5_best[6], label = "bestIndividual_r7")
plt.plot(run_5_best[7], label = "bestIndividual_r8")
plt.plot(run_5_best[8], label = "bestIndividual_r9")
plt.plot(run_5_best[9], label = "bestIndividual_r10")
plt.legend(loc="upper right")
plt.show()