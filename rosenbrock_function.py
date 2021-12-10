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
ARITHMETIC_STEP = 0.4
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

def arithmetic_recombination(offspring):
    # --- ARITHMETIC RECOMBINATION  (crossover)
    # --- Remember alpha of 0.5 will prouce twins 
    # Child1 = α.x + (1-α).y
    # Child2 = α.x + (1-α).y

    tempoff1 = individual()
    tempoff2 = individual()
    temp = individual()
    for i in range( 0, P, 2 ):
        tempoff1 = copy.deepcopy(offspring[i])
        tempoff2 = copy.deepcopy(offspring[i+1])
        temp = copy.deepcopy(offspring[i])

        crossprob = random.random() #! use to control how often genes get crossed
        #TODO implement crossover probability

        for j in range (0, N):
            tempoff1.gene[j] = ARITHMETIC_STEP * tempoff1.gene[j] + (1-ARITHMETIC_STEP) * tempoff2[j]
            tempoff2.gene[j] = ARITHMETIC_STEP * tempoff2.gene[j] + (1-ARITHMETIC_STEP) * tempoff1[j]

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

def gaussian_mutation(offspring):
    for i in range(P):
        newind = individual()
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < mut :
                ui = ui(gene)
                step = mut_step()
                gene = gene + np.sqrt(2) * STEP * (GMAX-(-GMIN)) * sp.erfcinv(ui)
            newind.gene.append(gene)
        offspring[i] = copy.deepcopy(newind)
    return offspring

def mut_step():
    return MUTATION/(GMAX-(-GMIN))

def ui(gene):
    ui = random.random()
    if ui >= 0.5: 
        ul = ul_or_ur(gene, "UL")
        return 2 * ul * (1 - 2*ui)
    else:
        ur = ul_or_ur(gene, "UR")
        return 2 * ur * (2*ui - 1)

    raise Exception("No U'i was returned - Something's gone wrong during mutation")


def ul_or_ur(gene, U):
    if U == "UL":  return  0.5*(sp.erf((-GMIN-gene)/(np.sqrt(2)*(GMAX-(-GMIN)*STEP))) + 1)
    elif U == "UR": return 0.5*(sp.erf((GMAX-gene)/(np.sqrt(2)*(GMAX-(-GMIN)*STEP))) + 1)
    else: raise Exception("UL / UR was not calculated - Something's gone wrong during mutation")



def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def gaussian_mutation(offspring, mut, step):
    offspring_fitness = []
    mean = 0
    sdeviation = 0

    for ft in range(P):
        offspring_fitness.append(offspring[ft].fitness)
        
    mean = np.mean(offspring_fitness)
    sdeviation = np.std(offspring_fitness)

    # norm_dist = normal_dist(offspring_fitness, mean, sdeviation)
    # plt.plot(offspring_fitness, norm_dist , color = 'red')
    # plt.xlabel('Data points')   
    # plt.ylabel('Probability Density')

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

def run(population, mut, step):
    plotPopulationMean = []
    plotBest = []

    for generations in range(0, G):

        offspring = selection(population)
        off_combined = recombination(offspring)
        off_mutation = gaussian_mutation(off_combined, mut, step)
        return
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

# table_range = [
#     [0.003, 0.005, 0.0010, 0.0015, 0.0020, 0.0025, 0.00275, 0.003],
#     [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
# ]

# for mut in range(0, len(table_range[0])):
#     print("!! ------------- MUT CHANGE ------------- !!\n")
#     for step in range(0, len(table_range[1])):
        
        # indent to here to generate table



popBest, popMean = run(seed_pop(),MUTATION,STEP)
exit()

iteration_plot_best_individual = []
iteration_plot_popMean = []
iteration_average = []

for i in range(10):

    # popBest, popMean = run(seed_pop(), table_range[0][mut], table_range[1][step])
    popBest, popMean = run(seed_pop(),MUTATION,STEP)
    print(f"{popBest[-6:]}")

    iteration_plot_best_individual.append(popBest)
    iteration_plot_popMean.append(popMean)
    iteration_average.append(popBest[-1])
    
_10_iteration_best_ind_average = sum(iteration_average)/10
iteration_average.clear()

popMean_sum = []
for bestMean in iteration_plot_popMean:
    popMean_sum.append(sum(bestMean))
best_popMean  = min(popMean_sum)
_10_iteration_lowest_popMean_index = popMean_sum.index(best_popMean)
# print("RUN USING: \t|MUT: {0} \t|STEP: {1}".format(table_range[0][mut], table_range[1][step]))
print(f"10 runs using same parameters\nAVERAGE: {_10_iteration_best_ind_average}\n")



# ------------ PLOTTING CODE
        
plt.xlabel('generations')
plt.ylabel('fitness')
plt.plot(iteration_plot_popMean[_10_iteration_lowest_popMean_index], label = "BEAST ITERATION AVERAGE")
plt.plot(iteration_plot_best_individual[0], label = "bestIndividual_r1")
plt.plot(iteration_plot_best_individual[1], label = "bestIndividual_r2")
plt.plot(iteration_plot_best_individual[2], label = "bestIndividual_r3")
plt.plot(iteration_plot_best_individual[3], label = "bestIndividual_r4")
plt.plot(iteration_plot_best_individual[4], label = "bestIndividual_r5")
plt.plot(iteration_plot_best_individual[5], label = "bestIndividual_r1")
plt.plot(iteration_plot_best_individual[6], label = "bestIndividual_r2")
plt.plot(iteration_plot_best_individual[7], label = "bestIndividual_r3")
plt.plot(iteration_plot_best_individual[8], label = "bestIndividual_r4")
plt.plot(iteration_plot_best_individual[9], label = "bestIndividual_r5")
plt.legend(loc="upper right")
plt.show()