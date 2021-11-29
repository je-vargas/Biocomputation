import GA_Classes as ga
import GA_Methods as methods
import GA_GenesAndFitness as fit
import Printing_Methods as hlp
import matplotlib.pyplot as plt
import random
import copy


P = 1000
N = 10
# MUTATIONSTEP = 1.0
# MUTATION = 0.15
# MUTATION = 0.02
MUTATION = 0.5
# MUTATION = 0.0
GMIN = -5.12
GMAX = 5.12
# GMIN = 0.0
# GMAX = 1.0

population = []
offspring = []

plotPopulationMean = []
plotBest = []

# start population
for x in range (0, P):
    tempgene=[]
    for y in range (0, N):
        tempgene.append(random.uniform(GMIN, GMAX))
    newind = ga.individual()
    newind.gene = copy.deepcopy(tempgene)
    newind.fitness = fit.candidate_fitness_cosFunc(newind.gene) #TODO UPDATE DEPENDING ON FITNESS FUNCT USED -> line 82
    population.append(newind)

for generations in range(0, 30):

    # --- SELECTION made to offspring array
    for i in range (0, P):
        parent1 = random.randint( 0, P-1 )
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint( 0, P-1 )
        off2 = copy.deepcopy(population[parent2])

        if off1.fitness < off2.fitness: #! FOR MAXIMATATION CHANGE TO >
            offspring.append( off1 )
        else:
            offspring.append( off2 )

    # --- RECOMBINATION  (crossover)
    tempoff1 = ga.individual()
    tempoff2 = ga.individual()
    temp = ga.individual()
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
    
    # --- MUTATION
    for i in range(0, P):
        newind = ga.individual()
        for j in range(0, N):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < MUTATION :
                alter = random.uniform(GMIN, GMAX)
                if random.randint(0, 1) :
                    gene = gene + alter
                    if gene > GMAX: gene = GMAX
                else :
                    gene = gene - alter
                    if gene < GMIN : gene = GMIN
            newind.gene.append(gene)
        newind.fitness = fit.candidate_fitness_cosFunc(newind.gene) #TODO UPDATE DEPENDING ON FITNESS FUNCT USED
        offspring[i] = copy.deepcopy(newind)


    # --- SORT POPULATION / OFFSPRING --> AND PERSIST BEST INDIVIDUAL
    population.sort(key=lambda ind: ind.fitness, reverse = True)
    popBest = population[-1] #! MAXIMATATION = 0, MINIMATATION = -1

    newPopulation = copy.deepcopy(offspring)
    newPopulation.sort(key=lambda ind: ind.fitness)
    newPopulation[-1] = popBest #! MAXIMATATION = 0, MINIMASATION = -1
    offspring.clear()

    # ----- COPY  OFFSPRING over to POPULATION ----- 
    offspringFit = []
    populationFit = []
    for i in range(0, P):
        offspringFit.append(newPopulation[i].fitness)
        populationFit.append(population[i].fitness)

    offFitness = sum(offspringFit)
    popFitness = sum(populationFit)
    
    offspringFit.clear()
    populationFit.clear()

    # if offFitness < popFitness: #! Only copy over offspring if it's better than population
    #     #TODO --> FOR MAXIMATATION >, MINIMASATION <  !We want less fittest pop
    #     population = copy.deepcopy(newPopulation)

    population = copy.deepcopy(newPopulation)

    # --- BEST-FITNESS / MEAN FITNESS
    popFitness = []
    for ind in population:
        popFitness.append(ind.fitness)

    best = min(popFitness) #TODO: UPDATE HERE FOR MAXIMATATION = max(), MINIMATATION = min() 
    popMean = (sum(popFitness)/P)
    plotBest.append(best)
    plotPopulationMean.append(popMean)

    popFitness.clear()

# ---------- Plot ----------
# print(plotBest)
print(f"popMean: \n{plotPopulationMean}")


plt.xlabel('generations')
plt.ylabel('fitness')
plt.plot(plotPopulationMean, label = "popAverage")
plt.plot(plotBest, label = "bestIndividual")
plt.legend(loc="upper right")
plt.show()
