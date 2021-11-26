import GA_Classes as ga
import GA_Methods as methods
import GA_GenesAndFitness as fit
import Printing_Methods as hlp
import matplotlib.pyplot as plt
import random
import copy


P = 6
N = 50
# MUTATION = 0.15
MUTATION = 0.02
# MUTATION = 0

population = []
offspring = []

plotPopulationMean = []
plotBest = []

# start population
for x in range (0, P):
    tempgene=[]
    for y in range (0, N):
        tempgene.append(random.randint(0,1))
    newind = ga.individual()
    newind.gene = copy.deepcopy(tempgene)
    newind.fitness = fit.candidate_fitness_binarySum(newind.gene)
    population.append(newind)

for generations in range(0, 50):

    # --- SELECTION made to offspring array
    for i in range (0, P):
        parent1 = random.randint( 0, P-1 )
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint( 0, P-1 )
        off2 = copy.deepcopy(population[parent2])
        if off1.fitness > off2.fitness:
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
    offspring = copy.deepcopy(methods.mutation(offspring, MUTATION))

    # --- SORT POPULATION / OFFSPRING --> AND PERSIST BEST INDIVIDUAL
    population.sort(key=lambda ind: ind.fitness, reverse = True)
    popBest = population[0]

    newPopulation = copy.deepcopy(offspring)
    newPopulation.sort(key=lambda ind: ind.fitness)
    newPopulation[0] = popBest
    offspring.clear()

    # ----- COPY  OFFSPRING over to POPULATION ----- 
    offspringFit = []
    populationFit = []
    for i in range(0, P):
        offspringFit.append(newPopulation[i].fitness)
        populationFit.append(population[i].fitness)

    offFitness = sum(offspringFit)
    popFitness = sum(populationFit)

    if offFitness > popFitness:
        population = copy.deepcopy(newPopulation)

    # --- BEST-FITNESS / MEAN FITNESS
    popFitness = []
    for ind in population:
        popFitness.append(ind.fitness)

    best = max(popFitness)
    popMean = (sum(popFitness)/P)
    plotBest.append(best)
    plotPopulationMean.append(popMean)

    popFitness.clear()

# ---------- Plot ----------
print(plotBest)
print(f"popMean: \n{plotPopulationMean}")


plt.xlabel('generations')
plt.ylabel('fitness')
plt.plot(plotPopulationMean, label = "popAverage")
plt.plot(plotBest, label = "bestIndividual")
plt.legend(loc="lower right")
plt.show()

