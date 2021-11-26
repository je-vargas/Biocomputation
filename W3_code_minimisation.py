import GA_Classes as gaClass
import GA_Methods as gaMethods
import GA_GenesAndFitness as ga
import Printing_Methods as hlp
import random
import copy
import matplotlib.pyplot as plt

# ------------------ selection ------------------

def tournametSelection_min(populationContainer):
    bestParents = list()
    fitness = 0
    for i in range(0, len(populationContainer)):
        randA = ga.random_int_generator(0, len(populationContainer))
        randB = ga.random_int_generator(0, len(populationContainer))
        parentA = populationContainer[randA]
        parentB = populationContainer[randB]

        if(parentA.fitness < parentB.fitness): # tweaked for minimisation
            bestParents.append(parentA)
        else:
            bestParents.append(parentB)

    for p in bestParents:
        fitness += p.fitness
    print(f"Tornament selection fitness: {fitness}")
    return bestParents

def tournametSelection_max(populationContainer):
    bestParents = list()
    fitness = 0
    for i in range(0, len(populationContainer)):
        randA = ga.random_int_generator(0, len(populationContainer))
        randB = ga.random_int_generator(0, len(populationContainer))
        parentA = populationContainer[randA]
        parentB = populationContainer[randB]

        if(parentA.fitness > parentB.fitness): # tweaked for minimisation
            bestParents.append(parentA)
        else:
            bestParents.append(parentB)

    for p in bestParents:
        fitness += p.fitness
    print(f"Tornament selection fitness: {fitness}")
    return bestParents

# ------------------ crossover ------------------

def single_point_cross_over(parents, geneSize):

    crossoverOffsprings = list()
    crossoverFitness = 0

    toff1 = gaClass.individual()
    toff2 = gaClass.individual()
    temp = gaClass.individual()

    for i in range( 0, len(parents), 2 ):
        toff1 = copy.deepcopy(parents[i])
        toff2 = copy.deepcopy(parents[i+1])
        temp = copy.deepcopy(parents[i])
        crosspoint = random.randint(1,geneSize)

        for j in range (crosspoint, geneSize):
            toff1.genes[j] = toff2.genes[j]
            toff2.genes[j] = temp.genes[j]
            
        parents[i] = copy.deepcopy(toff1)
        parents[i+1] = copy.deepcopy(toff2)
    
        #resetting parents fitness
        parents[i].fitness = ga.candidate_fitness_binarySum(parents[i].genes)
        parents[i+1].fitness = ga.candidate_fitness_binarySum(parents[i+1].genes)

    crossoverOffsprings.append(parents[i])
    crossoverOffsprings.append(parents[i+1])

    fitness = 0
    for p in crossoverOffsprings:
        fitness += p.fitness
    print(f"Crossover Fitness: {fitness}")

    return crossoverOffsprings

# ------------------ mutation ------------------

def mutation_on_binary(xssOffsprigs , mutationRate):
    mutatedOffspring = list()
    for candidate in xssOffsprigs:
        for i in range(0, len(candidate.genes)):
            if random.random() < mutationRate:
                if candidate.genes[i] == 1:
                    candidate.genes[i] = 0
                else:
                    candidate.genes[i] = 1
        candidate.fitness = ga.candidate_fitness_binarySum(candidate.genes)
        mutatedOffspring.append(candidate)

    fitness = 0
    for p in mutatedOffspring:
        fitness += p.fitness
    print(f"Mutation Fitness: {fitness}\n")
    
    return mutatedOffspring

def mutation_on_realNumbers(xssOffsprigs , mutationRate, rangeMin, rangeMax):
    mutatedOffsprings = list()
    for candidate in xssOffsprigs:
        for i in range(0, len(candidate.genes)):
            if (random.randint(0, 1) % 2):
                mutationStep = random.uniform(rangeMin, rangeMax)
                candidate.genes[i] += mutationStep
                if candidate.genes[i] > rangeMax:
                    candidate.genes[i] = rangeMax
            else:
                mutationStep = random.uniform(rangeMin, rangeMax)
                candidate.genes[i] -= mutationStep
                if candidate.genes[i] < rangeMin:
                    candidate.genes[i] = rangeMin
        mutatedOffsprings.append(candidate)
    return mutatedOffsprings

# ------------------ survivor selection ------------------

def survivor_selection_minimisation(offspringsList, currentPopContainer):

    offSpringMostFit = sorted(offspringsList, key=lambda x: x.fitness, reverse = True)
    bestParent = sorted(currentPopContainer, key=lambda x: x.fitness)

    # print("offspring to replace:") 
    # print(hlp.print_pop_container(offSpringMostFit))
    # print("best parent:")
    # print(hlp.print_pop_container(bestParent))
    
    print(f"MIN -- Offspring worst: {offSpringMostFit[0]}")
    print(f"MIN -- current best: {bestParent[0]}")
    offSpringMostFit[0] = bestParent[0]

    fitness = 0
    for p in offSpringMostFit:
        fitness += p.fitness
    print(f"Survivor Selection Fitness: {fitness}")
    
    return offSpringMostFit, bestParent[0]

def survivor_selection_maximatation(offspringsList, currentPopContainer):

    offSpringMostFit = sorted(offspringsList, key=lambda x: x.fitness)
    bestParent = sorted(currentPopContainer, key=lambda x: x.fitness, reverse=True)

    # print(f"offspring to replace: {offSpringMostFit}\nbest parent: {bestParent}")

    print("----- Persisting best individual from previous current pop -----")
    print(f"MIN -- Offspring worst: {offSpringMostFit[0]}")
    print(f"MIN -- current best: {bestParent[0]}")
    
    offSpringMostFit[0] = bestParent[0]

    fitness = 0
    for p in offSpringMostFit:
        fitness += p.fitness
    print(f"Survivor Selection Fitness: {fitness}")

    return offSpringMostFit, bestParent[0]


# ------------------ initiation ------------------
def init_pop(N, P):
    currentPop = gaClass.population()
    # currentPop.container = gaMethods.create_population_realNumbers(N, GENEMIN, GENEMAX, P)
    currentPop.container = gaMethods.create_population_binary(N, P)
    currentPop.fitness = ga.population_fitness(currentPop.container)
    return currentPop

def runs(nRuns, startPop):
    return 0


# ---------------------------- MAIN -----------> 
N = 50
P = 50
MUTRATE = 0
GENEMIN = -5.12
GENEMAX = 5.12



popMeanAverage = list()
bestGenerationIndividual = list()

pop = init_pop(N, P) # ----- WORKING

for genRun in range(0, 100):

    parentsSelection = tournametSelection_max(pop.container) # ----- WORKING
    combination = single_point_cross_over(parentsSelection, N) # ----- WORKING

    # mutation = mutation_on_realNumbers(combination, MUTRATE, GENEMIN, GENEMAX)
    # individualFitnessesCalculated = ga.offspring_fitness_cosFunc(mutation)
    
    mutation = mutation_on_binary(combination, MUTRATE) # ----- WORKING
    individualFitnessesCalculated = ga.offspring_fitness_binary(mutation) # ----- WORKING

    # survivorSelection = gaMethods.survivor_selection_maximatation(individualFitnessesCalculated, currentPop.container)
    survivorSelection, bestCandidate = survivor_selection_maximatation(individualFitnessesCalculated, pop.container) # ----- WORKING
    
    offspringPop = gaClass.population()
    offspringPop.container = survivorSelection
    offspringPop.fitness = ga.population_fitness(survivorSelection)

    # print(f"NewGeneration Fitness : {offspringPop.fitness}\nCurrent Pop Fitness: {pop.fitness}")

    popMeanAverage.append(pop.fitness/P)
    bestGenerationIndividual.append(bestCandidate.fitness)
    pop = copy.deepcopy(offspringPop)

    
    # if offspringPopulationFitness < pop.fitness:
    #     print(f"Offspring generation has improved\noffspring fitness: {offspringPopulationFitness}")
    #     pop.container = copy.deepcopy(survivorSelection)
    #     pop.fitness = offspringPopulationFitness


# # ----------- Graph Plotting ------------
plt.xlabel('generations')
plt.ylabel('fitness')
plt.plot(popMeanAverage, label = "popAverage")
plt.plot(bestGenerationIndividual, label = "bestIndividual")
plt.legend(loc="lower right")
plt.show()
