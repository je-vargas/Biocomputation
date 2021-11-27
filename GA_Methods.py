import GA_Classes as gaClass
import GA_GenesAndFitness as ga
import Printing_Methods as hlp
import random
import copy
import matplotlib.pyplot as plt
import numpy as np

# ------------------ mutation ------------------

def mutation(offspring, mutation):
    P = len(offspring)
    N = len(offspring[0].gene)
    for i in range( 0, P ):
        newind = gaClass.individual();
        newind.gene = []
        for j in range( 0, N ):
            gene = offspring[i].gene[j]
            mutprob = random.random()
            if mutprob < mutation:
                if( gene == 1):
                    gene = 0
                else:
                    gene = 1
            newind.gene.append(gene)
        newind.fitness = ga.candidate_fitness_binarySum(newind.gene)
        offspring[i] = copy.deepcopy(newind)
        
    return offspring

    def mutation_floats():
        for i in range(0, P):
            for j in range(0, N):
                if rand() < MUTRATE :
                    alter = random.uniform(GMIN, GMAX)
                    if random.randint(0, 1) :
                        offspring[i].gene[j] = copy.deepcopy(offspring[i].gene[j]+alter);
                        if offspring[i].gene[j] > GMAX: offspring[i].gene[j] = 1.0;
                    else :
                        offspring[i].gene[j] = offspring[i].gene[j]-alter;
                        if offspring[i].gene[j] < GMIN : offspring[i].gene[j] = 0.0;



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

# ------------------ --------- ------------------

def create_population_realNumbers(geneLength, geneRangeStart, geneRangeFinish ,populationSize):
    population = []
    for individual in range(0, populationSize):
        newCandidate = gaClass.individual()
        newCandidate.genes = ga.create_genes_realNumbers(geneLength, geneRangeStart, geneRangeFinish)
        newCandidate.fitness = ga.candidate_fitness_cosFunc(newCandidate.genes)
        population.append(newCandidate)
    return population

def create_population_binary(geneLength, populationSize):
    population = []
    for individual in range(0, populationSize):
        newCandidate = gaClass.individual()
        newCandidate.genes = ga.create_genes_binary(geneLength)
        newCandidate.fitness = ga.candidate_fitness_binarySum(newCandidate.genes)
        population.append(newCandidate)
    return population

def build_mating_pool(currentPopulation):
    matingPool = list()
    for currentIndividual in currentPopulation.container:
        for ind in range(0, currentIndividual.relative_fitness):
            matingPool.append(currentIndividual)
    return matingPool

def selected_population_returned_tournament(currentPopulation):

    popSize = currentPopulation.size
    newOffSpringPopulation = population(popSize)

    for i in range(0, popSize):

        parent1Index = random_number_geenerator(0, popSize)
        parent2Index = random_number_geenerator(0, popSize)

        parent1 = currentPopulation.container[parent1Index]
        offSpring1 = parent1

        parent2 = currentPopulation.container[parent2Index]
        offSpring2 = parent2

        if offSpring1.fitness > offSpring2.fitness:
            newOffSpringPopulation.container.append(offSpring1)
        else:
            newOffSpringPopulation.container.append(offSpring2)

    newOffSpringPopulation.population_fitness()

    return newOffSpringPopulation

def best_generation_individual(currentPopulation):
    bestIndividualIndex = 0
    best = currentPopulation[0].fitness
    
    
    for individual in range (0, len(currentPopulation)):
        if currentPopulation[individual].fitness > best:
            best = currentPopulation[individual].fitness
            bestIndividualIndex = individual

    return bestIndividualIndex

def worst_generation_individual(currentPopulation):
    worst = 1000
    indexOfWorstIndividual = 0
    worstIndividual = None

    worst = currentPopulation[0].fitness
    worstIndividual = currentPopulation[0]
    
    for individual in range (0, len(currentPopulation)):            

        if currentPopulation[individual].fitness < worst: 
            worst = currentPopulation[individual].fitness
            worstIndividual = currentPopulation[individual]
            indexOfWorstIndividual = individual

    return worstIndividual

def crossover_and_mutation(matingPool, genesLength, mutationRate, mutationStepMin, mutationStepMax, realNumFitness):
    "Defintion: Returns selected population object (new) with crossover and mutation "
    nextParent = 0
    matingPoolSize = len(matingPool)
    offSpringPopulation = population(matingPoolSize)

    for parent in range(0, matingPoolSize, 2):
        nextParent = parent + 1

        if nextParent > matingPoolSize-1:
            print("Next parent index is out of range {0}".format(nextParent))
            print("size of array : {0}".format(matingPoolSize))
            return
        else:
            # print("parents index {0} and {1}".format(i, nextParent))
            parentA = matingPool[parent]
            parentB = matingPool[nextParent]

            # mutate and cross over
            crossOverPoint = random_number_geenerator(0, genesLength)

            offSpringPopulation.crossOverPoint.append(crossOverPoint)
            offSpringPopulation.crossOverPoint.append(crossOverPoint)

            newChildren = cross_over(crossOverPoint, parentA, parentB)
            newChildren = mutation(newChildren, mutationRate, mutationStepMin, mutationStepMax)

            chA = newChildren['childA']
            chB = newChildren['childB']
            
            chA.candidate_fitness(realNumFitness)
            chB.candidate_fitness(realNumFitness)

            if chA == None or chB == None:
                print("Something has gone wrong during selection: line 232")

            offSpringPopulation.container.append(chA)
            offSpringPopulation.container.append(chB)

    return offSpringPopulation

def generations_run(generations, currentPopulation, genesLength, geneMin, geneMax, mutationRate, realNumFitness):
    populationAverageFitnessPlot = list()
    generationBestCandidate = list()
    
    currentPopulation.population_fitness() 

    for generation in range(0, generations):

        #select parents based on their fitness
        tournmanetSelect = tournmanetSelection(currentPopulation.container, genesLength)

        newPopulation = crossover_and_mutation(tournmanetSelect, genesLength, mutationRate, geneMin, geneMax, realNumFitness)
        
        newPopulation.population_fitness()

        populationAverageFitness = (currentPopulation.fitness / currentPopulation.size)

        worstIndividual = worst_generation_individual(currentPopulation.container) #// once selecction and reproduction 

        # helpful_print(generation, newPopulation, currentPopulation)

        if newPopulation.fitness < currentPopulation.fitness:

            currentPopulation = newPopulation # Can check here for best overall before copying over

        bestIndividualIndex = best_generation_individual(currentPopulation.container)

        # print("worstIndex: {0}\t| bestIndividual: {1}".format(worstIndividualIndex, bestIndividual.fitness))

        currentPopulation.container[bestIndividualIndex] = worstIndividual

        # copy local best ind over current pop worst 

        yPlotPopulationFitnessAverage.append(populationAverageFitness)
        yPlotBestindividual.append(worstIndividual.fitness)

        lastGeneration['population'] = currentPopulation
        lastGeneration['yPlot'] = yPlotPopulationFitnessAverage
        lastGeneration['yPlotBestIndividual'] = yPlotBestindividual

    return populationAverageFitnessPlot, generationBestCandidate