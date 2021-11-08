import random
import matplotlib.pyplot as plt
import numpy as np
# ------------------------- CLASS

class population:
    def __init__(self, size):
        self.container = list()
        self.__fitness = 0
        self.__size = size
        self.crossOverPoint = list()

    @property
    def size(self):
        return self.__size

    @property
    def fitness(self):
        return self.__fitness

    def work_out_population_fitness(self):
        self.__fitness = 0
        for individual in self.container:
            self.__fitness += individual.fitness

class individual_candidate:
    def __init__(self, geneLength):
        self.__genes = list()
        self.__fitness = 0
        self.__relativeFitnessAsPercentage = 0
        self.__genesLength = geneLength

    @property
    def fitness(self):
        return self.__fitness

    @property
    def relative_fitness(self):
        return self.__relativeFitnessAsPercentage

    @property
    def genes(self):
        return self.__genes

    def set_genes(self, newGenes):
        self.__genes = newGenes

    @property
    def gene_length(self):
        return self.__genesLength

    def set_relative_fitness(self, popFitness):
        relativeFitness = (self.__fitness / popFitness) * 100
        self.__relativeFitnessAsPercentage = round(relativeFitness)
        # print("indidivual's relative fitness: {relativeFitness} ".format(fitness = self.__fitness, relativeFitness = self.__relativeFitnessAsPercentage))

    def individuals_fitness(self):
        self.__fitness = 0
        for gene in self.genes:
            self.__fitness += gene

    def create_dna(self):
        for x in range(0, self.__genesLength):
            # random no between 0-1 because GeneticAlgorithm is using a Binary Encoding
            self.__genes.append(random.uniform(0.0, 1.0))

# ------------------------- METHODS

def random_number_geenerator(startingValue, upToValue):
    randomNumber = startingValue
    randomNumber = random.randint(startingValue, upToValue-1)
    return randomNumber

def DNA_bit_printer(individual):
    for bit in individual.genes:
        print(bit)

def population_DNA_printer(populationArray):
    for individual in populationArray:
        print("genes: {0}".format(individual.genes))

def generations_comparison_DNA_printer(newPopulation, oldPopulation):
    "Definition: prints new and old population for comparison"
    for individual in range(0, len(newPopulation.container)):
        print("popIndex: {0} | old pop: {1} | new pop: {2} | cross over: {3}".format(individual, oldPopulation.container[individual].genes, 
        newPopulation.container[individual].genes, newPopulation.crossOverPoint[individual]))
    print("\t\t\t| old pop fitness: {0}\t| new pop fitness: {1}".format(oldPopulation.fitness, newPopulation.fitness))

def create_individuals(geneLength, populationSize):
    # Creating random population of size P
    population = []
    for individual in range(0, populationSize):
        newindividual = individual_candidate(geneLength)
        newindividual.create_dna()
        newindividual.individuals_fitness()
        population.append(newindividual)
    return population

def calculate_relative_fitness_of_individuals(currentPopulation):
    for currentIndividual in currentPopulation.container:
        currentIndividual.set_relative_fitness(currentPopulation.fitness)

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

    newOffSpringPopulation.work_out_population_fitness()

    return newOffSpringPopulation

def cross_over(crossOverPoint, parentAGn, parentBGn):
    geneLength = parentAGn.gene_length
    newOffsprings = dict()

    pAGeneStart = parentAGn.genes[:crossOverPoint]
    pAGeneEnd = parentAGn.genes[crossOverPoint:]

    pBGeneStart = parentBGn.genes[:crossOverPoint]
    pBGeneEnd = parentBGn.genes[crossOverPoint:]

    childA = [*pAGeneStart, *pBGeneEnd]
    childB = [*pBGeneStart, *pAGeneEnd]

    offspring1 = individual_candidate(geneLength)
    offspring2 = individual_candidate(geneLength)

    offspring1.set_genes(childA)
    offspring2.set_genes(childB)

    newOffsprings['childA'] = offspring1
    newOffsprings['childB'] = offspring2

    return newOffsprings

def mutation_step(individualsGene, geneIndex, step):
    newGene = individualsGene
    if (random.randint(0, 1) % 2):
        newGene[geneIndex] += step
        if newGene[geneIndex] > 1.0:
            newGene[geneIndex] = 1.0
    else:
        newGene[geneIndex] -= step
        if newGene[geneIndex] < 0.0:
            newGene[geneIndex] = 0.0
    return newGene

def mutation(newOffsprings, mutationRate):
    childA = newOffsprings['childA']
    childB = newOffsprings['childB']

    childAGenes = childA.genes
    childBGenes = childB.genes


    for geneIndex in range(0, childA.gene_length):
        
        # if random.random() < mutationRate:
        mutationStepA = random.uniform(0.0, 1.0)
        newGenesA = mutation_step(childAGenes, geneIndex, mutationStepA)
            # print("mutation applied chA: {0}".format(childA.genes))

        # if random.random() < mutationRate:
        mutationStepB = random.uniform(0.0, 1.0)
        newGenesB = mutation_step(childBGenes, geneIndex, mutationStepB)
            # print("mutation applied chB: {0}".format(childB.genes))

        childA.set_genes(newGenesA)
        childB.set_genes(newGenesB)

    newOffsprings['childA'] = childA
    newOffsprings['childB'] = childB

    return newOffsprings

def best_generation_individual(currentPopulation):
    best = 0
    bestIndividualIndex = None
    
    for individual in range (0, len(currentPopulation)):
        if currentPopulation[individual].fitness > best:
            best = currentPopulation[individual].fitness
            bestIndividualIndex = individual

    return bestIndividualIndex

def worst_generation_individual(currentPopulation):
    worst = -1
    indexOfWorstIndividual = 0
    worstIndividual = None
    
    for individual in range (0, len(currentPopulation)):
        if individual == 0:
            worst = currentPopulation[individual].fitness
            worstIndividual = currentPopulation[individual]

        if currentPopulation[individual].fitness < worst: 
            worst = currentPopulation[individual].fitness
            worstIndividual = currentPopulation[individual]
            indexOfWorstIndividual = individual

    return worstIndividual

def crossover_and_mutation(matingPool, genesLength, mutationRate):
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
            newChildren = mutation(newChildren, mutationRate)

            chA = newChildren['childA']
            chB = newChildren['childB']
            
            chA.individuals_fitness()
            chB.individuals_fitness()

            if chA == None or chB == None:
                print("Something has gone wrong during selection: line 232")

            offSpringPopulation.container.append(chA)
            offSpringPopulation.container.append(chB)

    return offSpringPopulation

def tournmanetSelection(populationContainer, geneLength):
    selectedParents = list()
    for i in range(0, len(populationContainer)):
        randA = random_number_geenerator(0, len(populationContainer))
        randB = random_number_geenerator(0, len(populationContainer))
        parentA = populationContainer[randA]
        parentB = populationContainer[randB]

        if(parentA.fitness < parentB.fitness):
            selectedParents.append(parentA)
        else:
            selectedParents.append(parentB)
    return selectedParents

def helpful_print(generation, newPop, currentPop):
    print("generation: {0}\n".format(generation))
    # generations_comparison_DNA_printer(newPop, currentPop)
    # print("\n")
    print("new pop fitness : {0}\t| current pop fitess: {1}\n".format(newPop.fitness, currentPop.fitness))
    

def generations_run(generations, currentPopulation, genesLength, mutationRate):
    lastGeneration = dict()
    xGenerations = list()
    yPlotPopulationFitnessAverage = list()
    yPlotBestindividual = list()

    if len(currentPopulation.container) == 0:
        print("starting population container is empty")
        return
    
    currentPopulation.work_out_population_fitness() 

    for generation in range(0, generations):

        tournmanetSelect = tournmanetSelection(currentPopulation.container, genesLength)

        newPopulation = crossover_and_mutation(tournmanetSelect, genesLength, mutationRate)
        
        newPopulation.work_out_population_fitness()

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
        print(yPlotBestindividual)
        xGenerations.append(generation)

        lastGeneration['population'] = currentPopulation
        lastGeneration['xPlot'] = xGenerations
        lastGeneration['yPlot'] = yPlotPopulationFitnessAverage
        lastGeneration['yPlotBestIndividual'] = yPlotBestindividual

    return lastGeneration


# ---------------------------- MAIN -----------> 
GenesLen = 50
P = 50

# ----------- Initiation ------------
startingPopulation = population(P)
startingPopulation.container = create_individuals(GenesLen, startingPopulation.size)

# ----------- GA Running ------------
gn = generations_run(50, startingPopulation, GenesLen, 0.7)

# ----------- Graph Plotting ------------
plt.xlabel('generations')
plt.ylabel('fitness')
xGenerations = np.array(gn['xPlot'])
yPopulationFitness = np.array(gn['yPlot'])
yGenerationIndividual = np.array(gn['yPlotBestIndividual'])
plt.plot(xGenerations, yGenerationIndividual)
plt.plot(xGenerations, yPopulationFitness)
plt.show()


# --------- THIS IS USING ROULETE WHEEL -------------
    # #selection
    # calculate_relative_fitness_of_individuals(startingPopulation)
    # matingPool = build_mating_pool(startingPopulation)
    # matingPoolSize = len(matingPool)
    # --------- THIS IS USING ROULETE WHEEL -------------
