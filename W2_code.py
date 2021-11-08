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

    def gene_mutation(self, geneIndex):
        if bool(self.__genes[geneIndex]):
            self.__genes[geneIndex] = 0
        else:
            self.__genes[geneIndex] = 1

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
            if bool(gene):
                self.__fitness += 1

    def create_dna(self):
        for x in range(0, self.__genesLength):
            # random no between 0-1 because GeneticAlgorithm is using a Binary Encoding
            self.__genes.append(random.randint(0, 1))

# ------------------------- METHODS

def random_number_geenerator(startingValue, upToValue):
    randomNumber = startingValue
    randomNumber = random.randint(startingValue, upToValue-1)
    return randomNumber

def DNA_bit_printer(individual):
    for bit in individual.genes:
        print(bit)

def population_DNA_printer(populationArray):
    # print("\ngeneration: " + str(populationInteration))
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

    # print("cross point {0}".format(crossOverPoint))
    # print("parentA: {0}\nparentB: {1}".format(parentAGn.genes, parentBGn.genes))
    pAGeneStart = parentAGn.genes[:crossOverPoint]
    pAGeneEnd = parentAGn.genes[crossOverPoint:]

    pBGeneStart = parentBGn.genes[:crossOverPoint]
    pBGeneEnd = parentBGn.genes[crossOverPoint:]

    # print("genes arleady crossed")
    # print("PA crossed genes: {0} : {1}\nPB crossed genes: {2} : {3}\n".format(pAGeneStart, pBGeneEnd, pBGeneStart, pAGeneEnd))
    
    childA = [*pAGeneStart, *pBGeneEnd]
    childB = [*pBGeneStart, *pAGeneEnd]

    # print("PA crossed genes: {0}\nPB crossed genes: {1}\n".format(pAGeneStart, pBGeneStart))

    offspring1 = individual_candidate(geneLength)
    offspring2 = individual_candidate(geneLength)

    offspring1.set_genes(childA)
    offspring2.set_genes(childB)

    newOffsprings['childA'] = offspring1
    newOffsprings['childB'] = offspring2

    return newOffsprings

def mutation(newOffsprings, mutationRate):
    childA = newOffsprings['childA']
    childB = newOffsprings['childB']

    for geneIndex in range(0, childA.gene_length):
        if random.random() < mutationRate:
            childA.gene_mutation(geneIndex)
            # print("mutation applied chA: {0}".format(childA.genes))

        if random.random() < mutationRate:
            childB.gene_mutation(geneIndex)
            # print("mutation applied chB: {0}".format(childB.genes))

    newOffsprings['childA'] = childA
    newOffsprings['childB'] = childB

    return newOffsprings

def best_generation_individual(currentPopulation):
    best = 0
    bestIndividual = None
    
    for individual in currentPopulation:
        if individual.fitness > best:
            best = individual.fitness
            bestIndividual = individual

    return bestIndividual

def worst_generation_individual(currentPopulation):
    worst = -1
    indexOfWorstIndividual = 0
    
    for individual in range (0, len(currentPopulation)):
        if individual == 0:
            worst = currentPopulation[individual].fitness
            worstIndividual = currentPopulation[individual]

        if currentPopulation[individual].fitness < worst: 
            worst = currentPopulation[individual].fitness
            worstIndividual = currentPopulation[individual]
            indexOfWorstIndividual = individual

    return indexOfWorstIndividual

def crossover_and_mutation(matingPool, genesLength, mutationRate):
    "Defintion: Returns selected population object (new) with crossover and mutation "
    # work sheet 2 select 2 parents down the list and keeps going down
    nextParent = 0
    matingPoolSize = len(matingPool)
    offSpringPopulation = population(matingPoolSize)

    for i in range(0, matingPoolSize, 2):
        nextParent = i + 1

        if nextParent > matingPoolSize-1:
            print("Next parent index is out of range {0}".format(nextParent))
            print("size of array : {0}".format(matingPoolSize))
            return

        else:
            # print("parents index {0} and {1}".format(i, nextParent))
            parentA = matingPool[i]
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

        # offSpringPopulation.work_out_population_fitness()

    popfitness = offSpringPopulation.fitness
    # print("fitness of new population: {0}".format(popfitness))
    # print("\nNew population selected and being returned:\n")
    # population_DNA_printer(offSpringPopulation.container)

    return offSpringPopulation

def tournmanetSelection(populationContainer, geneLength):
    selectedParents = list()
    for i in range(0, len(populationContainer)):
        randA = random_number_geenerator(0, len(populationContainer))
        randB = random_number_geenerator(0, len(populationContainer))
        parentA = populationContainer[randA]
        parentB = populationContainer[randB]

        if(parentA.fitness > parentB.fitness):
            selectedParents.append(parentA)
        else:
            selectedParents.append(parentB)
    return selectedParents

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

        # print("Generation: {0}\n".format(generation))
        # population_DNA_printer(currentPopulation.container)

        # ------- !!! not using mating pool for W2 !!!! --------
            # for currentIndividual in currentPopulation.container:
            #     currentIndividual.set_relative_fitness(currentPopulation.fitness)
            #     print(str(currentIndividual.relative_fitness))

            # selectection -> for worksheet 2 is just selecting 2 parents down list and crossing over and adding mutation
        
        tournmanetSelect = tournmanetSelection(currentPopulation.container, genesLength)

        newPopulation = crossover_and_mutation(tournmanetSelect, genesLength, mutationRate)
        
        newPopulation.work_out_population_fitness()

        populationAverageFitness = (currentPopulation.fitness / currentPopulation.size)

        bestIndividual = best_generation_individual(currentPopulation.container) #// once selecction and reproduction 
        
        # 
            # print("\nNew population after Crss & Mut\n")
            # population_DNA_printer(newPopulation.container)
            # print("\n")
            # population_DNA_printer(newPopulation.container)

        # print("generation: {0}\n".format(generation))
        # generations_comparison_DNA_printer(newPopulation, currentPopulation)
        # print("\n")

        # print("new pop fitness : {0}\t| current pop fitess: {1}\n".format(newPopulation.fitness, currentPopulation.fitness))

        # local variable of best individual from currnt pop
        # if newPopulation.fitness > currentPopulation.fitness:
        currentPopulation = newPopulation

        worstIndividualIndex = worst_generation_individual(currentPopulation.container)

        # print("worstIndex: {0}\t| bestIndividual: {1}".format(worstIndividualIndex, bestIndividual.fitness))

        currentPopulation.container[worstIndividualIndex] = bestIndividual

        # copy local best ind over current pop worst 

        yPlotPopulationFitnessAverage.append(populationAverageFitness)
        yPlotBestindividual.append(bestIndividual.fitness)
        print(yPlotBestindividual)
        xGenerations.append(generation)

        lastGeneration['population'] = currentPopulation
        lastGeneration['xPlot'] = xGenerations
        lastGeneration['yPlot'] = yPlotPopulationFitnessAverage
        lastGeneration['yPlotBestIndividual'] = yPlotBestindividual

    return lastGeneration


# ---------------------------- MAIN


GenesLen = 50
P = 50

# create population container
startingPopulation = population(P)
startingPopulation.container = create_individuals(GenesLen, startingPopulation.size)

gn = generations_run(50, startingPopulation, GenesLen, 0.012)

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
