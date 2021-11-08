import random

GenesLen = 10
P = 20

# ------------------------- CLASS

class population: 
    def __init__(self, size):
        self.container = list()
        self.__fitness = 0
        self.__size=size
        self.__populationFitnessCalculated = False

    @property
    def size(self):
        return self.__size
    
    @property
    def fitness(self):
        return self.__fitness
    
    def work_out_population_fitness(self):
        if not self.__populationFitnessCalculated:
            for individual in self.container:
                self.__fitness += individual.fitness
            self.__populationFitnessCalculated = True

class individual_candidate:
    def __init__(self, geneLength):
        self.__genes = list()
        self.__fitness = 0
        self.__relativeFitnessAsPercentage = 0
        self.__genesLength = geneLength
        self.__fitnessCalculated = False

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
        if not self.__fitnessCalculated:
            for gene in self.genes:
                if bool(gene):
                    self.__fitness += 1
        self.__fitnessCalculated = True

    def create_dna(self):
        for x in range (0, self.__genesLength):
            self.__genes.append(random.randint(0,1)) # random no between 0-1 because GeneticAlgorithm is using a Binary Encoding
        
# ------------------------- METHODS
def random_number_geenerator(startingValue, upToValue ):
    randomNumber = startingValue
    randomNumber = random.randint( startingValue, upToValue-1 )
    return randomNumber

def individual_chrom_printer(individual):
    for bit in individual.genes:
        print(bit)

def population_chrom_printer(populationArray):
    # print("\ngeneration: " + str(populationInteration))
    for individual in populationArray:
        print(individual.genes)

def create_individuals(geneLength, populationSize):
    #Creating random population of size P
    population = []
    for individual in range (0, populationSize):
        newindividual = individual_candidate(geneLength)
        newindividual.create_dna()
        newindividual.individuals_fitness()
        population.append(newindividual)
    return population

def calculate_relative_fitness_of_population_individuals(currentPopulation):
    for currentIndividual in currentPopulation.container:
        currentIndividual.set_relative_fitness(currentPopulation.fitness)

def build_mating_pool(currentPopulation):
    matingPool = list()
    for currentIndividual in currentPopulation.container:
        for ind in range(0, currentIndividual.relative_fitness):
            matingPool.append(currentIndividual)
    return matingPool

def selected_population_returned(currentPopulation):

    popSize = currentPopulation.size
    newOffSpringPopulation = population(popSize)

    for i in range (0, popSize):

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

def population_generations(generations, currentPopulation):
    
    currentPopulation.work_out_population_fitness()
    population_chrom_printer(currentPopulation.container) # print to make sure container is not empty

    for generation in range(0, generations):

        #normalises data
        for currentIndividual in currentPopulation.container:
            currentIndividual.set_relative_fitness(currentPopulation.fitness)
            print(str(currentIndividual.relative_fitness))

        #create mating pool


        # selectection -> based on relative fitness
        selectedPopulation = selected_population_returned(currentPopulation)

        print("pop fitnessL " + str(currentPopulation.fitness))
        
        if selectedPopulation.fitness > currentPopulation.fitness:
            currentPopulation = selectedPopulation

        #reproduction cross over and mutation

    return currentPopulation

def cross_over(crossOverPoint,parentAGn, parentBGn):
    offsprings = {}
    geneLength = parentAGn.gene_length

    PAgene1 = parentAGn.genes[:crossOverPoint]
    PAgene2 = parentAGn.genes[crossOverPoint:]

    PBgene1 = parentBGn.genes[:crossOverPoint]
    PBgene2 = parentBGn.genes[crossOverPoint:]

    print(PAgene1.extend(PBgene2))
    print(PBgene1.extend(PAgene2))

    offspring1 = individual_candidate(geneLength) 
    offspring2 = individual_candidate(geneLength) 

    offspring1.set_genes(PAgene1)
    offspring2.set_genes(PBgene1)

    offsprings['childA'] = offspring2
    offsprings['childB'] = offspring2
    
    return offsprings

def mutation(offsprings, mutationRate):
    childA = offsprings['childA']
    childB = offsprings['childB']
    
    for geneIndex in range(0, childA.gene_length):
        if random.random() < mutationRate:
            childA.gene_mutation(geneIndex)
            childB.gene_mutation(geneIndex)
    offsprings['childA'] = childA
    offsprings['childB'] = childB
    return offsprings
    
def parent_selection_down_list_2(matingPool):
    #work sheet 2 select 2 parents down the list and keeps going down
    parents = list()
    nextParent = 0 
    matingPoolSize = len(matingPool)

    for i in range(0, matingPoolSize, 2):
        
        nextParent = i + 1
        print("next parent: {0}".format(nextParent))

        if nextParent == matingPoolSize:
            exit
        else: 
            parentA = matingPool[i]
            parentB = matingPool[nextParent]
            
            parents.append(parentB)
            parents.append(parentA)
            
    return parents
            
            


# ---------------------------- MAIN

a = [0, 1, 2]
print(parent_selection_down_list_2(a))

# # create population container
# newPopulation = population(10)
# newPopulation.container = create_individuals(GenesLen, newPopulation.size)
# newPopulation.work_out_population_fitness()

# # #selection
# calculate_relative_fitness_of_population_individuals(newPopulation)
# matingPool = build_mating_pool(newPopulation)
# matingPoolSize = len(matingPool)

# # parent_selection_down_list_2(matingPool)

# randomParentIndex1 = random_number_geenerator(0, matingPoolSize)
# randomParentIndex2 = random_number_geenerator(0, matingPoolSize)

# # reproduction 
# # select parents going down the list



# parentA = matingPool[randomParentIndex1]
# parentB = matingPool[randomParentIndex2]

# # chrossover and mutation
# crossOverPoint = random_number_geenerator(0, 5)
# offsprings = cross_over(crossOverPoint, parentA, parentB)
# #
#     # print(str(crossOverPoint))
#     # print("{0} \n{1}".format(parentA.genes,parentB.genes))
#     # print(offsprings)
# offsprings = mutation(offsprings, 0.2)




