import random 

class population: 
    def __init__(self):
        self.__population = []
        self.__fitness = 0

    @property
    def get_fitness(self):
        return self.__fitness

    def set_population(self, population):
        self.__population = population
    
    def population_fitness(self):
        for individual in self.population:
            self.fitness += individual.fitness
        return

class individual_candidate:
    def __init__(self):
        self.chromosome = []
        self.__fitness = 0

    def individual_fitness(self):
        for gene in self.chromosome:
            if gene:
                self.fitness += 1

    def chromosome_generator(self, chromosomeLength ):
        for x in range (0, chromosomeLength ):
            self.chromosome.append(random.randint(0,1)) # random no between 0-1 because GeneticAlgorithm is using a Binary Encoding

ind = individual_candidate()
ind.chromosome_generator(10)
print()