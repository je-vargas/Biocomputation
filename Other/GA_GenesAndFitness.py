from typing import Union, List
import numpy as np
import random

""" 
Available Methods:
    - population_fitness (popContainer : list()) -> int / float
    - candidate_fitness_cosFunc (candGenes : list()) ->  float
    - candidate_fitness_binarySum (candGenes : list()) -> int
    - random_int_generator () -> int
    - candidate_relative_fitness () -> int 
    - create_genes_realNumbers (ofLength : int, minRange : float, maxRange : float) -> list():float
    - create_genes_binary (ofLength : int) -> list():int

"""

def population_fitness(popContainer):
    popFitness = 0
    for individual in popContainer:
        popFitness += individual.fitness
        # if individual.fitness == 0 or individual.fitness == None:
        #     print(f"----- EMPTY ----- individual fitness: {individual.fitness} ") 
    
    # print(f"popFitness: {popFitness}")
    return popFitness

# def population_fitness_binarySum(container):
    #     population = gaClass.population()
    #     population.container = container
    #     popFitness = 0
    #     for individual in container:
    #         individual.fitness = candidate_fitness_binarySum(individual.genes)
    #         popFitness += individual.fitness
    #         population.fitness = popFitness
    #     return population

def offspring_fitness_cosFunc(offspring):
    individuals = list()
    for candidate in offspring:
        candidate.fitness = candidate_fitness_cosFunc(candidate.genes)
        individuals.append(candidate)
    return individuals

def offspring_fitness_binary(offspring):
    individuals = list()
    for candidate in offspring:
        candidate.fitness = candidate_fitness_binarySum(candidate.genes)
        individuals.append(candidate)
    return individuals
        


def candidate_fitness_cosFunc(candGenes) -> float:
    geneSize = len(candGenes)
    fixedSum = 10*geneSize
    fitness = 0
    for gene in candGenes:
        fitness += (((gene ** 2)) - (10 * np.cos(2 * np.pi * gene)))
    return fixedSum + fitness 

def candidate_fitness_binarySum(candGenes) -> int:
    fitness = 0
    for gene in candGenes:
        fitness += gene
    return fitness

def create_genes_realNumbers(ofLength, geneRangeMin, geneRangeMax) -> List[float]:
    newGenes = list()
    for x in range(0, ofLength):
        newGenes.append(random.uniform(geneRangeMin, geneRangeMax))
    return newGenes

def create_genes_binary(ofLength) -> List[int]:
    newGenes = list()
    for x in range(0, ofLength):
        newGenes.append(random.randint(0, 1))
    return newGenes

def random_int_generator(startingValue, upToValue) -> int:
    randomNumber = 0
    randomNumber = random.randint(startingValue, upToValue-1)
    return randomNumber

def candidate_relative_fitness() -> Union[int, float]:
    print("!!!! need to implement candidate relative fitness !!!!")
    return 0