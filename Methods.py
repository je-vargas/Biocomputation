from Other import GA_Classes as ga
import numpy as np

def seed_pop():
    population = []
    for x in range (0, P):
        tempgene=[]
        for y in range (0, N):
            tempgene.append(random.uniform(GMIN, GMAX))
        newind = ga.individual()
        newind.gene = copy.deepcopy(tempgene)
        # newind.fitness = fit.candidate_fitness_cosFunc(newind.gene) #TODO UPDATE DEPENDING ON FITNESS FUNCT USED -> line 82
        population.append(newind)
    return population


def selection_torn_min(population, P):
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