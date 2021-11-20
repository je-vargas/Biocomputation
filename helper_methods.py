"""
Available Methods:
    - DNA_bit_printer(list)
    - population_DNA_printer(list)
    - generations_comparison_DNA_printer(list(popContainer), list(popContainer))
    - helpful_print()
"""


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

def helpful_print(generation, newPop, currentPop):
    print("generation: {0}\n".format(generation))
    # generations_comparison_DNA_printer(newPop, currentPop)
    # print("\n")
    print("new pop fitness : {0}\t| current pop fitess: {1}\n".format(newPop.fitness, currentPop.fitness))