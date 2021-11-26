
import random
import matplotlib.pyplot as plt

N = 10

solutionInterations = 20

class solution:
    variable = [0]* N
    utility = 0 

#  METHOD --------------------------------

def untility_is_sum_of_array(individual):
    utility=0
    for i in range(N):
        utility = utility + individual.variable[i]
    return utility

def individual_array_random_populator (individual ,N):
    for j in range (N):
        individual.variable[j] = random.randint(0,100)
    return individual

individual = solution()

individual = individual_array_random_populator(individual, N)

individual.utility = untility_is_sum_of_array(individual)

newSolution = solution()

for interation in range (solutionInterations):
    for i in range(N):
        newSolution.variable[i] = individual.variable[i]

    change_point = random.randint(0, N-1)  # random point in array size 
    newSolution.variable[change_point] = random.randint(0,100)
    newSolution.utility = untility_is_sum_of_array( newSolution )

    if individual.utility <= newSolution.utility:
        individual.variable[change_point] = newSolution.variable[change_point]
        individual.utility = newSolution.utility
        print(newSolution.utility)
        plt.plot([newSolution.utility], [interation])




