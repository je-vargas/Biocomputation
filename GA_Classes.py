import Printing_Methods as prt

class population:
    '''
        properties: 
            - container
            - fitness
            - size
            - meanFitness
            - crossoverPoint
    '''
    def __init__(self):
        self.container = list()
        self.fitness = 0
        self.size = 0
        self.averagefitness = 0
        self.crossOverPoint = list()

    def __str__ (self):
        return f"Population:\n{prt.print_pop_container(self.container)}\nFitness: {self.fitness}\tSize: {self.size}"

class individual:
    '''
        properties: 
            - gene
            - fitness
            - relativeFitness
    '''
    def __init__(self):
        self.gene = []
        self.fitness = 0
        self.relativeFitness = 0

    def __str__ (self):
        return f"Genes:\n{self.gene}\nFitness: {self.fitness}\t| RelativeFitness: {self.relativeFitness}\n"