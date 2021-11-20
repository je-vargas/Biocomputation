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

class candidate:
    '''
        properties: 
            - genes
            - fitness
            - relativeFitness
    '''
    def __init__(self):
        self.genes = list()
        self.fitness = 0
        self.relativeFitness = 0