class population:
    def __init__(self):
        self.container = list()
        self.fitness = 0
        self.size = 0
        self.averagefitness = 0
        self.crossOverPoint = list()

class individual_candidate:
    def __init__(self):
        self.genes = list()
        self.fitness = 0
        self.relativeFitnessAsPercentage = 0

