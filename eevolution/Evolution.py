import random
import matplotlib.pyplot as plt

__author__ = 'erikvanegmond'


class Genome(object):
    def __init__(self, genes):
        self.genes = genes

    def __repr__(self):
        f = lambda g: "{:.2f}".format(g)
        return ", ".join(map(f, self.genes))

    def __delitem__(self, index):
        self.genes.pop(index)

    def __getitem__(self, index):
        return self.genes[index]

    def __setitem__(self, index, value):
        self.genes[index] = value

    @classmethod
    def randomgenes(cls):
        genes = [random.randint(0, 10) for _ in range(10)]
        return cls(genes)

    def mutate(self):
        self.genes[:] = [gene + random.gauss(0, 1) if random.random() > 0.7 else gene for gene in self.genes]


class Evaluation(object):
    @staticmethod
    def evaluate(genome):
        return float(sum(abs(x - i) for i, x in enumerate(genome)))


class Individual(object):
    num_individuals = 0

    def __init__(self, genome=None):
        if genome:
            self.genome = Genome(genome)
        else:
            self.genome = Genome.randomgenes()
        Individual.num_individuals += 1
        self.individual_ID = Individual.num_individuals
        self.fitness = None

    def __repr__(self):
        return "Individual {}. Fitness: {}, Genome: {}".format(self.individual_ID, self.evaluate(), self.genome)

    def evaluate(self):
        if not self.fitness:
            self.fitness = Evaluation.evaluate(self.genome)
        return self.fitness

    def mutate(self):
        self.genome.mutate()


class Population(dict):
    def __init__(self, population_size=50):
        self.population_size = population_size
        population_list = {Individual() for _ in range(population_size)}
        for individual in population_list:
            self[individual.individual_ID] = individual

    def __repr__(self):
        return "A population Object, {} individuals".format(len(self))

    def parent_selection(self, n_parents=2):
        parents = [self.tournament_selection(competition_size=20) for x in range(n_parents)]
        return parents

    def tournament_selection(self, competition_size=5, func=min):
        # TODO exclude previous winners
        candidates = random.sample(self.keys(), competition_size)
        winner = func(candidates, key=lambda x: self[x].evaluate())
        return self[winner]

    @staticmethod
    def cross_over(parents):
        genomes = [parent.genome.genes for parent in parents]
        child_genome = [random.choice(a) for a in list(zip(*genomes))]
        return Individual(child_genome)

    def add(self, new_member):
        """Adds a member to the population"""
        self[new_member.individual_ID] = new_member

    def maintain_popsize(self):
        while len(self) > self.population_size:
            loser = self.tournament_selection(func=max)
            self.pop(loser.individual_ID)

    def best(self):
        best_individual = None
        for id, individual in self.items():
            if best_individual:
                if best_individual.evaluate() > individual.evaluate():
                    best_individual = individual
            else:
                best_individual = individual
        return best_individual


class Evolution(object):
    def __init__(self, population_size=50):
        self.population = Population(population_size=population_size)

    def __repr__(self):
        return "An Evolutionary Algorithm"

    def run(self):
        bests = []
        for i in range(500):
            if i % 100 is 0:
                print("Generation: {}".format(i))
            n_children = random.randint(1, 15)
            for _ in range(n_children):
                parents = self.population.parent_selection()
                child = self.population.cross_over(parents)
                child.mutate()
                self.population.add(child)
            self.population.maintain_popsize()
            # print(self.population)
            bests.append(self.population.best().evaluate())
        # print(bests)
        print(self.population.best())
        plt.xkcd()
        plt.plot(bests)
        plt.axis([0, len(bests), 0, bests[0] + 1])
        plt.show()
