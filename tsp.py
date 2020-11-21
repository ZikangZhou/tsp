import copy
import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations


class Individual(object):

    def __init__(self, chromosome, fitness):
        self._chromosome = chromosome
        self._fitness = fitness

    def chromosome(self):
        return copy.deepcopy(self._chromosome)

    def fitness(self):
        return copy.deepcopy(self._fitness)

    def set_chromosome(self, chromosome):
        self._chromosome = copy.deepcopy(chromosome)

    def set_fitness(self, fitness):
        self._fitness = fitness


def get_distances(cities):
    chromosome_len = len(cities)
    distances = np.zeros([chromosome_len, chromosome_len])
    for i in range(chromosome_len):
        for j in range(chromosome_len):
            if i <= j:
                distances[i][j] = np.linalg.norm(cities[i] - cities[j])
                distances[j][i] = distances[i][j]
    return distances


class TSP(object):

    def __init__(self, distances, pop_size, crossover_rate, mutation_rate, max_generation, begin=None, end=None):
        self._pop_size = pop_size
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._begin = begin
        self._end = end
        if self._begin is None or self._end is None:
            self._chromosome_len = len(distances)
        else:
            self._chromosome_len = len(distances) - 2
        self._max_generation = max_generation
        self._pop = []
        self._mating_pop = []
        self._fitness_trend = np.zeros(self._max_generation)
        self._pop_archive = None
        self._distances = distances

    def initialize(self):
        for _ in range(self._pop_size):
            if self._begin is None or self._end is None:
                chromosome = np.random.permutation(self._chromosome_len)
            else:
                indices = list(range(len(self._distances)))
                indices.remove(self._begin)
                indices.remove(self._end)
                chromosome = np.random.permutation(indices)
            fitness = self.get_fitness(chromosome)
            self._pop.append(Individual(chromosome, fitness))
            self._mating_pop.append(Individual(chromosome, fitness))

    def evolve(self):
        for gen in range(self._max_generation):
            # Fitness Evaluation
            for n in range(self._pop_size):
                self._pop[n].set_fitness(self.get_fitness(self._pop[n].chromosome()))
            # Archiving
            for n in range(self._pop_size):
                self.archive(self._pop[n])
            # Tracking the fitness of the best individual
            self._fitness_trend[gen] = self._pop_archive.fitness()
            # Initialise the mating pool via tournament selection
            # elite is appended to the current population
            if gen != 0:
                self._pop.append(self._pop_archive)
            # Tournament selection with replacement
            for n in range(self._pop_size):
                self._mating_pop[n] = self.tournament()
            # Applying the genetic operators
            self.crossover()
            for n in range(self._pop_size):
                self.mutate(n)

        ground_truth = self.ground_truth()
        for i in range(len(self._fitness_trend)):
            if np.isclose(ground_truth, self._fitness_trend[i]):
                return i, ground_truth, self._fitness_trend
        return self._max_generation, self._fitness_trend[-1], self._fitness_trend

    def tournament(self):
        indices = np.random.choice(self._pop_size, size=2, replace=False)
        if self._pop[indices[0]].fitness() > self._pop[indices[1]].fitness():
            index = indices[0]
        else:
            index = indices[1]
        return Individual(self._pop[index].chromosome(), self._pop[index].fitness())

    def mutate(self, index):
        for i in range(self._chromosome_len):
            if np.random.rand() < self._mutation_rate:
                j = np.random.randint(self._chromosome_len)
                chromosome = self._pop[index].chromosome()
                tmp = chromosome[i]
                chromosome[i] = chromosome[j]
                chromosome[j] = tmp
                self._pop[index].set_chromosome(chromosome)

    def crossover(self):
        np.random.shuffle(self._mating_pop)
        if self._pop_size % 2 == 0:
            num_crossover = self._pop_size
        else:
            num_crossover = self._pop_size - 1
        for i in range(0, num_crossover, 2):
            if np.random.rand() < self._crossover_rate:
                indices = np.random.choice(self._chromosome_len, size=2, replace=True)
                begin, end = np.sort(indices)

                chromosome1 = self._mating_pop[i].chromosome()
                chromosome1[begin: end] = self._mating_pop[i + 1].chromosome()[begin: end]
                chromosome2 = self._mating_pop[i + 1].chromosome()
                chromosome2[begin: end] = self._mating_pop[i].chromosome()[begin: end]
                self._mating_pop[i].set_chromosome(chromosome1)
                self._mating_pop[i + 1].set_chromosome(chromosome2)
                self.resolve_conflict(i, i + 1, begin, end)
                self.resolve_conflict(i + 1, i, begin, end)
        self._pop = copy.deepcopy(self._mating_pop)

    def archive(self, individual):
        if self._pop_archive is None or self._pop_archive.fitness() < individual.fitness():
            self._pop_archive = Individual(individual.chromosome(), individual.fitness())

    def get_fitness(self, chromosome):
        fitness = 0.0
        for i in range(self._chromosome_len - 1):
            fitness += self._distances[chromosome[i]][chromosome[i + 1]]
        if self._begin is None or self._end is None:
            fitness += self._distances[chromosome[self._chromosome_len - 1]][chromosome[0]]
        else:
            fitness += self._distances[self._begin][chromosome[0]] + self._distances[chromosome[-1]][self._end]
        fitness = 1.0 / fitness
        return fitness

    def resolve_conflict(self, index1, index2, begin, end):
        while True:
            conflicts = []
            for i in range(self._chromosome_len):
                for j in range(i + 1, self._chromosome_len):
                    if self._mating_pop[index1].chromosome()[i] == self._mating_pop[index1].chromosome()[j]:
                        conflicts.append([self._mating_pop[index1].chromosome()[i], i, j])
            if len(conflicts) == 0:
                break
            for conflict in conflicts:
                chromosome = self._mating_pop[index1].chromosome()
                if begin <= conflict[1] <= end:
                    chromosome[conflict[2]] = self._mating_pop[index2].chromosome()[conflict[1]]
                else:
                    chromosome[conflict[1]] = self._mating_pop[index2].chromosome()[conflict[2]]
                self._mating_pop[index1].set_chromosome(chromosome)

    def ground_truth(self):  # 0.3871095186881932
        if self._begin is None or self._end is None:
            chromosomes = list(permutations(range(self._chromosome_len)))
        else:
            indices = list(range(len(self._distances)))
            indices.remove(self._begin)
            indices.remove(self._end)
            chromosomes = list(permutations(indices))
        fitness_list = []
        for chromosome in chromosomes:
            fitness_list.append(self.get_fitness(list(chromosome)))
        return max(fitness_list)


def grid_search():
    cities = np.array(
        [[0.3642, 0.7770],
         [0.7185, 0.8312],
         [0.0986, 0.5891],
         [0.2954, 0.9606],
         [0.5951, 0.4647],
         [0.6697, 0.7657],
         [0.4353, 0.1709],
         [0.2131, 0.8349],
         [0.3479, 0.6984],
         [0.4516, 0.0488]])
    pop_sizes = [20, 100]
    crossover_rates = [0.1, 0.4, 0.7, 1.0]
    mutation_rates = [0.02, 0.05, 0.08]
    for pop_size in pop_sizes:
        for crossover_rate in crossover_rates:
            for mutation_rate in mutation_rates:
                average_i = 0.0
                print(
                    f'pop_size={pop_size}, '
                    f'crossover_rate={crossover_rate}, '
                    f'mutation_rate={mutation_rate}')
                for _ in range(10):
                    tsp = TSP(get_distances(cities), pop_size, crossover_rate, mutation_rate, 500)
                    tsp.initialize()
                    i, fitness, fitness_trend = tsp.evolve()
                    average_i += i
                average_i /= 100
                print(f'reach optimal solution at iter {average_i} on average')


def specify_start_and_end():
    cities = np.array(
        [[0.3642, 0.7770],
         [0.7185, 0.8312],
         [0.0986, 0.5891],
         [0.2954, 0.9606],
         [0.5951, 0.4647],
         [0.6697, 0.7657],
         [0.4353, 0.1709],
         [0.2131, 0.8349],
         [0.3479, 0.6984],
         [0.4516, 0.0488]])

    tsp = TSP(get_distances(cities), 100, 0.4, 0.05, 50, 0, 9)
    tsp.initialize()
    i, fitness, fitness_trend = tsp.evolve()
    print(f'reach optimal solution at iter {i}, fitness={fitness}')
    plt.plot(fitness_trend, "r-", linewidth=2)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("./start_end")
    plt.show()


def vary_cities_num():
    distances = np.random.rand(50, 50)
    print(f'complete distances matrix: {distances}')
    pattern = ["b-", "g-", "r-", "c-", "y-"]
    for n in range(10, 51, 10):
        print(f'cities number={n}')
        tsp = TSP(distances[:n][:n], 100, 0.4, 0.05, 1000)
        tsp.initialize()
        i, fitness, fitness_trend = tsp.evolve()
        plt.plot(fitness_trend, pattern[int(n / 10) - 1], linewidth=2, label=f"{n} cities")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("./scale")
    plt.show()


if __name__ == '__main__':
    grid_search()
    specify_start_and_end()
    vary_cities_num()
    specify_start_and_end()
