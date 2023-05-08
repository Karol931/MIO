import numpy as np
import matplotlib.pyplot as plt
import random

SEED = 33
np.random.seed(SEED)

class Solution:
  def __init__(self, randomize_genes = False, gray_coding = False, gamma = None):
    self.genes = np.zeros(16, dtype=int)
    self.randomize_genes = randomize_genes
    self.gray_coding = gray_coding
    self.gamma = gamma
    if randomize_genes:
      for i in range(16):
        self.genes[i] = 0 if np.random.rand() < 0.5 else 1
        
    if gray_coding:
      for i in range(1, 16):
        self.genes[i] = self.genes[i] ^ self.genes[i-1]

  def decode(self):
    for i in reversed(range(1, 16)):
      self.genes[i] = self.genes[i] ^ self.genes[i-1]
    return int("".join(str(x) for x in self.genes), 2) / np.power(2, 16)

  def get_adaptation(self):
    if self.gray_coding:
      x = self.decode()
    else:
      x = int("".join(str(x) for x in self.genes), 2) / np.power(2, 16)
    return np.sin(100*x+0.1) + np.power(x, -x) + 0.2
  
  def crossover(self, other_solution):
    cut_position = np.random.randint(0, 7)
    new_solution = Solution(self.randomize_genes, self.gray_coding, self.gamma)
    new_solution.genes[0:cut_position] = self.genes[0:cut_position]
    new_solution.genes[cut_position:] = other_solution.genes[cut_position:]
    return new_solution

  def mutation(self):
    mutation_position = np.random.randint(0, 7)
    self.genes[mutation_position] = 0 if self.genes[mutation_position] == 1 else 1

def run(gray_coding=False, mutation_chance=0, gamma=None, randomize_genes = True, iterations = 50, population_size = 50, verbose = False, plot=True, **kwargs):
    population = [Solution(randomize_genes=randomize_genes, gray_coding=gray_coding, gamma=gamma) for i in range(population_size)]
    best_solution = Solution(randomize_genes=randomize_genes, gray_coding=gray_coding, gamma=gamma)
    best_solution_adaptation = 0.
    best_iteration_found = 0

    avgs, bests_local, bests_global = [], [], []

    for iteration in range(iterations):
        adaptations = [p.get_adaptation() for p in population]

        local_best_solution = population[adaptations.index(max(adaptations))]
        if local_best_solution.get_adaptation() > best_solution_adaptation:
            best_solution = local_best_solution
            best_solution_adaptation = local_best_solution.get_adaptation()
            best_iteration_found = iteration

        if verbose:
            print(f"Epoch: {iteration}; avg adaptation: {sum(adaptations)/len(adaptations)}; best adaptation: {max(adaptations)}"
                  f"best adaptation ever: {best_solution_adaptation} from iteration: {best_iteration_found}")

        avgs.append(sum(adaptations) / len(adaptations))
        bests_local.append(max(adaptations))
        bests_global.append(best_solution_adaptation)

        if not gamma:
            roulette_wheel = adaptations
            for i in range(len(roulette_wheel)):
                roulette_wheel[i] -= min(adaptations)
                roulette_wheel[i] /= (max(adaptations)-min(adaptations))
            parents = [random.choices(population, weights=roulette_wheel, k = 2) for i in range(population_size)]
        else:
            best_n = int(population_size * (gamma/100))
            sorted_population = sorted(population, key=lambda node: node.get_adaptation(), reverse=True)[:best_n]
            parents = [random.choices(sorted_population, k=2) for i in range(population_size)]

        children = [p[0].crossover(p[1]) for p in parents]
        for c in children:
            if np.random.rand() < mutation_chance:
                c.mutation()

        population = children


    adaptations = [p.get_adaptation() for p in population]
    local_best_solution = population[adaptations.index(max(adaptations))]
    if local_best_solution.get_adaptation() > best_solution_adaptation:
        best_solution = local_best_solution
        best_solution_adaptation = local_best_solution.get_adaptation()

    if verbose:
        print(f'---\nMutation chance: {mutation_chance}\nGamma: {gamma}')
        best_score = best_solution.decode()
        print('Best solution: ', best_solution.genes, ' = ', best_score, ' (correct: 0.39169)')
        print('Found in iteration: ', best_iteration_found)
        print('Largest function value found: ', best_solution_adaptation, ' (correct:  2.64)')

    if plot:
        if "header" in kwargs.keys():
            adnotation = kwargs["header"]

        fig = plt.figure()
        plt.plot(range(iterations), avgs, label='average')
        plt.plot(range(iterations), bests_local, label='best of generation')
        plt.plot(range(iterations), bests_global, label='global best')
        plt.legend()
        plt.title(f'{adnotation}: Gray coding: {gray_coding}, Mutation Chance: {mutation_chance}, Gamma: {gamma}')
        plt.xlabel('Iteration')
        plt.ylabel('Function Value')
        plt.show()

# Ad 1
header = "Ad 1.1"
gray_codings = [False, True]
for gray_coding in gray_codings:
    run(gray_coding=gray_coding, header=header)

# Ad 2
header = "Ad 1.2"
mutation_chances = [0, 0.1, 0.5, 1.0]
for mutation_chance in mutation_chances:
    run(mutation_chance=mutation_chance, header=header)

# Ad 3
header = "Ad 1.3"
gammas = [None, 20, 50]
for gamma in gammas:
    run(gamma=gamma, header=header)