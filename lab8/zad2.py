import numpy as np
import matplotlib.pyplot as plt
import time
import random
import zad1
np.random.seed(33)

def rastrigin(x):
    return x[0]**2 + x[1]**2 - 20*(np.cos(np.pi*x[0]) + np.cos(np.pi*x[1]) - 2)

def genetic_algorithm(f, bounds, n_pop=50, n_gen=200, p_mut=0.1):
    def random_individual(bounds):
        return [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    def fitness(individual):
        return f(individual)
    def tournament_selection(population, k=3):
        tournament = random.sample(population, k)
        return min(tournament, key=fitness)
    def crossover(parent1, parent2):
        point = random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    def mutate(individual, bounds):
        for i in range(len(individual)):
            if random.random() < p_mut:
                individual[i] = random.uniform(bounds[i][0], bounds[i][1])
        return individual

    population = [random_individual(bounds) for _ in range(n_pop)]

    for i in range(n_gen):
        parents = [tournament_selection(population) for _ in range(n_pop)]
        
        offspring = []
        for j in range(0, n_pop, 2):
            offspring.extend(crossover(parents[j], parents[j+1]))
        
        offspring = [mutate(child, bounds) for child in offspring]
        population = parents + offspring
        population = sorted(population, key=fitness)[:n_pop]

    return min(population, key=fitness)

swarm_solutions = zad1.swarm_solutions
swarm_times = zad1.swarm_times

bounds = [(-10, 10), (-10, 10)]
times = []

solutions = []
for _ in range(10):
    start = time.time()
    best_individual = genetic_algorithm(rastrigin, bounds)
    end = time.time()
    solutions.append(best_individual)
    times.append(end - start)

genetic_solution = np.mean(solutions, axis=0)
genetic_time = np.mean(times)
print(f"Avg: {genetic_solution}")
print(f"Avg time: {genetic_time}")

print("Swarm solutions:")
for solution in swarm_solutions:
    print(f"[{solution[0]}, {solution[1]}]")
print("\nGenetic algorithm solution: ")
print(genetic_solution)

print("\n", 20*"-", "Time comparison", 20*"-")

print("\nSwarm times: ")
for time in swarm_times:
    print(time)
print("\nGenetic time: ")
print(genetic_time)