import numpy as np
import matplotlib.pyplot as plt
import time
import random

np.random.seed(33)

def rastrigin(x: np.array):
    return x[0]**2 + x[1]**2 - 20*(np.cos(np.pi*x[0]) + np.cos(np.pi*x[1]) - 2)

def pso(cost_func, n_dim=2, num_particles=30, max_iter=200, w=0.5, c1=1, c2=2, verbose=False, plot=True):
    avgs = []
    bests_fitness = []

    particles = np.random.uniform(-10, 10, (num_particles, n_dim))
    velocities = np.zeros((num_particles, n_dim))

    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)
    if plot:
        fig = plt.figure(figsize=(10, 4))
    for i in range(max_iter):
        r1 = np.random.uniform(0, 1, (num_particles, n_dim))
        r2 = np.random.uniform(0, 1, (num_particles, n_dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        particles += velocities
        fitness_values = np.array([cost_func(p) for p in particles])
        
        avgs.append(np.mean(fitness_values))
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)
            
        bests_fitness.append(swarm_best_fitness)
        if verbose:
            print('Epoch:',i)
            print('Swarm best position:',swarm_best_position)
            print('Swarm best fitness:', swarm_best_fitness)

    if plot:
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.plot(bests_fitness)
        plt.title('Best Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        
        plt.subplot(1, 3, 2)
        plt.plot(avgs)
        plt.title('Average Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.show()

    return swarm_best_position, swarm_best_fitness

swarm_solutions = [] # global for comaprison with genetic algorithm
swarm_times = []

cognitive_social_scalings = [[0,2], [2, 0], [1, .5],  [2,2]]
all_avg = []
all_times = []

for c1, c2 in cognitive_social_scalings:
    solutions = []
    fitnesses = []
    times = []
    for i in range(10):
        start = time.time()
        solution, fitness = pso(rastrigin, c1=c1, c2=c2, plot=((i+1)%10==0))
        end = time.time()
        solutions.append(solution)
        fitnesses.append(fitness)
        times.append(end - start)
    swarm_solutions.append(np.mean(solutions, axis=0))
    print(f"Avg solution: {np.mean(solutions, axis=0)}, Avg fitness: {np.mean(fitnesses)}")
    print(f"Std solution: {np.std(solutions, axis=0)}, Std fintess: {np.std(fitnesses)}")
    print(f"Avg time: {np.mean(times)}")
    all_times.append(np.mean(times))
    swarm_times.append(np.mean(times))
    all_avg.append(np.mean(solutions, axis=0))

print("\n")
print(f"All average: {np.mean(all_avg, axis=0)}")
print(f"All average time: {np.mean(all_times)}")

ws = [1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.3, 0.5, 1]
all_avg = []
all_times = []

for w in ws:
    solutions = []
    fitnesses = []
    times = []
    for i in range(10):
        start = time.time()
        solution, fitness = pso(rastrigin, w=w, plot=((i+1)%10==0))
        end = time.time()
        solutions.append(solution)
        fitnesses.append(fitness)
        times.append(end - start)

    swarm_solutions.append(np.mean(solutions, axis=0))
    print(f"Avg solution: {np.mean(solutions, axis=0)}, Avg fitness: {np.mean(fitnesses)}")
    print(f"Std solution: {np.std(solutions, axis=0)}, Std fintess: {np.std(fitnesses)}")
    print(f"Avg time: {np.mean(times)}")
    all_times.append(np.mean(times))
    swarm_times.append(np.mean(times))
    all_avg.append(np.mean(solutions, axis=0))

print("\n")
print(f"All average: {np.mean(all_avg, axis=0)}")
print(f"All average time: {np.mean(all_times)}")