import numpy as np
import matplotlib.pyplot as plt
import random

SEED = 33
np.random.seed(SEED)

class Solution:
  def __init__(self, rand_num=False):
    self.A = [random.randint(0, 1) for _ in range(8)] if rand_num else [0] * 8
    self.B = [random.randint(0, 1) for _ in range(8)] if rand_num else [0] * 8
    self.C = [random.randint(0, 1) for _ in range(8)] if rand_num else [0] * 8
    self.D = [random.randint(0, 1) for _ in range(8)] if rand_num else [0] * 8

def f(a, b, c, d):
  if c > 0:
    return 5000 - ((600.0) * ((a - 20)**2) * ((b - 35)**2)) / c - ((a - 50)**2) * ((d - 48)**2) + d
  else:
    return 5000 - ((600.0) * ((a - 20)**2) * ((b - 35)**2)) / 1 - ((a - 50)**2) * ((d - 48)**2) + d

def to_decimal(arr):
  return int("".join(str(x) for x in arr), 2)

def gray_decode(n):
  n = int(''.join(map(str, n)), 2)
  m = n >> 1
  while m:
    n ^= m
    m >>= 1
  return n

def make_child(s1, s2):
  res = Solution()
  i = random.randint(1, 6)
  res.A = s1.A[:i] + s2.A[i:]
  i = random.randint(1, 6)
  res.B = s1.B[:i] + s2.B[i:]
  i = random.randint(1, 6)
  res.C = s1.C[:i] + s2.C[i:]
  i = random.randint(1, 6)
  res.D = s1.D[:i] + s2.D[i:]
  return res

def run(gamma=None):
  population = [Solution(rand_num=True) for _ in range(50)]
  eval_func = to_decimal
  average_scores = []
  best = None

  for iter in range(100):
    scores = list(map(lambda x: f(eval_func(x.A), eval_func(x.B), eval_func(x.C), eval_func(x.D)), population))
    average_scores.append(sum(scores) / len(scores))
    curr_best = population[np.array(scores).argmax()]

    if best is None or f(eval_func(best.A), eval_func(best.B), eval_func(best.C), eval_func(best.D)) < f(eval_func(curr_best.A), eval_func(curr_best.B), eval_func(curr_best.C), eval_func(curr_best.D)):
      best = curr_best

    if not gamma:
      parents = [random.choices(population=population, weights=np.array(scores) - min(scores), k=2) for _ in range(int(len(population) / 2))]
      children = [make_child(p1, p2) for p1, p2 in parents]
      children.extend([make_child(p2, p1) for p1, p2 in parents])

      for i in range(len(children)):
        if random.random() < 0.1:
          mgene = random.randint(0, 7)
          children[i].A[mgene] = 1 - children[i].A[mgene]
        if random.random() < 0.1:
          mgene = random.randint(0, 7)
          children[i].B[mgene] = 1 - children[i].B[mgene]
        if random.random() < 0.1:
          mgene = random.randint(0, 7)
          children[i].C[mgene] = 1 - children[i].C[mgene]
        if random.random() < 0.1:
          mgene = random.randint(0, 7)
          children[i].D[mgene] = 1 - children[i].D[mgene]

        population = children

    else:
      parents = sorted(zip(population, scores), key=lambda x: x[1])[int(len(population) * (gamma / 100)):]
      parents = list(map(lambda x: x[0], parents))
      parents2 = [random.choices(population=parents, k=2) for _ in range(50)]

      children = [make_child(p1, p2) for p1, p2 in parents2]
      children.extend([make_child(p2, p1) for p1, p2 in parents2])

      for i in range(len(children)):
        if random.random() < 0.5:
          mgene = random.randint(0, 7)
          children[i].A[mgene] = 1 - children[i].A[mgene]
        if random.random() < 0.5:
          mgene = random.randint(0, 7)
          children[i].B[mgene] = 1 - children[i].B[mgene]
        if random.random() < 0.5:
          mgene = random.randint(0, 7)
          children[i].C[mgene] = 1 - children[i].C[mgene]
        if random.random() < 0.5:
          mgene = random.randint(0, 7)
          children[i].D[mgene] = 1 - children[i].D[mgene]

      parents.extend(children)
      population = random.choices(population=parents, k=50)

  print(f"\nbest gene {eval_func(best.A),eval_func(best.B),eval_func(best.C),eval_func(best.D)}")
  print(f"best value {f(eval_func(best.A),eval_func(best.B),eval_func(best.C),eval_func(best.D))}")
  fig = plt.figure()
  plt.plot(average_scores)
  plt.title(f"Average Scores for gamma={gamma}")
  plt.xlabel('Iteration')
  plt.ylabel('Average Score')
  plt.show()

for gamma in [None, 0, 20, 50]:
  run(gamma)