import torch
import random
import numpy as np
from collections import defaultdict


def simhash_vec(input_tensor, seed=1024):
  input_tensor = input_tensor.float()
  torch.manual_seed(seed)
  proj_vec = torch.randn(input_tensor.shape)
  ans = torch.dot(input_tensor, proj_vec)
  if ans >= 0:
    return 1
  else:
    return 0


def hash_fac():
  num = random.randint(0, 2**20)
  return lambda x: simhash_vec(x, seed=num)


class LSHTable:

  def __init__(self, K, L):
    random.seed(0)
    self.hash_func_lst = [[hash_fac() for _ in range(K)] for _ in range(L)]
    self.tables = [defaultdict(list) for _ in range(L)]

  def insert(self, query, value):
    for i in range(len(self.tables)):
      hashed_key = tuple([func(query) for func in self.hash_func_lst[i]])
      if value not in self.tables[i][hashed_key]:
        self.tables[i][hashed_key].append(value)

  def query(self, query):
    all_values = []
    for i in range(len(self.tables)):
      hashed_key = tuple([func(query) for func in self.hash_func_lst[i]])
      all_values.extend(self.tables[i][hashed_key])
    return all_values

  # def remove(self, experience):
  #   combined_tensor = torch.cat(
  #       (torch.from_numpy(experience[0]).view(1, -1)[0],
  #        torch.from_numpy(experience[3]).view(1, -1)[0],
  #        torch.tensor([experience[1]]), torch.tensor([experience[2]])))
  #   # print(combined_tensor)
  #   for i in range(len(self.tables)):
  #     hashed_key = tuple(
  #         [func(combined_tensor) for func in self.hash_func_lst[i]])
  #     for j, (s, a, r, n_s, d) in enumerate(self.tables[i][hashed_key]):
  #       if np.array_equal(
  #           s, experience[0]
  #       ) and a == experience[1] and r == experience[2] and np.array_equal(
  #           n_s, experience[3]) and d == experience[4]:
  #         del self.tables[i][hashed_key][j]
  #         break
  #     if len(self.tables[i][hashed_key]) == 0:
  #       self.tables[i].pop(hashed_key)

  # def sample(self, num):
  #   table_idx = random.randint(0, len(self.tables) - 1)
  #   l = min(num, len(self.tables[table_idx]))
  #   sampled_rows = random.sample(self.tables[table_idx].keys(), l)
  #   return [random.choice(self.tables[table_idx][row]) for row in sampled_rows]
