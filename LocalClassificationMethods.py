#!/usr/bin/python

import numpy as np
import csv
from copy import deepcopy
from pprint import pprint
import random
import math
from scipy.stats import multivariate_normal


class FixedWindow():

  def __init__(self, xs, ys):
    self.xs = xs
    self.ys = ys
    self.window = 5

  def classify(self, point):
    closest = []

    for x, y in zip(self.xs, self.ys):
      dist = np.linalg.norm(point - x)

      if dist < self.window:
        closest.append(y)

    if len(closest) == 0:
      return random.choice([0, 1])

    guess = sum(closest) / len(closest)
    return round(guess)


class KNearestNeighbors():

  def __init__(self, xs, ys):
    self.xs = xs
    self.ys = ys
    self.k = 9

  def classify(self, point):
               #y,    distance
    closest = [(None, float("inf")) for i in range(self.k)]

    for x, new_y in zip(self.xs, self.ys):
      for i, (y, dist) in enumerate(closest):
        new_dist = np.linalg.norm(point - x)
        #pprint(("comparing", dist, new_dist))
        if new_dist < dist:
          #pprint(("replacing with", y, dist))
          closest[i] = (new_y, new_dist)
          break

    ys, dists = zip(*closest)
    #pprint((ys, dists))
    guess = sum(ys) / len(ys)
    return round(guess)

POS = 1
NEG = 0
class KernelDensity():

  def __init__(self, xs, ys):
    pos_xs = []
    neg_xs = []

    for x, y in zip(xs, ys):
      if y == POS:
        pos_xs.append(xs)
      elif y == NEG:
        neg_xs.append(xs)
      else:
        raise Exception(y)

    self.sep = {POS: pos_xs, NEG:neg_xs}

    self.prob_pos = float(len(pos_xs)) / len(xs)
    self.prob_neg = float(len(neg_xs)) / len(xs)

  def p(self, point, clazz):
    class_points = self.sep[clazz]

    diffs = [self.k(point, x) for x in class_points]

    return sum(diffs) / len(diffs)

  # indicator kernel
  def k(self, a, b):
    #return 1/math.sqrt(2*math.pi) * math.exp(-1 * np.linalg.norm(a - b) ** 2 / 2)
    #return math.exp(-1.0 * np.linalg.norm(a - b) ** 2 / 2)
    #diff = np.linalg.norm(a - b)
    #if diff < 5:
    #  return 1
    #else:
    #  return 0
    #return 1 / diff

    pprint((a, b, a.shape, b.shape))

    return multivariate_normal.pdf(a, mean=b, cov=np.eye(len(a)))


  def classify(self, point):
    pos = self.prob_pos * self.p(point, POS)
    neg = self.prob_neg * self.p(point, NEG)

    #pprint({"pos":pos, "neg":neg})

    if pos > neg:
      return POS
    else:
      return NEG


def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

K_FOLDS = 10
data_dir = "./data/"
spam_filename = data_dir + "spambase/spambase.data"
def cross_validate_spam(clazz):
  data = read_csv_as_numpy_matrix(spam_filename)[:4600,:]

  np.random.shuffle(data) #truffle shuffle

  num_crosses = K_FOLDS
  crosses = np.vsplit(data, K_FOLDS)
  total_error = 0

  for i in xrange(num_crosses):
    train = None
    for j in xrange(num_crosses):
      if i != j:
        if train == None:
          train = deepcopy(crosses[j])
        else:
          train = np.vstack((train, crosses[j]))

    test = crosses[i][:100]
    train = train[:5000]

    features = np.array(train[:,:56])
    truths = train[:,57].A1
    nb = clazz(features, truths)
    #nb.train()

    features = np.array(test[:,:56])
    truths = test[:,57].A1
    error = calculate_error(nb, features, truths)
    total_error += error

    pprint("cv: " + str(i))
    pprint(error)


  pprint("avg error")
  pprint(total_error / K_FOLDS)

def calculate_error(classifier, features, truths):
  errors = 0
  for item, truth in zip(features, truths):
    guess = classifier.classify(item)
    if guess != truth:
      errors +=1
  return float(errors) / len(truths)


if __name__ == "__main__":
  #pprint("Fixed Window")
  #cross_validate_spam(FixedWindow)
  #pprint("K Nearest Neighbors")
  #cross_validate_spam(KNearestNeighbors)
  pprint("Kernel Density")
  cross_validate_spam(KernelDensity)
