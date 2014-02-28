#!/usr/bin/python
# -*- coding: utf8 -*-

import csv, math
import numpy as np
from pprint import pprint
from scipy import linalg


STOP = .1

class GaussianMixtureModel():

  def __init__(self, num_gaussians, num_features):
    # list of gaussians
    # each gaussian is a tuple of mu and sigma

    self.gaussians = []
    self.gaussian_weights = []

    for _ in xrange(num_gaussians):
      mu = 0
      sigma = np.random.rand(num_features, num_features)
      self.gaussians.append((mu, sigma))
      self.gaussian_weights.append(1/num_gaussians)

    pprint("initial weights/gaussians")
    pprint(self.gaussian_weights)
    pprint(self.gaussians)

  def train(self, data, truth):
    self.features = data.T #TODO, ever used?
    self.items = data
    self.truth = truth

    self.last_likelyhood = self.likelyhood()

    i = 0

    while not self.convergence():
      gamma, n = self.set_expectations()
      self.maximize(gamma, n)
      pprint("no convergence" + str(i))
      i += 1

  def set_expectations(self):
    gamma = np.zeroes((len(self.items), len(self.gaussians)))
    n = np.zeroes(len(self.gaussians))

    for i in range(len(self.items)):
      densities_sum = self.density(self.items[i])

      for j in range(len(self.gaussians)):
        gamma[i,j] = w[j] * \
                     self.one_gaussian_density(self.items[i], self.gaussians[j]) \
                     / densities_sum

    for j in range(len(self.gaussians)):
      n[j] = sum(gamma[:,j])


  def maximize(self, gamma, n):
    new_mus = np.zeros(len(self.gaussians))
    new_sigmas = np.zeros(len(self.gaussians))

    y = self.truth #TODO ??

    for j in range(len(self.gaussians)):
      self.gaussian_weights[j] = n[j] / len(self.items)

      new_mus[j] = np.inner(gamma[:,j], y) / n[j]

      difference = y = new_mus
      new_sigmas[j] = np.dot(gamma[:,j], difference * difference.T) / n[j]

    self.gaussians = zip(new_mus, new_sigmas)

  def convergence(self):
    new_likelyhood = likelyhood()

    if abs(self.last_likelyhood - new_likelyhood) < STOP:
      self.last_likelyhood = new_likelyhood
      return true
    else:
      self.last_likelyhood = new_likelyhood
      return false


  def likelyhood(self):
    densities = [self.density(item) for item in self.items]
    pprint(densities)
    densities_logged = filter(lambda x: math.log(x, math.e), densities)
    likelyhood =  sum(densities_logged) / len(self.items)

    pprint("likelyhood")
    pprint(likelyhood)

    return likelyhood


  # density of x for a particulr gaussian, no weighting
  def one_gaussian_density(self, x, gaussian):
    mu = gaussian[0]
    sigma = gaussian[1]
    dimension = len(gaussian[1][0])

    #pprint(x)
    #pprint(gaussian)
    #pprint(-1/2 * (x - mu).T * np.linalg.pinv(sigma) * (x - mu))

    a_v = -1/2 * (x - mu).T * np.linalg.pinv(sigma) * (x - mu)
    b_v = linalg.expm(a_v)
    c_v = (2 * math.pi) ** dimension * np.linalg.det(sigma)
    d_v = c_v ** 1/2

    return (b_v/d_v)[0][0]

    #return (math.e ** (-1/2 * (x - mu).T * np.linalg.pinv(sigma) * (x - mu))) / \
    #       ((2 * math.pi) ** d * numpy.linalg.det(sigma.absolute)) ** 1/2

  #density over all gaussians/weights for x
  def density(self, x):
    densities_per_gaussian = [self.one_gaussian_density(x, gaussian)
                              for gaussian in self.gaussians]

    #pprint("densities")
    #pprint(densities_per_gaussian)

    return np.inner(self.gaussian_weights, densities_per_gaussian)


def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
                   delimiter=','))).astype('float')

data_dir1 = "../../data/HW1/"
data_dir3 = "../../data/HW3/"

def test_spam():
  spam_filename = data_dir1 + "spambase/spambase.data"
  data = read_csv_as_numpy_matrix(spam_filename)

  train = data[:4000,:]
  test = data[4001:,:]

  features = train[:,:56]
  truth = train[:,57]

  gmm = GaussianMixtureModel(3, 57)
  gmm.train(features, truth)

def test_two_gaussians():
  filename = data_dir3 + "2gaussian.txt"
  data = read_csv_as_numpy_matrix(filename)

  features = data[:,:1]
  truth = data[:,1]

  gmm = GaussianMixtureModel(3, 1)
  gmm.train(features, truth)

def test_three_gaussians():
  filename = data_dir3 + "3gaussian.txt"
  data = read_csv_as_numpy_matrix(filename)

  features = data[:,:1]
  truth = data[:,1]

  gmm = GaussianMixtureModel(3, 1)
  gmm.train(features, truth)

if __name__ == "__main__":
  test_two_gaussians()
  test_three_gaussians()
