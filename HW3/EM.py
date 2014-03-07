#!/usr/bin/python
# -*- coding: utf8 -*-

import csv, math
import numpy as np
from pprint import pprint
from scipy import linalg


STOP = .1

class GaussianMixtureModel():

  def __init__(self, num_gaussians):
    # list of gaussians
    # each gaussian is a tuple of mu and sigma

    self.num_gaussians = num_gaussians
    self.gaussians = []
    self.gaussian_weights = []

  def train(self, data):
    self.items = data
    pprint(data.shape)
    num_features = data.shape[1]

    # initialize
    for _ in xrange(self.num_gaussians):
      mu = np.zeros(num_features, dtype='float16')
      sigma = np.random.rand(num_features, num_features)
      #sigma = np.cov(data, rowvar=0)
      self.gaussians.append((mu, sigma))
      self.gaussian_weights.append(100.0/self.num_gaussians)

    pprint("initial weights/gaussians")
    pprint(self.gaussian_weights)
    pprint(self.gaussians)

    self.last_likelyhood = float("-inf")
    i = 0

    # iterate
    while not self.convergence():
      print("iteration", i)
      gamma, n = self.set_expectations()
      pprint("gamma, n")
      pprint((gamma, n))
      pprint((gamma.shape, n.shape))
      self.maximize(gamma, n)
      pprint("no convergence" + str(i))
      i += 1

  def set_expectations(self):
    gamma = np.zeros((len(self.items), len(self.gaussians)), dtype='float16')
    n = np.zeros(len(self.gaussians), dtype='float16')

    for i in range(len(self.items)):
      densities_sum = self.density(self.items[i])

      for j in range(len(self.gaussians)):
        gamma[i,j] = self.gaussian_weights[j] * \
                     self.one_gaussian_density(self.items[i], self.gaussians[j]) \
                     / densities_sum

    for j in range(len(self.gaussians)):
      n[j] = sum(gamma[:,j])

    return gamma, n


  def maximize(self, gamma, n):
    new_mus = np.zeros(len(self.gaussians), dtype='float16')
    new_sigmas = []

    y = self.items #TODO??

    for j in range(len(self.gaussians)):
      pprint(j)
      self.gaussian_weights[j] = n[j] / len(self.items)

      #print("gamma, y, n")
      #pprint((gamma.shape, y.shape, n.shape))
      #pprint((gamma[:,j], y, n[j]))
      #pprint((gamma[:,j].shape, y, n[j].shape))
      #pprint(np.dot(gamma[:,j], y) / n[j])
      #pprint(np.sum(np.dot(gamma[:,j], y)) / n[j])
      new_mus[j] = np.sum(np.dot(gamma[:,j], y)) / n[j]

      new_sigma = np.zeros(self.gaussians[0][1].shape, dtype='float16')
                  #same shape as prev sigmas
      for i in range(len(self.items)):
        #difference = y[i] - new_mus
        difference = y - new_mus[j]
        #new_sigmas[j] = np.dot(gamma[i,j], difference * difference.T) / n[j]
        #pprint((gamma[i,j], difference, difference.T))
        #pprint((gamma[i,j].shape, difference.shape, difference.T.shape))

        new_sigma += gamma[i,j] * difference.T * difference
        new_sigma = new_sigma / n[j]
      new_sigmas.append(new_sigma)

    self.gaussians = zip(new_mus, new_sigmas)

    pprint("new gaussians")
    pprint(self.gaussians)

  def convergence(self):
    new_likelyhood = self.likelyhood()

    if abs(self.last_likelyhood - new_likelyhood) < STOP:
      self.last_likelyhood = new_likelyhood
      return True
    else:
      self.last_likelyhood = new_likelyhood
      return False


  def likelyhood(self):
    densities = [self.density(item) for item in self.items]
    pprint(np.array(densities))
    densities_logged = filter(lambda x: math.log(x, math.e), densities)
    likelyhood =  sum(densities_logged) / len(self.items)

    pprint("likelyhood")
    pprint(likelyhood)

    return likelyhood


  # density of x for a particulr gaussian, no weighting
  def one_gaussian_density(self, x, gaussian):
    mu = gaussian[0]
    sigma = gaussian[1]
    dimension = len(mu)

    x = x.A1

    #pprint(x)
    #pprint(mu)
    #pprint(sigma)
    #pprint(-1/2 * (x - mu).T * np.linalg.pinv(sigma) * (x - mu))

    a_v = -1/2 * (x - mu).T * np.linalg.pinv(sigma) * (x - mu)
    b_v = linalg.expm(a_v)
    c_v = (2 * math.pi) ** dimension * np.linalg.det(sigma)
    pprint("det")
    pprint(np.linalg.det(sigma))
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
    #pprint(self.gaussian_weights)
    #pprint(np.inner(self.gaussian_weights, densities_per_gaussian))

    return np.inner(self.gaussian_weights, densities_per_gaussian)


def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
                   delimiter=','))).astype('float16')


import unittest
class TestGDA(unittest.TestCase):

  def test_initial_em_step(self):
    # three gaussians in 2 dimensions

    gmm = GaussianMixtureModel(3)

    filename = data_dir3 + "3gaussian.txt"
    data = read_csv_as_numpy_matrix(filename)

    gmm.last_likelyhood = float("-inf")
    gmm.items = data
    gmm.gaussian_weights = [33.333333333333336, 33.333333333333336, 33.333333333333336]
    gmm.gaussians = [(np.array([ 0.,  0.], dtype='float16'),
                      np.array([[ 0.88252301,  0.54174647],
                             [ 0.36648134,  0.5903604 ]])),
                     (np.array([ 0.,  0.], dtype='float16'),
                      np.array([[ 0.57356298,  0.1492326 ],
                             [ 0.41697418,  0.86501807]])),
                     (np.array([ 0.,  0.], dtype='float16'),
                      np.array([[ 0.3101249 ,  0.39345224],
                             [ 0.72807404,  0.9316527 ]]))]

    self.assertFalse(gmm.convergence())
    gamma, n = gmm.set_expectations()
    pprint((gamma, n))
    gmm.maximize(gamma, n)

    self.assertTrue(False)


data_dir3 = "../../data/HW3/"

def two_gaussians():
  filename = data_dir3 + "2gaussian.txt"
  data = read_csv_as_numpy_matrix(filename)

  gmm = GaussianMixtureModel(2)
  gmm.train(data)

def three_gaussians():
  filename = data_dir3 + "3gaussian.txt"
  data = read_csv_as_numpy_matrix(filename)

  gmm = GaussianMixtureModel(3)
  gmm.train(data)

if __name__ == "__main__":
  #two_gaussians()
  three_gaussians()
