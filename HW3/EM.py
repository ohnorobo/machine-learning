#!/usr/bin/python
# -*- coding: utf8 -*-

import csv, math
import numpy as np
from pprint import pprint
from scipy import linalg
from scipy.stats import multivariate_normal
from sklearn.cluster import k_means

STOP = .00001
#TOO_LOW = .01
BUMP = .01 #minimum variance value allowed

class GaussianMixtureModel():

  def __init__(self, num_gaussians):
    # list of gaussians
    # each gaussian is a tuple of mu and sigma

    self.num_gaussians = num_gaussians
    self.gaussians = []
    self.gaussian_weights = []

  def train(self, data):
    self.init_given_data(data)
    self.em()

  def init_given_data(self, data):
    self.items = data
    num_features = np.matrix(data).shape[1] #turn into matrix to get at widths of 1

    all_mus = self.get_initial_clusters(data, self.num_gaussians)

    # initialize
    for i in range(self.num_gaussians):
      mu = all_mus[i]
      # try initializing by kmeans instead
      sigma = np.cov(data, rowvar=0)
      self.gaussians.append((mu, sigma))
      self.gaussian_weights.append(1.0/self.num_gaussians)

    pprint("initial weights/gaussians")
    pprint(self.gaussian_weights)
    pprint(self.gaussians)

    self.last_likelyhood = float("-inf")

  def em(self):
    i = 0

    # iterate
    while not self.convergence():
      print("iteration", i)
      gamma, n = self.set_expectations()
      #pprint("gamma, n")
      #pprint((gamma, n))
      self.maximize(gamma, n)

      self.smooth_sigma_diagonal()

      #pprint("means, gaussians")
      #pprint(self.gaussian_weights)
      #pprint(self.gaussians)
      #pprint("no convergence " + str(i))
      i += 1

    pprint("converged on iteration " + str(i))
    pprint("final means/gaussians")
    pprint(self.gaussian_weights)
    pprint(self.gaussians)

  def get_initial_clusters(self, data, num_gaussians):
    centroids, label, inertia = k_means(data, n_clusters=num_gaussians)
    return centroids

  def smooth_sigma_diagonal(self):
    diag = BUMP * np.eye(self.gaussians[0][1].shape[0])
    #pprint(("diag", diag))
    for mus, sigmas in self.gaussians:
      sigmas += diag

  def set_expectations(self):
    gamma = np.zeros((len(self.items), len(self.gaussians)))
    n = np.zeros(len(self.gaussians))

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
    new_mus = [0] * self.num_gaussians
    new_sigmas = []

    y = self.items

    for j in range(len(self.gaussians)):
      #pprint("updating gaussian #" + str(j))
      self.gaussian_weights[j] = n[j] / len(self.items)

      new_mus[j] = (np.dot(gamma[:,j], y) / n[j]).A1
      #mus should be num_gaussians by num_features

      new_sigma_j = np.zeros(self.gaussians[0][1].shape)
                  #same shape as prev sigmas
      for i in range(len(self.items)):
        difference = y[i] - new_mus[j]
        new_sigma_j += gamma[i,j] * difference.T * difference

      new_sigma_j = new_sigma_j / n[j]
      #pprint(("new sigma j" , new_sigma_j))
      new_sigmas.append(new_sigma_j)

    self.gaussians = zip(new_mus, new_sigmas)


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
    #pprint(np.array(densities))
    densities_logged = filter(lambda x: math.log(x, math.e), densities)
    likelyhood =  sum(densities_logged) / len(self.items)

    #pprint("likelyhood")
    #pprint(likelyhood)

    return likelyhood

  def one_gaussian_density(self, x, gaussian):
    return multivariate_normal.pdf(x, mean=gaussian[0], cov=gaussian[1])

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

  def dont_initial_em_step(self):
    # three gaussians in 2 dimensions

    gmm = GaussianMixtureModel(3)

    filename = data_dir3 + "3gaussian.txt"
    data = read_csv_as_numpy_matrix(filename)

    gmm.last_likelyhood = float("-inf")
    gmm.items = data
    gmm.gaussian_weights = [33.333333333333336, 33.333333333333336, 33.333333333333336]
    gmm.gaussians = [(np.array([ 0.,  0.]),
                      np.array([[ 0.88252301,  0.54174647],
                             [ 0.36648134,  0.5903604 ]])),
                     (np.array([ 0.,  0.]),
                      np.array([[ 0.57356298,  0.1492326 ],
                             [ 0.41697418,  0.86501807]])),
                     (np.array([ 0.,  0.]),
                      np.array([[ 0.3101249 ,  0.39345224],
                             [ 0.72807404,  0.9316527 ]]))]

    self.assertFalse(gmm.convergence())
    gamma, n = gmm.set_expectations()
    #pprint((gamma, n))
    gmm.maximize(gamma, n)

    self.assertTrue(False)

  def test_given_covar(self):

    gmm = GaussianMixtureModel(3)

    filename = data_dir3 + "3gaussian.txt"
    data = read_csv_as_numpy_matrix(filename)

    gmm.last_likelyhood = float("-inf")
    gmm.items = data
    gmm.gaussian_weights = [33.333333333333336, 33.333333333333336, 33.333333333333336]
    gmm.gaussians = [(np.array([ 3.0,  3.0]),
                      np.array([[ 1.0,  0.0],
                             [ 0.0,  3.0 ]])),
                     (np.array([ 7.0,  4.0]),
                      np.array([[ 1.0,  0.5 ],
                             [ 0.5,  1.0]])),
                     (np.array([ 5.0, 7.0]),
                      np.array([[ 1.0 ,  0.2],
                             [ 0.2,  1.0 ]]))]

    gmm.em()

    self.assertTrue(False)


def test_cheng():
  '''
  [ 0.23  0.32  0.45]
  means:
  [[ 3.6   3.44]
  [ 6.64  4.63]
  [ 4.99  6.74]]
  covariances:
  [[[ 2.14  0.83]
  [ 0.83  4.2 ]]
  [[ 1.7  -0.19]
  [-0.19  2.37]]
  [[ 1.43  0.07]
  [ 0.07  1.92]]]
  '''

  gmm = GaussianMixtureModel(3)

  filename = data_dir3 + "3gaussian.txt"
  data = read_csv_as_numpy_matrix(filename)

  gmm.last_likelyhood = float("-inf")
  gmm.items = data
  gmm.gaussian_weights = np.array([ 0.23,  0.32,  0.45])
  gmm.gaussians = [(np.array([ 3.6,   3.44]),
                    np.array([[ 2.14,  0.83],
                              [ 0.83,  4.2 ]])),
                   (np.array([ 6.64,  4.63]),
                    np.array([[ 1.7,  -0.19],
                              [-0.19,  2.37]])),
                   (np.array([ 4.99,  6.74]),
                    np.array([[ 1.43,  0.07],
                              [ 0.07, 1.92]]))]

  gmm.em()




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
  #test_cheng()
