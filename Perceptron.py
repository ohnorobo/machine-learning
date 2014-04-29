#!/usr/bin/python
import numpy as np
from pprint import pprint
import csv
import math, time
import random

LEARNING_RATE = 1
num_iterations = 1000000

def get_perceptron(features, truth):
  w = np.zeros(features.shape[1])
  #w[-1] = 1
  w = np.matrix(w)

  for i in range(num_iterations):
    misclassified_points = 0

    for index, point in enumerate(features):
      if np.inner(point.A1, w.A1) <= 0.0:
        misclassified_points += 1
        #pprint(("adding point " + str(index), np.inner(point.A1, w.A1), point))
        w = np.matrix(w.A1 + point.A1)

    pprint("iteration: " + str(i))
    pprint("num_misclassified: " + str(misclassified_points))
    if misclassified_points == 0:
      pprint("final w")
      pprint(w.A1)
      return



#takes a matrix
# for every row where the last values is -1
# multiply entire row by -1
def flip(old_X):
  X = old_X.copy()
  pprint(X.shape)

  number_flipped = 0

  for i in range(X.shape[0]):
    if X[i,-1] == -1:
      number_flipped += 1
      X[i,:] = X[i,:] * -1
  pprint("number flipped: " + str(number_flipped))
  return X

def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
                   delimiter=','))).astype('float')

import unittest

data_dir = "./data/"
class TestLinearReg(unittest.TestCase):

  def test_flip(self):
    data = np.matrix('1 1 -1; 1 1 1')
    new_data = flip(data)
    self.assertEqual(new_data[0,2], 1)
    self.assertEqual(new_data[1,2], 1)
    self.assertEqual(new_data[0,0], -1)
    self.assertEqual(new_data[1,0], 1)

  def test_inner(self):
    x = [ -0.3852046 ,  -0.18301087,  -0.54516589,  -0.59832594, 1.        ]
    w = [ 0.57642699,  0.23646118,  0.3197695 ,  0.19114307,  2.        ]

    pprint(inner(x, w))

    self.assertTrue(False)

  def test_final_w(self):
    w = np.array([-0.05679759, -0.02521043, -0.01362577, -0.00960582,  2.0])

    spam_filename = data_dir + "perceptron.txt"
    data = read_csv_as_numpy_matrix(spam_filename)
    data = flip(data)

    for point in data:
      self.assertTrue(inner(point.A1, w) > 0)
      if inner(point.A1, w) <= 0:
        #print failing points
        pprint(inner(point.A1, w))
        pprint(point.A1)

    self.assertTrue(False)

  def test_all(self):
    spam_filename = data_dir + "perceptron.txt"
    data = read_csv_as_numpy_matrix(spam_filename)
    data = flip(data)
    features = data[:,:4]
    features = np.hstack((features,
                          np.matrix(np.ones(features.shape[0])).T))

    w = np.array([-0.05679759, -0.02521043, -0.01362577, -0.00960582,  2.0])

    wrong_count = 0

    for i, point in enumerate(features):
      if inner(point.A1, w) <= 0:
        pprint((i, inner(point.A1, w), point.A1))
        wrong_count += 1

    pprint(wrong_count)
    self.assertTrue(False)





def test_perceptron():
  spam_filename = data_dir + "perceptron.txt"
  data = read_csv_as_numpy_matrix(spam_filename)
  pprint(data)

  features = data[:,:4]
  truth = data[:,4]
  #add in bias
  features = np.hstack((features,
                        np.matrix(np.ones(features.shape[0])).T))

  data = np.hstack((features, truth))
  data = flip(data)
  pprint(data)

  #pprint("data")
  #pprint(data)
  features = data[:,:5]
  truth = data[:,5]

  plane = get_perceptron(features, truth)

#add bias before flipping hyperplane

if __name__ == "__main__":
  test_perceptron()
