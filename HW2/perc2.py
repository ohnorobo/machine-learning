#!/usr/bin/python
import numpy as np
from pprint import pprint
import csv
import math, time
import random

LEARNING_RATE = 1
num_iterations = 100

def get_perceptron(features, truth):
  features_with_bias = np.hstack((features,
                                  np.matrix(np.ones(features.shape[0])).T))
  features = features_with_bias

  w = np.zeros(features.shape[1])
  #w[-1] = 1
  w = np.matrix(w)

  for i in range(num_iterations):
    misclassified_points = 0

    for point in features:
      if inner(point.A1, w.A1) <= 0.0:
        #pprint("misclassed_point")
        #pprint(point.A1)
        #pprint("at")
        #pprint(w.A1)
        #pprint("inner")
        #pprint(inner(point.A1, w.A1))
        misclassified_points += 1
        w = np.matrix(w.A1 + point.A1)
      else:
        pass

    pprint("iteration: " + str(i))
    pprint("num_misclassified: " + str(misclassified_points))
    if misclassified_points == 0:
      pprint("final w")
      pprint(w.A1)
      return


def inner(xs, ys):
  sum = 0.0
  for x, y in zip(xs, ys):
    sum += x*y
  #pprint("inner: " + str(sum) + " " + str(xs))
  return sum



#takes a matrix
# for every row where the last values is -1
# multiply entire row by -1
def flip(old_X):
  X = old_X.copy()
  pprint(X.shape)

  for i in range(X.shape[0]):
    if X[i,-1] == -1:
      X[i,:] = X[i,:] * -1
  return X

def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
                   delimiter=','))).astype('float')

def normalize_data(data):
  a = data.T
  row_sums = a.sum(axis=1)
  row_mins = a.min(axis=1)
  new_matrix = np.zeros(a.shape)
  for i, (row, row_min, row_sum) in enumerate(zip(a, row_mins, row_sums)):
        new_matrix[i,:] = (row + row_min) / row_sum

  return new_matrix.T

import unittest

data_dir = "../../data/HW2/"
class TestLinearReg(unittest.TestCase):

  def test_flip(self):
    data = np.matrix('1 1 -1; 1 1 1')
    new_data = flip(data)
    self.assertEqual(new_data[0,2], 1)
    self.assertEqual(new_data[1,2], 1)
    self.assertEqual(new_data[0,0], -1)
    self.assertEqual(new_data[1,0], 1)

  def test_final_w(self):
    w = np.array([-0.05679759, -0.02521043, -0.01362577, -0.00960582,  1.0])

    spam_filename = data_dir + "perceptron.txt"
    data = read_csv_as_numpy_matrix(spam_filename)
    data = flip(data)

    for point in data:
      self.assertTrue(inner(point.A1, w) > 0)
      if inner(point.A1, w) <= 0:
        #print failing points
        pprint(inner(point.A1, w))
        pprint(point.A1)

    #self.assertTrue(False)



def test_perceptron():
  spam_filename = data_dir + "perceptron.txt"
  data = read_csv_as_numpy_matrix(spam_filename)
  data = flip(data)
  #data = normalize_data(data)

  pprint("data")
  pprint(data)

  features = data[:,:4]
  truth = data[:,4]
  plane = get_perceptron(features, truth)


if __name__ == "__main__":
  test_perceptron()
