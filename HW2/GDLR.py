#!/usr/bin/python
import numpy as np
from pprint import pprint
import csv
import math

LEARNING_RATE = 0.1
THRESHHOLD = 0.01

def get_gd_function(X, Y):
  thetas = np.random.random(X.shape[1])
  #find thetas by grad

  round = 0

  while True:

    pprint("round: " + str(round))
    #pprint(thetas)

    new_thetas = np.zeros(len(thetas))

    for i in range(len(thetas)):
      new_thetas[i] = thetas[i] - LEARNING_RATE * delta(X, Y, thetas, i)

    #if sum(map(abs, np.setdiff1d(thetas, new_thetas))) < THRESHHOLD:
    if sum(np.setdiff1d(thetas, new_thetas)) < THRESHHOLD:
      break
    else:
      #pprint("diff sum: " + str(sum(np.setdiff1d(thetas, new_thetas))))
      error = least_squares_error(lambda x: np.dot(x, thetas), X, Y)
      pprint("error: " + str(error/Y.size))

    thetas = new_thetas
    round += 1

  pprint("final")
  pprint(thetas)

  return lambda x: np.dot(x, thetas)

STOP_THRESH = 10
def stop(deltas):
  magnitude = np.linalg.norm(deltas)
  return magnitude < STOP_THRESH

def delta(X, Y, thetas, i):
  xs = X[i,:]
  theta = thetas[i]

  sum = 0
  for x, y in zip(xs, Y):
    sum += (h(theta, x) - y) * x

  return sum

def h(theta, x):
  return 1 / (1 + pow(math.e, -1 * theta * x))


def least_squares_error(regression, features, truths):
  error = 0

  for i in range(0,len(truths)):
    item = features[i,:]
    truth = truths[i]
    error += pow(abs(truth - regression(item)), 2)
  return error

def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
                   delimiter=','))).astype('float')

def guess_all(regression, features):
  guesses = []
  for item in features:
    guesses.append(regression(item.A1))
  return guesses

def normalize_columns(data):
  a = data.T
  row_maxes = a.max(axis=1)
  new_matrix = np.zeros(a.shape)
  for i, (row, row_max) in enumerate(zip(a, row_maxes)):
        new_matrix[i,:] = row / row_max

  return new_matrix.T

import unittest

data_dir = "../../data/HW1/"
class TestLinearReg(unittest.TestCase):

  def test_simple_case(self):
    data = np.matrix('3 3 -2; 5 0 3; 4 4 4')
    regression = get_gd_function(data[:,1:], data[:,0])
    error = least_squares_error(regression, data[:,1:], data[:,0])

def test_housing():
  housing_train_filename = data_dir + "housing/housing_train.txt"
  housing_test_filename = data_dir + "housing/housing_test.txt"

  train_data = read_csv_as_numpy_matrix(housing_train_filename)
  test_data = read_csv_as_numpy_matrix(housing_test_filename)

  all_data = np.vstack((train_data, test_data))
  all_data = normalize_columns(all_data)
  train_data = all_data[:433,:]
  test_data = all_data[433:,:]

  features = train_data[:,:12]
  truth = train_data[:,13]
  regression = get_gd_function(features, truth)

  features = test_data[:,:12]
  truth = test_data[:,13]
  error = least_squares_error(regression, features, truth)

  pprint("MSE housing")
  pprint(error / truth.size)

def test_spam():
  spam_filename = data_dir + "spambase/spambase.data"
  data = normalize_columns(read_csv_as_numpy_matrix(spam_filename))

  pprint(data)

  features = data[:,:56]
  truth = data[:,57]
  regression = get_gd_function(features, truth)

  error = least_squares_error(regression, features, truth)

  pprint("MSE spam")
  pprint(error / truth.size)

if __name__ == "__main__":
  #test_housing()
  test_spam()
