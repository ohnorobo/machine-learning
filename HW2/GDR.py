#!/usr/bin/python
import numpy as np
from pprint import pprint
import csv

LEARNING_RATE = .5

def get_gd_function(X, Y):
  ws = np.random.random(X.shape[1])
  #find ws by grad

  while True:

    pprint("old")
    pprint(ws)

    new_ws = np.array([0]*len(ws))

    for i in range(len(ws)):
      deltas = derivatives(ws, X)
      new_ws[i] = ws[i] - LEARNING_RATE * deltas[i]
      ws = new_ws
      pprint(ws)

    if stop(deltas):
      break

  pprint(ws)

  return lambda x: np.dot(x, ws)

STOP_THRESH = 10
def stop(deltas):
  magnitude = np.linalg.norm(deltas)
  return magnitude < STOP_THRESH


#get the derivative of function <ws . x> around x
def derivatives(ws, x):
  # derivative w/ respect to x0, x1, ... xn
  all = np.vstack((x, ws))
  pprint("all")
  pprint(all)
  pprint("grad")
  pprint(np.gradient(all)[0])
  return np.gradient(all)[0][:,-1]

def least_squares_error(regression, features, truths):
  error = 0

  for i in range(0,len(truths)):
    item = features[i,:].A1
    truth = truths.A1[i]
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
  data = read_csv_as_numpy_matrix(spam_filename)

  features = data[:,:56]
  truth = data[:,57]
  regression = get_gd_function(features, truth)

  error = least_squares_error(regression, features, truth)

  pprint("MSE spam")
  pprint(error / truth.size)

if __name__ == "__main__":
  test_housing()
  #test_spam()
