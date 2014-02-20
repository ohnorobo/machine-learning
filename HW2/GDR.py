#!/usr/bin/python
import numpy as np
from pprint import pprint
import csv, time

LEARNING_RATE = .00001
STOP = 10

def get_gd_function(X, Y):
  # add a column of 1s
  X_with_bias = np.hstack((X, np.matrix(np.ones(X.shape[0])).T))
  X = X_with_bias

  ws = np.matrix(np.ones(X.shape[1]))
  #ws = np.random.rand(X.shape[1])
  #find ws by grad
  iterations = 1

  while True:

    new_ws = np.array([0]*len(ws))

    deltas = derivatives(ws, X, Y)
    new_ws = ws - ( LEARNING_RATE * iterations * deltas )
    ws = new_ws

    pprint("error")
    pprint(least_squares_error_no_bias(ws.A1, X, Y))
    #time.sleep(1)

    #pprint(deltas)

    if np.absolute(deltas).mean() < STOP:
      break

  pprint(ws)

  return ws.A1

STOP_THRESH = 10
def stop(deltas):
  magnitude = np.linalg.norm(deltas)
  return magnitude < STOP_THRESH


#get the derivative of function <ws . x> around x
def derivatives(ws, x, y):
  hyp = np.dot(x, ws.A1)
  loss = hyp - y.T
  grad = np.dot(loss, x)

  return grad.A1

def least_squares_error(regression_weights, features, truths):
  error = 0

  for i in range(0,len(truths)):
    item = np.append(features[i,:], 1) #add in bias term
    truth = truths[i]

    #pprint({"truth": truth, "guess": np.inner(regression_weights, item)})
    #pprint(item)

    error += abs(truth - np.inner(regression_weights, item)) ** 2
  return error / truths.size

def least_squares_error_no_bias(regression_weights, features, truths):
  error = 0
  for i in range(0,len(truths)):
    item = features[i,:]
    truth = truths[i]
    #pprint({"truth": truth, "guess": np.inner(regression_weights, item)})
    error += abs(truth - np.inner(regression_weights, item)) ** 2
  return error / truths.size

def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
                   delimiter=','))).astype('float')

def guess_all(regression, features):
  guesses = []
  for item in features:
    guesses.append(regression(item.A1))
  return guesses

def normalize_data(data):
  a = data.T
  row_maxs = a.max(axis=1)
  row_mins = a.min(axis=1)
  new_matrix = np.zeros(a.shape)
  for i, (row, row_min, row_max) in enumerate(zip(a, row_mins, row_maxs)):
    if row_min < 0:
      new_matrix[i,:] = (row + row_min) / (row_max + row_min)
    else:
      new_matrix[i,:] = row / row_max
  return new_matrix.T

def standardize_data(data):
  a = data.T
  new_matrix = np.zeros(a.shape)
  truth_column_index = a.shape[0] - 1 #don't normalize labels

  mus = a.mean(axis=1)
  sigmas = a.std(axis=1)
  for i, (a, mu, sigma) in enumerate(zip(a, mus, sigmas)):
    #pprint(i)
    if i != truth_column_index:
      new_matrix[i,:] = (a - mu) / sigma
    else:
      new_matrix[i,:] = a
  return new_matrix.T


import unittest

class TestLinearReg(unittest.TestCase):

  def test_simple_case(self):
    data = np.matrix('3 3 -2; 5 0 3; 4 4 4')
    regression = get_gd_function(data[:,1:], data[:,0])
    error = least_squares_error(regression, data[:,1:], data[:,0])


data_dir = "../../data/HW1/"
def test_housing():
  global STOP
  STOP = 10

  housing_train_filename = data_dir + "housing/housing_train.txt"
  housing_test_filename = data_dir + "housing/housing_test.txt"

  train_data = read_csv_as_numpy_matrix(housing_train_filename)
  test_data = read_csv_as_numpy_matrix(housing_test_filename)

  all_data = np.vstack((train_data, test_data))
  all_data = standardize_data(all_data)
  train_data = all_data[:433,:]
  test_data = all_data[433:,:]

  features = train_data[:,:12]
  truth = train_data[:,13]

  print(features.shape, truth.shape)

  regression = get_gd_function(features, truth)
  pprint("regression")
  pprint(regression)

  error = least_squares_error(regression, features, truth)
  pprint("MSE housing training")
  pprint(error)

  features = test_data[:,:12]
  truth = test_data[:,13]
  error = least_squares_error(regression, features, truth)
  pprint("MSE housing testing")
  pprint(error)

def test_spam():
  spam_filename = data_dir + "spambase/spambase.data"
  data = normalize_data(read_csv_as_numpy_matrix(spam_filename))

  train = data[:4000,:]
  test = data[4001:,:]

  features = train[:,:56]
  truth = train[:,57]

  print(features.shape, truth.shape)
  pprint(features)
  pprint(truth)

  regression = get_gd_function(features, truth)
  pprint("regression")
  pprint(regression)

  error = least_squares_error(regression, features, truth)
  pprint("MSE spam train")
  pprint(error)

  features = test[:,:56]
  truth = test[:,57]
  error = least_squares_error(regression, features, truth)
  pprint("MSE spam test")
  pprint(error)

PORT_DATA = "../../data/HW2/ex3Data/"
def test_portland_housing():
  x_filename = PORT_DATA + "ex3x.dat"
  y_filename = PORT_DATA + "ex3y.dat"

  features = standardize_data(read_csv_as_numpy_matrix(x_filename))
  truth = np.matrix(standardize_data(read_csv_as_numpy_matrix(y_filename)))

  print(features.shape, truth.shape)

  regression = get_gd_function(features, truth)

  pprint("regression")
  pprint(regression)

  error = least_squares_error(regression, features, truth)

  pprint("Portland Housing Error")
  pprint(error)




if __name__ == "__main__":
  test_housing()
  #test_spam()
  #test_portland_housing()
