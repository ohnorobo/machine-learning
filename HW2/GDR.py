#!/usr/bin/python
import numpy as np
from pprint import pprint
import csv, time

LEARNING_RATE = .00000001

def get_gd_function(X, Y):
  # add a column of 1s
  X_with_bias = np.hstack((X, np.matrix(np.ones(X.shape[0])).T))
  X = X_with_bias

  ws = np.matrix(np.ones(X.shape[1]))
  #ws = np.random.rand(X.shape[1])
  #find ws by grad
  iterations = 1

  while True:

    #pprint("old")
    #pprint(ws)

    new_ws = np.array([0]*len(ws))

    deltas = derivatives(ws, X, Y)
    new_ws = ws - ( LEARNING_RATE * iterations * deltas )
    #new_ws = ws - LEARNING_RATE * deltas
    ws = new_ws

    pprint("error")
    pprint(least_squares_error(lambda x: np.dot(x, ws.A1), X, Y))
    #time.sleep(1)

    #iterations += .0005

    if stop(deltas):
      break

  pprint(ws)

  return lambda x: np.dot(x, ws.A1)

STOP_THRESH = 10
def stop(deltas):
  magnitude = np.linalg.norm(deltas)
  return magnitude < STOP_THRESH


#get the derivative of function <ws . x> around x
def derivatives(ws, x, y):
  # derivative w/ respect to x0, x1, ... xn
  #all = np.vstack((x, ws))
  #pprint("all")
  #pprint(all)
  #pprint("grad")
  #pprint(np.gradient(all)[0])
  #return np.gradient(all)[0][:,-1]
  #pprint(x.shape)
  #pprint(ws.shape)
  #for w, xi, yi in zip(ws, x.T, y):
  #  deltas.append(((np.dot(ws, xi.A1)) - yi) * xi)
  hyp = np.dot(x, ws.A1)
  #pprint("hyp")
  #pprint(hyp)
  #pprint(y)
  loss = hyp - y.T
  #pprint("loss")
  #pprint(loss)

  grad = np.dot(loss, x)

  #pprint("grad")
  #pprint(grad.A1)

  return grad.A1

def least_squares_error(regression, features, truths):
  error = 0

  for i in range(0,len(truths)):
    item = features[i,:].A1
    truth = truths.A1[i]
    error += pow(abs(truth - regression(item)), 2)
  return error / truth.size

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
  pprint(error)

def test_spam():
  spam_filename = data_dir + "spambase/spambase.data"
  data = normalize_data(read_csv_as_numpy_matrix(spam_filename))

  features = data[:,:56]
  truth = data[:,57]
  regression = get_gd_function(features, truth)

  error = least_squares_error(regression, features, truth)

  pprint("MSE spam")
  pprint(error)

if __name__ == "__main__":
  test_housing()
  #test_spam()
