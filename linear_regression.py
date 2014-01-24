#!/usr/bin/python
import numpy as np
from pprint import pprint
import inspect
import csv

def get_linear_reg_function(X, Y):
  thetas = (X.T * X).I * X.T * Y
  #pprint("thetas")
  #pprint(thetas.A1)
  return lambda x: np.dot(x, thetas.A1)

def least_squares_error(regression, features, truths):
  error = 0

  for i in range(0,len(truths)):
    item = features[i,:].A1
    truth = truths.A1[i]

    #pprint((truth, regression(item)))

    error += pow(abs(truth - regression(item)), 2)

  return error



def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

def mean_squared_error(guesses, truths):
  error = 0
  for guess, truth in zip(guesses, truths):
    error += pow(abs(guess - truth),2)
  return error / len(truths)


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
    regression = get_linear_reg_function(data[:,1:], data[:,0])
    error = least_squares_error(regression, data[:,1:], data[:,0])

  def test_housing(self):
    housing_train_filename = data_dir + "housing/housing_train.txt"
    housing_test_filename = data_dir + "housing/housing_test.txt"

    train_data = read_csv_as_numpy_matrix(housing_train_filename)
    test_data = read_csv_as_numpy_matrix(housing_test_filename)

    features = train_data[:,:12]
    truth = train_data[:,13]
    regression = get_linear_reg_function(features, truth)

    features = test_data[:,:12]
    truth = test_data[:,13]
    error = least_squares_error(regression, features, truth)

    pprint("MSE housing")
    pprint(error / truth.size)

    self.assertTrue(False)

  def test_spam(self):
    spam_filename = data_dir + "spambase/spambase.data"
    data = read_csv_as_numpy_matrix(spam_filename)

    features = data[:,:56]
    truth = data[:,57]
    regression = get_linear_reg_function(features, truth)

    error = least_squares_error(regression, features, truth)

    pprint("MSE spam")
    pprint(error / truth.size)

    self.assertTrue(False)
