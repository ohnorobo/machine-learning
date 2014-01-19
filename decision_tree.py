#!/usr/bin/python
import numpy as np
import unittest
import csv
from pprint import pprint


threshold = 3

class DecisionTreeNode:

  def __init__(self, features, truths):
    #pprint("features, init")
    #pprint(features)
    #pprint(truths)

    improvements, splits = self.all_splits(features, truths)

    best_improvement = max(improvements)
    self.featureindex = improvements.index(best_improvement)
    self.split = splits[self.featureindex]

    left_features, right_features, left_truth, right_truth =\
        self.split_features(features, truths)

    if best_improvement > threshold and len(left_truth) > 0 and len(right_truth) > 0:
      self.left = DecisionTreeNode(left_features, left_truth)
      self.right = DecisionTreeNode(right_features, right_truth)
    else:
      self.left = DecisionTreeLeaf(features, truths)
      self.right = DecisionTreeLeaf(features, truths)

  #returns the best split for each feature, and the improvement from each
  def all_splits(self, features, truths):
    improvements = []
    splits = []

    for feature in features.T:
      improvement, split = self.find_split(feature, truths)

      improvements.append(improvement)
      splits.append(split)

    #pprint("improvements")
    #pprint(improvements)
    return improvements, splits

  def find_split(self, feature, truths):
    split = lambda x: x < np.mean(feature)
    improved = len(truths)
    return improved, split

  def calculate_improvement(self, feature, truths, split):
    left_length = len(filter(split, feature))
    total_length = len(truths)
    right_length = total_length - left_length

    #more 'even' splits are preferred
    return min(left_length, right_length)

  def split_features(self, features, truths):
    left_set = []
    right_set = []
    left_truths = []
    right_truths = []

    for item, truth in zip(features, truths):
      item = item.A1
      if self.split(item[self.featureindex]):
        left_set.append(item)
        left_truths.append(truth)
      else:
        right_set.append(item)
        right_truths.append(truth)

    return np.matrix(left_set), np.matrix(right_set), left_truths, right_truths

  def classify(self, item):
    if self.split(item[self.featureindex]):
      return self.left.classify(item)
    else:
      return self.right.classify(item)

  def classify_all(self, items):
    return map(lambda x: self.classify(x.A1), items)


class DecisionTreeLeaf:

  def __init__(self, features, truths):
    self.solution = np.mean(truths)

  def classify(self, item):
    return self.solution


def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

def mean_squared_error(guesses, truths):
  error = 0
  for guess, truth in zip(guesses, truths):
    error += pow(abs(guess - truth),2)
  return error / truth.size


data_dir = "../../data/HW1/"
class TestLinearReg(unittest.TestCase):

  def test_simple_case(self):
    data = np.matrix('3 3 -2; 5 0 3; 4 4 4')
    features = data[:,1:]
    truth = data[:,0].A1

    #model = DecisionTreeNode(features, truth)
    #self.assertEqual(model.classify(np.array([3, -2])), 3)
    #self.assertEqual(model.classify(np.array([0, 3])), 5)
    #self.assertEqual(model.classify(np.array([4, 4])), 4)

  def test_housing(self):
    housing_train_filename = data_dir + "housing/housing_train.txt"
    housing_test_filename = data_dir + "housing/housing_test.txt"

    train_data = read_csv_as_numpy_matrix(housing_train_filename)
    test_data = read_csv_as_numpy_matrix(housing_test_filename)

    features = train_data[:,1:]
    truth = train_data[:,0].A1

    model = DecisionTreeNode(features, truth)

    features = test_data[:,1:]
    truth = test_data[:,0].A1
    guesses = model.classify_all(features)
    #pprint(zip(guesses, truth))
    pprint("MSE housing")
    pprint(mean_squared_error(guesses, truth))

    self.assertTrue(False)

  def test_spam(self):
    spam_filename = data_dir + "spambase/spambase.data"
    data = read_csv_as_numpy_matrix(spam_filename)

    features = data[:,1:]
    truth = data[:,0].A1

    model = DecisionTreeNode(features, truth)

    guesses = model.classify_all(features)
    #pprint(zip(guesses, truth))
    pprint("MSE spam")
    pprint(mean_squared_error(guesses, truth))

    self.assertTrue(False)

