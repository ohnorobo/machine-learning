#!/usr/bin/python
import numpy as np
import unittest
import csv
from pprint import pprint


threshold = 3

class DecisionTreeNode:

  def __init__(self, features, truths):
    pprint(features)
    pprint(truths)

    improvements, splits = self.all_splits(features, truths)

    best_improvement = max(improvements)
    self.featureindex = improvements.index(best_improvement)
    self.split = splits[self.featureindex]

    left_features, right_features, left_truth, right_truth =\
      self.split_features(self.split, self.featureindex, features, truths)

    if best_improvement > threshold: #child nodes
      self.left = DecisionTreeNode(left_features, left_truth)
      self.right = DecisionTreeNode(right_features, right_truth)
    else:
      self.left = DecisionTreeLeaf(left_features, left_truth)
      self.right = DecisionTreeLeaf(right_features, right_truth)

  #returns the best split for each feature, and the improvement from each
  def all_splits(self, features, truths):
    improvements = []
    splits = []

    for feature in features.T:
      improvement, split = self.find_split(feature, truths)

      improvements.append(improvement)
      splits.append(split)

    return improvements, splits

  def find_split(self, feature, truths):
    split = lambda x: x < mean(feature)
    improved = len(truths)
    return improved, split

  def calculate_improvement(self, feature, truths, split):
    left_length = len(filter(split, feature))
    total_length = len(truths)
    right_length = total_length - left_length

    #more 'even' splits are preferred
    return min(left_length, right_length)

  def split_features(self, split, featureindex, features, truths):
    left_set = []
    right_set = []
    left_truths = []
    right_truths = []

    for item, truth in zip(features, truths):
      item = item.A1
      pprint(item)
      pprint(featureindex)
      if split(item[featureindex]):
        left_set.append(item)
        left_truths.append(truth)
      else:
        right_set.append(item)
        right_truths.append(truth)

    return left_set, right_set, left_truths, right_truths

  def classify(item):
    result = self.decision_function(item[self.feature_index])

    if result:
      return self.left_child.classify(item)
    else:
      return self.right_child.classify(item)


class DecisionTreeLeaf:

  def __init__(self, features, truths):
    self.solution = mean(truths)

  def classify(self, item):
    return self.solution


def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

def mean(l):
  sum(l) / len(l)

data_dir = "../../data/HW1/"
class TestLinearReg(unittest.TestCase):

  '''
  def test_simple_case(self):
    data = np.matrix('3 3 -2; 5 0 3; 4 4 4')
    features = data[:,1:]
    truth = data[:,0]

    model = DecisionTreeNode(features, truth)
    self.assertTrue(model.classify(np.array([3, -2]), 3))
  '''

  def test_housing(self):
    housing_train_filename = data_dir + "housing/housing_train.txt"
    housing_test_filename = data_dir + "housing/housing_test.txt"

    train_data = read_csv_as_numpy_matrix(housing_train_filename)
    test_data = read_csv_as_numpy_matrix(housing_test_filename)

    features = train_data[:,1:]
    truth = train_data[:,0]

    model = DecisionTreeNode(features, truth)

  def test_spam(self):
    spam_filename = data_dir + "spambase/spambase.data"
    data = read_csv_as_numpy_matrix(spam_filename)

    features = data[:,1:]
    truth = data[:,0]

    model = DecisionTreeNode(features, truth)

