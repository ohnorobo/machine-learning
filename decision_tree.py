#!/usr/bin/python
import numpy as np
import unittest
import csv
import math
from pprint import pprint


THRESHHOLD = 0 #when to stop splitting
steps = 10.0 #how many different split thresholds to try for each feature

class DecisionTreeNode:
  # self.featureindex  #int, index of the feature to split on at this node
  # self.split         #float, threshold to split the feature at
  # self.left          #DecisionTreeNode/Leaf, left child
  # self.right         #DecisionTreeNode/Leaf, right child
  #
  # if the feature at featureindex is < split we go to the left
  # if it is >= we go to the right

  def __init__(self, features, truths):
    #if not sorted_features:
    #  sorted_features = sort_columns(features)

    improvements, splits = self.all_splits(features, truths)

    #pprint("splits / improvements")
    #featureindex, split, improvement, min-max
    #pprint([(x[0], x[1][0], x[1][1], minmax(x[1][2]))
    #       for x in enumerate(zip(splits, improvements, features.T))])

    best_improvement = max(improvements)
    self.featureindex = improvements.index(best_improvement)
    self.split = splits[self.featureindex]

    #pprint({"improvement":best_improvement,
    #        "index":self.featureindex,
    #        "split":self.split})

    left_features, right_features, left_truth, right_truth =\
        self.split_features(features, truths)

    if best_improvement > THRESHHOLD and len(left_truth) > 0 and len(right_truth) > 0:
      self.left = DecisionTreeNode(left_features, left_truth)
      self.right = DecisionTreeNode(right_features, right_truth)
    else:
      self.left = DecisionTreeLeaf(features, truths)
      self.right = DecisionTreeLeaf(features, truths)

  def structure(self):
    return {"0s": self.split,
            "1f": self.featureindex,
            "2l": self.left.structure(),
            "3r": self.right.structure()}

  #returns the best split for each feature, and the improvement from each
  def all_splits(self, features, truths):
    improvements = []
    splits = []

    i = 0
    for feature in features.T:
      #pprint("feature: " + str(i))
      i += 1
      improvement, split = self.find_split(feature, truths)

      improvements.append(improvement)
      splits.append(split)

    return improvements, splits

  #finds the best splitting function for a particular feature
  @staticmethod
  def find_split(feature, truths):
    feature = feature.A1

    possible_splits = zip(feature, truths)
    possible_splits.sort(key=lambda x: x[0]) #sort by feature value

    improvements = []

    left_sum = 0
    right_sum = sum(truths)
    left_count = 0
    right_count = len(truths)
    prev_feature = None

    for i in xrange(len(possible_splits)):
      curr_feature = possible_splits[i][0]
      truth = possible_splits[i][1]

      if i > 0:
        next_feature = possible_splits[i-1][0]

      if left_count == 0 or right_count == 0:
        improvements.append(0) #no improvement from splitting at an edge
      elif curr_feature == next_feature:
        improvements.append(0) #skip duplicate features
      else:
        mean_difference = abs(left_sum/left_count - right_sum/right_count)
        entropy = DecisionTreeNode.calculate_entropy(left_count, right_count)
        improvement = entropy * mean_difference
        improvements.append(improvement)

      left_count += 1
      right_count -= 1
      left_sum += truth
      right_sum -= truth

    best_improvement = max(improvements)
    best_index = improvements.index(best_improvement)
    best_split = possible_splits[best_index][0]

    #pprint(possible_splits)
    #pprint("best improvement index and split")
    #pprint(best_improvement)
    #pprint(best_index)
    #pprint(best_split)

    return best_improvement, best_split

  @staticmethod
  def calculate_entropy(left_len, right_len):
    total = 1.0 + left_len + right_len
    pl = left_len / total
    pr = right_len / total
    return pl * math.log(pl,2) * pr * math.log(pr,2)

  #calculate the improvement on a given feature with a given split
  @staticmethod
  def calculate_improvement(feature, truths, split):
    pass

  #split features and truths for the child nodes
  def split_features(self, features, truths):
    left_set = []
    right_set = []
    left_truths = []
    right_truths = []

    for item, truth in zip(features, truths):
      item = item.A1
      #pprint(item)
      #pprint(item[self.featureindex])
      #pprint(self.split)
      if item[self.featureindex] < self.split:
        left_set.append(item)
        left_truths.append(truth)
      else:
        right_set.append(item)
        right_truths.append(truth)

    return np.matrix(left_set), np.matrix(right_set), left_truths, right_truths

  def classify(self, item):
    if item[self.featureindex] < self.split:
      return self.left.classify(item)
    else:
      return self.right.classify(item)

  def classify_all(self, items):
    return map(lambda x: self.classify(x.A1), items)


class DecisionTreeLeaf:

  def __init__(self, features, truths):
    #pprint(truths)
    self.solution = np.mean(truths)

  def classify(self, item):
    return self.solution

  def structure(self):
    return {"s":self.solution}


def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

def mean_squared_error(guesses, truths):
  error = 0
  for guess, truth in zip(guesses, truths):
    error += pow(abs(guess - truth),2)
  return error / len(truths)

def mean(list):
  return sum(list) / len(list)

def sum_count(item, accum, split):
  #pprint("item")
  #pprint(item)
  if item[0] < split:
    accum["left_sum"] += item[1]
    accum["left_count"] += 1
  else:
    accum["right_sum"] += item[1]
    accum["right_count"] += 1
  return accum

def minmax(list):
  return (round(min(list.A1),2), round(max(list.A1),2))

#given a matrix return a list of arrays with the same elements, but each column sorted ascending
def sort_columns(m):
  sorted_columns = []
  for column in m:
    sorted_column = column.A1.sort()
    sorted_columns.append(sorted_column)

  #pprint(sorted_columns)
  return sorted_columns



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

  def test_split(self):
    data = np.matrix('0 1; 0 2; 1 3')
    features = data[:,1:]
    truth = data[:,0].A1

    model = DecisionTreeNode(features, truth)

    pprint("testing find split")
    improvement, split = model.find_split(data[:,1], truth)

    self.assertEqual(split, 3)

  def test_split_2(self):
    feature = np.matrix([0,0,0,1])
    truth = [0,0,1,1]

    pprint("testing find split")
    improvement, split = DecisionTreeNode.find_split(feature, truth)

    self.assertEqual(split, 1)

  def test_split_long(self):
    feature = np.matrix([0, 0, 0, 0, 1, 2])
    truth = [0, 0, 1, 0, 1, 1]
    imp, split = DecisionTreeNode.find_split(feature, truth)

    self.assertEqual(split, 1)
    self.assertEqual(imp, 1)



def test_housing():
  global THRESHHOLD
  THRESHHOLD = 0.01

  housing_train_filename = data_dir + "housing/housing_train.txt"
  housing_test_filename = data_dir + "housing/housing_test.txt"

  train_data = read_csv_as_numpy_matrix(housing_train_filename)
  test_data = read_csv_as_numpy_matrix(housing_test_filename)

  features = train_data[:,:12]
  truth = train_data[:,13].A1

  model = DecisionTreeNode(features, truth)

  features = test_data[:,:12]
  truth = test_data[:,13].A1
  guesses = model.classify_all(features)
  #pprint(zip(guesses, truth))

  #pprint(model.structure())
  pprint("MSE housing")
  pprint(mean_squared_error(guesses, truth))




def test_spam():
  global THRESHHOLD
  THRESHHOLD = 0.5
  spam_filename = data_dir + "spambase/spambase.data"
  data = read_csv_as_numpy_matrix(spam_filename)
  train = data[:4000,:]
  test = data[4001:,:]

  features = train[:,:56]
  truth = train[:,57].A1

  model = DecisionTreeNode(features, truth)

  features = test[:,:56]
  truth = test[:,57].A1

  guesses = model.classify_all(features)
  #pprint(sorted(zip(guesses, truth)))

  #pprint(model.structure())
  pprint("MSE spam")
  pprint(mean_squared_error(guesses, truth))



if __name__ == "__main__":
  test_housing()
  test_spam()
