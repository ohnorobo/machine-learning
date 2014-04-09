#!/usr/bin/python

import numpy as np
from pprint import pprint
import math
from copy import deepcopy


ITERATIONS = 10
SPLIT = 10 # testing data is 1/SPLIT of data

NEG = -1
POS = 1


def convert(classes, element):
  if element == classes[0] or element == 0.0:
    return NEG
  elif element == classes[1] or element == 1.0:
    return POS
  else:
    pprint(element)
    raise Exception("unclassified point")


class AdaBoost():

  def __init__(self, data, truths, feature_types):
    # data is a XxY matrix
    # truths is a 1xY matrix with discrete class values
    # feature_types is a list of
    #   - 'numeric'
    #   - or a list of labels for discrete values
    # which correspond to the data columns

    self.items = data
    self.feature_types = feature_types[:-1]
    self.truth_feature_types = feature_types[-1] # must be binary
    self.truths = self.convert_to_neg1_1s(truths, self.truth_feature_types)

    self.item_weights = np.array([1.0/len(data)]*len(self.items))

    self.classifiers = self.get_all_classifiers()
    self.classifier_weights = np.array([float('NaN')]*len(self.classifiers))
    #self.classifiers = []
    #self.classifier_weights = []

    #pprint(self.classifiers)

    self.train()

  # convert whatever weird symbols are being used for classes into 0/1
  def convert_to_neg1_1s(self, truths, feature_types):
    return map(lambda truth: convert(feature_types, truth), truths)

  def train(self):

    #pprint({"number of classifiers": len(self.classifiers)})

    # iterate through all classifiers
    #for i in range(ITERATIONS):
    for t, classifier in enumerate(self.classifiers):

      #classifier, t, error = self.choose_best_classifier_and_error()

      # compute error rate e_t
      error = self.error(classifier)
      #pprint({"error": error})

      # assign weight a_t to classifier f_t
      if error == 0:
        pprint(classifier)
        raise Exception("prefect feature, problem")
      else:
        a = math.log((1-error)/error)

      pprint({"error":error, "a":a, "classifier":classifier})
      self.classifier_weights[t] = a
      #self.classifiers.append(classifier)
      #self.classifier_weights.append(a)

      # add weight to misclassed points
      self.reweight_misclassed_points(a, classifier)
      # normalize point weights
      self.normalize_point_weights()

      #pprint(zip(self.classifiers, self.classifier_weights))
      #pprint(self.item_weights)

    a = zip(self.classifier_weights, self.classifiers)
    pprint(sorted(a, key=lambda x: abs(x[0])))

  #'''
  def choose_best_classifier(self):
    classifier_errors = [self.error(classifier) for classifier in self.classifiers]

    both = zip(classifier_errors, self.classifiers)
    both = sorted(both, key=lambda x: abs(x[0] - .5))

    pprint(both)

    classifier = both[0][1] #best classifier
    return classifier, self.classifiers.index(classifier) # and index

  def get_all_classifiers_per_feature(self, feature_index):
    feature_type = self.feature_types[feature_index]
    classifiers = []
    feature = self.items.T[feature_index].T

    if feature_type == 'numeric':
      values = self.items.T[feature_index]
      threshholds = sorted(set(values))
      for value in threshholds:
        classifiers.append(NumericDecisionStump(value, feature_index))
    else:
      for label in feature_type:
        classifiers.append(DiscreteDecisionStump(label, feature_index))
    return classifiers

  def get_all_classifiers(self):
    classifiers = []
    for i in range(len(self.feature_types)):
      c = self.get_all_classifiers_per_feature(i)
      classifiers.extend(c)
    return classifiers
  #'''

  def error(self, classifier):
    error = 0.0
    for i, (point, truth) in enumerate(zip(self.items, self.truths)):
      if self.misclassed(classifier, point, truth):
        error += self.item_weights[i]
    #pprint({"error":error, "total":sum(self.item_weights)})
    return error

  def reweight_misclassed_points(self, a, classifier):
    for i, (point, truth) in enumerate(zip(self.items, self.truths)):
      if self.misclassed(classifier, point, truth):
        self.item_weights[i] *= (math.e ** a)

  def misclassed(self, classifier, point, truth):
    guess = classifier.check(point) # 0 or 1
    #pprint({"guess": guess, "truth": truth})
    return guess == truth

  #sum weights should = 1
  def normalize_point_weights(self):
    self.item_weights *= 1.0/sum(self.item_weights)

  def choose_smallest_discriminant(self, data, n):
    # return n points with the smallest discriminant
    # and the rest of the points without

    d = sorted(data, key=lambda x: self.discriminant(x)) #TODO reverse?
    return data[:n], data[n:]

  def discriminant(self, item):
    d = 0
    for classifier, weight in zip(self.classifiers, self.classifier_weights):
      d += classifier.discriminant(item) * weight
    return d

  def classify(self, item):
    return sign(self.classify_prob(item))

  # returns a value, not really a prob
  def classify_prob(self, item):
    scores = [classifier.check(item) for classifier in self.classifiers]
    #pprint(zip(scores, self.classifier_weights))
    #pprint(np.inner(scores, self.classifier_weights))
    return np.inner(scores, self.classifier_weights)

def sign(x):
  if 0 < x:
    return 0
  if x <= 0:
    return 1

class DecisionStump():

  def __init__(self, value, value_index):
    self.value = value  # label or threshhold
    self.value_index = value_index  # index of the relevant feature in a given item

  def check(self, item):
    raise Exception("should be overridden")

class DiscreteDecisionStump(DecisionStump):
  def check(self, item):
    if isinstance(item, np.matrixlib.defmatrix.matrix):
      item = item.A1 #numpy type systems :(

    if item[self.value_index] == self.value:
      return POS
    else:
      return NEG

  def __repr__(self):
    return "[=" + str(self.value) + "] : " +\
           str(self.value_index)

class NumericDecisionStump(DecisionStump):
  def check(self, item):
    if isinstance(item, np.matrixlib.defmatrix.matrix):
      item = item.A1 # numpy type systems

    if self.value < item[self.value_index]:
      return NEG
    else:
     return POS

  def __repr__(self):
    return "[<" + str(self.value) + "] : " + str(self.value_index)

  def discriminant(self, item):
    pprint(item.shape)
    pprint(item)
    return abs(self.value - item[self.value_index])


class ECOC():
  # runs ecoc on multiclass input
  # in this case on the 20 classes of newsgroups

  def __init__(self, features, truths, feature_types):
    classes = sorted(set(truths))

    classifiers = []
    for clazz in classes:
      # make a bunch of classifiers which split
      new_truths = map(lambda x: 1 if x == clazz else 0, truths)

      feature_types_copy = deepcopy(feature_types)
      feature_types_copy[-1] = [0, 1]

      ada = AdaBoost(features, new_truths, feature_types_copy)

      classifiers.append(ada)

    self.classes = classes
    self.classifiers = classifiers

    # split each truths into into A and not-A for each label
    # train an ada classifier for each A/not-A pair

  def classify(self, item):
    # run each classifier and return the majority vote
    votes = [classifier.classify_prob(item) for classifier in self.classifiers]
    ind = np.argmax(votes)
    return self.classes[ind]


def read_config_file(config_filename, data_filename):
  f = open(config_filename)
  feature_types = []
  #datapoints, discrete, continuous = f[0].split()

  for line in f.readlines()[1:]:
    entries = line.strip().split()

    if int(entries[0]) < 0:
      feature_types.append("numeric")
    else:
      if len(entries) == 1:
        pass #skip, some lines are empty for whatever reason
      else:
        feature_types.append(entries[1:] + ['?']) #labels for discrete feature

  g = open(data_filename)
  data = []
  for line in g.readlines()[:-1]:
    entries = line.strip().split('\t')
    point = []

    for value, feature_type in zip(entries, feature_types):
      if feature_type == 'numeric':
        if value == "?":
          point.append(float("NaN")) # TODO ???
        else:
          #pprint(value)
          point.append(float(value))
      else: #discrete values
        if not (value in feature_type or value == "?"):
          pprint((value, feature_type))
          assert(value in feature_type)
        else:
          point.append(value)

    data.append(point)

  #return np.matrix(data), feature_types
  return np.matrix(data, dtype=float), feature_types #TODO this if for spambaset

TOP_DIR = "../../data/HW4/UCI/"
def read_in_data_config(name):
  config = TOP_DIR + name + "/" + name + ".config"
  data = TOP_DIR + name + "/" + name + ".data"
  data, config = read_config_file(config, data)
  return np.array(data), config


### Active Learning



START = 5.0/100 #how much data to start with
ADD = 5.0/100  #how much data to add each time
STOP = 1.0/2

def active_learning(name):
  data, feature_types = read_in_data_config(name)

  amt = START

  while amt < STOP:

    pprint({"amt": amt})

    split = round(len(data) * amt)
    train = data[:-split,:]
    test = data[-split+1:,:]

    run_cycle(train, test, AdaBoost, feature_types)

    amt += ADD

def active_learning_add_best(name):
  data, feature_types = read_in_data_config(name)

  amt = START

  pprint({"amt": amt})

  split = round(len(data) * amt)
  train = data[:-split,:]
  test = data[-split+1:,:]

  ada = run_cycle(train, test, AdaBoost, feature_types)

  while amt < STOP:
    amt += ADD
    pprint({"amt": amt})

    # find best points in testing set
    best, not_best = find_best(ada, test, round(ADD * len(data)))
    # add to training set
    train.extend(best)
    test = not_best

    ada = run_cycle(train, test, AdaBoost, feature_types)

def find_best(ada, data, n):
  # given a dataset and a classifier return it split into two datasets
  # the first has the n best points in the set
  # the second has the rest of the points

  return ada.choose_smallest_discriminant(data, n)


# problem 1

def test_data_sample(name, classifier_type):
  data, feature_types = read_in_data_config(name)
  np.random.shuffle(data)

  #data = data[:100] #trim down for testing

  split = round(len(data) / SPLIT)

  train = data[:-split,:]
  test = data[-split+1:,:]

  run_cycle(train, test, classifier_type, feature_types)


# train a classifier, calculate training and testing error
def run_cycle(train, test, classifier_type, feature_types):
  features = train[:,:train.shape[1]-1]
  truths = train[:,train.shape[1]-1]

  ada = classifier_type(features, truths, feature_types)

  error = calculate_error(ada, features, truths, feature_types)
  pprint(("training error", error))

  features = test[:,:test.shape[1]-1]
  truths = test[:,test.shape[1]-1]
  error = calculate_error(ada, features, truths, feature_types)
  pprint(("testing error", error))

  return ada


def calculate_error(ada, features, truths, feature_types):
  errors = 0

  for item, truth in zip(features, truths):
    guess = ada.classify(item)
    item_labels = feature_types[-1]

    if isinstance(guess, int):
      guess_label = item_labels[int(guess)]

    if guess != truth:
      errors +=1

  return float(errors) / len(truths)

if __name__ == "__main__":
  #titles = ["crx",
  #          "vote",
  #          #"band", # illegal value
  #          "monk",
  #          "tic",
  #          "spam", # slow
  #          "agr"] # slow

  titles = ["spam"]

  for title in titles:
    pprint(title)
    test_data_sample(title, AdaBoost)
    #active_learning(title)
    #active_learning_add_best(title)

  #multiclass_titles = ["bal",
  #                     "car",
  #                     "cmc",
  #                     "nur"]

  #for title in multiclass_titles:
  #  pprint(title)
  #  test_data_sample(title, ECOC)

