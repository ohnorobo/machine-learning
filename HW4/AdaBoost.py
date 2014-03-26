#!/usr/bin/python

import numpy as np
from pprint import pprint
import math


ITERATIONS = 100
SPLIT = 10 # testing data is 1/SPLIT of data


def convert(classes, element):
  if element == classes[0]:
    return 0.0
  elif element == classes[1]:
    return 1.0
  else:
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
    self.truths = self.convert_to_0_1s(truths, self.truth_feature_types)

    self.item_weights = np.array([1.0/len(data)]*len(self.items))

    self.classifiers = self.get_all_classifiers()
    self.classifier_weights = np.array([1.0]*len(self.classifiers))

    #pprint(self.classifiers)

    self.train()

  # convert whatever weird symbols are being used for classes into 0/1
  def convert_to_0_1s(self, truths, feature_types):
    return map(lambda truth: convert(feature_types, truth), truths)


  def train(self):

    # iterate through all classifiers
    for t, classifier in enumerate(self.classifiers):

      classifier.precompute() #recalculate results for classifier

      # compute error rate e_t
      error = self.error(classifier)

      # assign weight a_t to classifier f_t
      a = math.log((1-error)/error)
      pprint({"error":error, "a":a})
      self.classifier_weights[t] = a

      # add weight to misclassed points
      self.reweight_misclassed_points(a, classifier)
      # normalize point weights
      self.normalize_point_weights()

      #pprint(self.classifier_weights)
      #pprint(self.item_weights)

  def error(self, classifier):
    error = 0.0
    for i, (point, truth) in enumerate(zip(self.items, self.truths)):
      if self.misclassed(classifier, point, truth):
        error += self.item_weights[i]
    pprint({"error":error, "total":sum(self.item_weights)})
    return error

  def reweight_misclassed_points(self, a, classifier):
    for i, (point, truth) in enumerate(zip(self.items, self.truths)):
      if self.misclassed(classifier, point, truth):
        self.item_weights[i] *= math.e ** a

  def misclassed(self, classifier, point, truth):
    guess = classifier.check(point) # 0 or 1
    #pprint({"guess": guess, "truth": truth})
    return guess == truth

  #sum weights should = 1
  def normalize_point_weights(self):
    self.item_weights *= 1.0/sum(self.item_weights)

  def get_all_classifiers(self):
    classifiers = []
    for i in range(len(self.feature_types)):
      c = self.get_all_classifiers_per_feature(i)
      classifiers.extend(c)
    return classifiers

  def get_all_classifiers_per_feature(self, feature_index):
    feature_type = self.feature_types[feature_index]
    classifiers = []
    feature = self.items.T[feature_index].T

    if feature_type == 'numeric':
      values = self.items.T[feature_index].A1
      threshholds = sorted(set(values))
      for value in threshholds:
        classifiers.append(NumericDecisionStump(value, feature_index, feature,
                                                self.truths, self.item_weights))
    else:
      for label in feature_type:
        classifiers.append(DiscreteDecisionStump(label, feature_index, feature,
                                                 self.truths, self.item_weights))

    return classifiers

  def classify(self, item):
    scores = [classifier.check(item) for classifier in self.classifiers]
    return round(np.inner(self.classifier_weights, scores) / len(self.classifiers))


class DecisionStump():

  def __init__(self, value, value_index, feature, truths, item_weights):
    self.value = value  # label or threshhold
    self.value_index = value_index  # index of the relevant feature in a given item
    self.item_weights = item_weights  # weight of each item in the current iteration
    self.truths = truths  # true labels for each item
    self.feature = feature  # relevant feature

    self.precompute() #choose answers

  def check(self, item):
    raise Exception("should be overridden")

class DiscreteDecisionStump(DecisionStump):
  def check(self, item):
    if isinstance(item, np.matrixlib.defmatrix.matrix):
      item = item.A1 #numpy type systems :(

    if item[self.value_index] == self.value:
      return self.when_equal
    else:
      return self.when_not_equal

  def precompute(self):
    matches_label = {1:0, 0:0}
    doesnt_match_label = {1:0, 0:0}
    for feature_value, truth, weight in zip(self.feature, self.truths, self.item_weights):
      if feature_value == self.value:
        matches_label[truth] += weight
      else:
        doesnt_match_label[truth] += weight

    m0 = matches_label[0]
    m1 = matches_label[1]
    dm0 = matches_label[0]
    dm1 = matches_label[1]

    if m1 < m1:
      self.when_equal = 1
    else:
      self.when_equal = 0

    if dm0 < dm1:
      self.when_not_equal = 1
    else:
      self.when_not_equal = 0

    '''
    if m > .5:
      self.when_equal = 1
    else:
      self.when_equal = 0
    if dm > .5:
      self.when_not_equal = 1
    else:
      self.when_not_equal = 0
    '''


  def __repr__(self):
    return "[=" + str(self.value) + " " + str(self.matches_label) + "] : " +\
           str(self.value_index)

class NumericDecisionStump(DecisionStump):
  def check(self, item):
    if isinstance(item, np.matrixlib.defmatrix.matrix):
      item = item.A1 # numpy type systems

    if self.value < item[self.value_index]:
      return self.greater
    else:
      return self.lesser

  def precompute(self):
    gt_label = {1:0, 0:0}
    lt_label = {1:0, 0:0}
    for feature_value, truth, weight in zip(self.feature, self.truths, self.item_weights):
      if self.value < feature_value:
        gt_label[truth] += weight
      else:
        lt_label[truth] += weight

    gt0 = gt_label[0]
    gt1 = gt_label[1]
    lt0 = lt_label[0]
    lt1 = lt_label[1]

    if gt0 < gt1:
      self.greater = 1
    else:
      self.greater = 0

    if lt0 < lt1:
      self.lesser = 1
    else:
      self.lesser = 0

    '''
    if gt > .5:
      self.greater = 1
    else:
      self.greater = 0
    if lt > .5:
      self.lesser = 1
    else:
      self.lesser = 0
    '''


  def __repr__(self):
    return "[<" + str(self.value) + " " + str(self.greater) + "] : " + str(self.value_index)

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
          point.append(0.0) # TODO ???
        else:
          point.append(float(value))
      else: #discrete values
        if not (value in feature_type or value == "?"):
          #pprint((value, feature_type))
          assert(value in feature_type)
        else:
          point.append(value)

    data.append(point)

  return np.matrix(data), feature_types

TOP_DIR = "../../data/HW4/UCI/"
def test_data_sample(name):
  config = TOP_DIR + name + "/" + name + ".config"
  data = TOP_DIR + name + "/" + name + ".data"

  data, feature_types = read_config_file(config, data)
  np.random.shuffle(data)

  #pprint((data, feature_types))

  split = round(len(data) / SPLIT)

  train = data[:-split,:]
  test = data[-split+1:,:]

  #pprint(train.shape)

  features = np.array(train[:,:train.shape[1]-1])
  truths = train[:,train.shape[1]-1].A1

  #pprint((features, truths))

  ada = AdaBoost(data, truths, feature_types)

  features = np.array(test[:,:test.shape[1]-1])
  truths = test[:,test.shape[1]-1].A1

  errors = 0

  for item, truth in zip(features, truths):
    guess = ada.classify(item)
    item_labels = feature_types[-1]
    guess = item_labels[int(guess)]

    if guess != truth:
      errors +=1

  pprint(("error", float(errors)/len(truths)))


if __name__ == "__main__":
  print "CRX"
  test_data_sample("crx")
  #print "VOTE"
  #test_data_sample("vote")




