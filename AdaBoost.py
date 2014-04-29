#!/usr/bin/python

import numpy as np
from pprint import pprint
import math
from copy import deepcopy
from multiprocessing import Pool

NUM_CLASSIFIERS = 300
ITERATIONS = 100
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

    #self.classifiers = self.get_all_classifiers()
    #self.classifier_weights = np.array([float('NaN')]*len(self.classifiers))
    self.classifiers = []
    self.classifier_weights = []
    self.presort()

    #pprint(self.classifiers)

    self.train()
    self.release_sort()

  # convert whatever weird symbols are being used for classes into 0/1
  def convert_to_neg1_1s(self, truths, feature_types):
    return map(lambda truth: convert(feature_types, truth), truths)

  def train(self):

    #pprint({"number of classifiers": len(self.classifiers)})

    # iterate through all classifiers
    for i in range(ITERATIONS):
    #for t, classifier in enumerate(self.classifiers):

      classifier, t, error = self.choose_best_classifier_and_error()

      # compute error rate e_t
      error = self.error(classifier)
      #pprint({"error": error})

      # assign weight a_t to classifier f_t
      if error == 0 or error == 1:
        pprint(classifier)
        raise Exception("prefect feature, problem")
      else:
        a = math.log((1-error)/error)

      pprint({"error":error, "a":a, "classifier":classifier})
      #self.classifier_weights[t] = a
      self.classifiers.append(classifier)
      self.classifier_weights.append(a)

      # add weight to misclassed points
      self.reweight_misclassed_points(a, classifier)
      # normalize point weights
      self.normalize_point_weights()

      #pprint(zip(self.classifiers, self.classifier_weights))
      #pprint(self.item_weights)

    #self.pick_best_classifiers()

    a = zip(self.classifier_weights, self.classifiers)
    pprint(sorted(a, key=lambda x: abs(x[0])))

  def presort(self):
    sorted_features = []
    sorted_indexes = [] #indexes into truths/weights corrosponding to sorted feature

    for feature in self.items.T:
      both = zip(feature, range(0, len(feature)))
      s = sorted(both, key=lambda x: x[0]) #sort according to feature
      s_feature, s_index = zip(*s)

      sorted_features.append(s_feature)
      sorted_indexes.append(s_index)

    #pprint(np.array(sorted_features).shape)

    self.sorted_features = sorted_features
    self.sorted_indexes = sorted_indexes

  # release expensive data structures back to GC
  def release_sort(self):
    self.sorted_features = []
    self.sorted_indexes = []

  def choose_best_classifier_and_error(self):
    best_classifier = None
    best_error = .5

    #pprint(("num features", len(self.feature_types)))

    for i in range(len(self.feature_types)):
      c, e = self.get_best_classifier_and_error_per_feature(i)
      if abs(e - .5) > abs(best_error - .5):
        best_classifier = c
        best_error = e
    return best_classifier, i, best_error

  def get_best_classifier_and_error_per_feature(self, feature_index):
    if self.feature_types[feature_index] == 'numeric':
      c, e = self.get_best_classifier_and_error_per_feature_numeric(feature_index)
    else:
      c, e = self.get_best_classifier_and_error_per_feature_cat(feature_index)

    return c, e

  def get_best_classifier_and_error_per_feature_cat(self, feature_index):
    cats = self.feature_types[feature_index]
    feature = self.items.T[feature_index]

    best_error = .5
    best_label = cats[0]

    for cat in cats:
      wrong_in = 0
      wrong_out = 0

      for truth, label, weight in zip(self.truths, feature, self.item_weights):
        if label == cat and truth == NEG:
          wrong_in += weight
        if label != cat and truth == POS:
          wrong_out += weight

      new_error = wrong_in + wrong_out
      #pprint({"new_error": new_error})
      if abs(new_error - .5) > abs(best_error - .5):
        best_error = new_error
        best_label = cat
    return DiscreteDecisionStump(best_label, feature_index), best_error


  '''
  def get_best_classifier_and_error_per_feature_numeric(self, feature_index):
    feature_type = self.feature_types[feature_index]

    #pprint(feature_index)

    feature = self.sorted_features[feature_index]
    indexes = self.sorted_indexes[feature_index]

    best_error = .5
    best_cutoff = feature[0]

    # start at the lowest feature value, nothing is below it
    pos_above = sum(map(lambda x: x[1],
                    filter(lambda x: x[0] == POS, zip(self.truths, self.item_weights))))
    neg_below = 0

    for i, (feature_value, index) in enumerate(zip(feature, indexes)):
      truth = self.truths[index]
      weight = self.item_weights[index]

      if truth == POS:
        pos_above -= weight
      elif truth == NEG:
        neg_below += weight
      else:
        raise Exception(("weird stuff", truth))

      new_error = pos_above + neg_below
      if abs(new_error - .5) > abs(best_error - .5):
        best_error = new_error
        best_cutoff = feature[i]

    return NumericDecisionStump(best_cutoff, feature_index), best_error
  '''

  # special only-splitting-at-0 version for 20newsgroup data
  def get_best_classifier_and_error_per_feature_numeric(self, feature_index):
    split = 0
    feature = self.sorted_features[feature_index]
    indexes = self.sorted_indexes[feature_index]

    pos_below = 0.0
    neg_above = 0.0

    for i, (feature_value, index) in enumerate(zip(feature, indexes)):
      truth = self.truths[index]
      weight = self.item_weights[index]

      if feature_value <= split and truth == POS:
        pos_below += weight
      elif split < feature_value and truth == NEG:
        neg_above += weight

    return NumericDecisionStump(split, feature_index), pos_below+neg_above




  def error(self, classifier):
    error = 0.0
    for i, (point, truth) in enumerate(zip(self.items, self.truths)):
      if self.misclassed(classifier, point, truth):
        error += self.item_weights[i]
    return error

  def reweight_misclassed_points(self, a, classifier):
    for i, (point, truth) in enumerate(zip(self.items, self.truths)):
      if self.misclassed(classifier, point, truth):
        self.item_weights[i] *= (math.e ** a)

  def misclassed(self, classifier, point, truth):
    guess = classifier.check(point) # 0 or 1
    return guess == truth

  #sum weights should = 1
  def normalize_point_weights(self):
    self.item_weights *= 1.0/sum(self.item_weights)

  def choose_smallest_discriminant(self, data, n):
    # return n points with the smallest discriminant
    # and the rest of the points without

    pprint({"choosing":n, "from": len(data)})

    d = sorted(data, key=lambda x: abs(self.classify_prob(x)))#, reverse=True)
    # close to 0 is best
    return data[:n], data[n:]

  def classify(self, item):
    return sign(self.classify_prob(item))

  # returns a value, not really a prob
  def classify_prob(self, item):
    scores = [classifier.check(item) for classifier in self.classifiers]
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
    #pprint(item.shape)
    #pprint(item)
    return abs(self.value - item[self.value_index])




#NEWS = "/home/laplante/data/"
NEWS = "./data/20newsgroup/"
NUM_WORDS = 11350
# number of points:
# in train  = 11314
# in test = 7532
def read_20_newsgroup_data(fname):

  f = open(NEWS + fname)
  lines = f.readlines()

  data = np.zeros((len(lines), NUM_WORDS))
  truths = []

  for i, line in enumerate(lines):
    bits = line.split()
    truth = bits[0]
    points = bits[1:-3]

    truths.append(int(truth))

    for point in points:
      j, count = point.split(':')
      data[i,int(j)] = int(count)

  truths = np.array(truths).T
  data = np.array(data).T

  a = np.vstack((data, truths))
  a = a.T

  #pprint("column sums")
  #pprint(a.sum(axis=0).shape)
  #pprint(list(a.sum(axis=0)))

  return a


def delete_zeros(xs, ys):
  tf = xs.sum(axis=0)
  bad = []
  for i, val in enumerate(tf):
    if val <= 5: #remove all infrequent features
      bad.append(i)
  #pprint(("bad", bad))

  x = np.delete(xs, bad, axis=1)
  y = np.delete(ys, bad)

  return x, y


def print_examples(self):
  for i, cluster in enumerate(self.clusters()):
    pprint({"cluster": i}) 
    for point in cluster[:10]:
      pprint(row_index(self.xs, point)[0][0])


# columns = f1 ... f6
# rows = classes 0-7 (rel/comp/sale/auto/sport/med/space/pol)
           #     #               #              #
ECOC_8 = [[0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0],
          [0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1],
          [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0],
          [0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,0],
          [1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1],
          [1,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0],
          [1,1,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,1,0],
          [1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,1]]
'''
ECOC_8 = [[0,0,0],
          [0,0,1],
          [0,1,0],
          [0,1,1],
          [1,0,0],
          [1,0,1],
          [1,1,0],
          [1,1,1]]
'''
ECOC_26 = [[0,0,0,0,0,1],
           [0,0,0,0,1,0],
           [0,0,0,1,0,1],
           [0,0,0,1,1,1],
           [0,0,1,0,0,0],
           [0,0,1,1,1,0],
           [0,1,0,0,0,1],
           [0,1,0,0,1,1],
           [0,1,0,1,0,0],
           [0,1,1,0,1,0],
           [0,1,1,1,0,1],
           [0,1,1,1,1,1],
           [0,1,1,1,1,0], #
           [1,0,0,0,0,1],
           [1,0,0,1,0,0],
           [1,0,0,1,1,1],
           [1,0,1,0,0,1],
           [1,0,1,0,1,0],
           [1,0,1,1,0,0],
           [1,1,0,0,0,1],
           [1,1,0,0,1,0],
           [1,1,0,1,0,1],
           [1,1,0,1,1,1],
           [1,1,1,0,1,0],
           [1,1,1,1,0,0],
           [1,1,1,1,1,1]]


class ECOC():
  # runs ecoc on multiclass input
  # in this case on the 20 classes of newsgroups

  def __init__(self, features, truths, feature_types):
    classes = sorted(set(truths))
    num_classes = len(classes)
    self.ecoc = self.get_ecoc_matrix(num_classes)

    self.features = features
    self.truths = truths
    self.feature_types = feature_types

    pprint(self.features)
    pprint(set(self.truths))

    self.adas = [None]*len(self.ecoc)

    num = len(self.ecoc)

    pool = Pool(processes=5)
    self.adas = pool.map(train_single, zip([self]*num, range(num)))
    #self.train_single(i)
    #func = Process(target=self.train_single, args=(self, i))
    #func.start()

  def get_ecoc_matrix(self, m):
    if m == 8:
      return np.array(ECOC_8).T
    if m == 26:
      return np.array(ECOC_26).T
    else:
      raise Exception("only precomputed mat for 8")

  def convert_truths_to_1_0(self, truths, e_column, labels):
    pprint(truths.shape)
    new_truths = []
    for truth in truths:
      #new_truths.append(e_column[int(truth)])
      new_truths.append(e_column[labels.index(truth)])
    return np.array(new_truths)

  def classify(self, item):
    # run each classifier and return the vote
    votes = [ada.classify(item) for ada in self.adas]
    #return ecoc row that has the closest edit distance to votes
    return self.min_edit_distance(votes)

  def min_edit_distance(self, votes):
    distances = [self.calculate_edit_distance(row, votes) for row in self.ecoc.T]

    '''
    #pprint({"distances": distances})
    m = min(distances)
    count = distances.count(m)
    pprint(count)
    if count > 1:
      pprint(("tie", count, votes))
    '''

    return distances.index(min(distances))

  def calculate_edit_distance(self, row, votes):
    diff = 0
    assert(len(row) == len(votes))
    for a, b in zip(row, votes):
      if a != b:
        diff += 1
    return diff

#usually this would ba a method of ECOC, but it can't be for pickle reasons
def train_single(args):
  self = args[0]
  i = args[1]

  e_column = self.ecoc[i]
  new_truths = self.convert_truths_to_1_0(self.truths, e_column, self.feature_types[-1])

  pprint(("should be 0,1", set(new_truths)))

  new_feature_types = self.feature_types[:-1] #change class labels
  new_feature_types.append([0,1])

  ada = AdaBoost(self.features, new_truths, new_feature_types)
  return ada




def ecoc_news():
  train = read_20_newsgroup_data("train.txt")
  test = read_20_newsgroup_data("test.txt")

  np.random.shuffle(train)
  #train = train[:200]

  num_features = train.shape[0]-1
  feature_types = ["numeric"]*num_features
  feature_types.append([0,1,2,3,4,5,6,7]) #class labels

  run_cycle(train, test, ECOC, feature_types)



def ecoc_letter():
  data = read_letter_data()
  pprint(data)
  np.random.shuffle(data)

  split = round(len(data) * .9)
  train = data[:split,:]
  test = data[split:,:]

  pprint(train.shape)
  pprint(test.shape)

  num_features = train.shape[1]-1
  feature_types = ["numeric"]*num_features
  feature_types.append(['A','B','C','D','E','F','G','H',
                        'I','J','K','L','M','N','O','P',
                        'Q','R','S','T','U','V','W','X','Y','Z']) #class labels

  run_cycle(train, test, ECOC, feature_types)

LETTER = "./data/letter_recognition/letter-recognition.data"
def read_letter_data():
  f = open(LETTER)
  data = []

  for line in f:
    elements = line.strip().split(',')
    y = elements[0]
    point = [int(p) for p in elements[1:]]
    point.append(y)
    data.append(point)

  return np.array(data, dtype=object)







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
          #pprint((value, feature_type))
          assert(value in feature_type)
        else:
          point.append(value)

    data.append(point)

  #return np.matrix(data, dtype=object), feature_types
  return np.matrix(data, dtype=float), feature_types #TODO this if for spambaset

TOP_DIR = "./data/UCI/"
def read_in_data_config(name):
  config = TOP_DIR + name + "/" + name + ".config"
  data = TOP_DIR + name + "/" + name + ".data"
  data, config = read_config_file(config, data)
  return np.array(data), config







### Active Learning

#START = 5.0/1000 #how much data to start with
#ADD = 5.0/1000  #how much data to add each time
#STOP = 1.0/10 #when to stop

START = 5.0/100 #how much data to start with
ADD = 5.0/100  #how much data to add each time
STOP = 5.0/10 #when to stop


TEST_AMT = 1.0/10

def run_both(name):
  data, feature_types = read_in_data_config(name)
  np.random.shuffle(data)

  pprint("RANDOM")
  active_learning(data, feature_types)

  pprint("BEST")
  active_learning_add_best(data, feature_types)


def active_learning(data, feature_types):
  amt = START

  while amt <= STOP:

    pprint({"amt": amt})

    split = round(len(data) * amt)
    train = data[:split,:]
    test = data[-round(len(data) * TEST_AMT):,:]

    run_cycle(train, test, AdaBoost, feature_types)

    amt += ADD

def active_learning_add_best(data, feature_types):
  amt = START

  pprint({"amt": amt})

  split = round(len(data) * amt)
  train = data[:split,:]
  rest = data[split:-round(len(data) * TEST_AMT),:]
  test = rest[-round(len(data) * TEST_AMT):,:]

  #pprint({"split":split, "train":train.shape, "rest":rest.shape, "test":test.shape})

  ada = run_cycle(train, test, AdaBoost, feature_types)

  while amt < STOP:
    amt += ADD
    pprint({"amt": amt})

    # find best points in testing set
    best, not_best = find_best(ada, rest, round(ADD * len(data)))
    rest = not_best
    pprint(("shapes", best.shape, not_best.shape))
    pprint({"train shape":train.shape, "test shape":test.shape, "rest shape":rest.shape})
    # add to training set
    train = np.vstack((train, best))
    #test = not_best[-round(len(data) * TEST_AMT):,:]

    #pprint({"split":split, "train":train.shape, "rest":rest.shape, "test":test.shape})

    ada = run_cycle(train, test, AdaBoost, feature_types)






def find_best(ada, data, n):
  # given a dataset and a classifier return it split into two datasets
  # the first has the n best points in the set
  # the second has the rest of the points
  best, not_best = ada.choose_smallest_discriminant(data, n)
  return best, not_best


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

  #features, truths = delete_zeros(features, truths)

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

    #pprint((guess_label, guess, truth))

    if guess_label != truth:
      #if guess != truth:
      errors +=1

  return float(errors) / len(truths)








if __name__ == "__main__":
  titles = ["crx",
            "vote",
            #"band", # illegal value
            "monk",
            "tic",
            "spam", # slow
            "agr"] # slow

  titles = ["spam"]
  #titles = ["crx"]

  for title in titles:
    pprint(title)
    #test_data_sample(title, AdaBoost)
    run_both(title)


  ecoc_news()
  ecoc_letter()

  #multiclass_titles = ["bal",
  #                     "car",
  #                     "cmc",
  #                     "nur"]

  #for title in multiclass_titles:
  #  pprint(title)
  #  test_data_sample(title, ECOC)

