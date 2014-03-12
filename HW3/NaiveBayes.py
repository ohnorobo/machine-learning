#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
from pprint import pprint
from copy import deepcopy
import csv

K_FOLDS = 10


class NaiveBayes():

  def __init__(self, features, truths):
    # Makes a Naive bayes classifier modeling each feature as one of
    # "bernoulli"
    # "gaussian"
    # "histogram"

    self.items = features
    # truth values must be only 0s and 1s
    self.truths = truths

  def train(self):
    # per feature
    # calculate a number of classes each with a p(class) and p(feature_value | class)
    self.train_prob_per_class()
    self.train_features()

  # returns the probability of class 1 for the item
  # prob class 0 = (1 - prob)
  def classify(self, item):
    # p(class | item) = product( p(feature_value | class) * p(class) )
    # p(feature_value | class) = depends on subclass

    clazz = 0
    p0 = 1
    for i, feature_value in enumerate(item):
      p0 = p0 * \
           self.prob_per_feature_value(i, feature_value, clazz)
    p0 = p0 * self.prob_per_class(clazz)

    clazz = 1
    p1 = 1
    for i, feature_value in enumerate(item):
      p1 = p1 * \
           self.prob_per_feature_value(i, feature_value, clazz)
    p1 = p1 * self.prob_per_class(clazz)

    #pprint(("  ", p0, p1))
    return p1 / (p0 + p1)


  def classify_all(self, items):
    return [self.classify(item) for item in items]

  def avg_classification_error(self, items, truths):
    predictions = self.classify_all(items)
    differences = map(lambda x: abs(x[0]-x[1]), zip(predictions, truths))
    return sum(differences) / len(differences)


  def prob_per_feature_value(self, feature_index, feature_value, clazz):
    # p(feature_value | feature_index, clazz)
    raise Exception("should be overridden by subclasses")

  def train_features(self):
    # train whatever you need per feature
    raise Exception("should be overridden by subclasses")

  def train_prob_per_class(self):
    self.class_counts = {0: 0, 1: 0}
    for t in self.truths:
      self.class_counts[t] += 1

  def prob_per_class(self, clazz):
    # p(clazz)
    total_count = self.class_counts[0] + self.class_counts[1]
    return float(self.class_counts[clazz]) / total_count


class BernoulliNaiveBayes(NaiveBayes):
   # bernoulli - choose mean m for each feature
   # p(f<=m | 0)
   # p(f> m | 0)
   # p(f<=m | 1)
   # p(f> m | 1)

  #def __init__(self, features, truths):
  #  NaiveBayes.__init__(self, features, truths)

  def train_features(self):
     self.feature_means = column_means(self.items)
     num_features = len(self.feature_means)

     # f > m
     self.greater_counts = {0: [0]*num_features, 1: [0]*num_features}
     # f <= m
     self.lesser_counts = {0: [0]*num_features, 1: [0]*num_features}

     for item, truth in zip(self.items, self.truths):
       for i, (feature_value, feature_mean) in enumerate(zip(item, self.feature_means)):
         if feature_value > feature_mean:
           self.greater_counts[truth][i] += 1
         else:
           self.lesser_counts[truth][i] += 1

  def prob_per_feature_value(self, feature_index, feature_value, clazz):
    total_count_given_class = self.greater_counts[clazz][feature_index] + \
                              self.lesser_counts[clazz][feature_index]

    if feature_value > self.feature_means[feature_index]:
      feature_count_given_class = self.greater_counts[clazz][feature_index]
    else:
      feature_count_given_class = self.lesser_counts[clazz][feature_index]

    return float(feature_count_given_class) / total_count_given_class


class GaussianNaiveBayes(NaiveBayes):
   # gaussian, choose gaussian(m, sigma) for each feature (1Dimensional)
   # p(m_0, sigma_0 | 0)
   # p(m_0, sigma_0 | 1)
   # = gaussian.prob_density(m, sigma, value)

   pass

class HistogramNaiveBayes(NaiveBayes):
   # histogram, choose values [min, low-class-mean, mean, high-class-mean, max]
   # class means = means for just 0 or 1
   #
   # make 4 buckets, one for each interval between those values
   # b1 = [min, low-class-mean)
   # b2 = [low-class-mean, mean)
   # b3 = [mean, high-class-mean)
   # b4 = [high-class-mean, max)
   #
   # calculate prob of each class given 0 or 1

   pass


############# Utility Functions

def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

def column_means(a):
    return [float(sum(l))/len(l) for l in a.T]

############## Test

data_dir = "../../data/HW1/"
def cross_validate_spam(NaiveBayesClass):
  spam_filename = data_dir + "spambase/spambase.data"
  data = read_csv_as_numpy_matrix(spam_filename)[:4600,:]

  np.random.shuffle(data) #truffle shuffle

  num_crosses = K_FOLDS
  crosses = np.vsplit(data, K_FOLDS) 
  total_error = 0

  for i in xrange(num_crosses):
    train = None
    for j in xrange(num_crosses):
      if i != j:
        if train == None:
          train = deepcopy(crosses[j])
        else:
          train = np.vstack((train, crosses[j]))

    test = crosses[i]

    features = np.array(train[:,:56])
    truths = train[:,57].A1
    nb = NaiveBayesClass(features, truths)
    nb.train()

    features = np.array(test[:,:56])
    truths = test[:,57].A1
    error = nb.avg_classification_error(features, truths)
    total_error += error

    pprint("cv: " + str(i))
    pprint(error)

  pprint("avg error")
  pprint(total_error / K_FOLDS)

if __name__ =="__main__":
  pprint("Bernoulli")
  cross_validate_spam(BernoulliNaiveBayes)
  #pprint("Gaussian")
  #cross_validate_spam(GaussianNaiveBayes)
  #pprint("Histogram")
  #cross_validate_spam(HistogramNaiveBayes)

