#!/usr/bin/python
import numpy as np
#from scipy.stats import multivariate_normal
from scipy.stats import norm
#from sklearn.metrics import roc_curve, auc
import pylab as pl
from pprint import pprint
from copy import deepcopy
import pandas
import csv, sys
from ExpectationMaximization import GaussianMixtureModel as GMM

K_FOLDS = 10
#how much to widen gaussians with no standard deviation
SIGMA_SMOOTHING = .001
CUTOFF = .5
NUM_GAUSSIANS = 4



class NaiveBayes():

  def __init__(self, features, truths):
    # Makes a Naive bayes classifier

    # Subclasses:
    # bernoulli
    # gaussian
    # histogram
    # gaussian mixture model

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
    if p0 == 0 and p1 == 0:
      return .5
    else:
      return p1 / (p0 + p1)


  def classify_all(self, items):
    return [self.classify(item) for item in items]

  def avg_classification_error(self, items, truths):
    predictions = self.classify_all(items)
    differences = map(lambda x: abs(x[0]-x[1]), zip(predictions, truths))
    return sum(differences) / len(differences)

  def error_tables(self, items, truths):
    misclassified = 0
    false_pos = 0
    false_neg = 0

    ones = filter(lambda x: x == 1, truths)
    zeros = filter(lambda x: x == 0, truths)

    predictions = self.classify_all(items)

    for truth, prediction in zip(truths, predictions):

      if prediction > CUTOFF:
        guess = 1
      else:
        guess = 0

      if guess != truth:
        misclassified += 1

        if truth == 0:
          false_pos += 1
        elif truth == 1:
          false_neg += 1

    values = {"Misclassified": [float(misclassified) / len(truths)],
              "False Pos": [float(false_pos) / len(zeros)],
              "False Neg": [float(false_neg) / len(ones)]}

    print(pandas.DataFrame(values))

  def roc_curve_data(self, items, truths):
    predictions = self.classify_all(items)

    #taken from http://scikit-learn.org/0.11/auto_examples/plot_roc.html

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(truths, predictions)
    roc_auc = auc(fpr, tpr)
    print "\nAUC : %f" % roc_auc

    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC curve')
    pl.legend(loc="lower right")
    pl.show()

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

  def train_features(self):
    zeroes, ones = separate_classes(self.items, self.truths)

    mus0 = column_means(zeroes)
    #sigmas0 = self.smooth_sigmas(column_stds(zeroes))
    sigmas0 = self.smooth_sigmas(column_stds(self.items))

    mus1 = column_means(ones)
    #sigmas1 = self.smooth_sigmas(column_stds(ones))
    sigmas1 = self.smooth_sigmas(column_stds(self.items))

    self.gaussians = {0: zip(mus0, sigmas0), 1: zip(mus1, sigmas1)}

  def prob_per_feature_value(self, feature_index, feature_value, clazz):
    mu = self.gaussians[clazz][feature_index][0]
    sigma = self.gaussians[clazz][feature_index][1]

    return norm.pdf(feature_value, loc=mu, scale=sigma)

  def smooth_sigmas(self, sigmas):
    # add a slight bump if any sigmas = 0
    return map(lambda s: SIGMA_SMOOTHING if s == 0 else s, sigmas)

class GMMNaiveBayes(NaiveBayes):

  def train_features(self):
    zeroes, ones = separate_classes(self.items, self.truths)
    self.gmms = {0:[], 1:[]}

    for i, (feature_zeroes, feature_ones) in enumerate(zip(zeroes.T, ones.T)):
      print("training feature #" + str(i))

      gmm0 = GMM(NUM_GAUSSIANS)
      gmm1 = GMM(NUM_GAUSSIANS)

      feature_zeroes = np.matrix(feature_zeroes).T
      feature_ones = np.matrix(feature_ones).T

      pprint(feature_zeroes)
      pprint(feature_ones)

      pprint(np.matrix(feature_zeroes).shape)

      gmm0.train(feature_zeroes)
      gmm1.train(feature_ones)

      self.gmms[0].append(gmm0)
      self.gmms[1].append(gmm1)

      sys.stdout.flush() # to make print work correctly

  def prob_per_feature_value(self, feature_index, feature_value, clazz):
    gmm = self.gmms[clazz][feature_index]
    return gmm.density(feature_value)

def separate_classes(items, truths):
  # returns two arrays of items,
  # one for items where truth=0 and one where truth=1

  class0 = []
  class1 = []

  for item, truth in zip(items, truths):
    if truth == 0:
      class0.append(item)
    elif truth == 1:
      class1.append(item)
    else:
      raise Exception("truth != 0 or 1, was: " + str(truth))

  return np.array(class0), np.array(class1)



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

  def train_features(self):
    num_features = len(self.items.T)
    self.bucket_edges = [0]*num_features

    zeroes, ones = separate_classes(self.items, self.truths)

    for i, (feature, feature0, feature1) in enumerate(zip(self.items.T, zeroes.T, ones.T)):
      max_val = max(feature)
      min_val = min(feature)
      mean_val = sum(feature) / len(feature)
      mean0 = sum(feature0) / len(feature0)
      mean1 = sum(feature1) / len(feature1)

      if mean0 < mean1:
        self.bucket_edges[i] = [min_val, mean0, mean_val, mean1, max_val]
      else:
        self.bucket_edges[i] = [min_val, mean1, mean_val, mean0, max_val]

    self.bucket_counts = {0: [0]*num_features, 1: [0]*num_features}
    for i in range(num_features):
      self.bucket_counts[0][i] = [0, 0, 0, 0] #start with 0 in all buckets
      self.bucket_counts[1][i] = [0, 0, 0, 0] #start with 0 in all buckets

    for item, truth in zip(self.items, self.truths):
      for i, feature_value in enumerate(item):
        min_val, low_mean, mean, high_mean, max_val = self.bucket_edges[i]
        buckets = self.bucket_counts[truth][i]

        # we're never actually checking the min and max here
        # since it can only cause smoothing problems
        if feature_value < low_mean:
          buckets[0] += 1
        elif low_mean <= feature_value < mean:
          buckets[1] += 1
        elif mean <= feature_value < high_mean:
          buckets[2] += 1
        elif high_mean <= feature_value:
          buckets[3] += 1
        else:
          raise Exception("illegal feature value " + str(feature_value))

  def prob_per_feature_value(self, feature_index, feature_value, clazz):
    min_val, low_mean, mean, high_mean, max_val = self.bucket_edges[feature_index]
    buckets = self.bucket_counts[clazz][feature_index]

    if feature_value < low_mean:
      count = buckets[0]
    elif low_mean <= feature_value < mean:
      count = buckets[1]
    elif mean <= feature_value < high_mean:
      count = buckets[2]
    elif high_mean <= feature_value:
      count = buckets[3]
    else:
      raise Exception("illegal feature value " + str(feature_value))

    return float(count) / sum(buckets)


############# Utility Functions

def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

def column_means(a):
  return [float(sum(l))/len(l) for l in a.T]

def column_stds(a):
  return [np.std(l) for l in a.T]

def roc_curve(truths, predictions):
  # return fpr, tpr, threshholds
  both = zip(predictions, truths)
  both = sorted(both, key=lambda x: x[0]) #sort by predictions
  predictions, truths = zip(*both)

  fprs = []
  tprs = []

  for i, (prediction, truth) in enumerate(zip(predictions, truths)):

    if i != 0:
      tpr = len(filter(lambda x: x == 1, truths[:i]))
      # predicted negatives that are actually positive
      fpr = len(filter(lambda x: x == 0, truths[i:]))
      # predicted positives that are actually negative

      num_negatives = len(filter(lambda x: x==0, truths))
      num_positives = len(filter(lambda x: x==1, truths))

      fprs.append(float(fpr) / num_negatives)
      tprs.append(1 + -1*(float(tpr) / num_positives))

  return fprs, tprs, predictions[1:]

def auc(fprs, tprs):
  #returns area under curve defined by xs and ys

  both = zip(fprs, tprs)
  both = sorted(both, key=lambda x: x[0]) #sort by fprs
  fprs, tprs = zip(*both)

  trapezoids = []

  for i in range(2, len(fprs)):
    trapezoids.append((fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]))

  return sum(trapezoids) / 2

############## Test

data_dir = "./data/"
spam_filename = data_dir + "spambase/spambase.data"
def cross_validate_spam(NaiveBayesClass):
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

def error_tables(NaiveBayesClass):
  data = read_csv_as_numpy_matrix(spam_filename)[:4600,:]
  np.random.shuffle(data)
  train = data[:4000,:]
  test = data[4001:,:]

  features = np.array(train[:,:56])
  truths = train[:,57].A1
  nb = NaiveBayesClass(features, truths)
  nb.train()

  features = np.array(test[:,:56])
  truths = test[:,57].A1

  nb.error_tables(features, truths)
  roc_datapoints = nb.roc_curve_data(features, truths)

def do_all_the_things(clazz):
  #cross_validate_spam(clazz)
  error_tables(clazz)

if __name__ =="__main__":
  print("\nBernoulli")
  do_all_the_things(BernoulliNaiveBayes)
  print("\nGaussian")
  do_all_the_things(GaussianNaiveBayes)
  print("\nHistogram")
  do_all_the_things(HistogramNaiveBayes)
  #print("\nGaussian Mixture Model")
  #do_all_the_things(GMMNaiveBayes)

