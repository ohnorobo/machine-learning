#!/usr/bin/python
import numpy as np
from scipy.stats import multivariate_normal
from pprint import pprint

K_FOLDS = 10


class NaiveBayes():

  def __init__(self, features, truth, model="bernoulli")
    # Makes a Naive bayes classifier modeling each feature as one of
    # "bernoulli"
    # "gaussian"
    # "histogram"

    self.items = features
    self.truth = truth
    self.model = model
    self.classes = [0, 1] #built in


    def train(self):
      # per feature
      # calculate a number of classes each with a p(class) and p(feature_value | class)
      self.prob_per_class = self.calculate_prob_per_classes()
      self.prob_per_feature = 

    # returns the predicted value for item
    def test(self, item):
      # p(class | item) = product( p(feature_value | class) * p(class) )
      # p(feature_value | class) = 
      pass


   def prob_value_given_class(self, feature_index, feature_value, clazz):
     pass


   def prob_class(self, clazz):
     pass


class BernoulliNaiveBayes(NaiveBayes):
   # bernoulli - choose mean m for each feature
   # p(f<=m | 0)
   # p(f> m | 0)
   # p(f<=m | 1)
   # p(f> m | 1)

class GaussianNaiveBayes(NaiveBayes):
   # gaussian, choose gaussian(m, sigma) for each feature (1Dimensional)
   # p(m_0, sigma_0 | 0)
   # p(m_0, sigma_0 | 1)
   # = gaussian.prob_density(m, sigma, value)

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




def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),delimiter=','))).astype('float')

data_dir = "../../data/HW1/"
def cross_validate_spam(NaiveBayesClass):
  spam_filename = data_dir + "spambase/spambase.data"
  data = read_csv_as_numpy_matrix(spam_filename)[:4600,:]

  np.random_shuffle(data) #truffle shuffle

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

    features = train[:,:56]
    truth = train[:,57]
    nb = NaiveBayesClass(features, truth)

    features = test[:,:56]
    truth = test[:,57]
    error = nb.classify_all_error(features, truth)
    total_error += error / truth.size

    #pprint("cv: " + str(i))
    #pprint(error / truth.size)

  pprint("avg error")
  pprint(total_error / K_FOLDS)

if __name__ =="__main__":
  pprint("Bernoulli")
  cross_validate_spam(BernoulliNaiveBayes)
  pprint("Gaussian")
  cross_validate_spam(GaussianNaiveBayes)
  pprint("Histogram")
  cross_validate_spam(HistogramNaiveBayes)

