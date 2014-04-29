#!/usr/bin/python

import csv
import numpy as np
from pprint import pprint
from scipy.stats import multivariate_normal

data_dir1 = "./data/"

def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
  delimiter=','))).astype('float')

def column_means(a):
  return [float(sum(l))/len(l) for l in a.T]

spam_filename = data_dir1 + "spambase/spambase.data"
data = read_csv_as_numpy_matrix(spam_filename)

np.random.shuffle(data)

train = data[:4000,:]
test = data[4001:,:]

data_all = train
data_all = np.array(data_all)

class0 = []
class1 = []

for item in data_all:
  #pprint(item)
  if item[-1] == 0:
    class0.append(item)
  elif item[-1] == 1:
    class1.append(item)
  else:
    pprint(item)
    pprint("FAILURE")

class0 = np.array(class0)
class1 = np.array(class1)

features0 = class0[:,:56]
truth0 = class0[:,57]

features1 = class1[:,:56]
truth1 = class1[:,57]

# make means for each
mean0 = np.array(column_means(features0))
mean1 = np.array(column_means(features1))
#make a covariance matrix with both
covar = np.cov(data_all[:,:56], rowvar=0)
covar = np.matrix(covar)

pprint("means and covar")
pprint(mean0)
pprint(mean1)
pprint(covar)

print(len(mean0), len(mean1), covar.shape)

# testing data
features_test = test[:,:56]
truth_test = test[:,57]

num_correct = 0

for item, truth in zip(features_test, truth_test):
  prob0 = multivariate_normal.pdf(item, mean=mean0, cov=covar)
  prob1 = multivariate_normal.pdf(item, mean=mean1, cov=covar)

  if prob0 >= prob1:
    result = 0
  else:
    result = 1

  if result == truth.A1[0]:
    num_correct += 1

pprint(("correct", num_correct))
pprint(("total", len(truth_test)))

pprint(("percent", float(num_correct) / len(truth_test)))

