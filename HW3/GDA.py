#!/usr/bin/python

import csv
import numpy as np
from pprint import pprint
import scipy.stats

data_dir1 = "../../data/HW1/"


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

features_all = train[:,:56]

class0 = train[train[:,-1]!=0,:] #filter to only class 0
class1 = train[train[:,-1]!=1,:] #filter to only class 1

print(class0.shape, class1.shape)

features0 = class0[:,:56]
truth0 = class0[:,57]

features1 = class1[:,:56]
truth1 = class1[:,57]

# make means for each
mean0 = np.array(column_means(features0))
mean1 = np.array(column_means(features1))
#make a covariance matrix with both
covar = np.cov(features_all, rowvar=0)

pprint(mean0)
pprint(mean1)
pprint(covar)

print(len(mean0), len(mean1), covar.shape)


dist0 = scipy.stats.norm(mean0, covar)
dist1 = scipy.stats.norm(mean1, covar)

# predict testing data
features_test = test[:,:56]
truth_test = test[:,57]

num_correct = 0

for item, truth in zip(features_test, truth_test):
  prob0 = dist0.cdf(item)
  prob1 = dist1.cdf(item)

  #pprint((prob0, prob1))

  sum0 = 0
  sum1 = 1

  for line0, line1 in zip(prob0, prob1):
    for item0, item1 in zip(line0, line1):
      #pprint((item0, item1))
      #pprint((sum0, sum1))
      if item0 >= 0:
        sum0 += item0
      if item1 >= 0:
        sum1 += item1

  #pprint("sum")
  #pprint((sum0, sum1))

  if sum0 > sum1:
    result = 0
  else:
    result = 1

  if result == truth.A1[0]:
    num_correct += 1
    pprint(("correct", result, truth.A1[0]))
  else:
    pprint(("incorrect", result, truth.A1[0]))

pprint(("correct", num_correct))
pprint(("total", len(truth_test)))

