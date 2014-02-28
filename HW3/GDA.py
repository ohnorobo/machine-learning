#!/usr/bin/python

import csv
import numpy as np
from pprint import pprint

data_dir1 = "../../data/HW1/"


def read_csv_as_numpy_matrix(filename):
  return np.matrix(list(csv.reader(open(filename,"rb"),
  delimiter=','))).astype('float')

def column_means(a):
  return [float(sum(l))/len(l) for l in a.T]


spam_filename = data_dir1 + "spambase/spambase.data"
data = read_csv_as_numpy_matrix(spam_filename)

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

mean0 = np.array(column_means(features0))
mean1 = np.array(column_means(features1))
covar = np.cov(features_all, rowvar=0)

pprint(mean0)
pprint(mean1)
pprint(covar)

print(len(mean0), len(mean1), covar.shape)





# make means for each

#make a covariance matrix with both


# predict testing data

