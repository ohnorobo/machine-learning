#!/usr/bin/python
from pprint import pprint
import numpy as np

import sys
sys.path.append("./libsvm-3.17/")
from svmutil import *
import mnist

'''
svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

prob = svm_problem([1,-1], [[1,0,1], [-1,0,-1]])

param = svm_parameter()
param.kernel_type = LINEAR
param.C = 10

m=svm_train(prob, param)

m.predict([1,1,1])
'''

'''
# Construct problem in python format
# Dense data
#y, x = [1,-1], [[1,0,1], [-1,0,-1]]
# Sparse data
y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
prob  = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')
m = svm_train(prob, param)
'''


SUBSAMPLE = 10000

digits = [0,1,2,3,4,5,6,7,8,9]
DATAPATH = "../../data/HW5"

train_images, train_labels = mnist.read(digits, dataset='training', path=DATAPATH)
x = np.array(train_images).tolist() #svm requires a list
y = np.array(train_labels).T.astype(float).tolist()[0]

pprint(len(x))

x = x[:SUBSAMPLE]
y = y[:SUBSAMPLE]

#pprint(x)
#pprint(y)

#pprint(x[0])

prob  = svm_problem(y, x)

param = svm_parameter() #('-h 1') #shrinking
param.kernel_type = LINEAR
param.C = 10

m = svm_train(prob, param)

#pprint(m)

test_images, test_labels = mnist.read(digits, dataset='testing', path=DATAPATH)
xtest = np.array(test_images).tolist()
ytest = np.array(test_labels).T.astype(float).tolist()[0]

p_label, p_acc, p_val = svm_predict(ytest, xtest, m)

# pprint(p_label)
# pprint({"acc":p_acc, "val":p_val})

#errors = 0.0
#for guess, truth in zip(p_label, ytest):
#  if guess != truth:
#    errors += 1

#pprint(errors / len(ytest))




