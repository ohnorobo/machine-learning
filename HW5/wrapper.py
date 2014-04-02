#!/usr/bin/python
import sys
sys.path.append("./libsvm-3.17/")
from svmutil import *

svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

prob = svm_problem([1,-1], [[1,0,1], [-1,0,-1]])

param = svm_parameter()
param.kernel_type = LINEAR
param.C = 10

m=svm_train(prob, param)

m.predict([1,1,1])
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
