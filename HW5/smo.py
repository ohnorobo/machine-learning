#!/usr/bin/python
from pprint import pprint
import numpy as np
import mnist


# http://cs229.stanford.edu/materials/smo.pdf
class SMO:
  def __init__(self, xs, ys):
    self.x = xs
    self.y = ys

    self.a = np.random.rand(x.shape[0])
    self.b = np.random.rand(1)[0]
    self.c = .5 #constraint
    self.tolerance = .005

    self.loop()

  def loop(self):

    converged = False
    n = 0

    while not converged:
      pprint(("loop", n))

      # choose a_i, a_j to update
      i, j = self.get_indexes_to_update()

      pprint(("indexes", i, j))

      # reoptimize by changing only a_i and a_j
      converged = self.update_as(i, j)

      n += 1

  def update_as(self, i, j):
    x = self.x #just to make equations easier to read
    y = self.y
    a = self.a

    #pick upper and lower bounds
    if y[i] == y[j]:
      l = max(0, a[j] + a[i] - self.c)
      h = min(self.c, a[i] + a[j])
    else:
      l = max(0, a[j] - a[i])
      h = min(self.c, self.c + a[j] - a[i])

    #calculate errors
    e_i = self.predict(x[i]) - y[i]
    e_j = self.predict(x[j]) - y[j]
    eta = 2 * np.inner(x[i], x[j]) - np.inner(x[i], x[i]) - np.inner(x[j], x[j])

    if eta == 0:
      return #cannot progress on these indexes

    old_a_j = a[j] #save for future calculations
    old_a_i = a[i]
    old_b = self.b

    # optimize a_j
    a[j] = a[j] - ((y[j] * (e_i - e_j)) / eta)

    #make sure a_j is within [l,h]
    if h < a[j]:
      a[j] = h
    elif a[j] < l:
      a[j] = l
    #else:
      # a_j remains the same

    # optimize a_i
    a[i] = a[i] + y[j] * y[i] * (old_a_j - a[j])

    # update b
    b1 = self.b \
           - e_i \
           - y[i] * (a[i] - old_a_i) * np.inner(x[i], x[i]) \
           - y[j] * (a[j] - old_a_j) * np.inner(x[i], x[j])
    b2 = self.b \
           - e_j \
           - y[i] * (a[i] - old_a_i) * np.inner(x[i], x[j]) \
           - y[j] * (a[j] - old_a_j) * np.inner(x[j], x[j])

    if 0 < a[i] < self.c:
      self.b = b1
    elif 0 < a[j] < self.c:
      self.b = b2
    else:
      self.b = (b1 + b2) / 2

    self.b = self.b[0] #get single value out of b

    pprint({"old a i":old_a_i, "old a j":old_a_j,
            "new a_i":a[i], "new a j":a[j],
            "old b":old_b, "new_b":self.b})

    # return whether we converged
    if abs(a[i] - old_a_i) < self.tolerance and abs(a[j] - old_a_j) < self.tolerance:
      return True
    else:
      return False

  def get_indexes_to_update(self):
    # return 2 random indexes
    return np.random.randint(0,self.a.shape[0],2)

  def predict(self, point):
    pprint({"x": x.shape, "point":point.shape})

    #s = 0
    inners = [np.inner(x[i], point) for i in range(len(x))]
    #for a_i, y_i, inner_i in zip(self.a, self.y, inners):
    #  s += a_i * y_i * inner_i
    s = np.sum(self.a, self.y.T, inners)

    return s + self.b

  def test_all(self, xs, ys):
    guesses = [self.predict(point) for point in xs]
    sq_error = 0

    for y, guess in zip(self.y, guesses):
      diff = abs(y - guess)
      sq_error += diff ** 2

    mean_sq_error = sq_error / len(ys)
    pprint(("mean squared error", mean_sq_error))




## main

DATAPATH = "../../data/HW5"
digits = [0,1,2,3,4,5,6,7,8,9]

train_images, train_labels = mnist.read(digits, dataset='training', path=DATAPATH)
x = np.array(train_images)
y = np.array(train_labels).astype(float)

pprint((x, y))
pprint((x.shape, y.shape))

smo = SMO(x, y)

test_images, test_labels = mnist.read(digits, dataset='testing', path=DATAPATH)
xtest = np.array(test_images)
ytest = np.array(test_labels).astype(float)

smo.test_all(xtest, ytest)
