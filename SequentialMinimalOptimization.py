#!/usr/bin/python
from pprint import pprint
import numpy as np
import mnist

MAX_PASSES = 15


# http://cs229.stanford.edu/materials/smo.pdf
class SMO:
  def __init__(self, xs, ys):
    self.x = xs
    self.y = ys

    self.a = np.zeros(x.shape[0])
    self.b = 0
    self.c = .5 #constraint
    self.tolerance = .005

    self.loop()

  def loop(self):

    converged = False
    passes = 0

    while passes < MAX_PASSES:
      pprint(("pass", passes))
      for i in range(len(self.a)): #iterate over all ai

        e_i = self.predict(self.x[i]) - self.y[i]
        if self.editable(i, e_i):

          # choose a_j to update
          j = self.get_index_to_update(i)
          e_j = self.predict(self.x[j]) - self.y[j]
          #while not self.editable(j, e_j):
          #  j = self.get_index_to_update(i)
          #  e_j = self.predict(self.x[j]) - self.y[j]

          #pprint(("indexes", i, j))

          # reoptimize by changing only a_i and a_j
          converged = self.update_as(i, j, e_i, e_j)


      #pprint((self.a, self.b))
      #self.print_support_vectors()
      #pprint(("b", self.b))
      self.test_all(self.x, self.y)
      passes += 1

  def get_bounds(self, i, j):
    a = self.a

    #pick upper and lower bounds
    if y[i] == y[j]:
      l = max(0, a[j] + a[i] - self.c)
      h = min(self.c, a[i] + a[j])
    else:
      l = max(0, a[j] - a[i])
      h = min(self.c, self.c + a[j] - a[i])

    return (l, h)

  def editable(self, i, e_i):
    return ((e_i*self.y[i] < -1*self.tolerance and self.a[i] < self.c)
            or (e_i*self.y[i] > self.tolerance and self.a[i] > 0))



  def update_as(self, i, j, e_i, e_j):
    x = self.x #just to make equations easier to read
    y = self.y
    a = self.a

    l, h = self.get_bounds(i, j)

    #calculate errors
    #e_i = self.predict(x[i]) - y[i]
    #e_j = self.predict(x[j]) - y[j]
    eta = 2 * np.inner(x[i], x[j]) - np.inner(x[i], x[i]) - np.inner(x[j], x[j])

    if eta >= 0 or l == h:
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

    '''
    pprint(("calculateing b", self.b,
           e_i,
           y[i] * (a[i] - old_a_i) * np.inner(x[i], x[i]),
           y[j] * (a[j] - old_a_j) * np.inner(x[i], x[j]),
           e_j,
           y[i] * (a[i] - old_a_i) * np.inner(x[i], x[j]),
           y[j] * (a[j] - old_a_j) * np.inner(x[j], x[j])))
    '''

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

    #self.b = self.b[0] #get single value out of b

    #pprint({"old a i":old_a_i, "old a j":old_a_j,
    #        "new a_i":a[i], "new a j":a[j],
    #        "old b":old_b, "new_b":self.b})

    # return whether we converged
    if abs(a[i] - old_a_i) < self.tolerance and abs(a[j] - old_a_j) < self.tolerance:
      return True
    else:
      return False

  def get_index_to_update(self, i):
    j = i
    while j == i: # j cannot equal i
      j = np.random.randint(0,self.a.shape[0]) #pick a random j
    return j

  def print_support_vectors(self):
    vectors = []
    for i, a in enumerate(self.a):
      if a != 0:
        vectors.append((i, a))

    pprint(vectors)

  def predict(self, point):
    inners = np.array([np.inner(x[i], point) for i in range(len(x))])
    s = np.sum(self.a * self.y * inners)

    #pprint((self.a.shape, self.y.shape, inners.shape))
    #pprint((self.a, self.y, inners))

    return s + self.b

  def test_all(self, xs, ys):
    guesses = [self.predict(point) for point in xs]
    error = 0

    #pprint(zip(guesses, ys))

    for y, guess in zip(self.y, guesses):
      if guess > 0:
        guess = 1
      else:
        guess = -1

      if y != guess:
        error += 1
        #pprint((y, guess))
      else:
        #pprint((y, guess))
        pass

    pprint(("error", float(error) /len(self.y)))

class OneToManySMO():

  def __init__(self, xs, ys, classes):
    self.classes = classes
    self.smos = []

    for clazz in classes:
      one_vs = self.make_one_vs_many(clazz, ys)
      pprint(("classifying vs", clazz))
      smo = SMO(x, one_vs)
      self.smos.append(smo)

  def make_one_vs_many(self, clazz, ys):
    return map(lambda y: 1 if y == clazz else -1, ys)

  def test_all(self, xs, ys):
    error = 0
    for x, y in zip(xs, ys):
      probs = [smo.predict(x) for smo in self.smos]

      #pprint(probs)

      i = probs.index(max(probs))
      guess = self.classes[i]

      if guess != y:
        error += 1
        #pprint((guess, y))
      else:
        #pprint((guess, y))
        pass

    pprint(("error %", float(error) / len(ys)))


def count_unique(keys):
  uniq_keys = np.unique(keys)
  bins = uniq_keys.searchsorted(keys)
  pprint(zip(uniq_keys, np.bincount(bins)))


## main

DATAPATH = "./data/mnist/"
digits = [0,1,2,3,4,5,6,7,8,9]

TRAIN_AMT = 1000
TEST_AMT = 1000

train_images, train_labels = mnist.read(digits, dataset='training', path=DATAPATH)
x = np.array(train_images)[:TRAIN_AMT]
y = np.array(train_labels).astype(float).T[0][:TRAIN_AMT]

count_unique(y)

#pprint((x, y))
pprint((x.shape, y.shape))

digits = [2]

smo = OneToManySMO(x, y, digits)

test_images, test_labels = mnist.read(digits, dataset='testing', path=DATAPATH)
xtest = np.array(test_images)[:TEST_AMT]
ytest = np.array(test_labels).astype(float).T[0][:TEST_AMT]

smo.test_all(xtest, ytest)
