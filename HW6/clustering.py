#!/usr/bin/python
import numpy as np
from pprint import pprint


ITERATIONS = 10

def most_common(lst):
  return max(set(lst), key=lst.count)

class KMeans():

  def __init__(self, xs, ys, num_classes):
    self.centroids = np.random.rand(num_classes, xs.shape[1])
    pprint(self.centroids.shape)
    self.xs = xs
    self.ys = ys

    self.train()
    self.select_labels()

  def select_labels(self):
    clusters = self.clusters_y()
    labels = [most_common(cluster) for cluster in clusters]

    self.labels = labels


  def train(self):
    for r in range(ITERATIONS):
      pprint(r)
      pprint(self.centroids)
      new_clusters = self.clusters()
      self.centroids = self.new_centroids(new_clusters)

  def find_closest_centroid_index(self, point):
    distances = [np.linalg.norm(centroid - point) for centroid in self.centroids]
    #pprint(distances)
    return distances.index(min(distances))

  def clusters(self):
    clusters = [[]] * len(self.centroids)

    for x in self.xs:
      closest = self.find_closest_centroid_index(x)
      clusters[closest].append(x)

    return clusters

  def clusters_y(self):
    clusters = [[]] * len(self.centroids)

    for x, y in zip(self.xs, self.ys):
      closest = self.find_closest_centroid_index(x)
      clusters[closest].append(y)

    return clusters

  def new_centroids(self, clusters):
    centroids = []
    l = len(self.xs)

    for cluster in clusters:
      centroid = np.sum(cluster) / l
      centroids.append(centroids)

    return centroids

  def classify(self, point):
    index = self.find_closest_centroid_index(point)
    return self.labels[index]


class Hierarchical():

  def __init__(self, xs, ys, num_classes):
    pass

  def classify(self, point):
    pass



NEWS = "../../data/HW4/20newsgroup/"
NUM_WORDS = 11350
# number of points:
# in train  = 11314
# in test = 7532
def read_20_newsgroup_data(fname):

  f = open(NEWS + fname)
  lines = f.readlines()

  data = np.zeros((len(lines), NUM_WORDS))
  truths = []

  for i, line in enumerate(lines):
    bits = line.split()
    truth = bits[0]
    points = bits[1:-3]

    truths.append(int(truth))

    for point in points:
      j, count = point.split(':')
      data[i,int(j)] = int(count)

  truths = np.array(truths).T
  data = np.array(data).T

  a = np.vstack((data, truths))
  a = a.T

  return a


# train a classifier, calculate training and testing error
def run_cycle(train, test, classifier_type):
  features = train[:,:train.shape[1]-1]
  truths = train[:,train.shape[1]-1]

  classifier = classifier_type(features, truths, 20)

  error = calculate_error(classifier, features, truths)
  pprint(("training error", error))

  features = test[:,:test.shape[1]-1]
  truths = test[:,test.shape[1]-1]
  error = calculate_error(classifier, features, truths)
  pprint(("testing error", error))

  return classifier

def calculate_error(classifier, features, truths):
  errors = 0
  for item, truth in zip(features, truths):
    guess = classifier.classify(item)
    if guess != truth:
      errors +=1
  return float(errors) / len(truths)

##main
train = read_20_newsgroup_data("train.txt")
test = read_20_newsgroup_data("test.txt")

np.random.shuffle(train)
train = train[:1000]

run_cycle(train, test, KMeans)
run_cycle(train, test, Hierarchical)



