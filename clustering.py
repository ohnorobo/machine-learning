#!/usr/bin/python
import numpy as np
from pprint import pprint


ITERATIONS = 10

def most_common(lst):
  return max(set(lst), key=lst.count)


class Clustering():

  def classify(self, point):
    index = self.find_closest_centroid_index(point)
    return self.labels[index]

  def find_closest_centroid_index(self, point):
    distances = [np.linalg.norm(centroid - point) for centroid in self.centroids]
    #pprint(distances)
    return distances.index(min(distances))

  def clusters_y(self):
    clusters = [[] for x in xrange(0,len(self.centroids))]

    for x, y in zip(self.xs, self.ys):
      closest = self.find_closest_centroid_index(x)
      clusters[closest].append(y)

    return clusters

  def select_labels(self):
    clusters = self.clusters_y()
    labels = [most_common(cluster) for cluster in clusters]

    self.labels = labels

  def calculate_centroid(self, cluster):
    return np.sum(cluster, axis=0) / len(cluster)




class KMeans(Clustering):

  def __init__(self, xs, ys, num_classes):
    self.xs = xs
    self.ys = ys

    self.centroids = self.pick_starting_centroids(num_classes)
    pprint(self.centroids.shape)

    self.train()
    self.select_labels()

  '''
  def pick_starting_centroids(self, num_classes):
    mins = self.xs.min(axis=0)
    maxs = self.xs.max(axis=0)

    centroids = []
    for i in range(num_classes):
      centroid = []
      for mmin, mmax in zip(mins, maxs):
        centroid.append(np.random.uniform(mmin, mmax))
      centroids.append(centroid)
    return np.array(centroids)
  '''

  def pick_starting_centroids(self, num_classes):
    #pick 20 random points as initial centroids
    pick = np.random.randint(len(self.xs), size=num_classes)
    centroids = self.xs[pick,:]
    return centroids

  def train(self):
    for r in range(ITERATIONS):
      pprint(r)
      pprint(self.centroids)
      new_clusters = self.clusters()
      self.centroids = self.new_centroids(new_clusters)

  def clusters(self):
    clusters = [[] for x in xrange(0,len(self.centroids))]

    for x in self.xs:
      closest = self.find_closest_centroid_index(x)
      clusters[closest].append(x)

    return clusters

  def new_centroids(self, clusters):
    centroids = []

    for cluster in clusters:
      centroid = self.calculate_centroid(cluster)
      pprint(("cluster shape", len(cluster)))
      pprint(("centroid shape", centroid.shape))
      centroids.append(centroid)

    a_centroids = np.array(centroids, dtype=float)

    pprint((a_centroids, a_centroids.shape))

    return a_centroids

HIGH = 10000.0

class Hierarchical(Clustering):

  def __init__(self, xs, ys, num_classes):
    self.xs = xs
    self.ys = ys
    self.num_classes = num_classes
    self.clusters = self.initial_clusters()
    self.centroids = self.initial_centroids()
    self.proximity_matrix = self.get_proximity_matrix()

    self.train()
    self.select_labels()

    pprint(self.labels)

  def initial_clusters(self):
    c = []
    for point in self.xs:
      c.append([point]) #put each point in a 1-element list
    return c

  def initial_centroids(self):
    c = []
    for point in self.xs:
      c.append(point)
    return c

  def get_proximity_matrix(self):
    m = np.zeros((len(self.centroids), len(self.centroids)))

    for i, centroid_a in enumerate(self.centroids):
      for j, centroid_b in enumerate(self.centroids):
        if i == j:
          #pprint(i)
          m[i,j] = float(HIGH)
        else:
          m[i,j] = np.linalg.norm(centroid_a - centroid_b)
    return m


  def train(self):
    while len(self.centroids) > self.num_classes:
      pprint(("num centroids and classes", len(self.centroids), self.num_classes))
      pprint(("prox matrix", self.proximity_matrix, self.proximity_matrix.shape))
      a,b = np.unravel_index(self.proximity_matrix.argmin(), self.proximity_matrix.shape)
      pprint({"a":a, "b":b, "argmin":self.proximity_matrix.argmin(),
              "value":self.proximity_matrix[a,b]})

      merged = self.clusters[a]
      merged.extend(self.clusters[b])
      merged_centroid = self.calculate_centroid(merged)
      #pprint(merged)

      del self.clusters[min(a,b)]
      del self.clusters[max(a,b)-1]
      self.clusters.append(merged)

      del self.centroids[min(a,b)]
      del self.centroids[max(a,b)-1]
      self.centroids.append(merged_centroid)

      self.proximity_matrix = self.update_proximity_matrix(self.proximity_matrix,
          merged_centroid, a, b)


  def update_proximity_matrix(self, old_prox, new_centroid, a, b):
    old_prox = np.delete(old_prox, [a,b], 0) #delete rows
    old_prox = np.delete(old_prox, [a,b], 1) #delete cols

    # add a line of zeroes on the right and bottom edges
    mid = np.hstack((old_prox, np.zeros((old_prox.shape[0], 1), dtype=old_prox.dtype)))
    pprint(("mid", mid, mid.shape))
    new_prox = np.vstack((mid, np.zeros((1, mid.shape[1]), dtype=mid.dtype)))

    pprint(("expanded", new_prox, new_prox.shape))

    old_length = len(old_prox) - 1
    new_length = len(new_prox) - 1

    #fill them in with new comparisons
    new_prox[new_length,new_length] = float(HIGH)

    for i, centroid in enumerate(self.centroids[:-1]):
      diff = np.linalg.norm(centroid - new_centroid)

      pprint(("checking", diff, i))

      new_prox[new_length,i] = diff
      new_prox[i,new_length] = diff

    pprint(("new prox", new_prox, new_prox.shape))

    return new_prox







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

  pprint(set(truths))

  classifier = classifier_type(features, truths, 8)

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
train = train[:100]
train = train[:1000,:] #lose most features

#run_cycle(train, test, KMeans)
run_cycle(train, test, Hierarchical)



