#!/usr/bin/python
# -*- coding: utf8 -*-


STOP = .1

class GaussianMixtureModel():

  def __init__(self, num_gaussians):
    # list of gaussians
    # each gaussian is a tuple of mu and sigma

    self.gaussians = []
    self.gaussian_weights = []

    for _ in xrange(num_gaussians):
      mu = 
      sigma = 
      self.gaussians.append((mu, sigma))
      self.gaussian_weights.append(1/num_gaussians)

  def train(data, truth):
    self.features = data.T #TODO, ever used?
    self.items = data
    self.truth = truth

   self.last_likelyhood = self.likelyhood()

    while not self.convergence():
      gamma, n = self.set_expectations()
      self.maximize(gamma, n)

  def set_expectations():
    gamma = np.zeroes((len(self.items), len(self.gaussians)))
    n = np.zeroes(len(self.gaussians))

    for i in range(len(self.items)):
      densities_sum = self.density(self.items[i])

      for j in range(len(self.gaussians)):
        gamma[i,j] = w[j] * \
                     self.one_gaussian_density(self.items[i], self.gaussians[j]) \
                     / densities_sum

    for j in range(len(self.gaussians)):
      n[j] = sum(gamma[:,j])


  def maximize(gamma, n):
    new_mus = np.zeros(len(self.gaussians))
    new_sigmas = np.zeros(len(self.gaussians))

    y = self.truth #TODO ??

    for j in range(len(self.gaussians))
      self.gaussian_weights[j] = n[j] / len(self.items))

      new_mus[j] = np.inner(gamma[:,j], y) / n[j]

      difference = y = new_mus
      new_sigmas[j] = np.dot(gamma[:,j], difference * difference.T) / n[j]

    self.gaussians = zip(new_mus, new_sigmas)

  def convergence():
    new_likelyhood = likelyhood()

    if abs(self.last_likelyhood - new_likelyhood) < STOP:
      self.last_likelyhood = new_likelyhood
      return true
    else
      self.last_likelyhood = new_likelyhood
      return false


  def likelyhood():
    densities = [self.density(item) for item in self.items]
    densities_logged = filter(math.ln, densities)
    return sum(densities_logged) / len(self.items)


  # density of x for a particulr gaussian, no weighting
  def one_gaussian_density(x, gaussian):
    mu = gaussian[0]
    sigma = gaussian[1]

    return (math.e, ** (-1/2 * (x - mu).T * sigma.inverse * (x - mu))) /
           (2 * math.pi) ** d * sigma.absolute) ** 1/2

  #density over all gaussians/weights for x
  def density(self, x):
    densities_per_gaussian = [self.one_gaussian_density(x, gaussian)
                              for gaussian in self.gaussians]
    return np.inner(self.gaussian_weights, densities_per_gaussian)





