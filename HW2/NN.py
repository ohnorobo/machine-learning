#!/usr/bin/python

from pprint import pprint
import numpy as np



class NeuralNetwork:

  def __init__(self, inp, out, hidden_layers, hidden_size):
    # inp - inputs, list of numbers
    # out - outputs, list of numbers, same length as inputs
    # hidden_layers, # of hidden layers
    # hidden size, # of nodes in each hidden layer
    self.inp = inp
    self.out = out

    if len(inp) != len(out):
      raise Exception("different number of inputs and outputs "
                       + len(inp) + " " + len(out))

    # connections between node layers
    # each connection set is an x by y matrix of values
    # x - input node
    # y - output node
    self.connections = [] #list of matricies, 1 for each set of connections
    num_connections = hidden_layers + 1

    for n in xrange(num_connections):
      if n == 0: #first
        self.connections.append(np.random.rand(len(inp), hidden_size))
      elif n == num_connections - 1: #last
        self.connections.append(np.random.rand(hidden_size, len(out)))
      else: #middle
        self.connections.append(np.random.rand(hidden_size, hidden_size))

  def train(self):
    #modify linear connections until convergence
    pass


  def test(self, inps):
    pass


if __name__ == "__main__":
  inp = [10000000,
         1000000,
         100000,
         10000,
         1000,
         100,
         10,
         1]
  out = inp #autoencoder
  nn = NeuralNetwork(inp, out, 1, 3)
  pprint(nn.connections)
  nn.train()

  test_outputs = nn.test(inp)
  pprint("inputs, result")
  pprint(zip(inp, test_outputs))
