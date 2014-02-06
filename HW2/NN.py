#!/usr/bin/python

from pprint import pprint
import numpy as np



class NeuralNetwork:

  def __init__(self, inps, outs, hidden_layers, hidden_size):
    # inp - inputs, list of numbers
    # out - outputs, list of numbers, same length as inputs
    # hidden_layers, # of hidden layers
    # hidden size, # of nodes in each hidden layer
    self.inps = inps
    self.outs = outs
    self.hidden_size = hidden_size

    if len(inps) != len(outs):
      raise Exception("different number of inputs and outputs "
                       + len(inps) + " " + len(outs))

    # connections between node layers
    # each connection set is an x by y matrix of values
    # x - input node
    # y - output node
    self.connections = [] #list of matricies, 1 for each set of connections
    self.num_connections = hidden_layers + 1
    self.threshholds = np.random.rand(hidden_layers, hidden_size)

    for n in xrange(self.num_connections):
      if n == 0: #first
        self.connections.append(np.random.rand(len(self.inps),
                                               self.hidden_size))
      elif n == self.num_connections - 1: #last
        self.connections.append(np.random.rand(self.hidden_size,
                                               len(self.outs)))
      else: #middle
        self.connections.append(np.random.rand(self.hidden_size,
                                               self.hidden_size))

  def train(self):
    #modify linear connections until convergence
    pass


  def test(self, inp):
    pprint("input: " + str(inp))

    prev_values = []
    for possible_input in self.inps:
      if inp == possible_input:
        prev_values.append(1)
      else:
        prev_values.append(0)

    pprint(prev_values)

    for connection_index in range(self.num_connections):
      current_links = self.connections[connection_index]

      if connection_index == 0: #first
        num_prev_nodes = len(self.inps)
        num_next_nodes = self.hidden_size
      elif connection_index == self.num_connections - 1: #last
        num_prev_nodes = self.hidden_size
        num_next_nodes = len(self.outs)
      else: #middle
        num_prev_nodes = self.hidden_size
        num_next_nodes = self.hidden_size

      next_values = np.zeros(num_next_nodes)

      for j in range(num_next_nodes):
        for i in range(num_prev_nodes):
          next_values[j] += prev_values[i] * current_links[i][j]

      pprint(next_values)

      prev_values = next_values

    output_index = np.argmax(prev_values)
    return self.outs[output_index]





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

  test_outputs = map(lambda x: nn.test(x), inp)
  pprint("inputs, result")
  pprint(zip(inp, test_outputs))
