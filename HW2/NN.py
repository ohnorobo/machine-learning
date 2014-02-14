#!/usr/bin/python

from pprint import pprint
import numpy as np
import math


ITERATIONS = 10000
RATE = 0.1

EIGHT_BIT = [[1,0,0,0,0,0,0,0],
             [0,1,0,0,0,0,0,0],
             [0,0,1,0,0,0,0,0],
             [0,0,0,1,0,0,0,0],
             [0,0,0,0,1,0,0,0],
             [0,0,0,0,0,1,0,0],
             [0,0,0,0,0,0,1,0],
             [0,0,0,0,0,0,0,1]]


class NeuralNetwork:

  def __init__(self, inps, outs, hidden):
    # inp    - number of inputs
    # out    - number of outputs
    # hidden - number of hidden nodes
    self.input_size = inps
    self.output_size = outs
    self.hidden_size = hidden

    # weights to hidden nodes
    self.w_hidden = (np.matrix(np.random.rand(hidden, inps)) - .5) / 50
    # weights to output nodes
    self.w_output= (np.matrix(np.random.rand(outs, hidden)) - .5) / 50

    #pprint(self.w_hidden)
    #pprint(self.w_output)

  def train(self, initial_inputs, final_outputs):
    for iteration in range(ITERATIONS):
      for inp, out in zip(initial_inputs, final_outputs):

        ##########
        ##### feed forward
        hidden_outputs = np.zeros(self.hidden_size)
        output_outputs = np.zeros(self.output_size)

        # calculate output values for all hidden and output nodes
        #pprint(self.w_hidden.shape)
        #pprint(self.w_output.shape)
        for h in range(self.hidden_size):
          hidden_outputs[h] = sigmoid(np.inner(self.w_hidden[h], inp))
        for k in range(self.output_size):
          output_outputs[k] = sigmoid(np.inner(self.w_output[k], hidden_outputs))

        #pprint("outputs")
        #pprint(inp)
        #pprint(hidden_outputs)
        #pprint(output_outputs)

        ##########
        ##### propagate backward
        hidden_errors = np.zeros(self.hidden_size)
        output_errors = np.zeros(self.output_size)

        # calculate output error
        for k in range(self.output_size):
          output_errors[k] = output_outputs[k] * \
                             (1 - output_outputs[k]) * \
                             (out[k] - output_outputs[k])

        #pprint(self.w_output)
        #pprint(self.w_output[h])

        for h in range(self.hidden_size):
          hidden_errors[h] = hidden_outputs[h] * \
                            (1 - hidden_outputs[h]) * \
                            np.inner(self.w_output[:,h].T, output_errors)

        #pprint("errors")
        #pprint(hidden_errors)
        #pprint(output_errors)

        ##TODO print gradients for all weights

        ##########
        ##### update each weight
        for k in range(self.output_size):
          for h in range(self.hidden_size):
            self.w_output[k,h] += RATE * output_errors[k] * hidden_outputs[h]
        for h in range(self.hidden_size):
          for i in range(self.input_size):
            self.w_hidden[h,i] += RATE * hidden_errors[h] * inp[i]


      pprint("weights")
      pprint(self.w_hidden)
      pprint(self.w_output)


      #for testing the autoencoder
      pprint("iteration: " + str(iteration))
      pprint("number correct: " + \
             str(self.calculate_number_correct(EIGHT_BIT, EIGHT_BIT)))
      #pprint(self.w_hidden)
      #pprint(self.w_output)

    pprint("final weights")
    pprint(self.w_hidden)
    pprint(self.w_output)




  def test(self, inp):
    input_nodes = inp
    hidden_nodes = np.zeros(self.hidden_size)
    output_nodes = np.zeros(self.output_size)

    # calculate hidden values
    for j in range(self.hidden_size):
      hidden_nodes[j] = sigmoid(np.inner(self.w_hidden[j], input_nodes))

    for k in range(self.output_size):
      output_nodes[k] = sigmoid(np.inner(self.w_output[k], hidden_nodes))

    clean_output = np.zeros(self.output_size)
    max_index = np.argmax(output_nodes)
    clean_output[max_index] = 1

    return clean_output


  def calculate_number_correct(self, inp, out):
    test_outputs = map(lambda x: nn.test(x), inp)
    pairs = zip(map(np.argmax, out), map(np.argmax, test_outputs))
    #pprint(test_outputs)
    pprint(map(np.argmax, test_outputs))
    total = len(filter(lambda t: t[0] == t[1], pairs))
    return total

def sigmoid(y):
  return 1 / (1 + math.e ** y)


if __name__ == "__main__":
  inp = EIGHT_BIT
  out = EIGHT_BIT
  nn = NeuralNetwork(8, 8, 3)
  nn.train(inp, out)

  test_outputs = map(lambda x: nn.test(x), inp)
  pprint("inputs, result")
  #pprint(zip(inp, test_outputs))
  #pprint(zip(map(np.argmax, inp), map(np.argmax, test_outputs)))
  #pprint(nn.calculate_number_correct(inp, inp))
