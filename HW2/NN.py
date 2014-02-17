#!/usr/bin/python

from pprint import pprint
import numpy as np
import math
from copy import deepcopy


ITERATIONS = 10000
RATE = 10

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
          '''
          pprint("w_hidden[h]")
          pprint(self.w_hidden[h])
          pprint("inp")
          pprint(inp)
          pprint("inner")
          pprint(np.inner(self.w_hidden[h], inp))
          pprint("sigmoid")
          pprint(sigmoid(np.inner(self.w_hidden[h], inp)))
          '''

        for k in range(self.output_size):
          output_outputs[k] = sigmoid(np.inner(self.w_output[k], hidden_outputs))
          '''
          pprint("w_output[k]")
          pprint(self.w_output[k])
          pprint("hidden out")
          pprint(hidden_outputs)
          pprint("inner")
          pprint(np.inner(self.w_output[k], hidden_outputs))
          pprint("sigmoid")
          pprint(sigmoid(np.inner(self.w_output[k], hidden_outputs)))
          '''

        '''
        pprint("outputs")
        pprint(inp)
        pprint(hidden_outputs)
        pprint(output_outputs)
        '''

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
          '''
          pprint("w_output[:,h]T")
          #pprint(self.w_output[:,h])
          pprint(self.w_output[:,h].T)
          '''

        '''
        pprint("errors")
        pprint(hidden_errors)
        pprint(output_errors)
        '''

        '''
        ##TODO print gradients for all weights
        pprint("gradients")
        pprint("output")
        for k in range(self.output_size):
          for h in range(self.hidden_size):
            pprint((k, h, RATE * output_errors[k] * hidden_outputs[h]))
        pprint("hidden")
        for h in range(self.hidden_size):
          for i in range(self.input_size):
            pprint((h, i, RATE * hidden_errors[h] * inp[i]))
        '''

        ##########
        ##### update each weight
        for k in range(self.output_size):
          for h in range(self.hidden_size):
            self.w_output[k,h] += RATE * output_errors[k] * hidden_outputs[h]
        for h in range(self.hidden_size):
          for i in range(self.input_size):
            self.w_hidden[h,i] += RATE * hidden_errors[h] * inp[i]

      '''
      pprint("weights")
      pprint(self.w_hidden)
      pprint(self.w_output)
      '''


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
    test_outputs = map(lambda x: self.test(x), inp)
    pairs = zip(map(np.argmax, out), map(np.argmax, test_outputs))
    #pprint(test_outputs)
    pprint(map(np.argmax, test_outputs))
    total = len(filter(lambda t: t[0] == t[1], pairs))
    return total

def sigmoid(y):
  return 1 / (1 + math.e ** (-1 * y))

import unittest
class TestNeuralNetwork(unittest.TestCase):

  def test_initial_run(self):
    global ITERATIONS
    ITERATIONS = 1

    w_output = [[0.9858889462004379, 0.40755675712967815, -0.11565939691638046 ],
        [0.8178073553033031, -0.6165992666872547, -0.4829456889555309 ],
        [0.4549304150942926, 0.07202285996320211, 0.5806383729984419 ],
        [0.274806571693089, 0.4408120517746435, -0.3110808519944225 ],
        [-0.8189218821186048, 0.39974398676156925, -0.23753194735879007 ],
        [0.1838646543154787, 0.10395666056270134, -0.13156208036205208 ],
        [-0.9584635025269407, 0.8109215282766791, -0.7522210924698075 ],
        [-0.8737585606775781, -0.44115283842559505, -0.39419437805056023 ]]
    w_hidden = [[ 0.6510528870161744, -0.36270888004803103, -0.39700943966442764,
                   0.6952022039964331, -0.25421811203698624, -0.284821556403712,
                   -0.5561762623544277, 0.170584845772344 ],
                 [0.831265025422474, -1.0585914517383777, -0.13020007821675617,
                  0.1221860545238008, -0.38320339261256975, -0.4703515572719222,
                  -0.5807132604354268, -0.35410185236914815 ],
                 [0.7178341974330525, 0.9451762328772098, -0.18679079492828293,
                  -0.1564177980482053, 0.6964245915897823, -0.4486206432299291,
                  0.026641809519414437, -0.6186685331223171 ]]

    w_output = np.matrix(w_output)
    w_hidden = np.matrix(w_hidden)

    old_w_output = deepcopy(w_output)
    old_w_hidden = deepcopy(w_hidden)

    nn = NeuralNetwork(8, 8, 3)
    nn.w_hidden = w_hidden
    nn.w_output = w_output

    nn.train(EIGHT_BIT, EIGHT_BIT)

    pprint("weight diff")
    pprint("outputs")
    pprint(nn.w_output - old_w_output)
    pprint("hidden")
    pprint(nn.w_hidden - old_w_hidden)

    self.assertTrue(False)



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
