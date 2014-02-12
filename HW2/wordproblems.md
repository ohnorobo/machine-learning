## problem 1

table

    decision tree
    spam
    test: 0.22420294697213022
    train: 0.015287836100290475
    housing
    test: 55.9635359129718
    train: 8.7948615499824019

    regression, normal
    spam
    test: 0.22745483944643188
    train: 0.11278389586859197
    housing
    test: 21.483099127900058
    train: 28.793140665812356

    regression, grad decent
    spam
    test:
    train:
    housing
    test:
    train:

    log regression:
    spam
    test:
    train:
    housing
    test:
    train:


## problem 2

perceptron output:


## problem 3

### b

Viewed as a encoder/decoder algorithm the training procedure is learning how to encode/decode the data effectivly. Depending on the number of hidden nodes the training tries to find a procedure for reducing the i bits of the input into j hidden bits, while still being able to recover the i output bits at the end. To effectivly train an encoder you would have to train it multiple times, each time lowering the number of hidden nodes, eventually choosing the encoder with the smallest number of nodes which could nevertheless encode/decode the data accurately. In the 8-3-8 example the fact that every number only contains a single '1' is being encoded, and allows for the compression, in a more realistic dataset the encoder finds logical relationships betwen features (f1 and f2 implies f3), (f4 and not f5 implies f2), etc. or linear relationships like f4 ~ f3*4 + f2.

### c

No, each layer of the network can only contain as many 'bits' of information as there are nodes in they layer. The 10000000 representation is an expansion of the numbers 0-7. The densest way to represent those numbers is in binary. Representing a numbern in binary requires ceiling(log(n)) bits, so representing the eight inputs requires at least log(8) = 3 bits, fewer than that and information will be lost.


## problem 4

### DHS ch5 pb2 (pg 271)

#### Figure 5.3 illustrates the two most popular methods for designing a c-category classiﬁer from linear boundarysegments. Another method is to save the full (c,2) linear ωi/ωj boundaries, and classifyanypoint bytaking a vote based on all these boundaries. Prove whether the resulting decision regions must be convex. If theyneed not be convex, construct a non-pathological example yielding at least one non-convex decision region.

    f1 f2 | solution
    ------+---------
     1 1  | 0
     1 0  | 1
     0 1  | 1
     0 0  | 0


## problem 7

### Andrew Ng's lecture

How to design things quickly by building quick-n-dirty algorithms and diagnosing their problems instead of architecting elaborate systems from the get-go. (this is a good way to solve practical problems, not to do novel ml research)

How to choose good disgnostic procedures when your algorithms do have problems.

* helicopter example

* don't just try changing things randomly

### Some basic diagnostic procedures and examples:

#### Analysis

ablative analysis -

* How to decide which parts of your system are the most helpful. Try system without each component so you know which are helpful and which aren't

error analysis -

* look at specific errors to try and see what your're missing in the data (how your human intuition doesn't map to what the computer is doing) and architect more features using that information

#### Training error vs Testing Error

high variance -

* If training error is much lower than your testing error then youre overfitting. Try more (different) training data or using a more tialored (smaller) feature set.

high bias -

* If both training and testing error are too high (and similar) then you don't have enough power, try adding more features and different features

Accuracy is not high enough

bad opimization algorithm -

* if your error is not minimized/objective is not maximized that means something is wrong with your algorithm, you should change your features/use a new algorithm

bad optimization objective -

* when your testing accuracy is low but your error is minimized/objective is maximized this means your testing objective doesn't map well to the accuracy you're trying to gain. Use different learning rate



## problem 8

### DHS ch6 pb1

#### Show that if the transfer function of the hidden units is linear, a three-layer network is equivalent to a two-layer one. Explain why, therefore, that a three-layer network with linear hidden units cannot solve a non-linearly separable problem such as XOR or n-bit parity.
