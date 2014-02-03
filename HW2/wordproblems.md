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

#### writeup

How to design things quickly by building quick-n-dirty algorithms and diagnosing their problems instead of architecting elaborate systems from the get-go.
(this is a good way to solve practical problems, not to do novel ml research)

How to choose good disgnostic procedures when your algorithms do have problems.
  helicopter example
  don't just try changing things randomly

Some basic diagnostic procedures and examples:

  Analysis
  ablative analysis -
    how to decide which parts of your system are the most helpful
    try system without each component so you know which are helpful and which aren't
  error analysis -
    look at specific errors to try and see what your're missing in the data
    (how your human intuition doesn't map to what the computer is doing)
    and architect more features using that information

  Training error vs Testing Error
  high variance -
    If training error is much lower than your testing error then youre overfitting.
    Try more (different) training data or using a more tialored (smaller) feature set.
  high bias -
    If both training and testing error are too high (and similar) 
    then you don't have enough power, try adding more features and different features

  Accuracy is not high enough
  bad opimization algorithm -
    if your error is not minimized/objective is not maximized that means something is wrong
    with your algorithm, you should change your features/use a new algorithm
  bad optimization objective -
    when your testing accuracy is low but your error is minimized/objective is maximized
    this means your testing objective doesn't map well to the accuracy you're trying to gain
    Use different learning rate



## problem 8

### DHS ch6 pb1

#### Show that if the transfer function of the hidden units is linear, a three-layer network is equivalent to a two-layer one. Explain why, therefore, that a three-layer network with linear hidden units cannot solve a non-linearly separable problem such as XOR or n-bit parity.
