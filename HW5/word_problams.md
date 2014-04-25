##problem 5

###Explain why 0 ≤ α ≤ C/m is a constraint in the dual optimization with slack variables. 

(HINT: read Chris Burges tutorial first)

###Distinguish three cases, and explain them in terms of the classification and constraints:

    a) 0 = α ;
    b) 0 < α < C/m ;
    c) α = C/m.

###This has been discussed in class and in SVM notes; a detailed rigurous explanation is expected.


##problem 6

###Consider the following 6 points in 2D, for two classes:

    class 0:   (1,1)   (2,2)    (2,0)

    class 1:   (0,0)   (1,0)    (0,1)

### a) Plot these 6 points, construct the optimal hyperplane by inspection and intuition (give th e W,b) and calculate the margin.

     |
    \| o o
     \
     +\o
     | \
    -+-+\---
     |   \
     0 1 2 ...

    hyperplane = -x + 1.5

intersection of y=x and y=-x+1.5 is (.75, .75), the distance between (.75, .75) and (1, 1) is root(2 * (.25)*2) or approx .353553...


### b) Which points are support vectors ?

    +(1, 0)
    +(0, 1)
    o(1, 1)

### c) Construct the hyperplane by solving the dual optimization problem using the Lagrangian. Compare with part (a).


