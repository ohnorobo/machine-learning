##problem 5

###Explain why 0 ≤ α ≤ C/m is a constraint in the dual optimization with slack variables. 

(HINT: read Chris Burges tutorial first)

###Distinguish three cases, and explain them in terms of the classification and constraints:

    a) 0 = α ;
    b) 0 < α < C/m ;
    c) α = C/m.

###This has been discussed in class and in SVM notes; a detailed rigurous explanation is expected.

    w = sum(a*y*x)
    0 = sum(a*y)

    let u_i = f(x_i) where f is the final svm and x_i is a point

a)

α = 0 means the point is outside the margin.

It also means y_i*u_i should be > 1

b)

0 < α < C/m
C is the slack allowing for margin failure, since the points are probably not linearly seperable. So an alpha > 0 is inside the margin and contributes to that slack.

It also means y_i*u_i should = 1


c)

α = C/m are the strongest support vectors. They are the 'pushiest' and exert the most force on the line. Setting c to high may result in too many pushy support vectors and a slow solution.

It also means y_i*u_i should be < 1



##problem 6

###Consider the following 6 points in 2D, for two classes:

    class 0:   (1,1)   (2,2)    (2,0)

    class 1:   (0,0)   (1,0)    (0,1)

### a) Plot these 6 points, construct the optimal hyperplane by inspection and intuition (give th e W,b) and calculate the margin.

     |
    \|   o
     \
     +\o
     | \
    -+-+\o--
     |   \
     0 1 2 ...

    hyperplane = -x + 1.5

intersection of y=x and y=-x+1.5 is (.75, .75), 

the distance between (.75, .75) and (1, 1) is root(2 * (.25)^2)

or approx .353553...


### b) Which points are support vectors ?

    +(1, 0)
    +(0, 1)
    o(1, 1)
    o(2, 0)

### c) Construct the hyperplane by solving the dual optimization problem using the Lagrangian. Compare with part (a).

    x = (1,1)  α = 1  y = 1
    x = (2,2)  α = 0  y = 1
    x = (2,0)  α = 1  y = 1
    x = (0,0)  α = 0  y = -1
    x = (1,0)  α = 1  y = -1
    x = (0,1)  α = 1  y = -1

I'm picking 1 for the alpha's here since they're all the same distance from the line, so they all have to be equal. 1 just makes for easier calculation

    w = sum(y * a * x)
    b = w * x - y

    w = (1,1) + (2,0) + (-1,0) + (0,-1)
    w = (2,0)

    w = -1

    if the line is as (X,X) and b equals ?
      (1, 0) - 1.5
      (0, 1) - 1.5
      (1, 1) - 2.0
      (2, 0) - 2.0

    so average b is 1.75










