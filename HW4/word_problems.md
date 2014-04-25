# problem 6

What is the VC dimension for the class of hyphothesis of:

## a) unions of two rectangles

4

any 4 points are either

* evenly split and you can either just box the two you want, one in each rectangle
* 1-3 split, and you can just box the point you want in 1 rectangle

+-+-+ cannot be shattered by two rectangles

## b) circles

VC = 3

Any three points can be used as points of a triangle, a circle can seperate any two points of a triangle from the last

4 is not possible since

+-
-+

cannot be split by a circle

## c) triangles

3

+-+-

cannot be split by a triangle

## d) multidimensional "sphere" given by f(x) = sign [(x-c)(x-c) -b] in the Euclidian space with m dimensions ℝ m .

point in R1 = 2

circle in R2 = 4

sphere in r3 = also 4, as long as the 4 points are in a plane the 'cross section' of the shpere is stil a circle


