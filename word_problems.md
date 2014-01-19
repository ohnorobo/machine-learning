#HW1

##3

###repeated queries

If a decision is made a some point in the path than only items with that decision =true will be on the left, and =false on the right. So if further down the path the same decision is made again if will encounter either all =true items, or =false items. That means the second decision is totally redundant, and cannot possible add any information (since all items will go either to only the left or only the right). So the tree with the second decision removed (and only the child decisions of whichever side all the items go to) is equivalent. All of the repeated queries can be removed this way, so an equivalent tree with no repeats exists.


###a
prove that a binary tree is equivalent to unequal branching

A single unequal branching can be approximated by a binary tree of arbitrary depth. (Although for some branchings the binary tree would be infinite.) This is by creating more nodes and 'binary searching' for the cutoff point used in the unequal branching. Since each unequal node can be replaced replacing them one by one will result in a fully binary tree which has the same classification behavior as the origional tree. (Although, again, the tree may be infinite.)


###b
upper and lower limits

    levels = [log(2, b) + 1, b+1]

lower bound - degenerate binary tree, with all the branches coming off one trunk

    o
    |\
    o o
    |\
    o o
    |\
    o o
    |\
    o o

upper bound - fully balanced binary tree

    leaves = 2^height - 1
    nodes = 2^(height+1) - 1
    nodes = 2*leaves - 1

          o
        /   \
       o     o
      / \   / \
     o   o  o  o

###c
upper and lower limits

    nodes = [2(b-1), 2b+1]

##4 - entropy decrease
###a

    bits = log(2, possible_outcomes)

The best we can do on an individual split is cleave the results exactly in half by finding a perfect 50% split.

if we do that the original entropy was

    b = log(2, a)

and the new entropy is

    b - 1 = log(2, a/2)

###b

with arbitarry branching we split the possible outcomes into n sections.

to the original entropy is

    b_old = log(2, a)

and the new entropy is

    b_new = log(2, a/n)

###5
normal equation solutions

to get to y = ax + b

data is

    X = [x0 ... xn]
    Y = [y0 ...yn]

General formula

    = inverse(transpose(X) * X) * transpose(X) * Y



