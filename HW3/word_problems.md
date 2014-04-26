
## problem 7

### a. prove 

    p(a|b,c) = p(b|a,c) * p(a|c) / p(b|c)
             = p(c|a,b) * p(a|b) / p(c|b)


    p(a∣b ∩ c) = 
    p(a ∩ b ∩ c) / p(b ∩ c)
    (p(a ∩ b ∩ c) / p(c)) / (p(b ∩ c) / p(c))
    p(a ∩ b | c) / p(b | c)
    p(b | a ∩ c) * p(a | c) / p(b | c)


### b. You are given a coin which you know is either fair or double-headed. You believe that the a priori odds of it being fair are F to 1; i.e., you believe that the a priori probability of the coin being fair is F/(F+1) . You now begin to flip the coin in order to learn more. Obviously, if you ever see a tail, you know immediately that the coin is fair. As a function of F, how many heads in a row would you need to see before becoming convinced that there is a better than even chance that the coin is double-headed?

F = prob of coin being fair

start with F=1

likelyness of getting n heads given a fair coin 

    p(h|f) = (1/2)^n

likelyness that coin is fair given n heads

    p(f|h) = ( (1/2)^n * 1/2 ) / (1/2)^n

likelyness that coin is bad given n heads

   p(b|h) = 1 - p(f|h)




## problem 8

### a. Somebody tosses a fair coin and if the result is heads, you get nothing, otherwise you get $5. How much would you be pay to play this game? What if the win is $500 instead of $5?

any amount less than win/2


### b. Suppose you play instead the following game: At the beginning of each game you pay an entry fee of $100. A coin is tossed until a head appears, counting n = the number of tosses it took to see the first head. Your reward is 2^n (that is: if a head appears first at the 4th toss, you get $16). Would you be willing to play this game (why)?

win = (2^1 * (1 - 1/2 ^ 1)) + (2^2 * (1 - 1/2 ^ 2) + (2^3 * (1 - 1/2 ^ 3) * ...

win = sum(2^n * (1 - 1/2 ^ n)) from n=1 to infinity

Yes you would want to play, 2^n*(1-(1/2)^n) has an upward curvature, the reward increaces much more quickly than the chance of wining decreaces.

### c. Lets assume you answered "yes" at part b (if you did not, you need to fix your math on expected values). What is the probability that you make a profit in one game?

You make a profit of $28 ($128 - $100) anytime the first head is 7th or more.

So the probability of getting 6 tails on a row is

(1/2)^6, or 1/64


## problem 9

### DHS CH2, Pb 43

    p(w_j) = prior probabilities

    x = {0,1}_i

    p_ij = p(x_i = 1 | w_j)

### a. meaning pf p_ij

    w_j = prior probabilities for a particular class

    x_i = feature values for a particular point

    p_ij = probability contribution that point x is in class j given the feature value i

### b. show minimum error for g_j(x)

    g_j(x) = sum(x_i * ln(p_ij / 1 - p_ij)) + sum(ln(1 - p_ij) + ln(P_w))

    The first term ln(p_ij / 1- p_ij) means that probabilities close to .5 go to 0, while probabilities near 0 or 1 go to -inf and +inf. This means a matching x_i gets a large positive term in the sum, while a non-matching x_i gets a lerge negative term.

    The second term is an inverse of the probability at p_ij, summed with the state of natue for that w_j

