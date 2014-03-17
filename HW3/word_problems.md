
# problem 7

## a. prove bayes theorem

## b. You are given a coin which you know is either fair or double-headed. You believe that the a priori odds of it being fair are F to 1; i.e., you believe that the a priori probability of the coin being fair is F/(F+1) . You now begin to flip the coin in order to learn more. Obviously, if you ever see a tail, you know immediately that the coin is fair. As a function of F, how many heads in a row would you need to see before becoming convinced that there is a better than even chance that the coin is double-headed?


# problem 8

## a. Somebody tosses a fair coin and if the result is heads, you get nothing, otherwise you get $5. How much would you be pay to play this game? What if the win is $500 instead of $5?

any amount less than win/2


## b. Suppose you play instead the following game: At the beginning of each game you pay an entry fee of $100. A coin is tossed until a head appears, counting n = the number of tosses it took to see the first head. Your reward is 2^n (that is: if a head appears first at the 4th toss, you get $16). Would you be willing to play this game (why)?

win = (2^1 * (1 - 1/2 ^ 1)) + (2^2 * (1 - 1/2 ^ 2) + (2^3 * (1 - 1/2 ^ 3) * ...

win = sum(2^n * (1 - 1/2 ^ n)) from n=1 to infinity

Yes you would want to play, 2^n*(1-1/2^n) has an upward curvature, the reward increaces much more quickly than the chance of wining decreaces.

## c. Lets assume you answered "yes" at part b (if you did not, you need to fix your math on expected values). What is the probability that you make a profit in one game?

## d. [ExtraCredit] After about how many games (estimate) the probability of making a profit overall is bigger than 50% ?


# problem 9

## DHS CH2, Pb 43
