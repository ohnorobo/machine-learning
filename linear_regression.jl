#!/usr/bin/julia

function get_linear_reg_function(X, Y)
  #println(X)
  #println(Y)
  #println(X')
  #println(X' * X)
  #println(inv(X' * X))
  #println(X' * Y)

  θs = inv(X' * X) * X' * Y

  println("θs")
  println(typeof(θs))
  println(θs)

  return x -> sum(cross(x, θs))
end


# error = 1/2 Σ( | hx - yx |^2 )
function least_squares_error(regression, features::Array{Float64,2}, truth::Array{Float64,1})
  totalerror = 0
  for i = 1:size(truth, 1)
    item = features'[:,i]
    println("item")
    println(typeof(item))
    println(item)
    println("truth")
    println(truth[i,:])
    println("guess")
    println(regression(item))
    error = abs(truth[i,:] - regression(item))^2
    totalerror = totalerror + error
  end
  return totalerror
end




#testing
function test_toy()
  # http://easycalculation.com/statistics/learn-regression.php
  data = [1.0 1.0 2 3 4 5; 2.0 2 10 4 5 6]
  features = data[:,2:]
  truth = data[:,1]

  reg = get_linear_reg_function(features, truth)
  error = least_squares_error(reg, features, truth)

  println(reg)
  println(error)
end

data_dir = "../../data/HW1/"
function test_spam()
  # spam
  spam_data_file = string(data_dir, "spambase/spambase.data")

  println("reading in $spam_data_file")
  spam_data = readcsv(spam_data_file)
end

function test_housing()
  housing_train_data_file = string(data_dir, "housing/housing_train.txt")
  housing_test_data_file = string(data_dir, "housing/housing_test.txt")

  println("reading in $housing_train_data_file")
  housing_train_data = readcsv(housing_train_data_file)
  housing_test_data = readcsv(housing_test_data_file)

  reg = get_linear_reg_function(housing_train_data[:,2:], housing_train_data[:,1])
  error = least_squares_error(reg, housing_test_data[:,2:], housing_test_data[:,1])
  println(error)
end

#main
#test_toy()
test_housing()
