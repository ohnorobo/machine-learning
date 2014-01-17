#!/usr/bin/julia

function get_linear_reg_function(X, Y)
  println(X)
  println(Y)
  println(X')
  println(X' * X)
  println(inv(X' * X))
  println(X' * Y)

  θs = inv(X' * X) * X' * Y

  println("θs")
  println(typeof(θs))
  println(θs)

  return x -> sum(cross(x, θs))
end


# error = 1/2 Σ( | hx - yx |^2 )
function calculate_error(regression, features::Array{Float64,2}, truth::Array{Float64,1})
  totalerror = 0
  for i = 1:size(truth, 1)
    item = features'[:,1]
    println("item")
    println(typeof(item))
    println(item)
    println("truth")
    println(truth[i,:])
    error = abs(truth[i,:] - regression(item))^2
    totalerror = totalerror + error
  end
  return totalerror
end



#testing

# http://easycalculation.com/statistics/learn-regression.php
data = [60 61 62 63 65; 3.1 3.6 3.8 4 4.1]
#data = [60 3.1; 61 3.6; 62 3.8; 63 4; 65 4.1]
truth = [1.0; 2.0]

reg = get_linear_reg_function(data, truth)
error = calculate_error(reg, data, truth)

println(reg)
println(error)

