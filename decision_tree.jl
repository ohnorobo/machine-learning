#!/usr/bin/julia

# problem 1

threshhold = 10 # decreace in entropy under which we stop splitting

abstract DecisionTree

type DecisionTreeNode <:DecisionTree
  decision_function  #boolean function to decide which child to choose
  feature_index  #which feature to split on
  leftsubtree::DecisionTree
  rightsubtree::DecisionTree
end

function train(featureset, labels)
  #calculate entropy/split for each feature
  improvements, splits = all_splits(featureset, labels)

  #pick the best split
  best_improvement = maximum(improvements)
  best_index = findin(improvements, best_improvement)
  best_split = splits[best_index]

  #split features
  left_featureset, right_featureset = split_featureset(best_split, best_index, featureset)
  left_labels, right_labels = split_labels(best_split, best_index, featureset, labels)

  if best_improvement > threshhold  #child nodes
    leftchild = train(left_featureset, left_labels)
    rightchild = train(right_featureset, right_labels)
  else   #child leaves
    leftchild = train(left_labels)
    rightchild = train(right_labels)
  end

  return DecisionTreeNode(best_split, best_index, leftchild, rightchild)
end

#splits a featureset into two featuresets with a boolean function
function split_featureset(split, feature_index, featureset)
  left_set = Array{Float64,2}[]
  right_set = Array{Float64,2}[]

  for i=size(featureset, 2) #for each item
    if split(featureset[i:feature_index])
      push!(left_set, featureset[i])
    else
      push!(right_set, featureset[i])
    end
  end

  return left_set, right_set
end

# find the optimal (boolean) splitting function for a numeric feature
# which returns the best decreace in entropy
function find_split(features::Array{Float64, 1}, labels)
  improvement = 0.0
  return improvement, x -> x > mean(features)
end


#return all the possible splits (1/feature) and the improvement caused by that split
function all_splits(featureset, labels)
  improvements = Float64[]
  splits = Function[]

  for i=size(featureset, 1) #for each feature
    feature = featureset[:,i]
    println(feature)
    (improvement, split) = find_split(feature, labels)
    push!(improvements, improvement)
    push!(splits, split)
  end

  return (improvements, splits)
end


# how much we improve by splitting on a particular feature
# also returns the splitting function
function information_gain_and_split(features, labels, feature_index)
  split = find_split(features[feature_index], labels)
  return 0, split
end

function classify(model::DecisionTreeNode, item)
  result = model.decision_function(item)
  if result
    return classify(model.leftsubtree, remaining_features)
  else
    return classify(model.rightsubtree, remaining_features)
  end
end


type DecisionTreeLeaf <:DecisionTree
  classification
end

function train(labels)
  classification = mode(labels)
  return DecisionTreeLeaf(classification)
end

function classify(model::DecisionTreeLeaf, item)
  return model.classification
end



function train_decision_tree(features, labels)
  return train(features, labels)
end



function test_toy()
  data = [1.0 2 3 1; 1 3 2 0; 2 1 3 0]
  features = data[:,1:3]
  println("features")
  println(features)
  truth = data[:,4]
  println("truth")
  println(truth)
  model = train_decision_tree(features, truth)
  println("model")
  println(model)
  guess = classify(model, [ 2.0 3 1]) 
  println("guess")
  println(guess)
  # guess should be 0
end

data_dir = "../../data/HW1/"
function test_spambase()
  # spam
  spam_data_file = string(data_dir, "spambase/spambase.data")

  println("reading in $spam_data_file")
  spam_data = readcsv(spam_data_file)
end

function test_housing()
  # housing
  housing_train_data_file = string(data_dir, "housing/housing_train.txt")
  housing_test_data_file = string(data_dir, "housing/housing_test.txt")

  println("reading in $housing_train_data_file")
  housing_train_data = readcsv(housing_train_data_file)
  housing_test_data = readcsv(housing_test_data_file)

  model = train_decision_tree(housing_train_data[2:], housing_train_data[1])
  guesses = classify(model, housing_test_data[2:])
  println("Guesses")
  println(guesses)
  println("Truth")
  println(housing_test_data[1])
end



# main

improvement, split = find_split([ 3.0, 2, 3], [ 1.0, 0, 0])
println(split)

test_toy()
