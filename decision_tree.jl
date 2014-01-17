#!/usr/bin/julia

# problem 1

# training_features = [[1.1, 1.2, 1.3, 0],
#                      [4.5, 0.0, 5.3, 1],
#                      [1.1, 5.4, 6.7, 1]]
# training_labels = ["a", "b", "c", "s"]
# model = train_decision_tree(training_features, training_labels)
# test_feature_set = [1.1, 1.1, 1.1]
# classification = classify(model, test_feature_set)
# 1

threshhold = 10 # decreace in entropy under which we stop splitting


abstract DecisionTree

type DecisionTreeNode <:DecisionTree
  decision_function
  leftsubtree::DecisionTree
  rightsubtree::DecisionTree
end

function train(model::DecisionTreeNode, featureset, labels)
  #calculate entropy/split for each feature
  improvements, splits = all_splits(featureset, labels)

  #pick the best split
  best_improvement = max(improvements)
  best_index = index(improvements, best_improvement)
  best_split = splits[best_index]

  #left_featureset, right_featureset = split_featureset(best_split)


  decision_function = split
  #if the split is < threshhold make 2 leaves
  #if the split it > threshhold make 2 nodes
  #train both children
end

# find the optimal (boolean) splitting function for a numeric feature
# which returns the best decreace in entropy
function find_split(features::Array{Float64, 1}, labels)
  return x -> x > mean(features)
end


#return all the possible splits (1/feature) and the improvement caused by that split
function all_splits(featureset, labels)
  improvements = Array[]
  splits = Array[]

  for i=size(featureset, 1)
    feature = featureset[:,i]
    println(feature)
    improvement, split = find_split(feature, labels)
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

function classify(model::DecisionTreeNode, featureset)
  (result, remaining_features) = model.decision_function(featureset)
  if result
    return classify(model.leftsubtree, remaining_features)
  else
    return classify(model.rightsubtree, remaining_features)
  end
end


type DecisionTreeLeaf <:DecisionTree
  classification
end

function train(model::DecisionTreeLeaf, featureset, labels)
  model.classification = mode(labels)
end

function classify(model::DecisionTreeLeaf, test_feature_set)
  return model.classification
end



function train_decision_tree(features, labels)
  node = DecisionTreeNode(x -> x, DecisionTreeLeaf(0), DecisionTreeLeaf(0))
  train(node, features, labels)
  return node
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

a = find_split([ 3.0, 2, 3], [ 1.0, 0, 0])
println(a)

test_toy()
