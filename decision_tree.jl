#!/usr/bin/julia







#main
data_dir = "../../data/HW1/"

#spam
spam_data_file = string(data_dir, "spambase/spambase.data")

println("reading in $spam_data_file")
spam_data = readcsv(spam_data_file)
#println(spam_data)

#housing
housing_train_data_file = string(data_dir, "housing/housing_train.txt")
housing_test_data_file = string(data_dir, "housing/housing_test.txt")

println("reading in $housing_train_data_file")
housing_data = readcsv(housing_train_data_file)
#println(housing_data)


