# Call: $ ./bagged_trees <#trees> <max depth> <train-set-file> <test-set-file> 
######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# April of 2019
# Bootstrap Aggregation (Bagging)
######################################################################################

import numpy as np
import sys
import json # Load data sets
import DecisionTree as dt # implementation of DecisionTree

np.random.seed(0) # ensure program is deterministic

printOutput = True

######################################################################################
# Get inputs and load .json data

# Receive arguments using sys
numTrees = int(sys.argv[1])
maxDepth = int(sys.argv[2])
trainingSetPath = sys.argv[3]
testSetPath = sys.argv[4]

# Load training and test set
# metadata + data
with open(trainingSetPath) as f:
    trainSet = json.load(f)
with open(testSetPath) as f:
    testSet = json.load(f)

# Extract metadata
features = trainSet["metadata"]["features"]

# Get data
trainingData = np.array(trainSet["data"])
testData = np.array(testSet["data"])

trainData_size = len(trainingData)

######################################################################################
# Generate bootstrap training sets (indices)

bootstrapIndices = np.zeros((trainData_size,numTrees)).astype(int)
for i in range(numTrees):
    bootstrapIndices[:,i] = \
        np.random.choice(trainData_size, size=trainData_size, replace=True)

# print indices of your bootstrapped samples

if False:#printOutput:
    for row in bootstrapIndices:
        for j in range(numTrees-1):
            print(row[j], end= ",")
        print(row[-1])
    print("\n", end = "")

######################################################################################
# Train models

models = [] # init list for trained models
for i in range(numTrees):
    # build decision tree
    mydt = dt.DecisionTree()

    # train data
    train_X = trainingData[bootstrapIndices[:,i]]
    train_y = train_X[:,-1]
    train_X = train_X[:,:-1]

    # training
    mydt.fit(train_X, train_y, features, max_depth=maxDepth)

    # append to models list
    models.append(mydt)

######################################################################################
# Testing

# separate features from class
test_X = testData[:,:-1]
test_y = testData[:,-1]

individual_prob = [] # init list for computed probabilities
for tree in models: # predict for all trained models
    individual_prob.append(tree.predict(test_X, prob=True))

# get average from results
avg_prob = np.mean(individual_prob, axis = 0)

# print results
if printOutput:
    correct_predictions = 0
    for i in range(len(avg_prob)):
        for j in range(numTrees):
            individual_prediction = \
                features[-1][-1][np.argmax(individual_prob[j][i])]
            print(individual_prediction, end= ",")
        bagging_prediction = features[-1][-1][np.argmax(avg_prob[i])]
        print(bagging_prediction, end= ",")
        print(test_y[i], end= "\n")

        # correct preditction?
        if bagging_prediction == test_y[i]: correct_predictions += 1
    
    print("\n"+str(correct_predictions/len(avg_prob)))



    







