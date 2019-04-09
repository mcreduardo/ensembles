# Call: $ ./boosted_trees <max trees> <max depth> <train-set-file> <test-set-file>
######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# April of 2019
# Multi-class AdaBoost to generate an ensemble of decision trees
######################################################################################

import numpy as np
import sys
import json # Load data sets
import DecisionTree as dt # implementation of DecisionTree

printOutput = True

######################################################################################
# Get inputs and load .json data

# Receive arguments using sys
maxTrees = int(sys.argv[1])
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
numClasses = len(features[-1][-1])

######################################################################################
# Multi-class AdaBoost
# Following Hastie proposed implementation
# https://web.stanford.edu/~hastie/Papers/samme.pdf

# init weights
wi = np.ones(trainData_size)/trainData_size

classifiers = [] # list to store classifiers
weights = [] # list to store classifiers
classifiers_scores = [] # list to store scores of classifiers
for i in range(maxTrees):

    # Fit a classifier T^(m)(X) to the training data using weights wi
    tree = dt.DecisionTree()
    train_X = trainingData[:,:-1]
    train_y = trainingData[:,-1]
    tree.fit(train_X, train_y, features, max_depth=maxDepth, instance_weights=wi)

    # compute error
    predictions = tree.predict(train_X, prob=False)
    error = np.sum(wi * (predictions != train_y)) / np.sum(wi)

    # compute score
    alpha = np.log( (1-error)/error * (numClasses - 1) )

    # if score is negative: break  (discard this classifier)
    #if alpha < 0:
    if error >= 1 - (1/numClasses):
        break
    classifiers.append(tree)
    classifiers_scores.append(alpha)
    weights.append(wi)

    # update weights
    wi = wi * np.exp( alpha * (predictions != train_y) )
    wi = wi/np.sum(wi) # normalize

numClassifiers = len(classifiers)

# print weights and scores
if printOutput:
    for i in range(len(wi)):
        for j in range(numClassifiers - 1):
            print("%.12f"%weights[j][i], end=",")
        print("%.12f"%weights[numClassifiers-1][i])
    print("")
    for j in range(numClassifiers - 1):
        print("%.12f"%classifiers_scores[j], end=",")
    print("%.12f"%classifiers_scores[numClassifiers-1])
    print("")



######################################################################################
# Testing

# separate features from class
test_X = testData[:,:-1]
test_y = testData[:,-1]

individual_predictions = []
for tree in classifiers: # predict for all trained models
    individual_predictions.append(tree.predict(test_X, prob=False))

# compute overrall prediction
classes = features[-1][-1]
predicted = []
correct_predictions = 0
for i in range(len(test_y)):
    aux = np.zeros(len(classes))
    for j in range(numClassifiers):
        aux[classes.index(individual_predictions[j][i])] += \
            classifiers_scores[j]
    boosted_prediction = classes[np.argmax(aux)]
    predicted.append(boosted_prediction)
    # correct preditction?
    if boosted_prediction == test_y[i]: correct_predictions += 1

# print predictions and accuracy
if printOutput:
    for i in range(len(test_y)):
        for j in range(numClassifiers):
            print(individual_predictions[j][i], end=",")
        print(predicted[i], end= ",")
        print(test_y[i], end= "\n")
    print("\n"+str(correct_predictions/len(test_y)))

    
    
