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