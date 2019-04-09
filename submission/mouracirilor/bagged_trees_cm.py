######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# April of 2019
# Compute confusion matrix of bagged trees
######################################################################################

import numpy as np
import json # Load data sets
import DecisionTree as dt # implementation of DecisionTree

######################################################################################

def bagged_trees_cm(numTrees, maxDepth, trainingSetPath, testSetPath, printAcc):

    np.random.seed(0) # ensure program is deterministic

    # Load training and test set
    # metadata + data
    with open(trainingSetPath) as f:
        trainSet = json.load(f)
    with open(testSetPath) as f:
        testSet = json.load(f)

    # Extract metadata
    features = trainSet["metadata"]["features"]

    # init conf matrix
    classes = features[-1][-1]
    numClasses = len(classes)
    confusion_matrix = np.zeros((numClasses,numClasses)).astype(int)

    # Get data
    trainingData = np.array(trainSet["data"])
    testData = np.array(testSet["data"])

    trainData_size = len(trainingData)

    # Generate bootstrap training sets (indices)
    bootstrapIndices = np.zeros((trainData_size,numTrees)).astype(int)
    for i in range(numTrees):
        bootstrapIndices[:,i] = \
            np.random.choice(trainData_size, size=trainData_size, replace=True)


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

    # Testing
    # separate features from class
    test_X = testData[:,:-1]
    test_y = testData[:,-1]

    individual_prob = [] # init list for computed probabilities
    for tree in models: # predict for all trained models
        individual_prob.append(tree.predict(test_X, prob=True))

    # get average from results
    avg_prob = np.mean(individual_prob, axis = 0)

    correct_predictions = 0
    predicted = []
    for i in range(len(avg_prob)):
        for j in range(numTrees):
            individual_prediction = \
                features[-1][-1][np.argmax(individual_prob[j][i])]
        bagging_prediction = features[-1][-1][np.argmax(avg_prob[i])]
        predicted.append(bagging_prediction)

        # correct preditction?
        if bagging_prediction == test_y[i]: correct_predictions += 1
    
    # compute confusion matrix
    for i in range(len(test_y)):
        confusion_matrix\
            [classes.index(predicted[i])][classes.index(test_y[i])] += 1

    if printAcc: 
        print(str(numTrees)+","+str(correct_predictions/len(test_y))+";")

    return confusion_matrix

        

