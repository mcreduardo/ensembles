######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# April of 2019
# Compute confusion matrix of boosted trees
######################################################################################

import numpy as np
import json # Load data sets
import DecisionTree as dt # implementation of DecisionTree

######################################################################################

def boosted_trees_cm(maxTrees, maxDepth, trainingSetPath, testSetPath, printAcc):

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

    # Testing

    # separate features from class
    test_X = testData[:,:-1]
    test_y = testData[:,-1]

    individual_predictions = []
    for tree in classifiers: # predict for all trained models
        individual_predictions.append(tree.predict(test_X, prob=False))

    # compute overrall prediction
    predicted = []
    correct_predictions = 0
    for i in range(len(test_y)):
        aux = np.zeros(len(classes))
        for j in range(numClassifiers):
            aux[classes.index(individual_predictions[j][i])] += \
                classifiers_scores[j]
        boosted_prediction = classes[np.argmax(aux)]
        predicted.append(boosted_prediction)
        if boosted_prediction == test_y[i]: correct_predictions += 1

    # compute confusion matrix
    for i in range(len(test_y)):
        confusion_matrix\
            [classes.index(predicted[i])][classes.index(test_y[i])] += 1

    if printAcc: 
        print(str(numClassifiers)+","+str(correct_predictions/len(test_y))+";")
        

    return confusion_matrix

        
        
