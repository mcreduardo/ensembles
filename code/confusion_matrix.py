# Call: $ ./confusion_matrix <bag|boost> <# trees> <max tree depth> <train-set-file> <test-set-file>
######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# April of 2019
# Confusion matrix
######################################################################################

import numpy as np
import sys

# implementations for computation of CMs
from boosted_trees_cm import boosted_trees_cm
from bagged_trees_cm import bagged_trees_cm

printOutput = True

######################################################################################
# Get inputs and load .json data

# Receive arguments using sys
bag_or_boost = sys.argv[1]
numTrees = int(sys.argv[2])
maxDepth = int(sys.argv[3])
trainingSetPath = sys.argv[4]
testSetPath = sys.argv[5]

# bag or boost?
if bag_or_boost == "boost":
    # compute boost CM
    confusion_matrix = boosted_trees_cm(\
        maxTrees=numTrees, maxDepth=maxDepth,\
        trainingSetPath=trainingSetPath, testSetPath=testSetPath)

elif bag_or_boost == "bag":
    # compute bag CM
    confusion_matrix = bagged_trees_cm(\
        numTrees=numTrees, maxDepth=maxDepth,\
        trainingSetPath=trainingSetPath, testSetPath=testSetPath)

else:
    # not bag or boost
    print("error: method "+ bag_or_boost +"not bag or boost.")

# print CM
if printOutput:
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            if j == confusion_matrix.shape[1] - 1:
                print(confusion_matrix[i][j], end="\n")
            else:
                print(confusion_matrix[i][j], end=",")
    print("")
