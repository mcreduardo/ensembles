# Call: $ ./ensemble_acc <train-set-file> <test-set-file>
######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# April of 2019
# Generate data for plotting ensemble acc. vs ensemble size
######################################################################################

import numpy as np
import sys

# implementations for computation of CMs
from boosted_trees_cm import boosted_trees_cm
from bagged_trees_cm import bagged_trees_cm

######################################################################################
# Get inputs and load .json data

# Receive arguments using sys
bag_or_boost = sys.argv[3]
maxDepth = int(sys.argv[4])
print(maxDepth)
trainingSetPath = sys.argv[1]
testSetPath = sys.argv[2]


for i in range(1,20):
    if i%3 == 0:
        if bag_or_boost == "boost":
            boosted_trees_cm(\
                maxTrees=i, maxDepth=maxDepth,\
                trainingSetPath=trainingSetPath, testSetPath=testSetPath,\
                printAcc=True)
        elif bag_or_boost == "bag":
            bagged_trees_cm(\
                numTrees=i, maxDepth=maxDepth,\
                trainingSetPath=trainingSetPath, testSetPath=testSetPath,\
                printAcc=True)
