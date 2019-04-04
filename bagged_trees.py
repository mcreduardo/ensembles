#



# Import modules
import DecisionTree as dt
import json
import numpy as np
# Get training data
train = json.load(open(’a_training_set.json’, ’r’))
train_meta = train[’metadata’][’features’]
train_data = np.array(train[’data’])
train_X = train_data[:,:-1]
train_y = train_data[:,-1]
# Build and train a decision tree:
mytree = dt.DecisionTree()
mytree.fit(train_X, train_y, train_meta, max_depth=5)
# look at the structure of the trained tree:
print(mytree)
# Get test data
test = json.load(open(’a_test_set.json’, ’r’))
test_data = np.array(test[’data’])
test_X = test_data[:,:-1]
test_y = test_data[:,-1]
# Predict the test labels:
predicted_y = mytree.predict(test_X, prob=True)


if __name__=="__main__":

    import sys
    import json
    args = sys.argv

    depth = int(args[1])
    train_file = args[2]
    test_file = args[3]

    # Example usage of the DecisionTree class: 
    mydt = DecisionTree()
    train = json.load(open(train_file,"r"))
    train_X = np.array(train['data'])
    train_y = train_X[:,-1]
    train_X = train_X[:,:-1]
    meta = train['metadata']['features']
    mydt.fit(train_X, train_y, meta, max_depth=depth)

    print(mydt)
    
    test = json.load(open(test_file,"r"))
    test_X = np.array(test['data'])
    test_y = test_X[:,-1]
    test_X = test_X[:,:-1]
    
    preds = mydt.predict(test_X, prob=False)
   
    print( (preds == test_y).sum() / preds.shape[0] )