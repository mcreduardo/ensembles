#!/bin/bash

python3 code/bagged_trees.py 5 2 datasets/digits_train.json datasets/digits_test.json > output/bagged_trees_5_2_digits.txt
python3 code/bagged_trees.py 2 3 datasets/digits_train.json datasets/digits_test.json > output/bagged_trees_2_3_digits.txt

python3 code/bagged_trees.py 5 2 datasets/heart_train.json datasets/heart_test.json > output/bagged_trees_5_2_heart.txt
python3 code/bagged_trees.py 2 3 datasets/heart_train.json datasets/heart_test.json > output/bagged_trees_2_3_heart.txt

python3 code/bagged_trees.py 5 2 datasets/mushrooms_train.json datasets/mushrooms_test.json > output/bagged_trees_5_2_mushrooms.txt
python3 code/bagged_trees.py 2 3 datasets/mushrooms_train.json datasets/mushrooms_test.json > output/bagged_trees_2_3_mushrooms.txt

python3 code/bagged_trees.py 5 2 datasets/winequality_train.json datasets/winequality_test.json > output/bagged_trees_5_2_winequality.txt
python3 code/bagged_trees.py 2 3 datasets/winequality_train.json datasets/winequality_test.json > output/bagged_trees_2_3_winequality.txt


python3 code/boosted_trees.py 5 2 datasets/digits_train.json datasets/digits_test.json > output/boosted_trees_5_2_digits.txt
python3 code/boosted_trees.py 2 3 datasets/digits_train.json datasets/digits_test.json > output/boosted_trees_2_3_digits.txt

python3 code/boosted_trees.py 5 2 datasets/heart_train.json datasets/heart_test.json > output/boosted_trees_5_2_heart.txt
python3 code/boosted_trees.py 2 3 datasets/heart_train.json datasets/heart_test.json > output/boosted_trees_2_3_heart.txt

python3 code/boosted_trees.py 5 2 datasets/mushrooms_train.json datasets/mushrooms_test.json > output/boosted_trees_5_2_mushrooms.txt
python3 code/boosted_trees.py 2 3 datasets/mushrooms_train.json datasets/mushrooms_test.json > output/boosted_trees_2_3_mushrooms.txt

python3 code/boosted_trees.py 5 2 datasets/winequality_train.json datasets/winequality_test.json > output/boosted_trees_5_2_winequality.txt
python3 code/boosted_trees.py 2 3 datasets/winequality_train.json datasets/winequality_test.json > output/boosted_trees_2_3_winequality.txt



python3 code/confusion_matrix.py bag 5 2 datasets/digits_train.json datasets/digits_test.json > output/confusion_matrix_bag_5_2_digits.txt
python3 code/confusion_matrix.py boost 5 2 datasets/digits_train.json datasets/digits_test.json > output/confusion_matrix_boost_5_2_digits.txt





