#!/bin/bash



python3 code/ensemble_acc.py datasets/digits_train.json datasets/digits_test.json bag 3
python3 code/ensemble_acc.py datasets/digits_train.json datasets/digits_test.json bag 5
python3 code/ensemble_acc.py datasets/digits_train.json datasets/digits_test.json bag 7


python3 code/ensemble_acc.py datasets/digits_train.json datasets/digits_test.json boost 3
python3 code/ensemble_acc.py datasets/digits_train.json datasets/digits_test.json boost 5
python3 code/ensemble_acc.py datasets/digits_train.json datasets/digits_test.json boost 7
