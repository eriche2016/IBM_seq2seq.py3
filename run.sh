#!/bin/bash

# prepare dataset 
if [ 1 -eq 0 ]; then 
    ./scripts/toy.sh
fi 

# run toy experiment
if [ 1 -eq 1 ]; then 
    TRAIN_PATH=data/toy_reverse/train/data.txt
    DEV_PATH=data/toy_reverse/dev/data.txt
    # Start training
    python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --cuda
fi 
