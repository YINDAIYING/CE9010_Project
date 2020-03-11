#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

partition=${1}

srun -u --partition=${partition} \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    -w BJ-IDC1-10-10-30-54 \
    python test.py --name fed_ft_ResNet50_train_all_70_global_1_local
