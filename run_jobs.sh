#!/bin/bash
python spl_dml.py --cifar10 --visdom --batch-size 32 --momentum 0.5 --curriculum-epochs 30 --log-interval 100 --name cifar10spl --spld

python spl_dml.py --cifar10 --visdom --batch-size 32 --momentum 0.5 --epochs 30 --log-interval 100 --name cifar10spl


