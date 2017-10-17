#!/bin/bash
python spl_dml.py --batch-size 64 --momentum 0.5 --curriculum-epochs 30 --log-interval 100 --mnist --spl --spld
