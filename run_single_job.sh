#!/bin/bash
python spl_dml.py  --batch-size 32 --momentum 0.5 --curriculum-epochs 30 --log-interval 100 --name fashionmnistspl --fashionmnist --visdom
