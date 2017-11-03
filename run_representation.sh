# SPLDML
python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --magnet --data_augmentation --max-iter 100 --plot
python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --magnet --max-iter 100 --plot

python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --magnet --data_augmentation --max-iter 200 --plot
python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --magnet --max-iter 200 --plot

python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --magnet --data_augmentation --max-iter 50 --plot
python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --magnet --max-iter 50 --plot
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --en-scheduler --shallow-model vgg --magnet --data_augmentation --max-iter 100

#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 64 --en-scheduler --shallow-model vgg --spldml --max-iter 100
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 64 --learning_rate1 1e-3 --shallow-model vgg --spldml --max-iter 100

# Random
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --random --data_augmentation
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --en-scheduler --shallow-model vgg --random --data_augmentation

#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 64 --learning_rate1 1e-3 --shallow-model vgg --random
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 64 --en-scheduler --shallow-model vgg --random

# SPLD
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --learning_rate1 1e-3 --shallow-model vgg --spl --data_augmentation --max-iter 100
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 128 --en-scheduler --shallow-model vgg --spl --data_augmentation --max-iter 100

#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 64 --learning_rate1 1e-3 --shallow-model vgg --spl --max-iter 100
#python spld_iclr.py --model resnet18 --dataset cifar10 --dropout_rate 0.3 --batch_size 64 --en-scheduler --shallow-model vgg --spl --max-iter 100