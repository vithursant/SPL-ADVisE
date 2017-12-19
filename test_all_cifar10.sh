python  trainer.py \
        --dataset cifar10 \
        --data_augmentation \
        --model preresnet \
        --depth 110 \
        --epochs 164 \
        --schedule 81 122 \
        --gamma 0.1 \
        --wd 1e-4 \
        --train-batch 128 \
        --learning_rate1 0.1 \
        --checkpoint checkpoints/cifar10/random-preresnet110-lr01-b128-dataaug \
        --random

python  trainer.py \
        --dataset cifar10 \
        --data_augmentation \
        --model preresnet \
        --depth 110 \
        --epochs 164 \
        --schedule 81 122 \
        --gamma 0.1 \
        --wd 1e-4 \
        --train-batch 128 \
        --learning_rate1 0.1 \
        --checkpoint checkpoints/cifar10/spld-preresnet110-lr01-b128-dataaug \
        --spld

python  trainer.py \
        --dataset cifar10 \
        --data_augmentation \
        --model preresnet \
        --depth 110 \
        --epochs 164 \
        --schedule 81 122 \
        --gamma 0.1 \
        --wd 1e-4 \
        --embedding-model vgg16 \
        --train-batch 128 \
        --learning_rate1 0.1 \
        --checkpoint checkpoints/cifar10/leap-preresnet110-vgg16-lr01-b128-dataaug \
        --leap

#python trainer.py --data_augmentation --model resnet18 --dataset cifar10 --train-batch 128 --spld  --learning_rate1 0.001 --checkpoint checkpoints/cifar10/spld-resnet18-lr001-b128-dataaug
#python trainer.py --model resnet18 --embedding-model vgg16 --dataset cifar10 --train-batch 128 --leap  --learning_rate1 0.001 --checkpoint checkpoints/cifar10/leap-resnet18-vgg16-lr001-b128
