# python  trainer.py \
#         --dataset cifar100 \
#         --num-classes 100 \
#         --data_augmentation \
#         --model densenet \
#         --depth 100 \
#         --growthRate 12 \
#         --epochs 300 \
#         --schedule 150 225 \
#         --gamma 0.1 \
#         --wd 1e-4 \
#         --train-batch 64 \
#         --learning_rate1 0.1 \
#         --checkpoint checkpoints/cifar100/random-densenet-bc-100-12-lr01-b128-dataaug \
#         --random
#
# python  trainer.py \
#         --dataset cifar100 \
#         --num-classes 100 \
#         --data_augmentation \
#         --model densenet \
#         --depth 100 \
#         --growthRate 12 \
#         --epochs 300 \
#         --schedule 150 225 \
#         --gamma 0.1 \
#         --wd 1e-4 \
#         --train-batch 64 \
#         --learning_rate1 0.1 \
#         --checkpoint checkpoints/cifar100/spld-densenet-bc-100-12-lr01-b128-dataaug \
#         --spld

python  trainer.py \
        --dataset cifar100 \
        --num-classes 100 \
        --data_augmentation \
        --model densenet \
        --depth 100 \
        --growthRate 12 \
        --epochs 300 \
        --schedule 150 225 \
        --gamma 0.1 \
        --wd 1e-4 \
        --train-batch 64 \
        --learning_rate1 0.1 \
        --learning_rate2 0.001 \
        --embedding-model vgg16 \
        --checkpoint checkpoints/cifar100/leapdensenet-bc-100-12-vgg16-lr01-b128-dataaug-magnetloss \
        --leap \
        --plot

#python trainer.py --data_augmentation --model resnet18 --dataset cifar10 --train-batch 128 --spld  --learning_rate1 0.001 --checkpoint checkpoints/cifar10/spld-resnet18-lr001-b128-dataaug
#python trainer.py --model resnet18 --embedding-model vgg16 --dataset cifar10 --train-batch 128 --leap  --learning_rate1 0.001 --checkpoint checkpoints/cifar10/leap-resnet18-vgg16-lr001-b128
