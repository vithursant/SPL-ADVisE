# python  trainer.py \
#         --dataset cifar100 \
#         --num-classes 100 \
#         --data_augmentation \
#         --model wideresnet \
#         --depth 28 \
#         --widen-factor 10 \
#         --dropout_rate 0.3 \
#         --epochs 200 \
#         --schedule 60 120 160 \
#         --gamma 0.1 \
#         --wd 5e-4 \
#         --train-batch 128 \
#         --learning_rate1 0.1 \
#         --checkpoint checkpoints/cifar100/random-wideresnet-wf-10-d-28-lr01-b128-dataaug-iclr \
#         --random

python  trainer.py \
        --dataset cifar100 \
        --num-classes 100 \
        --data_augmentation \
        --model wideresnet \
        --depth 28 \
        --widen-factor 10 \
        --dropout_rate 0.3 \
        --epochs 200 \
        --schedule 60 120 160 \
        --gamma 0.1 \
        --wd 5e-4 \
        --train-batch 128 \
        --learning_rate1 0.1 \
        --checkpoint checkpoints/cifar100/spld-wideresnet-wf-10-d-28-lr01-b128-dataaug-iclr \
        --spld