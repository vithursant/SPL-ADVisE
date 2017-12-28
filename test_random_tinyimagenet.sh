# python  trainer.py \
#          --dataset tinyimagenet \
#          --num-classes 200 \
#          --data_augmentation \
#          --model resnet18 \
#          --depth 100 \
#          --growthRate 12 \
#          --epochs 300 \
#          --schedule 150 225 \
#          --gamma 0.1 \
#          --wd 1e-4 \
#          --train-batch 256 \
#          --learning_rate1 0.001 \
#          --checkpoint checkpoints/tinyimagenet/random-resnet18-bc-100-12-lr0001-b256-dataaug-iclr \
#          --random

 python  trainer.py \
          --dataset tinyimagenet \
          --num-classes 200 \
          --data_augmentation \
          --model resnet18 \
          --depth 100 \
          --growthRate 12 \
          --epochs 300 \
          --schedule 150 225 \
          --gamma 0.1 \
          --wd 1e-4 \
          --train-batch 256 \
          --learning_rate1 0.001 \
          --checkpoint checkpoints/tinyimagenet/random-resnet18-bc-100-12-lr0001-b256-dataaug-iclr \
          --random
