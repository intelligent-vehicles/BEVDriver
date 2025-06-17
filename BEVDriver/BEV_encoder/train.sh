#!/bin/bash -l
#SBATCH --job-name=train_encoder        
#SBATCH --nodes=1                     
#SBATCH --ntasks=1         
#SBATCH --export=NONE            
#SBATCH --time=24:00:00    
#SBATCH --gres=gpu:4
#SBATCH --output=train_output_new.txt
#SBATCH --error=train_error_new.txt

GPU_NUM=8
DATASET_ROOT=('dataset')

conda activate bevdriver

./BEV_encoder/distributed_train.sh $GPU_NUM "${DATASET_ROOT[@]}"  --dataset carla --train-towns 1 2 3 4 5 6 7 10  --val-towns 1 \
    --train-weathers 0 1 2 3 4 5 6 7 8 9  --val-weathers 10 11 12 13 \
    --model bevdriver_encoder_train --sched cosine --epochs 100 --warmup-epochs 10 --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment bevdriver_encoder_train \
    --pretrained --output "encoder-models" 

conda deactivate