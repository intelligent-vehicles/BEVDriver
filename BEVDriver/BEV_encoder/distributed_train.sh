#!/bin/bash -l
#SBATCH --job-name=distributed_train_encoder         
#SBATCH --nodes=1                     
#SBATCH --ntasks=1         
#SBATCH --export=NONE            
#SBATCH --time=24:00:00    
#SBATCH --gres=gpu:1                          
#SBATCH --output=train_dist_output.txt
#SBATCH --error=train_dist_error.txt

GPU_NUM=1
shift
python3 -m torch.distributed.launch --nproc_per_node=$GPU_NUM BEV_encoder/train.py "$@"
