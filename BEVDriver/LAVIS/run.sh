#!/bin/bash -l
#SBATCH --job-name=run_bevdriver
#SBATCH --nodes=1                     
#SBATCH --ntasks=1          
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks-per-node=1        
#SBATCH --time=24:00:00             
#SBATCH --output=run_bevdriver_output.txt
#SBATCH --error=run_bevdriver_error.txt

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# export CUDA_VISIBLE_DEVICES=0
CONFIG_PATH='lavis/projects/bevdriver/train_modular.yaml'

conda activate bevdriver

${CONDA_PREFIX}/bin/python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=12345 \
    train.py \
    --cfg-path $CONFIG_PATH

conda deactivate