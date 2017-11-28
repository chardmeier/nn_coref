#! /bin/bash

#SBATCH -p gpu
#SBATCH --mem 48g
#SBATCH -t 10:00:00
#SBATCH --gres gpu:1

module load cuda cudnn

echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

nvidia-smi

th=$USERAPPL/torch/install/bin/th

export LD_LIBRARY_PATH=$CUDALIB

cd $HOME/coref/nn_coref/nn
$th vanilla_mr.lua -gpuid $CUDA_VISIBLE_DEVICES -Ha 128 -save
