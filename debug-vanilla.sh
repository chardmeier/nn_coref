#! /bin/bash

#SBATCH -p gpu
#SBATCH --mem 48g
#SBATCH -t 10:00:00
#SBATCH --gres gpu:1

th=$USERAPPL/torch/install/bin/th

modeldir=$WRKDIR/nn_coref-models

cd $HOME/coref/nn_coref/nn
$th debug_vanilla_mr.lua -Ha 128 -loadAndPredict \
	-pwDevFeatPrefix test_small -anaDevFeatPrefix test_small \
	-savedPWNetFi $modeldir/simple-vanilla-700-128.model-vanilla-pw \
	-savedNANetFi $modeldir/simple-vanilla-700-128.model-vanilla-na

