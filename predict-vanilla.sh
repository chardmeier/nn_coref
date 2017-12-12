#! /bin/bash

#SBATCH -p gpu
#SBATCH --mem 48g
#SBATCH -t 10:00:00
#SBATCH --gres gpu:1

if [ $# -lt 1 ]
then
	echo "Usage: $0 exp [args...]" 1>&2
	exit 1
fi
exp=$1
shift

module load cuda cudnn

echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

nvidia-smi

th=$USERAPPL/torch/install/bin/th

export LD_LIBRARY_PATH=$CUDALIB

modeldir=$WRKDIR/nn_coref-models
bpsdir=bps

cd $HOME/coref/nn_coref/nn
$th vanilla_mr.lua -gpuid 0 -Ha 128 -loadAndPredict \
	-pwDevFeatPrefix dev_small -anaDevFeatPrefix dev_small \
	-savedPWNetFi $modeldir/$exp-vanilla-700-128.model-vanilla-pw \
	-savedNANetFi $modeldir/$exp-vanilla-700-128.model-vanilla-na \
	-bpfi $bpsdir/$exp-vanilla-700-128.bps \
	"$@"

