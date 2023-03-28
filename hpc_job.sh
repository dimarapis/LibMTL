#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J autolambda
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u dimara@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/gpu_%J.out
#BSUB -e logs/gpu_%J.err
# -- end of LSF options --

# Load the cuda module
#module load numpy/1.21.1-python-3.8.11-openblas-0.3.17
#module load cuda/11.4

#Activate virtual env
unset PYTHONPATH
unset PYTHONHOME
source ~/miniconda3/bin/activate
conda activate autolambda
pip install -e .
#python trainer_dense.py --dataset sim_warehouse --task all --gpu 0 --weight autolconfigs/trainer_dwa_none.yaml

#python trainer_dense.py --dataset nyuv2 --task all --gpu 0 
#python trainer_dense.py --dataset nyuv2 --task all --weight equal --grad_method pcgrad --gpu 0
#python trainer_dense_single.py 

python examples/nyu/train_nyu.py --weighting EW --arch HPS --dataset_path dataset/nyuv2 --gpu_id 0 --scheduler step

## submit by using: bsub < jobscript.sh

#run1 seed0
#run2 seed29

#Equal results:  TEST: Seg 5.5368 0.1867 Depth 0.4211 0.4211 Normal 272.9856 0.329
#DWA results: TEST: Seg 6.1210 0.1824 Depth 0.4139 0.4139 Normal 272.9822 0.3755

#grad_method options: 

# graddrop, pcgrad, cagrad
