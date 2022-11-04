#!/bin/bash
#SBATCH --job-name=Three_couple
#SBATCH -p gpu1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
# Record current host & date.
hostname; date
# Initialize cuda environement.
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
source /lustre/grp/gjhlab/lvbj/lyz_grp/Software/anaconda3/bin/activate WHT
source /lustre/grp/gjhlab/lvbj/.bashrc
# Initialize cuda environement.
# Train Speech_decode.
cd ..
python Prepro.py
