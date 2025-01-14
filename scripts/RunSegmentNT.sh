#!/bin/bash

#SBATCH -p defq
#SBATCH --account=awdu223_dgxnt24
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH -o /pscratch/awdu223_dgxnt24/OutputLogs/RunJaxGetProb.out

export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

srun -N${SLURM_NNODES} --ntasks=${SLURM_NPROCS} --gpus-per-task=${SLURM_GPUS_PER_TASK} --gpu-bind=none \
    --container-image=awdu223/nucleotidetransformerjax \
    --container-mounts=/pscratch/awdu223_dgxnt24/BernardoExonsProject:/run \
    -e /pscratch/awdu223_dgxnt24/ErrorLogs/JaxGetProbabilities-%j.err \
    -o /pscratch/awdu223_dgxnt24/OutputLogs/JaxGetProbabilities-%j.out \
        python3 -u /run/scripts/JaxGetProbabilities.py
