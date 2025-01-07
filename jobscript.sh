#!/usr/bin/env bash

#SBATCH -A NAISS2023-5-522 -p alvis

#SBATCH -N 1 --gpus-per-node=A100:4

#SBATCH -t 3-00:00:00

#SBATCH -J "MNMG PyTorch"





#FREE_PORT=`comm -23 <(seq "8888" "8988" | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf -n 1`
#echo "Tensorboard URL: https://proxy.c3se.chalmers.se:${FREE_PORT}/`hostname`/"
#tensorboard --path_prefix /`hostname`/ --bind_all --port $FREE_PORT --logdir=./tensorboard-log-$SLURM_JOB_ID &

#cd /mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/TinyIMGNT/


#python /mimer/NOBACKUP/groups/naiss2023-5-522/ali/mixed_precision_one_shot_training/TinyIMGNT/uptimize_with_scaler_and_constraint.py
#srun -N $SLURM_JOB_NUM_NODES --export=ALL --ntasks-per-node=4 bash -c " NCCL_DEBUG=INFO python -m torch.distributed.run --node_rank="'$SLURM_NODEID'" --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d uptimize_with_scaler_and_constraint.py"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$SLURM_GPUS_ON_NODE uptimize_with_scaler_and_constraint.py