#!/bin/bash -l
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=4                # Request 4 tasks (1 per GPU)
#SBATCH --gpus=4                  # Request 4 GPUs
#SBATCH --time=08:00:00           # Adjust the time limit as needed
#SBATCH --account=p200932          # Account to be charged
#SBATCH --qos=default              # QOS setting
#SBATCH --mail-type=ALL            # Send mail on all events (begin, end, fail)
#SBATCH --mail-user=prateek.mathur@insight-centre.org
#SBATCH --job-name=controlled_diff_ct2mri # Job name
#SBATCH --partition=gpu            # Specify the GPU partition
#SBATCH --ntasks-per-node=4        # Number of tasks per node (equal to the number of GPUs)

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load env/staging/2023.1
module load Python/3.11.3-GCCcore-12.3.0
module load TensorRT/10.0.0-foss-2023a-CUDA-12.2.0
module load mpi4py/3.1.4

# Activate the pre-existing virtual environment
source ~/env_setup/diffusion/bin/activate

# Set environment variable for distributed training
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DIFFUSION_BLOB_LOGDIR="/project/home/p200932/models/controlled_diff_ct2mri/"
# Set MASTER_ADDR to the current node hostname and assign MASTER_PORT
#export MASTER_ADDR=$(hostname -s)
#export MASTER_PORT=$(shuf -i 10000-20000 -n 1)

#echo "Using MASTER_ADDR: $MASTER_ADDR and MASTER_PORT: $MASTER_PORT"

# Run distributed training using SLURM and torchrun
srun python train_ms.py --root_dir /project/home/p200932/Isles2024Dataset/ \
                                     --csv_file train_diffusion.csv \
                                     --batch_size 20 \
                                     --lr 1e-4 \
				     --resume_checkpoint /project/home/p200932/models/controlled_diff_ct2mri/model110000.pt \
				     --opt_checkpoint /project/home/p200932/models/controlled_diff_ct2mri/opt110000.pt \
                                     --save_interval 5000 \
                                     --log_interval 1000 \
                                     --ema_rate 0.9999 \
                                     --use_fp16 False \
                                     --class_cond False \
                                     --rescale_learned_sigmas False \
                                     --learn_sigma False \
				     --application modality_conversion \
                                     --segmentation_model /home/users/u101835/controlled-diffusion/notebooks/ct2mri_unet.pth
