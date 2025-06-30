#!/bin/bash -l
#SBATCH --nodes=1                 # Request 1 node
#SBATCH --ntasks=4                # Request 4 tasks (1 per GPU)
#SBATCH --gpus=4                  # Request 4 GPUs
#SBATCH --time=12:00:00           # Adjust the time limit as needed
#SBATCH --account=#######          # Account to be charged
#SBATCH --qos=default              # QOS setting
#SBATCH --mail-type=ALL            # Send mail on all events (begin, end, fail)
#SBATCH --mail-user=prateek.mathur@insight-centre.org
#SBATCH --job-name=ms_diffusion    # Job name
#SBATCH --partition=gpu            # Specify the GPU partition
#SBATCH --ntasks-per-node=4        # Number of tasks per node (equal to the number of GPUs)

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load Python/3.11.3-GCCcore-12.3.0
module load mpi4py/3.1.4

# Activate the pre-existing virtual environment
source ~/env_setup/ms_diff/bin/activate

# Set environment variable for distributed training
export OMP_NUM_THREADS=4  # Number of threads per process (adjust based on the CPU cores available)

# Run distributed training using SLURM and torch.distributed.launch
srun python -m torch.distributed.run --nproc_per_node=4 train_ms.py --root_dir /project/home/p200468/ms_slice_data/ \
                                     --csv_file training_data_diffusion.csv \
                                     --batch_size 16 \
                                     --lr 1e-4 \
                                     --save_interval 5000 \
                                     --log_interval 1000 \
                                     --ema_rate 0.9999 \
                                     --use_fp16 True \
                                     --class_cond True \
                                     --rescale_learned_sigmas False \
                                     --learn_sigma False \
                                     --segmentation_model /home/users/u101835/ms-diffusion/basic_unet/checkpoint.pt
