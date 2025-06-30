#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

export DIFFUSION_BLOB_LOGDIR="/mnt/recsys/prateek/models/controlled_diff_ct2mri/"

echo "Starting training on a single GPU in background..."
logfile="continue_training_reduced_lr.log"
# Run the training script in background with nohup to keep it running after logout
# Redirect both stdout and stderr to continue_training.log
nohup python train_ms.py --root_dir "/mnt/recsys/prateek/Isles2024Dataset_v2/" \
                                     --csv_file "train_diffusion.csv" \
                                     --batch_size 12 \
                                     --lr 1e-5 \
                                     --resume_checkpoint "/mnt/recsys/prateek/models/controlled_diff_ct2mri/model285000.pt" \
                                     --save_interval 5000 \
                                     --log_interval 1000 \
                                     --ema_rate 0.9999 \
                                     --use_fp16 False \
                                     --class_cond False \
                                     --rescale_learned_sigmas False \
                                     --learn_sigma False \
                                     --application modality_conversion \
                                     --segmentation_model "/home/prateek/controlled-diffusion/notebooks/ct2mri_unet.pth" \
                                     > $logfile 2>&1 &

# Get the process ID
TRAINING_PID=$!

echo "Training started with PID: $TRAINING_PID"
echo "Logs are being written to: $logfile"
echo "You can monitor the training with: tail -f $logfile"
echo "You can check if the process is still running with: ps -p $TRAINING_PID"
echo "To stop the training, use: kill $TRAINING_PID"
echo ""
echo "You can now safely disconnect from the server. The training will continue running." 