#!/bin/bash

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Set the model name and other parameters
MODEL_NAME="vit_50"  # Change to 'unet', 'vit_50', etc., as needed
EPOCHS=100
LEARNING_RATE=2e-5
OPTIMIZER="AdamW"
LOSS="f0.5"  # Choose from 'f0.5', 'f1', 'f2'
DATA_DIRECTORY="/mnt/recsys/prateek/isles_synthetic_data"
BATCH_SIZE=64
TRAIN_FILE="train_real_0_synth_100.csv"
VALID_FILE="valid.csv"
GPU="cuda:0"  # This will now be limited to GPU 0 due to CUDA_VISIBLE_DEVICES
RUN_NAME="real_0_synth_100"
VIS_FREQ=10
AUGMENT=true

# Log file for output
LOG_FILE="train_${MODEL_NAME}_${RUN_NAME}.log"

# Run the training script in the background with low CPU and IO priority
nohup nice -n 10 ionice -c2 -n7 python train_baselines.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --loss $LOSS \
    --data_directory $DATA_DIRECTORY \
    --batch_size $BATCH_SIZE \
    --training_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --gpu $GPU \
    --run_name $RUN_NAME \
    --vis_freq $VIS_FREQ \
    --augment > $LOG_FILE 2>&1 &

# Print process ID
echo "Training started in the background with PID: $!"
echo "CUDA_VISIBLE_DEVICES is set to 0"
echo "Check progress using: tail -f $LOG_FILE" 