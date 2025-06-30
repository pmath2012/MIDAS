#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python train_ms.py --root_dir /home/prateek/ms_project/ms_slice_data/ \
                    --csv_file training_data_diffusion.csv \
                    --batch_size 4 \
                    --lr 1e-4 \
                    --save_interval 500 \
                    --log_interval 10 \
                    --ema_rate 0.9999 \
                    --use_fp16 False \
                    --world_size 1 \
                    --class_cond True \
                    --rescale_learned_sigmas False \
                    --learn_sigma False \
                    --segmentation_model /home/prateek/ms-diffusion/basic_unet/checkpoint.pt