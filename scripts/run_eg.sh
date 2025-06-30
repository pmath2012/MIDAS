#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python create_samples.py --root_dir /mnt/recsys/prateek/synthetic_ms_data \
                    --csv_file synthetic_data_part1.csv \
                    --batch_size 10 \
                    --checkpoint /home/prateek/ms-diffusion/saved_models/ema_0.9999_080000.pt