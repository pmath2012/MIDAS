import os
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from PIL import Image
import torch
import lpips
import pandas as pd
import csv

# Paths
ROOT_DIR = "/mnt/recsys/prateek/Isles2024Dataset_v2/"
CSV_FILE = "test.csv"
GENERATED_DIR = "isles_gen/pix2pix/"
OUTPUT_CSV = "evaluation_metrics_pix2pix.csv"

# Load test.csv
test_df = pd.read_csv(os.path.join(ROOT_DIR, CSV_FILE))

# LPIPS setup
lpips_fn = lpips.LPIPS(net='alex')  # or 'vgg'
lpips_fn = lpips_fn.cuda() if torch.cuda.is_available() else lpips_fn

metrics = []

for idx, row in test_df.iterrows():
    real_path = os.path.join(ROOT_DIR, row["path_B"])
    gen_path = os.path.join(GENERATED_DIR, os.path.basename(row["path_B"]))
    if not os.path.exists(gen_path):
        print(f"Generated image not found: {gen_path}")
        continue
    if not os.path.exists(real_path):
        print(f"Real image not found: {real_path}")
        continue
    fake_img = np.array(Image.open(gen_path).convert('L'))
    real_img = np.array(Image.open(real_path).convert('L'))

    # Normalize to [0, 1]
    fake_img_norm = fake_img.astype(np.float32) / 255.0
    real_img_norm = real_img.astype(np.float32) / 255.0

    # MSE
    mse = mean_squared_error(real_img_norm, fake_img_norm)
    # PSNR
    psnr = peak_signal_noise_ratio(real_img_norm, fake_img_norm, data_range=1.0)
    # SSIM
    ssim = structural_similarity(real_img_norm, fake_img_norm, data_range=1.0)
    # LPIPS (expects 3-channel, normalized to [-1, 1])
    fake_lpips = np.stack([fake_img_norm]*3, axis=2)
    real_lpips = np.stack([real_img_norm]*3, axis=2)
    fake_tensor = torch.from_numpy(fake_lpips).permute(2,0,1).unsqueeze(0).float()
    real_tensor = torch.from_numpy(real_lpips).permute(2,0,1).unsqueeze(0).float()
    if torch.cuda.is_available():
        fake_tensor = fake_tensor.cuda()
        real_tensor = real_tensor.cuda()
    lpips_val = lpips_fn(fake_tensor, real_tensor).item()

    metrics.append({
        'image': os.path.basename(row["path_B"]),
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips_val
    })

# Compute averages and stds
avg_mse = np.mean([m['mse'] for m in metrics])
std_mse = np.std([m['mse'] for m in metrics])
avg_psnr = np.mean([m['psnr'] for m in metrics])
std_psnr = np.std([m['psnr'] for m in metrics])
avg_ssim = np.mean([m['ssim'] for m in metrics])
std_ssim = np.std([m['ssim'] for m in metrics])
avg_lpips = np.mean([m['lpips'] for m in metrics])
std_lpips = np.std([m['lpips'] for m in metrics])

print(f"Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
print(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f}")

# Write per-image and average metrics to CSV
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    fieldnames = ['image', 'mse', 'psnr', 'ssim', 'lpips']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for m in metrics:
        writer.writerow(m)
    writer.writerow({'image': 'MEAN', 'mse': avg_mse, 'psnr': avg_psnr, 'ssim': avg_ssim, 'lpips': avg_lpips})
    writer.writerow({'image': 'STD', 'mse': std_mse, 'psnr': std_psnr, 'ssim': std_ssim, 'lpips': std_lpips})

print(f"Results saved to {OUTPUT_CSV}") 
