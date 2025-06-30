# Updated compute_image_metrics.py script with grayscale handling

import os
import re
import argparse
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error
from scipy.stats import wasserstein_distance
from skimage import io
from tqdm import tqdm

try:
    import lpips
    import torch
except ImportError:
    lpips = None
    torch = None
    print("Warning: lpips not installed. LPIPS metric will be skipped.")

def extract_subject_id(filename):
    # Try sub-strokeXXXX-YYYYY
    match = re.search(r'(sub-stroke\d{4}-\d{5})', filename)
    if match:
        return match.group(1)
    # Try sub-strokeXXXX_YYYYY (e.g., test_sub-stroke0109_00071_dwi.png)
    match = re.search(r'sub-stroke(\d{4})_(\d{5})', filename)
    if match:
        return f"sub-stroke{match.group(1)}-{match.group(2)}"
    return None

def normalize_image(img):
    """Convert to float32 in [0,1] range, handling uint8 or float inputs."""
    img = img.astype(np.float32)
    if img.dtype == np.float32 and img.max() <= 1.0:
        return img
    if img.max() > 1.0:
        img /= 255.0
    return img

def main():
    parser = argparse.ArgumentParser(description='Compute SSIM, PSNR, MAE, and LPIPS for grayscale images.')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory with ground truth images')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory with predicted images')
    parser.add_argument('--comp_name', type=str, required=True, help='Comparison name for output CSV')
    args = parser.parse_args()

    # Build maps from subject_id to file path
    gt_files = {extract_subject_id(f): os.path.join(args.gt_dir, f)
                for f in os.listdir(args.gt_dir) if extract_subject_id(f)}
    target_files = {extract_subject_id(f): os.path.join(args.target_dir, f)
                    for f in os.listdir(args.target_dir) if extract_subject_id(f)}

    # Remove any None keys (shouldn't be present due to the if above, but for safety)
    gt_files = {k: v for k, v in gt_files.items() if k is not None}
    target_files = {k: v for k, v in target_files.items() if k is not None}

    common_ids = sorted(set(gt_files) & set(target_files))
    print(f"Found {len(common_ids)} common subject_ids.")

    if lpips is not None:
        lpips_fn = lpips.LPIPS(net='alex').eval()
    results = []

    for subject_id in tqdm(common_ids):
        gt_img = io.imread(gt_files[subject_id])
        pred_img = io.imread(target_files[subject_id])

        # If comp_name is 'resvit', convert to single channel if RGB
        if args.comp_name.lower() == 'resvit':
            if gt_img.ndim == 3 and gt_img.shape[-1] == 3:
                gt_img = gt_img.mean(axis=-1)
            if pred_img.ndim == 3 and pred_img.shape[-1] == 3:
                pred_img = pred_img.mean(axis=-1)

        # normalize to float32 [0,1]
        gt = normalize_image(gt_img)
        pred = normalize_image(pred_img)

        try:
            # Compute SSIM/PSNR/MAE on the single-channel array
            ssim_val = ssim(gt, pred, data_range=1.0)
            psnr_val = psnr(gt, pred, data_range=1.0)
            mae_val  = mean_absolute_error(gt.flatten(), pred.flatten())
            # Compute EMD
            emd_val = wasserstein_distance(gt.flatten(), pred.flatten())
        except Exception as e:
            print(f"Error for subject_id: {subject_id}")
            print(f"GT shape: {gt.shape}, Pred shape: {pred.shape}")
            raise

        # Compute LPIPS if available: replicate channels for network
        if lpips is not None and torch is not None:
            gt_rgb   = np.stack([gt]*3, axis=-1)
            pred_rgb = np.stack([pred]*3, axis=-1)
            # convert to [-1,1] tensor
            gt_tensor = torch.from_numpy(gt_rgb.transpose(2,0,1)).unsqueeze(0).float() * 2 - 1
            pr_tensor = torch.from_numpy(pred_rgb.transpose(2,0,1)).unsqueeze(0).float() * 2 - 1
            with torch.no_grad():
                lpips_val = float(lpips_fn(gt_tensor, pr_tensor))
        else:
            lpips_val = None

        results.append({
            'subject_id': subject_id,
            'ssim': ssim_val,
            'psnr': psnr_val,
            'mae': mae_val,
            'lpips': lpips_val,
            'emd': emd_val
        })

    out_csv = f"{args.comp_name}_metrics.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")

if __name__ == '__main__':
    main()

