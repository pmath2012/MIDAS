import os
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from improved_diffusion.script_util import create_conditional_model_and_diffusion, args_to_dict
from scripts.train_ms import model_and_diffusion_defaults
from improved_diffusion.ms_datasets import PERF_LOAD_MAX, PERF_NUM_MAX

# --------- CONFIG ---------
# Set these paths as needed
MODEL_PATH = "/path/to/model.pt/"
ROOT_DIR = "/path/to/data/"
CSV_FILE = "train.csv"
OUTPUT_DIR = "/path/to/output"
BATCH_SIZE = 64  # Slightly increased for better GPU utilization
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # For parallel image loading
USE_CACHE = True  # Enable image caching

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache for loaded images to avoid repeated disk I/O
_image_cache = {}

def read_and_normalize(root_dir, path, normalize=True, mask=False, use_cache=True):
    """Optimized image loading with caching"""
    if use_cache and USE_CACHE:
        cache_key = f"{root_dir}_{path}_{normalize}_{mask}"
        if cache_key in _image_cache:
            return _image_cache[cache_key]
    
    img = cv2.imread(os.path.join(root_dir, path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {os.path.join(root_dir, path)}")
    
    if mask:
        img = np.where(img.astype(np.float32) > 128, 1, 0).astype(np.float32)
    else:
        if normalize:
            img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    if use_cache and USE_CACHE:
        _image_cache[cache_key] = img
    
    return img

def get_sample(row, root_dir):
    """Optimized sample preparation"""
    mask = read_and_normalize(root_dir, row["label"], mask=True)
    params = {
        "lesion_load": torch.tensor(row['lesion_load'] / PERF_LOAD_MAX, dtype=torch.float32),
        "num_lesions": torch.tensor(row['num_lesions'] / PERF_NUM_MAX, dtype=torch.float32),
        "lesion": torch.tensor(row['lesion'], dtype=torch.float32),
    }
    imgA = read_and_normalize(root_dir, row["path_A"])
    mask_tensor = torch.from_numpy(mask)
    img_tensor = torch.from_numpy(imgA).unsqueeze(0)
    mask_tensor = torch.where(mask_tensor > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    mask_tensor = mask_tensor.unsqueeze(0)
    params["mask"] = mask_tensor
    params["t0"] = img_tensor
    return params

def create_batch(samples, device):
    """Optimized batch creation"""
    masks = torch.stack([s["mask"] for s in samples]).to(device)
    t0s = torch.stack([s["t0"] for s in samples]).to(device)
    lesion_loads = torch.stack([s["lesion_load"] for s in samples]).to(device)
    num_lesions = torch.stack([s["num_lesions"] for s in samples]).to(device)
    lesions = torch.stack([s["lesion"] for s in samples]).to(device)
    
    model_kwargs = {
        "mask": masks,
        "t0": t0s,
        "lesion_load": lesion_loads,
        "num_lesions": num_lesions,
        "lesion": lesions,
    }
    return model_kwargs

def save_image_batch(images, filenames, output_dir):
    """Optimized batch image saving"""
    for img, filename in zip(images, filenames):
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        img = img.permute(1, 2, 0).squeeze().cpu().numpy()
        out_path = os.path.join(output_dir, filename)
        if not cv2.imwrite(out_path, img):
            print(f"Error writing {out_path}")

def preload_images_parallel(test_df, root_dir, num_workers=4):
    """Preload all images in parallel to reduce I/O bottlenecks"""
    print("Preloading images in parallel...")
    
    def load_row_images(row):
        try:
            mask = read_and_normalize(root_dir, row["label"], mask=True, use_cache=False)
            imgA = read_and_normalize(root_dir, row["path_A"], use_cache=False)
            return row.name, mask, imgA
        except Exception as e:
            print(f"Error loading images for row {row.name}: {e}")
            return row.name, None, None
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_row_images, row) for _, row in test_df.iterrows()]
        
        # Store preloaded images in cache
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preloading"):
            row_idx, mask, imgA = future.result()
            if mask is not None and imgA is not None:
                _image_cache[f"{root_dir}_{test_df.iloc[row_idx]['label']}_True_True"] = mask
                _image_cache[f"{root_dir}_{test_df.iloc[row_idx]['path_A']}_True_False"] = imgA

def generate_filename_from_mask(mask_path):
    """Generate filename from mask path: mask_#.png -> image_dwi_#.png"""
    mask_basename = os.path.basename(mask_path)
    
    # Extract the number from mask_#.png
    if mask_basename.startswith('mask_') and mask_basename.endswith('.png'):
        number_part = mask_basename[5:-4]  # Remove 'mask_' prefix and '.png' suffix
        return f"image_dwi_{number_part}.png"
    else:
        # Fallback: just replace 'mask' with 'image_dwi'
        return mask_basename.replace('mask', 'image_dwi')

def main():
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Image caching: {USE_CACHE}")
    
    # 1. Load model and diffusion
    print("Loading model and diffusion...")
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path=MODEL_PATH,
        output_dir=None,
        segmentation_model=None,
        learn_sigma=False,
        use_fp16=False,  # Keep FP32 for image quality
        csv_file="test.csv",
        root_dir="/path/to/data/",
        image_size=256,
        rescale_learned_sigmas=False,
        class_cond=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults["rescale_learned_sigmas"] = False
    defaults["application"] = "modality_conversion"
    args = type('Args', (), defaults)()
    
    model, diffusion = create_conditional_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()

    # 2. Load test dataframe
    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(ROOT_DIR, CSV_FILE))
    print(f"Loaded {len(test_df)} test samples")

    # 3. Preload images in parallel (optional - comment out if memory is limited)
    # preload_images_parallel(test_df, ROOT_DIR, NUM_WORKERS)

    # 4. Choose sample function
    sample_fn = diffusion.p_sample_loop

    # 5. Generate images in batches with parallelization optimizations
    # Generate filenames from mask names instead of path_B
    filenames = [generate_filename_from_mask(row["label"]) for _, row in test_df.iterrows()]
    results = []
    
    print(f"Generating {len(test_df)} images in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(test_df), BATCH_SIZE), desc="Generating images"):
        batch_rows = test_df.iloc[i:i+BATCH_SIZE]
        actual_batch_size = len(batch_rows)
        
        # Prepare samples
        samples = [get_sample(row, ROOT_DIR) for _, row in batch_rows.iterrows()]
        model_kwargs = create_batch(samples, DEVICE)
        
        with torch.no_grad():
            gen_imgs = sample_fn(
                model,
                shape=(actual_batch_size, 1, 256, 256),
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
        
        # Save images with mask-based filenames
        batch_filenames = [generate_filename_from_mask(batch_rows.iloc[j]["label"]) 
                          for j in range(actual_batch_size)]
        save_image_batch(gen_imgs, batch_filenames, OUTPUT_DIR)
        
        # Update results
        for filename in batch_filenames:
            results.append({"filename": filename})
        
        # Clear cache periodically to prevent memory buildup
        if i % (BATCH_SIZE * 10) == 0:
            torch.cuda.empty_cache()
            if len(_image_cache) > 1000:  # Limit cache size
                _image_cache.clear()
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)
    print(f"Generated {len(results)} images successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 
