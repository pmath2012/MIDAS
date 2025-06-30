import os
import cv2
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from skimage import measure, io
import re


def strip_small_lesions(target_slice, x_len=3, y_len=3):
    labels = measure.label(target_slice, background=0)
    props = measure.regionprops_table(labels, properties=('label','bbox','axis_major_length','axis_minor_length'))
    props = {k: np.array(v) for k, v in props.items()}
    remove = [(x < x_len) and (y < y_len) for x, y in zip(props['axis_major_length'], props['axis_minor_length'])]
    labels_to_remove = props['label'][remove].tolist() if len(remove) > 0 else []
    labels_to_remove.append(0)  # background
    new_labels = np.isin(labels, labels_to_remove, invert=True)
    return new_labels.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser('Reconstitute ISLES predicted slices into 3D stacks')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name (subdir in pred_dir and output_dir)')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted PNGs (output_dir/experiment_name)')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save 3D stacks (output_dir/experiment_name)')
    parser.add_argument('--strip', action='store_true', help='Strip small lesions from slices')
    parser.add_argument('--external', action='store_true', help='Enable external mode for train/valid/test and sub-strokecase support')
    return parser.parse_args()


def main():
    args = parse_args()
    # If experiment is ground_truth, do not append to pred_dir
    if args.experiment_name == 'ground_truth':
        pred_dir = args.pred_dir
    else:
        pred_dir = os.path.join(args.pred_dir, args.experiment_name)
    out_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    # Find all predicted PNGs
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
    # Parse patient ID and slice index from filenames
    patient_slices = {}
    for fname in pred_files:
        if args.external:
            # External: match (test|train|valid) and sub-stroke or sub-strokecase
            match = re.match(r'(?:test|train|valid)_(sub-stroke(?:case)?\d{4})_(\d{5})_(?:dwi_pred|mask)\.png', fname)
        else:
            # Internal: match only test_ and sub-stroke
            match = re.match(r'test_(sub-stroke\d{4})_(\d{5})_(?:dwi_pred|mask)\.png', fname)
        if not match:
            print(f"Skipping unrecognized filename: {fname}")
            continue
        patient_id, slice_idx = match.group(1), int(match.group(2))
        if patient_id not in patient_slices:
            patient_slices[patient_id] = {}
        patient_slices[patient_id][slice_idx] = fname

    files_out = []
    for patient_id, slices_dict in patient_slices.items():
        if not slices_dict:
            print(f"No slices found for patient {patient_id}, skipping.")
            continue
        max_idx = max(slices_dict.keys())
        min_idx = min(slices_dict.keys())
        N = max_idx - min_idx + 1
        vol = np.zeros((256, 256, N), dtype=np.uint8)
        for idx, fname in slices_dict.items():
            img = io.imread(os.path.join(pred_dir, fname))
            img = (img > 0).astype(np.uint8)
            if args.strip:
                img = strip_small_lesions(img)
            vol[:, :, idx - min_idx] = img
        # Use suffix from the first file for output name
        suffix = '_mask' if any(f.endswith('_mask.png') for f in slices_dict.values()) else '_pred'
        nii_name = f"{patient_id}{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), os.path.join(out_dir, nii_name))
        print(f"Saved {nii_name} with shape {vol.shape}")
        files_out.append({'patient': patient_id, 'file': nii_name, 'num_slices': N})
    pd.DataFrame(files_out).to_csv(os.path.join(out_dir, 'stacks_index.csv'), index=False)
    print(f"Saved stacks_index.csv in {out_dir}")

if __name__ == '__main__':
    main() 