import torch
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from models.vision_transformer import ViT
from glasses.models.segmentation.unet import UNet
from datasets import get_isles_dataloader
from sklearn.metrics import precision_score, recall_score


def parse_args():
    parser = argparse.ArgumentParser('Test ISLES model')
    parser.add_argument('--model_name', type=str, default='unet', choices=['unet', 'vit_50'], help='Model name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--data_directory', type=str, required=True, help='Root directory for images/masks')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predicted segmentations')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name for subdirectory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU to use')
    return parser.parse_args()


def save_predictions(preds, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for pred, fname in zip(preds, filenames):
        # pred: [H, W] numpy array, fname: string
        pred_img = (pred > 0.5).astype(np.uint8) * 255  # Ensure binary mask
        cv2.imwrite(os.path.join(output_dir, fname), pred_img)


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection) / (y_true_f.sum() + y_pred_f.sum() + 1e-8)


def main():
    args = parse_args()
    device = args.gpu if torch.cuda.is_available() else 'cpu'

    # Model selection
    if args.model_name == 'unet':
        model = UNet(in_channels=1, num_classes=1)
    elif args.model_name == 'vit_50':
        model = ViT(
            in_channels=1,
            num_classes=1,
            with_pos='learned',
            resnet_stages_num=5,
            backbone='resnet50',
            pretrained=False
        )
    else:
        raise ValueError(f"Unsupported model name {args.model_name}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Dataloader (no augmentation, shuffle False)
    test_loader = get_isles_dataloader(
        csv_file=args.test_csv,
        root_dir=args.data_directory,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        augment=False
    )

    # Read filenames from CSV for saving
    import pandas as pd
    df = pd.read_csv(args.test_csv)
    image_filenames = df['image'].tolist()
    mask_filenames = df['label'].tolist()  # Not used, but could be for evaluation

    all_preds = []
    all_names = []
    all_metrics = []
    idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()  # [B, 1, H, W]
            gt_masks = batch['mask'].cpu().numpy() if 'mask' in batch else None
            for i in range(preds.shape[0]):
                pred_mask = preds[i, 0]  # [H, W]
                # Use the original image filename, but save as .png
                base_name = os.path.splitext(os.path.basename(image_filenames[idx]))[0]
                out_name = f"{base_name}_pred.png"
                all_preds.append(pred_mask)
                all_names.append(out_name)
                # Compute metrics if ground truth is available
                if gt_masks is not None:
                    gt_mask = gt_masks[i, 0]
                    dice = dice_coef(gt_mask, pred_mask)
                    precision = precision_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
                    recall = recall_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
                    all_metrics.append({
                        'filename': base_name,
                        'dice': dice,
                        'precision': precision,
                        'recall': recall
                    })
                idx += 1

    # Set experiment-specific output directory
    exp_output_dir = os.path.join(args.output_dir, args.experiment_name)

    save_predictions(all_preds, all_names, exp_output_dir)
    print(f"Saved {len(all_preds)} predicted segmentations to {exp_output_dir}")

    # Save results CSV if ground truth is available
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        results_csv_path = os.path.join(exp_output_dir, 'results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved slice-wise metrics to {results_csv_path}")

if __name__ == '__main__':
    main() 