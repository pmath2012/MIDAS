import argparse
import torch
import sys
import os
sys.path.append("..")
from glasses.models.segmentation.unet import UNet
from models.vision_transformer import ViT
from utils.utils import get_loss_function
from datasets import get_train_val_dataloaders
from train.training import train_epochs

def check_keys(model, pretrain_path):
    # Check for matching keys
    pretrain = torch.load(pretrain_path)
    pretrained_keys = set(pretrain.keys())
    model_keys = set(model.state_dict().keys())

    missing_keys = model_keys - pretrained_keys
    unexpected_keys = pretrained_keys - model_keys

    if len(missing_keys) > 0:
        print("Missing keys in model:", missing_keys)
    else: 
        print("No Missing keys")

    if len(unexpected_keys) > 0:
        print("Unexpected keys in pre-trained model:", unexpected_keys)
    else:
        print("No unexpected keys")

def parse_args():
    parser = argparse.ArgumentParser(description='Train ISLES baselines')
    parser.add_argument('--model_name', type=str, default='unet', 
                       choices=['unet', 'vit_50'], help='Model name to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use')
    parser.add_argument('--pretrain', action='store_true', help='Use pre-trained model')
    parser.add_argument('--pretrain_path', type=str, default="", help='Path to pre-trained model')
    parser.add_argument('--loss', type=str, choices=['f0.5', 'f1', 'f2', 'dice', 'DiceFocalLoss', 'DiceCELoss', 'BCE', 'fce_0.5', 'fce_1', 'fce_2'], default='f0.5', help='Mask loss function')
    parser.add_argument('--data_directory', type=str, default='', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--training_file', type=str, default='train.csv', help='Training file')
    parser.add_argument('--validation_file', type=str, default='valid.csv', help='Validation file')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU to use')
    parser.add_argument('--run_name', type=str, default=None, help='Unique run identifier')
    parser.add_argument('--vis_freq', type=int, default=10, help='Visualization frequency (every N epochs)')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs)')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()

    print('-'*100)
    print(f"\n\n\nTraining {args.model_name} for ISLES\n\n\n")
    print('-'*100)

    learning_rate = args.learning_rate
    loss = get_loss_function(args.loss)
    epochs = args.epochs
    pretrain = args.pretrain
    pretrain_path = args.pretrain_path
    data_directory = args.data_directory
    train_file = args.training_file
    valid_file = args.validation_file
    batch_size = args.batch_size
    device = args.gpu if torch.cuda.is_available() else 'cpu'
    run_name = args.run_name
    vis_freq = args.vis_freq
    augment = args.augment
    patience = args.patience
    
    # Model selection
    if args.model_name == 'unet':
        model = UNet(in_channels=1, num_classes=1)  # Single channel for DWI images
    elif args.model_name == 'vit_50':
        model = ViT(
            in_channels=1, 
            num_classes=1, 
            with_pos='learned',
            resnet_stages_num=5,
            backbone='resnet50',
            pretrained=True
        )
    else:
        raise ValueError(f"Unsupported model name {args.model_name}")

    if pretrain:
        check_keys(model, pretrain_path)
        model.load_state_dict(torch.load(pretrain_path), strict=False)
    
    # Get dataloaders using our ISLES implementation
    train_dataloader, valid_dataloader = get_train_val_dataloaders(
        train_csv=f"{data_directory}/{train_file}",
        val_csv=f"{data_directory}/{valid_file}",
        root_dir=data_directory,
        batch_size=batch_size,
        augment_train=augment
    )
    
    # Optimizer selection
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    # Model checkpoint path
    if run_name is not None:
        ckpt = f"checkpoints/{args.model_name}_{run_name}_{args.loss}_epochs_{epochs}.pth"
    else:
        ckpt = f"checkpoints/{args.model_name}_{args.loss}_epochs_{epochs}.pth"
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Augmentation: {augment}")
    print(f"Loss function: {args.loss}")
    print(f"Run name: {run_name}")
    print(f"Visualization frequency: {vis_freq}")
    print(f"Early stopping patience: {patience}")
    print('-'*100)

    # Start training using our enhanced training function
    history = train_epochs(
        model=model,
        trainloader=train_dataloader,
        valloader=valid_dataloader,
        optimizer=optimizer,
        criterion=loss,
        num_epochs=epochs,
        device=device,
        save_path=ckpt,
        run_name=run_name,
        vis_freq=vis_freq,
        patience=patience
    )
    
    print(f"\nTraining completed! Best model saved to: {ckpt}")
    print(f"Final validation Dice: {max(history['val_dice']):.4f}")
    print('-'*100)
