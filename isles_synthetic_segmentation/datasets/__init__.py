import torch
from torch.utils.data import DataLoader
import sys
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Clear any existing paths and add only the project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets.lesions_dataset import ISLESDataset
from utils.group_transforms import Normalize, AugmentTransform

class TransformPipeline:
    """Pipeline to apply multiple transforms sequentially."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

def get_isles_dataloader(csv_file, root_dir, batch_size=4, shuffle=True, 
                        num_workers=4, transform=None, augment=False):
    """
    Create a DataLoader for the ISLES dataset.
    
    Args:
        csv_file (str): Path to the CSV file with image and label paths
        root_dir (str): Root directory containing the images
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        transform (callable, optional): Additional transforms to apply
        augment (bool): Whether to apply augmentation transforms
    
    Returns:
        DataLoader: PyTorch DataLoader for the ISLES dataset
    """
    
    # Create transform pipeline
    transforms = []
    
    # Add normalization
    transforms.append(Normalize(output_intensity=1))
    
    # Add augmentation if requested
    if augment:
        transforms.append(AugmentTransform(
            v_flip=True, 
            h_flip=True, 
            elastic_transform=True, 
            p=0.5
        ))
    
    # Add any additional transforms
    if transform is not None:
        transforms.append(transform)
    
    # Create transform pipeline
    transform_pipeline = TransformPipeline(transforms) if transforms else None
    
    # Create dataset
    dataset = ISLESDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform_pipeline
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

def get_train_val_dataloaders(train_csv, val_csv, root_dir, batch_size=4, 
                             num_workers=4, augment_train=True):
    """
    Create separate train and validation dataloaders.
    
    Args:
        train_csv (str): Path to training CSV file
        val_csv (str): Path to validation CSV file
        root_dir (str): Root directory containing the images
        batch_size (int): Batch size for the dataloaders
        num_workers (int): Number of worker processes
        augment_train (bool): Whether to apply augmentation to training data
    
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    
    train_loader = get_isles_dataloader(
        csv_file=train_csv,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augment=augment_train
    )

    val_loader = get_isles_dataloader(
        csv_file=val_csv,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        augment=False
    )

    return train_loader, val_loader
