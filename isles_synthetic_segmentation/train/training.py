import torch
import numpy as np
from torchmetrics.functional import dice
from torchmetrics.functional.classification import accuracy, binary_f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

def mb(x):                 # helper to pretty-print MiB
    return x / 1024**2
print_every = 50

def visualize_batch(images, masks, predictions, epoch, run_name, save_dir='logs/visualizations'):
    """
    Create and save a grid visualization of validation results.
    
    Args:
        images: Batch of input images [B, C, H, W]
        masks: Batch of ground truth masks [B, C, H, W]
        predictions: Batch of model predictions [B, C, H, W]
        epoch: Current epoch number
        run_name: Unique run identifier
        save_dir: Directory to save visualizations
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    predictions = torch.sigmoid(predictions).cpu().numpy()  # Apply sigmoid for binary predictions
    
    batch_size = images.shape[0]
    
    # Select middle samples: batch_size/2 - 2 to batch_size/2 + 2
    start_idx = max(0, batch_size // 2 - 2)
    end_idx = min(batch_size, batch_size // 2 + 2)
    num_samples = end_idx - start_idx
    
    # Select middle samples from the batch
    images = images[start_idx:end_idx]
    masks = masks[start_idx:end_idx]
    predictions = predictions[start_idx:end_idx]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Input image
        img = images[i, 0]  # Take first channel
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Input Image {start_idx + i + 1}')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        mask = masks[i, 0]
        axes[i, 1].imshow(mask, cmap='Reds', alpha=0.7)
        axes[i, 1].set_title(f'Ground Truth {start_idx + i + 1}')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = predictions[i, 0]
        axes[i, 2].imshow(pred, cmap='Blues', alpha=0.7)
        axes[i, 2].set_title(f'Prediction {start_idx + i + 1}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{run_name}_epoch_{epoch}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def train_inner_loop(model, data, criterion, optimizer, device='cpu', train=True):
    """
    Training/validation loop for single image segmentation.
    
    Args:
        model: The segmentation model
        data: Dictionary containing 'image' and 'mask'
        criterion: Loss function
        optimizer: Optimizer (None for validation)
        device: Device to run on
        train: Whether this is training or validation
    
    Returns:
        Dictionary of metrics
    """
    running_metrics = {'loss': 0.0, 'accuracy': 0.0, 'f1': 0.0, 'dice': 0.0}
    image, labels = data['image'], data['mask']
    image = image.to(device)
    labels = labels.to(device)
    
    if train:
        optimizer.zero_grad()
    
    # forward pass
    outputs = model(image)
    loss = criterion(outputs, labels)
    running_metrics['loss'] = loss.item()

    with torch.no_grad():
        preds = outputs.detach()
        running_metrics['accuracy'] = accuracy(preds, labels, task='binary').cpu().item()
        running_metrics['f1'] = binary_f1_score(preds, labels).cpu().item()
        running_metrics['dice'] = dice(preds, labels).cpu().item()
    
    if train:
        loss.backward()
        optimizer.step()
    
    return running_metrics, image, labels, outputs

def train_model(model, trainloader, optimizer, criterion, device='cpu'):
    """
    Training function for ISLES segmentation model.
    
    Args:
        model: The segmentation model
        trainloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Tuple of (loss, accuracy, dice, f1)
    """
    model.train()
    print('Training')
    epoch_metrics = {'loss': [], 'accuracy': [], 'f1': [], 'dice': []}
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        running_metrics, _, _, _ = train_inner_loop(model, data, criterion, optimizer, device, train=True)

        # memory estimation:
        if i % print_every == 0:
            reserved = torch.cuda.memory_reserved(device)
            active = torch.cuda.memory_allocated(device)
            print(f"[batch {i:>4}] reserved = {mb(reserved):7.1f} MB"
                  f" | active = {mb(active):7.1f} MB")
        
        for key in running_metrics.keys():
            epoch_metrics[key].append(running_metrics[key])
    
    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])

    return epoch_loss, epoch_acc, epoch_dice, epoch_f1

def validate_model(model, testloader, criterion, device='cpu', epoch=None, run_name=None, vis_freq=10):
    """
    Validation function for ISLES segmentation model.
    
    Args:
        model: The segmentation model
        testloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number (for visualization)
        run_name: Unique run identifier (for visualization)
        vis_freq: Frequency of visualization (every N epochs)
    
    Returns:
        Tuple of (loss, accuracy, dice, f1)
    """
    model.eval()
    print('Validation')
    epoch_metrics = {'loss': [], 'accuracy': [], 'f1': [], 'dice': []}
    
    # For visualization
    vis_batch = None
    vis_images = None
    vis_masks = None
    vis_predictions = None
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            running_metrics, images, masks, predictions = train_inner_loop(
                model, data, criterion, None, device, train=False
            )
            
            # Store first batch for visualization
            if i == 0:
                vis_images = images
                vis_masks = masks
                vis_predictions = predictions
            
            for key in running_metrics.keys():
                epoch_metrics[key].append(running_metrics[key])

    # loss and accuracy for the complete epoch
    epoch_loss = np.mean(epoch_metrics['loss'])
    epoch_acc = np.mean(epoch_metrics['accuracy'])
    epoch_dice = np.mean(epoch_metrics['dice'])
    epoch_f1 = np.mean(epoch_metrics['f1'])
    
    # Visualization
    if epoch is not None and run_name is not None and epoch % vis_freq == 0:
        if vis_images is not None:
            visualize_batch(vis_images, vis_masks, vis_predictions, epoch, run_name)
            print(f'Visualization saved for epoch {epoch}')
    
    return epoch_loss, epoch_acc, epoch_dice, epoch_f1

def train_epochs(model, trainloader, valloader, optimizer, criterion, 
                num_epochs, device='cpu', save_path=None, run_name=None, vis_freq=10, patience=20):
    """
    Complete training loop for multiple epochs with early stopping.
    
    Args:
        model: The segmentation model
        trainloader: Training data loader
        valloader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of training epochs
        device: Device to run on
        save_path: Path to save best model
        run_name: Unique run identifier for logging and visualization
        vis_freq: Frequency of visualization (every N epochs)
        patience: Number of epochs to wait for improvement before early stopping
    
    Returns:
        Dictionary containing training history
    """
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"isles_run_{timestamp}"
    
    print(f"Starting training run: {run_name}")
    print(f"Early stopping patience: {patience} epochs")
    
    best_val_dice = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_dice': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_dice': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # Training
        train_loss, train_acc, train_dice, train_f1 = train_model(
            model, trainloader, optimizer, criterion, device
        )
        
        # Validation
        val_loss, val_acc, val_dice, val_f1 = validate_model(
            model, valloader, criterion, device, epoch, run_name, vis_freq
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_dice'].append(train_dice)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_dice'].append(val_dice)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # Check for improvement in validation loss (for early stopping)
        if val_loss < best_val_loss:
            patience_counter = 0
            print(f'Validation loss improved to {val_loss:.4f}')
            save_model = True
        else:
            patience_counter += 1
            print(f'Validation loss did not improve. Patience: {patience_counter}/{patience}')
            save_model = False
        
        # Save model on first epoch, or if validation loss improves
        if save_path and (epoch == 0 or save_model):
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
                'history': history,
                'run_name': run_name
            }, save_path)
            if epoch == 0:
                print('Saving initial model')
            else:
                print(f'Model saved with validation Loss: {val_loss:.4f}')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {patience} epochs without improvement')
            print(f'Best validation loss: {best_val_loss:.4f}')
            print(f'Best validation dice: {best_val_dice:.4f}')
            break
    
    return history
