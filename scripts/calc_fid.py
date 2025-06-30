import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

class FIDImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, -1

def get_data_loader(image_dir, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])
    dataset = FIDImageDataset(image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def compute_fid_and_kid(real_dir, gen_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(feature=2048).to(device)

    real_loader = get_data_loader(real_dir, batch_size)
    gen_loader = get_data_loader(gen_dir, batch_size)

    # Update FID/KID with real images
    for inputs, _ in tqdm(real_loader, desc="Processing real images"):
        inputs = inputs.to(device)
        fid.update(inputs, real=True)
        kid.update(inputs, real=True)

    # Update FID/KID with generated images
    for inputs, _ in tqdm(gen_loader, desc="Processing generated images"):
        inputs = inputs.to(device)
        fid.update(inputs, real=False)
        kid.update(inputs, real=False)

    fid_score = fid.compute().item()
    kid_mean, kid_var = kid.compute()
    return fid_score, kid_mean.item(), kid_var.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID and KID between two image directories.")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated images")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    args = parser.parse_args()

    fid, kid_mean, kid_var = compute_fid_and_kid(args.real_dir, args.gen_dir, args.batch_size)
    print(f"FID: {fid:.4f}")
    print(f"KID (mean): {kid_mean:.4f}")
    print(f"KID (variance): {kid_var:.4f}")

# Example usage:
# For diffusion:
# python scripts/calc_fid.py --real_dir notebooks/isles_gen/real --gen_dir notebooks/isles_gen/synth --batch-size 16
# For ResVit:
# python scripts/calc_fid.py --real_dir notebooks/generated_eg/real --gen_dir notebooks/generated_eg/synth --batch-size 16 