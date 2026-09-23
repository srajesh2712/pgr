import os
import glob
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
# -------------------------------------------------------------------
# 1. DEFINE A SIMPLE PYTORCH CONVOLUTIONAL FEATURE EXTRACTION MODEL
# -------------------------------------------------------------------
class SatelliteChangeCNN(nn.Module):
    def __init__(self):
        super(SatelliteChangeCNN, self).__init__()
        # Convolutional layer extracts spatial edges, textures, and cloud boundaries
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        features = self.relu(self.conv(x))
        return features

# -------------------------------------------------------------------
# 2. IMAGE PREPROCESSING PIPELINE
# -------------------------------------------------------------------
transform = T.Compose([
    T.Resize((256, 256)),  # Standardize all satellite images to 256x256
    T.ToTensor()           # Convert PIL Image to PyTorch Tensor [3, 256, 256]
])

# -------------------------------------------------------------------
# 3. LOAD ALL SATELLITE IMAGES FROM FOLDER
# -------------------------------------------------------------------
# Replace 'satellite_data' with your actual image directory path
script_dir = Path(__file__).resolve().parent
image_dir = script_dir / "data"

image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

if len(image_paths) < 2:
    raise ValueError(f"Found {len(image_paths)} images in {image_dir}. Need at least 2!")
# Load images and convert into a batch tensor: [N, 3, 256, 256]
image_tensors = [transform(Image.open(p).convert("RGB")) for p in image_paths]
batch_tensor = torch.stack(image_tensors)

# -------------------------------------------------------------------
# 4. EXTRACT SPATIAL FEATURES & COMPUTE CHANGE DETECTION
# -------------------------------------------------------------------
model = SatelliteChangeCNN()
model.eval()  # Set to evaluation mode

with torch.no_grad():
    # Get feature maps for all images: Shape [N, 16, 256, 256]
    feature_maps = model(batch_tensor)
    
    # Compute absolute difference between consecutive image features
    # (Frame t+1 minus Frame t)
    feature_diffs = torch.abs(feature_maps[1:] - feature_maps[:-1])
    
    # Average across feature channels to produce a 2D spatial heatmap: [N-1, 256, 256]
    change_heatmaps = feature_diffs.mean(dim=1)

# -------------------------------------------------------------------
# 5. VISUALIZE RESULTS IN MATPLOTLIB
# -------------------------------------------------------------------
num_comparisons = len(image_paths) - 1
fig, axes = plt.subplots(num_comparisons, 3, figsize=(12, 4 * num_comparisons))

# Handle single vs multiple subplot axes indexing
if num_comparisons == 1:
    axes = [axes]

for i in range(num_comparisons):
    img1 = Image.open(image_paths[i])
    img2 = Image.open(image_paths[i+1])
    heatmap = change_heatmaps[i].numpy()

    # Column 1: Time t
    axes[i][0].imshow(img1)
    axes[i][0].set_title(f"Time {i+1}: {os.path.basename(image_paths[i])}")
    axes[i][0].axis("off")

    # Column 2: Time t+1
    axes[i][1].imshow(img2)
    axes[i][1].set_title(f"Time {i+2}: {os.path.basename(image_paths[i+1])}")
    axes[i][1].axis("off")

    # Column 3: Detected Spatial Change Heatmap
    im = axes[i][2].imshow(heatmap, cmap="jet")
    axes[i][2].set_title("CNN Detected Change (Heatmap)")
    axes[i][2].axis("off")
    plt.colorbar(im, ax=axes[i][2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()