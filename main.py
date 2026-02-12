import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn

def load_and_normalize_image(directory_path, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    image_paths = []
    for filename in os.listdir(directory_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_paths.append(os.path.join(directory_path, filename))

    print(f"Found {len(image_paths)} images in {directory_path}")

    return normalize_image(image_paths)


def normalize_image(image_path, size=256):

    if isinstance(image_path, str):
        image_path = [image_path]

    images = []

    for path in image_path:
        image = Image.open(path).convert('RGB').resize((size, size))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        images.append(img_array)

    return np.stack(images)


def conv_block(in_channels, out_channels, dropout=0.1):
    """Convolutional block (3x3 Filter) with batch normalization and ReLU activation."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout) #added dropout, ie turn off neurons
    )

def down_block(in_channels, out_channels, dropout=0.1):
    """Imagine taking a photo and making it smaller and smaller (like zooming out):
Big photo (lots of details) → Medium photo → Tiny photo
down_block squishes the image to be smaller
Why? To find the BIG patterns (like "this is a face" instead of "this is a single hair")
Goes down the left side of the V"""
    return nn.Sequential(
        conv_block(in_channels, out_channels, dropout),
        conv_block(out_channels, out_channels, dropout),
        nn.MaxPool2d(2)
    )

def up_block(in_channels, out_channels, dropout=0.1):
    """Imagine taking that tiny photo and blowing it back up:
Tiny photo → Medium photo → Big photo
up_block makes the image bigger again
Why? To rebuild the image back to full size with all the patterns we learned
Goes up the right side of the V"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        conv_block(in_channels, out_channels, dropout),
        conv_block(out_channels, out_channels, dropout)
    )

""" RGB image: in_channels=3 (Red, Green, Blue)
Grayscale image: in_channels=1 (just brightness)
RGB + Depth: in_channels=4 (Red, Green, Blue, Depth) """


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout=0.1):
        super(UNet, self).__init__()

        # Encoder (going DOWN the V)
        self.enc1 = down_block(in_channels, 64, dropout)  # 3 -> 64 channels
        self.enc2 = down_block(64, 128, dropout)  # 64 -> 128
        self.enc3 = down_block(128, 256, dropout)  # 128 -> 256

        # Bottleneck (bottom of the V)
        self.bottleneck = conv_block(256, 512)

        # Decoder (going UP the V)
        self.up1 = up_block(512, 256)  # 512 -> 256
        self.up2 = up_block(512, 128)  # 512 -> 128 (512 because of skip!)
        self.up3 = up_block(256, 64)  # 256 -> 64

        # Final output layer
        self.final = nn.Conv2d(128, out_channels, kernel_size=1)  # 128 -> 3 (RGB)


    def forward(self, x):
        # Encoder - save features for skip connections
        e1 = self.enc1(x)  # Size: H/2 x W/2
        e2 = self.enc2(e1)  # Size: H/4 x W/4
        e3 = self.enc3(e2)  # Size: H/8 x W/8

        # Bottleneck
        b = self.bottleneck(e3)  # Size: H/8 x W/8

        # Decoder - use skip connections
        d1 = self.up1(b)
        d1 = torch.nn.functional.interpolate(d1, size=e3.shape[2:])
        d1 = torch.cat([d1, e3], dim=1)

        d2 = self.up2(d1)
        d2 = torch.nn.functional.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.up3(d2)
        d3 = torch.nn.functional.interpolate(d3, size=e1.shape[2:])
        d3 = torch.cat([d3, e1], dim=1)

        # Final output
        out = self.final(d3)
        return out



model = UNet(in_channels=3, out_channels=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet(in_channels=3, out_channels=3, dropout=0.1)
model = model.to(device) # IF found clashes write model = Unet(...).to(device)

z = torch.randn(1, 3, 256, 256).to(device)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

criterion = nn.MSELoss() #Amplifies the differences between the predicted and actual output
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Adam optimizer - (Adapt based on gradients)

target_img = load_and_normalize_image('/Users/knightnm/PythonProject/AIE/images')
target_tensor = torch.from_numpy(target_img).to(device)
print(target_img.shape)


num_iterations = 3000 #Deep image prior iterations

for i in range(num_iterations):
    # Forward pass: noise -> network -> output
    output = model(z)

    # Calculate loss
    loss = criterion(output, target_tensor)

    # Backward pass: calculate gradients
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()

    # Print progress
    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")
