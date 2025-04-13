import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset:
class DeblurDataset(Dataset):
    def __init__(self, blur_images, sharp_images, transform=None):
        self.blur_images = blur_images
        self.sharp_images = sharp_images
        self.transform = transform

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):
        blurred_path = self.blur_images[idx]
        sharp_path = self.sharp_images[idx]

        blurred = cv2.imread(blurred_path, cv2.IMREAD_COLOR)
        sharp = cv2.imread(sharp_path, cv2.IMREAD_COLOR)

        if blurred is None:
            print(f"Warning: Blurred image wasn’t read: {blurred_path}")
            blurred = np.zeros((512, 512, 3), dtype=np.uint8)
        if sharp is None:
            print(f"Warning: Sharp image wasn’t read: {sharp_path}")
            sharp = np.zeros((512, 512, 3), dtype=np.uint8)

        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        # Check size and crop to 512x512
        h, w, _ = blurred.shape
        if h < 512 or w < 512:
            print(f"Warning: Image size too small, padding applied. Blurred: {blurred_path}, Sharp: {sharp_path}")
            blurred = cv2.copyMakeBorder(blurred, 0, max(0, 512 - h), 0, max(0, 512 - w), cv2.BORDER_CONSTANT, value=[0, 0, 0])
            sharp = cv2.copyMakeBorder(sharp, 0, max(0, 512 - h), 0, max(0, 512 - w), cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Cropping
        h, w, _ = blurred.shape
        top = (h - 512) // 2
        left = (w - 512) // 2
        blurred = blurred[top:top+512, left:left+512]
        sharp = sharp[top:top+512, left:left+512]

        # Normalization and tensor conversion
        if self.transform:
            blurred = self.transform(blurred)
            sharp = self.transform(sharp)

        return blurred, sharp

# 2. Architecture
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            ResBlock(64),
            ResBlock(64)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.middle(features)
        reconstructed = self.decoder(features)
        return reconstructed

# 3. Training
def train_model(model, dataloader, optimizer, criterion_recon, epochs=20):
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(epochs):
        epoch_loss = 0
        for blurred, sharp in dataloader:
            blurred = blurred.float().to(device)
            sharp = sharp.float().to(device)

            optimizer.zero_grad()

            output = model(blurred)
            loss_recon = criterion_recon(output, sharp)
            loss = loss_recon
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"D:\Hoschule Muenchen\Digital Bilderverarbeitung\Models\Epochs\deblur_model_epoch{epoch + 1}.pth")
        print(f"Model {epoch + 1} saved")

# 4. Visualization
def visualize_results(model, dataloader):
    model.eval()
    with torch.no_grad():
        for blurred, sharp in dataloader:
            blurred = blurred.float().to(device)
            sharp = sharp.float().to(device)

            output = model(blurred)

            blurred = blurred.cpu().numpy().transpose(0, 2, 3, 1)
            sharp = sharp.cpu().numpy().transpose(0, 2, 3, 1)
            output = output.cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(len(blurred)):
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.title("Blurred")
                plt.imshow((blurred[i] * 0.5 + 0.5).clip(0, 1))

                plt.subplot(1, 3, 2)
                plt.title("Sharp")
                plt.imshow((sharp[i] * 0.5 + 0.5).clip(0, 1))

                plt.subplot(1, 3, 3)
                plt.title("Output")
                plt.imshow((output[i] * 0.5 + 0.5).clip(0, 1))

                plt.show()
            break

# 5. Main Code
if __name__ == "__main__":
    # Directories list
    directories = [
        # r"D:\Hoschule Muenchen\Digital Bilderverarbeitung\deblur18GB\DBlur\Gopro\test",
        r"D:\Hoschule Muenchen\Digital Bilderverarbeitung\deblur18GB\DBlur\Helen\validation"
    ]

    blur_images = []
    sharp_images = []

    # Collect all images from directories
    for base_path in directories:
        blur_folder = os.path.join(base_path, "blur")
        sharp_folder = os.path.join(base_path, "sharp")

        for file in os.listdir(blur_folder):
            blur_path = os.path.join(blur_folder, file)
            sharp_path = os.path.join(sharp_folder, file)
            if os.path.exists(blur_path) and os.path.exists(sharp_path) and file.lower().endswith((".jpg", ".png", ".jpeg")):
                blur_images.append(blur_path)
                sharp_images.append(sharp_path)
            else:
                if not os.path.exists(blur_path):
                    print(f"Warning: File not found in blur: {blur_path}")
                if not os.path.exists(sharp_path):
                    print(f"Warning: File not found in sharp: {sharp_path}")

    print(f"Number of image pairs: {len(blur_images)}")

    # Check if n_blur = n_sharp
    assert len(blur_images) == len(sharp_images), f"Number of files in blur ({len(blur_images)}) and sharp ({len(sharp_images)}) folders do not match!"

    # Initialize Dataset
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalization
    dataset = DeblurDataset(blur_images, sharp_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model
    model = DeblurNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_recon = nn.MSELoss()

    # Training
    train_model(model, dataloader, optimizer, criterion_recon, epochs=40)

    # Visualize results
    visualize_results(model, dataloader)
