# mnist_inpainting_final.py
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np

# ---------------------- PARAMETERS ----------------------
TRAIN_MASK_MODE = "mixed"
TRAIN_SQUARE_SIZE = 7
TRAIN_LINE_THICKNESS = 7
BATCH_SIZE = 64
NUM_EPOCHS = 1000
LR = 5e-5
LAMBDA_RECON = 1000
MODEL_PATH = "generator_trained.pth"
CONTROL_MODEL_FILE= 1 # 1: use existing model, 0: delete and train again
# ---------------------- CONTROL ----------------------
if CONTROL_MODEL_FILE == 0 and os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print(f"🗑️ Existing model deleted: {MODEL_PATH}")

# -----------------------------------------------------

random.seed(42)
torch.manual_seed(42)

# ---------- DATA ----------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------- MASK FUNCTİON ----------
def add_mask(image):
    c, h, w = image.shape
    img = image.clone()
    mask = torch.ones_like(img)

    mode = TRAIN_MASK_MODE
    if mode == "random_square":
        s = TRAIN_SQUARE_SIZE
        top = random.randint(0, h - s)
        left = random.randint(0, w - s)
        img[:, top:top+s, left:left+s] = 0.0
        mask[:, top:top+s, left:left+s] = 0.0
    elif mode == "horizontal":
        th = TRAIN_LINE_THICKNESS
        y = (h // 2) - (th // 2)
        img[:, y:y+th, :] = 0.0
        mask[:, y:y+th, :] = 0.0
    elif mode == "vertical":
        tw = TRAIN_LINE_THICKNESS
        x = (w // 2) - (tw // 2)
        img[:, :, x:x+tw] = 0.0
        mask[:, :, x:x+tw] = 0.0
    elif mode == "mixed":
        choice = random.choice(["random_square", "horizontal", "vertical"])
        if choice == "random_square":
            s = TRAIN_SQUARE_SIZE
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
            img[:, top:top+s, left:left+s] = 0.0
            mask[:, top:top+s, left:left+s] = 0.0
        elif choice == "horizontal":
            th = TRAIN_LINE_THICKNESS
            y = (h // 2) - (th // 2)
            img[:, y:y+th, :] = 0.0
            mask[:, y:y+th, :] = 0.0
        else:
            tw = TRAIN_LINE_THICKNESS
            x = (w // 2) - (tw // 2)
            img[:, :, x:x+tw] = 0.0
            mask[:, :, x:x+tw] = 0.0
    return img, mask


# ---------- MODELS ----------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128+64, 64, 4, stride=2, padding=1)
        self.dec3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        d1 = self.relu(self.dec1(e3))
        d1_cat = torch.cat([d1, e1], dim=1)
        d2 = self.relu(self.dec2(d1_cat))
        out = self.tanh(self.dec3(d2))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# ---------- TRAİN AND UPLOAD ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNetGenerator().to(device)
D = Discriminator().to(device)

criterion_GAN = nn.BCELoss()
criterion_recon = nn.L1Loss()
optimizer_G = optim.Adam(G.parameters(), lr=LR)
optimizer_D = optim.Adam(D.parameters(), lr=LR)

if os.path.exists(MODEL_PATH):
    G.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    G.eval()
    print("✅ The trained model has been loaded and will not be trained again.")
else:
    print("🚀 The model is being trained...")
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(dataloader)
        for images, _ in loop:
            images = images.to(device)
            masked_images, masks = zip(*[add_mask(img.clone()) for img in images])
            masked_images = torch.stack(masked_images).to(device)
            masks = torch.stack(masks).to(device)
            
            real_labels = torch.ones(images.size(0),1).to(device)
            fake_labels = torch.zeros(images.size(0),1).to(device)
            
            D.zero_grad()
            outputs_real = D(images)
            outputs_fake = D(G(masked_images).detach())
            loss_D = criterion_GAN(outputs_real, real_labels) + criterion_GAN(outputs_fake, fake_labels)
            loss_D.backward()
            optimizer_D.step()
            
            G.zero_grad()
            fake_images = G(masked_images)
            loss_recon_masked = criterion_recon(fake_images * (1-masks), images * (1-masks))
            loss_G = criterion_GAN(D(fake_images), real_labels) + LAMBDA_RECON * loss_recon_masked
            loss_G.backward()
            optimizer_G.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())
    torch.save(G.state_dict(), MODEL_PATH)
    print("✅ Model training completed and saved")

# ---------- TEST WİTH YOUR DATA----------
def test_custom_image(image_path):
    img = Image.open(image_path).convert('L').resize((28,28))
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        completed = G(tensor).cpu().squeeze()
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Girdi (Your masked number)"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(completed, cmap='gray'); plt.title("The completed version of the model"); plt.axis("off")
    plt.show()

# ---------- OPERATING ----------
# Try with your handwriting:
test_custom_image(r"D:\kods\testdata.png") #Load your own data



