# mnist_inpainting_fixed.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------- AYARLAR ----------------------
# TRAIN_MASK_MODE options: "horizontal", "vertical", "random_square", "mixed"
TRAIN_MASK_MODE = "vertical"
TRAIN_SQUARE_SIZE = 7       # sadece random_square için
TRAIN_LINE_THICKNESS = 7    # yatay/dikey çizgi kalınlığı
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR = 1e-4
LAMBDA_RECON = 500
# -----------------------------------------------------

# reproducibility (opsiyonel)
random.seed(42)
torch.manual_seed(42)

# ---------- 1. Veri Hazırlama ----------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---------- 1b. Maske fonksiyonu (eğitimde kullanılacak) ----------
def add_mask(image):
    """
    image: tensor (1,28,28)
    döndürür: (masked_image, mask) ; mask: 1 for preserved, 0 for masked region
    """
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
        # center line
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

    else:
        # fallback: small random square
        s = TRAIN_SQUARE_SIZE
        top = random.randint(0, h - s)
        left = random.randint(0, w - s)
        img[:, top:top+s, left:left+s] = 0.0
        mask[:, top:top+s, left:left+s] = 0.0

    return img, mask

# ---------- 2. Model Tanımları ----------

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 4, stride=2, padding=1)   # 28->14
        self.enc2 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 14->7
        self.enc3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)# 7->7
        self.relu = nn.ReLU()
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 7->14
        self.dec2 = nn.ConvTranspose2d(128+64, 64, 4, stride=2, padding=1) # 14->28
        self.dec3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)               # 28->28
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Encoder
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        
        # Decoder
        d1 = self.relu(self.dec1(e3))
        # Skip connection: concat e1 ile d1
        d1_cat = torch.cat([d1, e1], dim=1)
        d2 = self.relu(self.dec2(d1_cat))
        out = self.tanh(self.dec3(d2))
        return out

# CNN Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),  # 28->14
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 14->7
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


# ---------- 3. Model ve Optimizasyon ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = UNetGenerator().to(device)
D = Discriminator().to(device)

# ----- GPU kontrol -----
print("Generator device:", next(G.parameters()).device)
print("Discriminator device:", next(D.parameters()).device)

criterion_GAN = nn.BCELoss()
criterion_recon = nn.L1Loss()
optimizer_G = optim.Adam(G.parameters(), lr=LR )
optimizer_D = optim.Adam(D.parameters(), lr=LR )
lambda_recon = LAMBDA_RECON 
num_epochs = NUM_EPOCHS

# ---------- 4. Eğitim Döngüsü ----------
for epoch in range(num_epochs):
    loop = tqdm(dataloader)
    for images, _ in loop:
        images = images.to(device)
        masked_images, masks = zip(*[add_mask(img.clone()) for img in images])
        masked_images = torch.stack(masked_images).to(device)
        masks = torch.stack(masks).to(device)
        
        real_labels = torch.ones(images.size(0),1).to(device)
        fake_labels = torch.zeros(images.size(0),1).to(device)
        
        # ----- Discriminator -----
        D.zero_grad()
        outputs_real = D(images)
        outputs_fake = D(G(masked_images).detach())
        loss_D = criterion_GAN(outputs_real, real_labels) + criterion_GAN(outputs_fake, fake_labels)
        loss_D.backward()
        optimizer_D.step()
        
        # ----- Generator -----
        G.zero_grad()
        fake_images = G(masked_images)
        loss_recon_masked = criterion_recon(fake_images * (1-masks), images * (1-masks))
        loss_G = criterion_GAN(D(fake_images), real_labels) + lambda_recon * loss_recon_masked
        loss_G.backward()
        optimizer_G.step()
        
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(D_loss=loss_D.item(), G_loss=loss_G.item())

# ---------- 5. Test & Görselleştirme ----------
G.eval()
# 1) 0-9 için örnekler: (kullanılan TRAIN_MASK_MODE ile maskelenmiş)
plt.figure(figsize=(12, 14))
for digit in range(10):
    indices = [i for i, (img, label) in enumerate(train_dataset) if label == digit]
    idx = random.choice(indices)
    sample, _ = train_dataset[idx]
    masked_sample, mask = add_mask(sample.clone())
    with torch.no_grad():
        completed = G(masked_sample.unsqueeze(0).to(device)).cpu().squeeze()

    plt.subplot(10, 3, digit*3 + 1)
    plt.imshow(sample.squeeze(), cmap='gray'); plt.title(f"Original ({digit})"); plt.axis("off")
    plt.subplot(10, 3, digit*3 + 2)
    plt.imshow(masked_sample.squeeze(), cmap='gray'); plt.title("Masked"); plt.axis("off")
    plt.subplot(10, 3, digit*3 + 3)
    plt.imshow(completed.squeeze(), cmap='gray'); plt.title("Completed"); plt.axis("off")
plt.tight_layout()
plt.show()

# 2) Yatay ve dikey karşılaştırma (kontrollü maskeler)
sample, _ = train_dataset[0]  # sabit örnek
def add_custom_mask(image, top, left, mask_h, mask_w):
    img = image.clone()
    img[:, top:top+mask_h, left:left+mask_w] = 0.0
    mask = torch.ones_like(img)
    mask[:, top:top+mask_h, left:left+mask_w] = 0.0
    return img, mask

# Horizontal (1x28)
mask_pos_yatay = (14, 0)
mask_h_y, mask_w_y = 1, 28
masked_y, mask_y = add_custom_mask(sample, *mask_pos_yatay, mask_h_y, mask_w_y)
with torch.no_grad():
    completed_y = G(masked_y.unsqueeze(0).to(device)).cpu().squeeze()

# Vertical (28x1)
mask_pos_dikey = (0, 14)
mask_h_v, mask_w_v = 28, 1
masked_v, mask_v = add_custom_mask(sample, *mask_pos_dikey, mask_h_v, mask_w_v)
with torch.no_grad():
    completed_v = G(masked_v.unsqueeze(0).to(device)).cpu().squeeze()

plt.figure(figsize=(9,6))
plt.subplot(2,3,1); plt.imshow(sample.squeeze(), cmap='gray'); plt.title("Original"); plt.axis("off")
plt.subplot(2,3,2); plt.imshow(masked_y.squeeze(), cmap='gray'); plt.title("Masked (Horizontal)"); plt.axis("off")
plt.subplot(2,3,3); plt.imshow(completed_y.squeeze(), cmap='gray'); plt.title("Completed (Horizontal)"); plt.axis("off")
plt.subplot(2,3,4); plt.imshow(sample.squeeze(), cmap='gray'); plt.title("Original"); plt.axis("off")
plt.subplot(2,3,5); plt.imshow(masked_v.squeeze(), cmap='gray'); plt.title("Masked (Vertical)"); plt.axis("off")
plt.subplot(2,3,6); plt.imshow(completed_v.squeeze(), cmap='gray'); plt.title("Completed (Vertical)"); plt.axis("off")
plt.tight_layout()
plt.show()
