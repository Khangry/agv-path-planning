import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# 1. Định nghĩa Dataset đọc từ file COCO
class DragonFruitCOCODataset(Dataset):
    def __init__(self, root, annFile, transform=None, augment=False):
        self.root = root
        self.coco = COCO(annFile) # Đọc file JSON
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.augment = augment

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Đọc ảnh gốc
        path = coco.loadImgs(img_id)[0]['file_name']
        file_name = os.path.basename(path) 
        image = Image.open(os.path.join(self.root, file_name)).convert('RGB')

        # Tạo mask trống (toàn màu đen)
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        
        # Vẽ các vùng đã gán nhãn (Polygon) vào mask
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))

        mask = Image.fromarray(mask)

        # Data Augmentation (Chỉ áp dụng cho tập train)
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Color Jitter (chỉ thay đổi màu sắc ảnh, không ảnh hưởng mask)
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = jitter(image)

        if self.transform:
            image = self.transform(image)
            # Mask chỉ resize, không normalize
            mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

    def __len__(self):
        return len(self.ids)

# 2. Cấu hình Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Bạn có thể thử tăng lên 12 hoặc 16 nếu VRAM còn dư
EPOCHS = 30
LEARNING_RATE = 1e-4

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load dữ liệu (Đường dẫn đến thư mục ảnh và file JSON)
full_dataset = DragonFruitCOCODataset(
    root="dataset/images", 
    annFile="dataset/result.json", 
    transform=data_transforms
)

# Chia dữ liệu: 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Cố định random seed để chia dữ liệu luôn giống nhau
generator = torch.Generator().manual_seed(42)
train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)

# Tạo tập train (có Augmentation) và tập val (không Augmentation)
train_ds = torch.utils.data.Subset(
    DragonFruitCOCODataset(root="dataset/images", annFile="dataset/result.json", transform=data_transforms, augment=True),
    train_indices
)
val_ds = torch.utils.data.Subset(
    DragonFruitCOCODataset(root="dataset/images", annFile="dataset/result.json", transform=data_transforms, augment=False),
    val_indices
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# 4. Khởi tạo Mô hình DeepLabV3
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1)) # 2 lớp: Nền và Đường đi
model.to(device)

# 5. Training
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Thêm Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Thêm GradScaler cho AMP (Automatic Mixed Precision)
scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

print(f"Đang huấn luyện trên: {device}")
print("Bắt đầu huấn luyện với dữ liệu COCO...")

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # --- TRAINING LOOP ---
    model.train()
    total_train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Sử dụng AMP để giảm VRAM và tăng tốc
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(imgs)['out']
            loss = criterion(outputs, masks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # --- VALIDATION LOOP ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                outputs = model(imgs)['out']
                loss = criterion(outputs, masks)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Cập nhật Scheduler
    scheduler.step(avg_val_loss)
    
    # Lưu mô hình tốt nhất
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "dragon_fruit_path_coco_best.pth")
        print(f"  --> Đã lưu mô hình tốt nhất mới (Val Loss: {best_val_loss:.4f})")

# 6. Lưu mô hình epoch cuối
torch.save(model.state_dict(), "dragon_fruit_path_coco_last.pth")
print("Hoàn tất! Đã lưu mô hình epoch cuối thành công!")

# Vẽ biểu đồ Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
plt.title('Biểu đồ Loss theo Epoch - Ruộng Thanh Long')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_chart.png') # Lưu biểu đồ thành ảnh
print("Đã lưu biểu đồ loss_chart.png")
