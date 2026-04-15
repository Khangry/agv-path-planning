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


# 1. Định nghĩa Dataset đọc từ file COCO
class DragonFruitCOCODataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile) # Đọc file JSON
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Đọc ảnh gốc
        path = coco.loadImgs(img_id)[0]['file_name']
        # Lưu ý: Label Studio có thể để path là "upload/1/abc.jpg", ta cần lấy tên file cuối cùng
        file_name = os.path.basename(path) 
        image = Image.open(os.path.join(self.root, file_name)).convert('RGB')

        # Tạo mask trống (toàn màu đen)
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        
        # Vẽ các vùng đã gán nhãn (Polygon) vào mask
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))

        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            # Mask chỉ resize, không normalize
            mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

    def __len__(self):
        return len(self.ids)

# 2. Cấu hình Hyperparameters
device = torch.device("cuda")
BATCH_SIZE = 8  # Bạn có thể thử tăng lên 12 hoặc 16 nếu VRAM còn dư
EPOCHS = 30
LEARNING_RATE = 1e-4


data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load dữ liệu (Đường dẫn đến thư mục ảnh và file JSON)
dataset = DragonFruitCOCODataset(
    root="dataset/images", 
    annFile="dataset/result.json", 
    transform=data_transforms
)

# Chia dữ liệu: 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

# 4. Khởi tạo Mô hình DeepLabV3
model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1)) # 2 lớp: Nền và Đường đi
model.to(device)

# 5. Training (Rút gọn)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print(f"Đang huấn luyện trên: {torch.cuda.get_device_name(0)}")
print("Bắt đầu huấn luyện với dữ liệu COCO...")

train_losses = []


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# 6. Lưu mô hình
torch.save(model.state_dict(), "dragon_fruit_path_coco.pth")
print("Đã lưu mô hình thành công!")

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
plt.title('Biểu đồ Loss theo Epoch - Ruộng Thanh Long')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_chart.png') # Lưu biểu đồ thành ảnh
plt.show()