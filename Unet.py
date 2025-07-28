# Unet_with_metrics_gpu_optimized.py

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pycocotools.coco import COCO
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# GPU 최적화 설정
torch.backends.cudnn.benchmark = True  # 입력 크기가 일정할 때 성능 향상

def make_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

# 1. COCO 세그멘테이션용 Dataset 클래스 정의
class CocoSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, target_transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, info['file_name'])
        image = Image.open(path).convert('RGB')

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
        mask = Image.fromarray(mask * 255)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

# 2. 전처리 정의
IMG_SIZE = (256, 256)
image_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mask_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# 3. U-Net 모델 정의 (출력은 로짓)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool  = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottom= DoubleConv(512, 1024)
        self.up4   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upc4  = DoubleConv(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upc3  = DoubleConv(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upc2  = DoubleConv(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upc1  = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        b  = self.bottom(self.pool(d4))
        u4 = torch.cat([self.up4(b), d4], dim=1); u4 = self.upc4(u4)
        u3 = torch.cat([self.up3(u4), d3], dim=1); u3 = self.upc3(u3)
        u2 = torch.cat([self.up2(u3), d2], dim=1); u2 = self.upc2(u2)
        u1 = torch.cat([self.up1(u2), d1], dim=1); u1 = self.upc1(u1)
        return self.final(u1)  # 로짓 반환

# 4. 학습·검증·테스트 루프

def train_and_evaluate(train_loader, val_loader, test_loader, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 안전한 AMP용 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    print("▶▶▶ START TRAINING ◀◀◀")
    for epoch in range(1, num_epochs+1):
        # Training
        model.train()
        running_train = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train += loss.item()
        avg_train = running_train / len(train_loader)

        # Validation
        model.eval()
        running_val, correct_val, total_val = 0.0, 0, 0
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.no_grad(), autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            running_val += loss.item()
            preds = torch.sigmoid(logits)
            correct_val += (preds > 0.5).float().eq(masks).sum().item()
            total_val += masks.numel()
        avg_val = running_val / len(val_loader)
        val_acc = correct_val / total_val

        # Test
        running_test, correct_test, total_test = 0.0, 0, 0
        for imgs, masks in tqdm(test_loader, desc=f"Epoch {epoch}/{num_epochs} [Test]", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.no_grad(), autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            running_test += loss.item()
            preds = torch.sigmoid(logits)
            correct_test += (preds > 0.5).float().eq(masks).sum().item()
            total_test += masks.numel()
        avg_test = running_test / len(test_loader)
        test_acc = correct_test / total_test

        # Epoch summary
        print(f"Epoch {epoch}/{num_epochs}  "
              f"Train Loss: {avg_train:.4f}  "
              f"Val Loss:   {avg_val:.4f}  "
              f"Val Acc:    {val_acc:.4f}  "
              f"Test Loss:  {avg_test:.4f}  "
              f"Test Acc:   {test_acc:.4f}")

# 5. 실행 및 결과 호출
if __name__ == '__main__':
    # Dataset 준비
    train_ds = CocoSegmentationDataset(
        img_dir='data/images/train2017',
        ann_file='data/annotations/instances_train2017.json',
        transform=image_transform,
        target_transform=mask_transform
    )
    full_val = CocoSegmentationDataset(
        img_dir='data/images/val2017',
        ann_file='data/annotations/instances_val2017.json',
        transform=image_transform,
        target_transform=mask_transform
    )
    vsize = len(full_val) // 2
    tsize = len(full_val) - vsize
    val_ds, test_ds = random_split(full_val, [vsize, tsize], generator=torch.Generator().manual_seed(42))

    # DataLoader 생성 (최적화)
    bsz = 8  # 배치 크기 증가
    train_loader = make_dataloader(train_ds, batch_size=bsz, shuffle=True)
    val_loader   = make_dataloader(val_ds,   batch_size=bsz, shuffle=False)
    test_loader  = make_dataloader(test_ds,  batch_size=bsz, shuffle=False)

    # Training & Evaluation 실행
    train_and_evaluate(train_loader, val_loader, test_loader, num_epochs=5)
