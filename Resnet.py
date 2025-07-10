# 2025_07_10_BAIL_조현준



# 0. 라이브러리 및 설정
import os, time, random, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt 

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models

from sklearn.model_selection import train_test_split

SEED        = 42
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size        = 32
mean        = (0.4914, 0.4822, 0.4466)
std         = (0.2470, 0.2435, 0.2616)
batch_size  = 128
EPOCHS      = 80

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed()




# 1. 이미지 전처리
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomCrop(resize, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase):
        return self.data_transform[phase](img)
transform = ImageTransform(size, mean, std)


# train transform 함수
def train_transform(x):
    return transform(x, 'train')


if __name__ == "__main__":

    # 2. 전체 val set → val/test로 분할
    val_raw = datasets.CIFAR10(root='./data', train=False, download=True)
    val_images = [(img, label) for img, label in zip(val_raw.data, val_raw.targets)]
    val_data, test_data = train_test_split(
        val_images, test_size=0.5, stratify=[label for _, label in val_images], random_state=SEED
    )



    # 3. 사용자 정의 Dataset (val/test)
    class CIFAR10Subset(Dataset):
        def __init__(self, data, transform=None, phase='val'):
            self.data = data
            self.transform = transform
            self.phase = phase
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            img, label = self.data[idx]
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img, self.phase)
            return img, label



    # 4. DataLoader 구성
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                    transform=train_transform)
    val_dataset   = CIFAR10Subset(val_data,  transform=transform, phase='val')
    test_dataset  = CIFAR10Subset(test_data, transform=transform, phase='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=256,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=256,
                            shuffle=False, num_workers=0, pin_memory=True)



    # 5. 모델 정의
    def build_resnet50_cifar():
        model = models.resnet50(weights=None)
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(model.fc.in_features, 10)
        return model
    model = build_resnet50_cifar().to(device)



    # 6. 손실함수&옵티마이저&스케줄러
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)



    # 7. 학습&평가 함수
    def train_fn(model, loader, optimizer, criterion, device):
        model.train()
        loss_total, correct, total = 0, 0, 0
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward(); optimizer.step()
            loss_total += loss.item()
            correct    += (pred.argmax(1) == y).sum().item()
            total      += y.size(0)
        return loss_total / len(loader), correct / total

    @torch.no_grad()
    def evaluate(model, loader, criterion, device):
        model.eval()
        loss_total, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss_total += loss.item()
            correct    += (pred.argmax(1) == y).sum().item()
            total      += y.size(0)
        return loss_total / len(loader), correct / total

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}




    # 8. 학습 루프
    best_valid_acc = 0.0
    for epoch in range(EPOCHS):
        st = time.time()
        train_loss, train_acc = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # 기록
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), "resnet-best.pt")

        print(f"[{epoch+1:03}] {int(time.time()-st)}s | "
              f"Train {train_loss:.4f}/{train_acc*100:5.2f}% | "
              f"Val {val_loss:.4f}/{val_acc*100:5.2f}% (best {best_valid_acc*100:5.2f}%)")



    # 9. 테스트 성능
    model.load_state_dict(torch.load("resnet-best.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")



    #  학습 그래프 시각화
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.array(history['train_acc'])*100, label='Train Acc')
    plt.plot(np.array(history['val_acc'])*100,   label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
