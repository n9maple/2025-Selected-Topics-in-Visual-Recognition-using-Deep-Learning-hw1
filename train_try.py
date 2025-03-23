import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import ResNet152_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from FocalLoss import FocalLoss
from Dataset import TrainDataset
import numpy as np

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_dir, device):
    best_acc = 0.0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        print(f"\nEpoch {epoch+1}/{num_epochs} - Training...")
        start_time = time.time()
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_progress.set_postfix(loss=running_loss / (train_progress.n + 1), acc=100.0 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1} Completed - Train Acc: {train_acc:.2f}% - Time: {time.time() - start_time:.1f}s")

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", unit="batch")
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc = 100.0 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}% - Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
        torch.save(model.state_dict(), f"{save_dir}/model_epoch{epoch}.pth")
        
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    np.save("train_losses.npy", np.array(train_losses))
    np.save("val_losses.npy", np.array(val_losses))
    print("Training and validation losses saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--criterion", type=str, choices=["cross_entropy", "focal"], default="focal", help="Loss function")
    parser.add_argument("--dropout", type=str, choices=["on", "off"], default="on", help="Add dropout layer before the output layer")
    parser.add_argument("--save_dir", type=str, default="save_model", help="Save model to the directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(500),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.CenterCrop([500, 500]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TrainDataset(data_dir="data/train", transform=train_transforms)
    val_dataset = TrainDataset(data_dir="data/val", transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Loading model...")
    model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    if args.dropout == "on":
        model.fc = nn.Sequential(nn.Dropout(p=0.5, inplace=True), nn.Linear(model.fc.in_features, 100))
    else:
        model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) if args.criterion == "cross_entropy" else FocalLoss(gamma=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, args.num_epochs, args.save_dir, device)