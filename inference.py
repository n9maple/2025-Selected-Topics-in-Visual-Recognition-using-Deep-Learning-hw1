import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from Dataset import TestDataset

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer
test_transforms = transforms.Compose([
    transforms.CenterCrop([500, 500]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# test data directory
test_dir = "data/test"

# build DataLoader
test_dataset = TestDataset(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# loading model
model = models.resnet152(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Dropout(p = 0.5, inplace=True), 
    nn.Linear(model.fc.in_features, 100))
model.load_state_dict(torch.load("best_model_1.pth", map_location=device))
model.to(device)
model.eval()

predictions = []
with torch.no_grad():
    for images, image_names in tqdm(test_loader, desc="Predicting Test Data"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for img_name, pred_label in zip(image_names, predicted.cpu().numpy()):
            predictions.append((img_name, pred_label))

# output csv
df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
df.to_csv("prediction.csv", index=False)
print("Predictions saved to prediction.csv!")
