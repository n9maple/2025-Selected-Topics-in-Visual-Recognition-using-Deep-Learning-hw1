import argparse
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from Dataset import TestDataset
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="data/test",
        help="The directory where the test data are",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./save_model/model_epoch19.pth",
        help="The path of the model weight",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Save prediction result to the directory",
    )
    parser.add_argument(
        "--nodropout",
        action="store_true",
        help="Delete the dropout layer if on (need to be the same as traing model)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Choose the device to train"
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="The output file will be named 'val_predictions.csv' if on",
    )
    args = parser.parse_args()
    # device
    device = args.device

    # Transformer
    test_transforms = transforms.Compose(
        [
            transforms.CenterCrop([500, 500]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # test data directory
    test_dir = args.test_data_dir

    # build DataLoader
    test_dataset = TestDataset(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # loading model
    model = models.resnet152(weights=None)
    if not args.nodropout:
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True), nn.Linear(model.fc.in_features, 100)
        )
    else:
        model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc="Predicting Data"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for img_name, pred_label in zip(image_names, predicted.cpu().numpy()):
                predictions.append((img_name, pred_label))

    # output csv
    df = pd.DataFrame(predictions, columns=["image_name", "pred_label"])
    if args.validation:
        df.to_csv(os.path.join(args.save_dir, "val_prediction.csv"), index=False)
        print("Predictions saved to val_prediction.csv!")
    else:
        df.to_csv(os.path.join(args.save_dir, "prediction.csv"), index=False)
        print("Predictions saved to prediction.csv!")
