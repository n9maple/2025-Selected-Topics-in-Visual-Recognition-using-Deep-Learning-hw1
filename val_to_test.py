import os
import shutil
import pandas as pd
from tqdm import tqdm

val_dir = "data/val"

val_all_dir = "data/val_all"


os.makedirs(val_all_dir, exist_ok=True)


image_list = []


for class_id in tqdm(range(100), desc="Copying val images"):
    class_path = os.path.join(val_dir, str(class_id))

    for filename in os.listdir(class_path):
        old_path = os.path.join(class_path, filename)
        new_filename = f"{class_id}_{filename}"
        new_path = os.path.join(val_all_dir, new_filename)

        shutil.copy2(old_path, new_path)

        image_name = os.path.splitext(new_filename)[0]
        image_list.append((image_name, class_id))


df = pd.DataFrame(image_list, columns=["image_name", "pred_label"])
df.to_csv("val_groundtruth.csv", index=False)
print("Validation dataset copied, and ground truth CSV saved as val_groundtruth.csv!")
