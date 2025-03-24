import os
import shutil
import pandas as pd
from tqdm import tqdm

# 原始 val 資料夾
val_dir = "data/val"
# 新的 val_all 資料夾 (所有圖片將放在這裡)
val_all_dir = "data/val_all"

# 確保新的資料夾存在
os.makedirs(val_all_dir, exist_ok=True)

# 存儲 image_name 和對應的標籤
image_list = []

# 遍歷 0~99 類別資料夾
for class_id in tqdm(range(100), desc="Copying val images"):
    class_path = os.path.join(val_dir, str(class_id))
    
    # 遍歷該類別的所有圖片
    for filename in os.listdir(class_path):
        old_path = os.path.join(class_path, filename)
        new_filename = f"{class_id}_{filename}"  # 確保不同類別的圖片不重名
        new_path = os.path.join(val_all_dir, new_filename)

        shutil.copy2(old_path, new_path)  # 複製圖片
        
        # 取得檔名 (去除副檔名)，並記錄標籤
        image_name = os.path.splitext(new_filename)[0]
        image_list.append((image_name, class_id))

# 轉為 DataFrame 並存成 CSV
df = pd.DataFrame(image_list, columns=["image_name", "pred_label"])
df.to_csv("val_groundtruth.csv", index=False)
print("Validation dataset copied, and ground truth CSV saved as val_groundtruth.csv!")
