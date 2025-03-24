import pandas as pd
import argparse

def calculate_accuracy(prediction_file, groundtruth_file):
    pred_df = pd.read_csv(prediction_file)
    gt_df = pd.read_csv(groundtruth_file)
    merged_df = pred_df.merge(gt_df, on='image_name', suffixes=('_pred', '_gt'))
    correct_predictions = (merged_df['pred_label_pred'] == merged_df['pred_label_gt']).sum()
    
    total_predictions = len(merged_df)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default='val_predictions.csv', help="The path of predicion file (it will be named 'val_predictions.csv' by default)")
    parser.add_argument("--grond_path", type=str, default='val_groundtruth.csv', help="The path of groundtruth file (it will be named 'val_groundtruth.csv' by default)")
    args = parser.parse_args()

    calculate_accuracy(args.pred_path, args.grond_path)
