import datetime
import shutil
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import KFold

ksplit , ds_yamls = None, []


def split():
    # Proceed to retrieve all label files for your dataset.
    dataset_path = Path('F:/BrowserDownloads/datasets/CTC-CAF')  # replace with 'path/to/dataset' for your custom data
    labels = sorted(dataset_path.rglob("labels2/*.txt"))  # all data in 'labels'

    # get the classes
    yaml_file = 'F:/BrowserDownloads/datasets/CTC-CAF/data.yaml'
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']
    cls_idx = sorted(classes.keys())

    # Initialize an empty pandas DataFrame
    indx = [l.stem for l in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    # Count the instances of each class-label present in the annotation files.
    for label in labels:
        lbl_counter = Counter()

        with open(label, 'r') as lf:
            lines = lf.readlines()

        for l in lines:
            # classes for YOLO label uses integer at first position of each line
            if l != '\n':
                lbl_counter[int(l.split(' ')[0])] += 1
            # lbl_counter[0] = 2, lbl_counter[1] = 10
            # {'0':2, '1':10} ??

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

    # K-Fold
    global ksplit
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))

    folds = [f'split_{n}' for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)

    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'

    # calculate the distribution of class labels for each fold
    # as a ratio of the classes present in val to those present in train.
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1E-7)
        fold_lbl_distrb.loc[f'split_{n}'] = ratio

    save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val')
    save_path.mkdir(parents=True, exist_ok=True)

    images = sorted((dataset_path / 'images').rglob("*.jpg"))  # change file extension as needed
    global ds_yamls

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'path': save_path.as_posix() + '/split_k',
                'train': 'train',
                'val': 'val',
                'names': classes
            }, ds_y)

    # copy the images and labels into the respective directory ('train' or 'val') for each split.
    for image, label in zip(images, labels):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / 'images'
            lbl_to_path = save_path / split / k_split / 'labels'

            # Copy image and label files to new directory
            # Might throw a SamefileError if file already exists
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")


def main():
    model = YOLO('yolov8n.pt', task='detect')
    ksplit = 5
    #global ksplit
    #global ds_yamls
    ds_yamls = ['F:/BrowserDownloads/datasets/CTC-CAF/2023-08-29_5-Fold_Cross-val/split_1/split_1_dataset.yaml',
                'F:/BrowserDownloads/datasets/CTC-CAF/2023-08-29_5-Fold_Cross-val/split_2/split_2_dataset.yaml',
                'F:/BrowserDownloads/datasets/CTC-CAF/2023-08-29_5-Fold_Cross-val/split_3/split_3_dataset.yaml',
                'F:/BrowserDownloads/datasets/CTC-CAF/2023-08-29_5-Fold_Cross-val/split_4/split_4_dataset.yaml',
                'F:/BrowserDownloads/datasets/CTC-CAF/2023-08-29_5-Fold_Cross-val/split_5/split_5_dataset.yaml']
    results = {}
    for k in range(ksplit):
        dataset_yaml = ds_yamls[k]
        model.train(data=dataset_yaml, batch=16, epochs=10, imgsz=640,
                              project='ctc-caf', name='yolov8n_640_5fold-test')  # Include any training arguments
        results[k] = model.metrics  # save output metrics for further analysis


if __name__ == '__main__':
    # split()
    main()
