import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T

BASE_DIR = Path(__file__).resolve().parent.parent

def find_summary_csv():
    candidates = [
        BASE_DIR / "data" / "preprocessing_summary.csv",
        Path("d:/SEM5_EDI/data/preprocessing_summary.csv"),
        Path("d:/EDI_PPMI_dataset/data/preprocessing_summary.csv")
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

def find_ppmi_dir():
    candidates = [
        BASE_DIR / "PPMI",
        Path("d:/EDI_PPMI_dataset/PPMI"),
        BASE_DIR / "raw_dataset" / "PPMI",
        Path("d:/SEM5_EDI/raw_dataset/PPMI")
    ]
    for c in candidates:
        if c.exists() and len(os.listdir(c)) > 0:
            return c
    return candidates[0]

def find_manifest_csv():
    candidates = [
        BASE_DIR / "dataset_manifest.csv",
        Path("d:/SEM5_EDI/dataset_manifest.csv"),
        Path("d:/EDI_PPMI_dataset/dataset_manifest.csv")
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

def create_patient_level_splits(manifest_csv=None,
                                ppmi_dir=None,
                                summary_csv=None,
                                out_json=None,
                                seed=42):
    """
    Creates strict patient/subject level train/val/test splits to avoid data leakage.
    """
    if manifest_csv is None:
        manifest_csv = find_manifest_csv()
    if ppmi_dir is None:
        ppmi_dir = find_ppmi_dir()
    if summary_csv is None:
        summary_csv = find_summary_csv()
    if out_json is None:
        out_json = BASE_DIR / "data" / "dataset_splits.json"

    manifest = pd.read_csv(manifest_csv)
    
    valid_subjects = set()
    if os.path.exists(summary_csv):
        summary_df = pd.read_csv(summary_csv)
        for _, row in summary_df.iterrows():
            sub = str(row["subject_id"]).strip()
            prep_size_str = str(row["prep_size"])
            try:
                dims = [int(x.strip()) for x in prep_size_str.strip("()[]").split(",") if x.strip()]
                depth = min(dims)
                if depth >= 15:
                    valid_subjects.add(sub)
            except Exception:
                valid_subjects.add(sub)
    else:
        valid_subjects = set(os.listdir(ppmi_dir))

    records = []
    for sub in sorted(valid_subjects):
        match = manifest[manifest["subject_id"].astype(str) == str(sub)]
        if len(match) > 0:
            diag_str = str(match.iloc[0]["diagnosis"])
            label = 1 if "Parkinson" in diag_str else 0
        else:
            continue
        records.append({"subject_id": sub, "diagnosis": diag_str, "label": label})

    df = pd.DataFrame(records).drop_duplicates(subset=["subject_id"])
    
    # Enforce 50/50 Class Balance (Equal number of PD and Healthy Control subjects)
    df_pd = df[df["label"] == 1]
    df_hc = df[df["label"] == 0]
    n_samples = min(len(df_pd), len(df_hc))
    
    df_pd_bal = df_pd.sample(n=n_samples, random_state=seed)
    df_hc_bal = df_hc.sample(n=n_samples, random_state=seed)
    df_balanced = pd.concat([df_pd_bal, df_hc_bal]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Stratified Split at SUBJECT Level
    train_df, test_val_df = train_test_split(df_balanced, test_size=0.30, random_state=seed, stratify=df_balanced["label"])
    val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=seed, stratify=test_val_df["label"])

    splits = {
        "train": train_df["subject_id"].tolist(),
        "val": val_df["subject_id"].tolist(),
        "test": test_df["subject_id"].tolist(),
        "labels": dict(zip(df_balanced["subject_id"], df_balanced["label"]))
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"50/50 Patient-level splits created successfully (Filtered {len(df_balanced)} subjects: {n_samples} PD vs {n_samples} HC):")
    print(f"  Train subjects: {len(splits['train'])} (PD: {train_df['label'].sum()}, HC: {len(train_df)-train_df['label'].sum()})")
    print(f"  Val subjects:   {len(splits['val'])} (PD: {val_df['label'].sum()}, HC: {len(val_df)-val_df['label'].sum()})")
    print(f"  Test subjects:  {len(splits['test'])} (PD: {test_df['label'].sum()}, HC: {len(test_df)-test_df['label'].sum()})")
    print(f"Saved splits to {out_json}")

    return splits

class MRISliceDataset(Dataset):
    """
    PyTorch Dataset loading 2.5D MRI slices from preprocessed 3D NIfTI volumes.
    Supports Data Augmentation for training set generalization.
    """
    def __init__(self, subject_ids, labels, data_dir=None, is_train=False, transform=None):
        self.subject_ids = subject_ids
        self.labels = labels
        self.data_dir = Path(data_dir) if data_dir else BASE_DIR / "data" / "nifti_processed"
        self.is_train = is_train

        if transform:
            self.transform = transform
        elif is_train:
            # Data Augmentation pipeline for Generalization
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.RandomRotation(degrees=10),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Deterministic pipeline for Val / Test
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sub_id = self.subject_ids[idx]
        label = self.labels[sub_id]
        
        file_path = self.data_dir / f"{sub_id}_prep.nii.gz"
        if not file_path.exists():
            file_path = Path("d:/EDI_PPMI_dataset/data/nifti_processed") / f"{sub_id}_prep.nii.gz"
        if not file_path.exists():
            return torch.zeros((3, 224, 224), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        
        z_mid = arr.shape[0] // 2
        if self.is_train:
            offset = np.random.randint(-2, 3)
            z_mid = max(1, min(arr.shape[0] - 2, z_mid + offset))

        slice1 = arr[max(0, z_mid - 1)]
        slice2 = arr[z_mid]
        slice3 = arr[min(arr.shape[0] - 1, z_mid + 1)]

        slice_stacked = np.stack([slice1, slice2, slice3], axis=-1)
        
        s_min, s_max = slice_stacked.min(), slice_stacked.max()
        if s_max > s_min:
            slice_uint8 = ((slice_stacked - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
        else:
            slice_uint8 = np.zeros_like(slice_stacked, dtype=np.uint8)

        img_tensor = self.transform(slice_uint8)
        return img_tensor, torch.tensor(label, dtype=torch.long)

    def get_volume_triplets(self, idx):
        sub_id = self.subject_ids[idx]
        label = self.labels[sub_id]
        
        file_path = self.data_dir / f"{sub_id}_prep.nii.gz"
        if not file_path.exists():
            file_path = Path("d:/EDI_PPMI_dataset/data/nifti_processed") / f"{sub_id}_prep.nii.gz"
        if not file_path.exists():
            return torch.zeros((1, 3, 224, 224), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        img = sitk.ReadImage(str(file_path))
        arr = sitk.GetArrayFromImage(img)
        
        z_mid = arr.shape[0] // 2
        offsets = [-6, -3, 0, 3, 6]
        triplet_tensors = []

        for off in offsets:
            z_curr = max(1, min(arr.shape[0] - 2, z_mid + off))
            s1 = arr[max(0, z_curr - 1)]
            s2 = arr[z_curr]
            s3 = arr[min(arr.shape[0] - 1, z_curr + 1)]
            stacked = np.stack([s1, s2, s3], axis=-1)
            
            s_min, s_max = stacked.min(), stacked.max()
            if s_max > s_min:
                slice_u8 = ((stacked - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
            else:
                slice_u8 = np.zeros_like(stacked, dtype=np.uint8)

            t = self.transform(slice_u8)
            triplet_tensors.append(t)

        volume_tensor = torch.stack(triplet_tensors, dim=0) # [3, 3, 224, 224]
        return volume_tensor, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    splits = create_patient_level_splits()
    train_dataset = MRISliceDataset(splits["train"], splits["labels"])
    print(f"Sample dataset size: {len(train_dataset)}")
