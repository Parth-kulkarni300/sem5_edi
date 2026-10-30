import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import SimpleITK as sitk

def convert_dicom_to_nifti(dicom_dir):
    """
    Reads a directory containing DICOM slices and returns a 3D SimpleITK Image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        # Fallback to searching all dcm files in folder
        dcm_files = sorted(glob.glob(os.path.join(dicom_dir, "**", "*.dcm"), recursive=True))
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
        dicom_names = dcm_files
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def reorient_to_ras(image):
    """
    Reorients the 3D Sitk image to standard RAS (Right, Anterior, Superior) orientation.
    """
    return sitk.DICOMOrient(image, 'RAS')

def resample_isotropic(image, new_spacing=(1.0, 1.0, 1.0)):
    """
    Resamples a 3D Sitk image to isotropic voxel spacing (default: 1.0mm x 1.0mm x 1.0mm).
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    
    return resample.Execute(image)

def normalize_intensity(image_arr, method="zscore"):
    """
    Normalizes numpy intensity array of brain MRI.
    """
    # Exclude background zero pixels from stats calculation
    mask = image_arr > 0
    if not np.any(mask):
        mask = np.ones_like(image_arr, dtype=bool)

    if method == "zscore":
        mean_val = np.mean(image_arr[mask])
        std_val = np.std(image_arr[mask]) + 1e-8
        norm_arr = (image_arr - mean_val) / std_val
        norm_arr[~mask] = 0
    elif method == "minmax":
        min_val = np.min(image_arr[mask])
        max_val = np.max(image_arr[mask])
        norm_arr = (image_arr - min_val) / (max_val - min_val + 1e-8)
        norm_arr[~mask] = 0
    else:
        norm_arr = image_arr

    return norm_arr.astype(np.float32)

def skull_strip_simple(image, lower_threshold=10):
    """
    Applies Otsu / Connected Component thresholding to produce a clean brain mask.
    """
    # Threshold out background air
    mask = sitk.OtsuThreshold(image, 0, 1)
    # Fill holes and keep largest component
    mask = sitk.VotingBinaryIterativeHoleFilling(mask, radius=[2,2,2])
    mask = sitk.RelabelComponent(sitk.ConnectedComponent(mask)) == 1
    # Multiply raw image by binary brain mask
    masked_image = sitk.Mask(image, mask)
    return masked_image, mask

def preprocess_subject(subject_id, dicom_dir, raw_out_dir, prep_out_dir):
    """
    Complete processing pipeline for a single subject.
    """
    raw_out_dir = Path(raw_out_dir)
    prep_out_dir = Path(prep_out_dir)
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    prep_out_dir.mkdir(parents=True, exist_ok=True)

    raw_nifti_path = raw_out_dir / f"{subject_id}_raw.nii.gz"
    prep_nifti_path = prep_out_dir / f"{subject_id}_prep.nii.gz"

    if raw_nifti_path.exists() and prep_nifti_path.exists() and os.path.getsize(prep_nifti_path) > 1000:
        try:
            prep_img = sitk.ReadImage(str(prep_nifti_path))
            raw_img = sitk.ReadImage(str(raw_nifti_path))
            return {
                "subject_id": subject_id,
                "raw_nifti": str(raw_nifti_path),
                "prep_nifti": str(prep_nifti_path),
                "original_size": raw_img.GetSize(),
                "prep_size": prep_img.GetSize(),
                "prep_spacing": prep_img.GetSpacing(),
                "status": "SUCCESS"
            }
        except Exception:
            pass # Re-run if file reading fails

    # Step 1: Load DICOM
    img_3d = convert_dicom_to_nifti(dicom_dir)
    sitk.WriteImage(img_3d, str(raw_nifti_path))

    # Step 2: Reorient to RAS
    img_ras = reorient_to_ras(img_3d)

    # Step 3: Resample to 1.0mm Isotropic Voxels
    img_resampled = resample_isotropic(img_ras, (1.0, 1.0, 1.0))

    # Step 4: Skull Stripping / Brain Masking
    img_brain, brain_mask = skull_strip_simple(img_resampled)

    # Step 5: Convert to Numpy & Normalize Intensity
    arr = sitk.GetArrayFromImage(img_brain)
    norm_arr = normalize_intensity(arr, method="zscore")

    # Step 6: Convert back to Sitk & Save Processed NIfTI Volume
    out_img = sitk.GetImageFromArray(norm_arr)
    out_img.CopyInformation(img_brain)
    sitk.WriteImage(out_img, str(prep_nifti_path))

    return {
        "subject_id": subject_id,
        "raw_nifti": str(raw_nifti_path),
        "prep_nifti": str(prep_nifti_path),
        "original_size": img_3d.GetSize(),
        "prep_size": out_img.GetSize(),
        "prep_spacing": out_img.GetSpacing(),
        "status": "SUCCESS"
    }

def process_all_subjects(ppmi_dir="d:/EDI_PPMI_dataset/PPMI", output_base="d:/EDI_PPMI_dataset/data"):
    ppmi_path = Path(ppmi_dir)
    subjects = [d for d in ppmi_path.iterdir() if d.is_dir()]
    
    print(f"Starting MRI preprocessing pipeline for {len(subjects)} subjects...\n")
    
    raw_dir = Path(output_base) / "nifti_raw"
    prep_dir = Path(output_base) / "nifti_processed"
    
    results = []
    for i, sub_dir in enumerate(subjects, 1):
        sub_id = sub_dir.name
        print(f"[{i}/{len(subjects)}] Processing Subject {sub_id}...")
        try:
            res = preprocess_subject(sub_id, sub_dir, raw_dir, prep_dir)
            results.append(res)
            print(f"  --> Success! Saved preprocessed volume {res['prep_size']} to {prep_dir.name}/{sub_id}_prep.nii.gz")
        except Exception as e:
            print(f"  --> [ERROR] Processing Subject {sub_id} failed: {e}")
            results.append({"subject_id": sub_id, "status": f"FAILED: {e}"})

    df = pd.DataFrame(results)
    summary_path = Path(output_base) / "preprocessing_summary.csv"
    df.to_csv(summary_path, index=False)
    print("\n" + "="*60)
    print(f"PREPROCESSING COMPLETE! Saved summary to {summary_path}")
    print("="*60)
    return df

if __name__ == "__main__":
    process_all_subjects()
