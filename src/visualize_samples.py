import os
from pathlib import Path
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

def generate_comparison_plot(subject_id="101480",
                             raw_nifti="d:/EDI_PPMI_dataset/data/nifti_raw/101480_raw.nii.gz",
                             prep_nifti="d:/EDI_PPMI_dataset/data/nifti_processed/101480_prep.nii.gz",
                             out_plot="d:/EDI_PPMI_dataset/data/plots/mri_preprocessing_comparison.png"):
    """
    Plots a side-by-side comparison of raw DICOM 3D volume vs preprocessed brain volume.
    """
    Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(raw_nifti) or not os.path.exists(prep_nifti):
        print(f"[Warning] NIfTI files for subject {subject_id} not ready yet.")
        return

    img_raw = sitk.ReadImage(raw_nifti)
    arr_raw = sitk.GetArrayFromImage(img_raw)
    
    img_prep = sitk.ReadImage(prep_nifti)
    arr_prep = sitk.GetArrayFromImage(img_prep)

    z_raw = arr_raw.shape[0] // 2
    z_prep = arr_prep.shape[0] // 2

    slice_raw = arr_raw[z_raw]
    slice_prep = arr_prep[z_prep]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(f"NeuroLens AI — MRI Preprocessing Pipeline (Subject {subject_id})", fontsize=14, fontweight="bold", y=0.98)

    # 1. Raw Volume Slice
    im0 = axes[0].imshow(slice_raw, cmap="gray")
    axes[0].set_title(f"1. Raw DICOM Slice\n(Size: {arr_raw.shape})", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Preprocessed & Skull-Stripped
    im1 = axes[1].imshow(slice_prep, cmap="gray")
    axes[1].set_title(f"2. Preprocessed & Skull-Stripped\n(1.0mm Isotropic, RAS, Z-Score)", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. 2.5D Stacked ResNet Input
    slice_rgb = np.stack([arr_prep[z_prep-1], arr_prep[z_prep], arr_prep[z_prep+1]], axis=-1)
    # Min-max scale for display
    s_min, s_max = slice_rgb.min(), slice_rgb.max()
    slice_rgb_norm = (slice_rgb - s_min) / (s_max - s_min + 1e-8)
    
    axes[2].imshow(slice_rgb_norm)
    axes[2].set_title("3. 2.5D 3-Channel Input\n(ResNet18 Input Tensor)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Generated comparison visualization: {out_plot}")

if __name__ == "__main__":
    generate_comparison_plot()
