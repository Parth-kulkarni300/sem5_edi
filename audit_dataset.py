import os
import sys
import glob
import pandas as pd
import numpy as np
import pydicom

def parse_dicom_folder(folder_path):
    dcm_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm') or f.endswith('.dcm.gz')]
    if not dcm_files:
        # Check subdirectories
        dcm_files = glob.glob(os.path.join(folder_path, "**", "*.dcm"), recursive=True)
    
    return dcm_files

def run_audit(raw_dir=r"d:\SEM5_EDI\raw_dataset", output_dir=r"d:\SEM5_EDI"):
    print(f"Scanning raw dataset directory: {raw_dir}")
    os.makedirs(output_dir, exist_ok=True)

    manifest_rows = []
    
    # Walk directory to find DICOM series or files
    dcm_files = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.dcm') or f.endswith('.nii') or f.endswith('.nii.gz'):
                dcm_files.append(os.path.join(root, f))

    print(f"Found total files: {len(dcm_files)}")

    # Group DICOM files by directory / series
    series_map = {}
    for fpath in dcm_files:
        sdir = os.path.dirname(fpath)
        if sdir not in series_map:
            series_map[sdir] = []
        series_map[sdir].append(fpath)

    print(f"Grouped into {len(series_map)} series folders.")

    for sdir, files in series_map.items():
        sample_file = files[0]
        rel_path = os.path.relpath(sdir, raw_dir)
        
        # Default metadata values
        subj_id = "UNKNOWN"
        img_id = "UNKNOWN"
        diagnosis = "UNKNOWN"
        modality = "MR"
        img_desc = "Structural T1 MRI"
        dims = "UNKNOWN"
        spacing = "UNKNOWN"
        qc_status = "PASS"
        qc_reason = "OK"

        # Try parsing subject ID and image ID from path or filename
        fname = os.path.basename(sample_file)
        parts = rel_path.replace("\\", "/").split("/")
        
        for part in parts:
            if part.isdigit() and len(part) >= 4:
                subj_id = part
            elif "PPMI_" in part or "PPMI" in part:
                # e.g., PPMI_3178_...
                sub_parts = part.split("_")
                for sp in sub_parts:
                    if sp.isdigit() and len(sp) >= 4:
                        subj_id = sp
                        break

        # Check filename for Image ID (e.g. _I1530044.dcm)
        if "_I" in fname:
            try:
                img_id = fname.split("_I")[1].split(".")[0]
            except Exception:
                pass
        
        # Read header via pydicom if DICOM
        try:
            dcm = pydicom.dcmread(sample_file, stop_before_pixels=True)
            if hasattr(dcm, "PatientID") and dcm.PatientID:
                subj_id = dcm.PatientID
            if hasattr(dcm, "SeriesInstanceUID") and dcm.SeriesInstanceUID:
                if img_id == "UNKNOWN":
                    img_id = dcm.SeriesInstanceUID.split(".")[-1]
            if hasattr(dcm, "Modality") and dcm.Modality:
                modality = dcm.Modality
            if hasattr(dcm, "SeriesDescription") and dcm.SeriesDescription:
                img_desc = dcm.SeriesDescription
            
            # Dimensions & Voxel Spacing
            rows = getattr(dcm, "Rows", 0)
            cols = getattr(dcm, "Columns", 0)
            n_slices = len(files)
            dims = f"({rows}, {cols}, {n_slices})"

            pix_space = getattr(dcm, "PixelSpacing", [1.0, 1.0])
            slice_thick = getattr(dcm, "SliceThickness", 1.0)
            spacing = f"({float(pix_space[0]):.2f}, {float(pix_space[1]):.2f}, {float(slice_thick):.2f})"

            if rows == 0 or cols == 0 or n_slices == 0:
                qc_status = "FAIL"
                qc_reason = "Zero dimensions or missing slices"
        except Exception as e:
            qc_status = "FAIL"
            qc_reason = f"DICOM read error: {str(e)}"

        manifest_rows.append({
            "subject_id": subj_id,
            "image_id": img_id,
            "diagnosis": diagnosis,
            "file_path": rel_path,
            "modality": modality,
            "image_description": img_desc,
            "image_dimensions": dims,
            "voxel_spacing": spacing,
            "quality_status": qc_status,
            "quality_reason": qc_reason,
            "slice_count": len(files)
        })

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, "dataset_manifest.csv")
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Saved dataset manifest to {manifest_path}")

    # Generate Audit Summary Report
    total_scans = len(df_manifest)
    unique_subjects = df_manifest["subject_id"].nunique()
    qc_pass = (df_manifest["quality_status"] == "PASS").sum()
    qc_fail = (df_manifest["quality_status"] == "FAIL").sum()
    
    pd_count = (df_manifest["diagnosis"] == "Parkinson's Disease").sum()
    control_count = (df_manifest["diagnosis"] == "Healthy Control").sum()

    audit_summary = {
        "Metric": [
            "Total MRI Scans / Series",
            "Unique Subjects",
            "PD Count (Parkinson's)",
            "Control Count (Healthy)",
            "Unknown Class Count",
            "QC Pass Scans",
            "QC Fail Scans",
            "Duplicate Subject IDs",
            "Distinct Image Dimensions",
            "Distinct Voxel Spacings"
        ],
        "Value": [
            total_scans,
            unique_subjects,
            pd_count,
            control_count,
            total_scans - (pd_count + control_count),
            qc_pass,
            qc_fail,
            total_scans - unique_subjects if total_scans > unique_subjects else 0,
            str(df_manifest["image_dimensions"].unique().tolist()),
            str(df_manifest["voxel_spacing"].unique().tolist())
        ]
    }
    
    df_audit = pd.DataFrame(audit_summary)
    audit_path = os.path.join(output_dir, "dataset_audit.csv")
    df_audit.to_csv(audit_path, index=False)
    print(f"Saved dataset audit to {audit_path}")

    # Save QC Report
    df_qc = df_manifest[["subject_id", "image_id", "file_path", "quality_status", "quality_reason"]]
    qc_path = os.path.join(output_dir, "qc_report.csv")
    df_qc.to_csv(qc_path, index=False)
    print(f"Saved QC report to {qc_path}")

if __name__ == "__main__":
    raw_dir = r"d:\SEM5_EDI\raw_dataset"
    if len(sys.argv) > 1:
        raw_dir = sys.argv[1]
    run_audit(raw_dir)
