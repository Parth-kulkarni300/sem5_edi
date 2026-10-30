import os
import sys
import time
import gdown

folder_url = "https://drive.google.com/drive/folders/1Ie50txbANgAxJjWKJruBhmZwOX6YssNR"
output_dir = r"d:\SEM5_EDI\raw_dataset"

os.makedirs(output_dir, exist_ok=True)

print(f"Starting download from {folder_url} to {output_dir}...")

max_retries = 5
for attempt in range(1, max_retries + 1):
    try:
        print(f"--- Attempt {attempt} of {max_retries} ---")
        gdown.download_folder(
            url=folder_url,
            output=output_dir,
            quiet=False,
            use_cookies=False
        )
        print("Download completed successfully!")
        break
    except Exception as e:
        print(f"Attempt {attempt} encountered error: {e}")
        if attempt < max_retries:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Max retries reached.")

# Summary of downloaded files
all_files = []
for root, dirs, files in os.walk(output_dir):
    for f in files:
        all_files.append(os.path.join(root, f))

print(f"\nTotal downloaded files: {len(all_files)}")
