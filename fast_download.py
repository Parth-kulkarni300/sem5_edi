import os
import re
import json
import requests
import concurrent.futures

folder_id = "1Ie50txbANgAxJjWKJruBhmZwOX6YssNR"
output_dir = r"d:\SEM5_EDI\raw_dataset"
os.makedirs(output_dir, exist_ok=True)

print("Testing fast multithreaded download...")
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

# Fetch main folder page
res = session.get(f"https://drive.google.com/drive/folders/{folder_id}")
html = res.text

# Extract ds:1 or initial items
print("Main page fetched. Page length:", len(html))
