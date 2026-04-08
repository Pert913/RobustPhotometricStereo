import json
import os

def list_failed_folders(json_path, root_dir):
    # 1. Đọc danh sách folder sạch
    with open(json_path, 'r') as f:
        valid_folders = set(json.load(f))
    
    # 2. Tìm tất cả folder cam_xxxx thực tế
    failed = []
    for root, dirs, files in os.walk(root_dir):
        if "local_normal.exr" in files or "binary_mask.exr" in files:
            full_path = os.path.abspath(root)
            if full_path not in valid_folders:
                failed.append(full_path)
    
    # 3. In kết quả
    print(f"=== DANH SÁCH {len(failed)} FOLDER BỊ LỖI ===")
    for path in failed:
        print(f"❌ {path}")

# Chạy để xem tên 5 folder bị lỗi
list_failed_folders("valid_samples.json", r"E:\PhotoStereo")