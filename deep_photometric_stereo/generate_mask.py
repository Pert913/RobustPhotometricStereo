"""
Generate a robust binary mask from a sequence of Photometric Stereo images.
Method: Max-Intensity Projection across all N images.
"""
import cv2
import numpy as np
import glob
import os
import argparse

def generate_robust_mask(input_dir, output_path, threshold=15):
    # 1. Tìm tất cả các file ảnh trong thư mục
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"Lỗi: Không tìm thấy ảnh nào trong thư mục {input_dir}")
        return

    print(f"Đang xử lý {len(image_paths)} ảnh để tạo mask...")

    # 2. Thuật toán Max-Intensity Projection (Gộp ảnh sáng nhất)
    max_img = None
    
    for path in image_paths:
        # Đọc ảnh dưới dạng xám (Grayscale)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        if max_img is None:
            max_img = img
        else:
            # So sánh pixel-by-pixel, giữ lại pixel sáng hơn
            max_img = np.maximum(max_img, img)

    # 3. Làm mờ nhẹ ảnh đã gộp để khử nhiễu
    blurred = cv2.GaussianBlur(max_img, (5, 5), 0)

    # 4. Phân ngưỡng (Thresholding)
    # Vì max_img đã gom hết các điểm sáng nhất, phông nền vẫn sẽ tối (< threshold),
    # còn toàn bộ vật thể chắc chắn sẽ sáng rực lên (> threshold).
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # 5. Dọn dẹp viền (Morphology)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Lấp lỗ hổng
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Xóa nhiễu nền

    # 6. Lưu kết quả
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    cv2.imwrite(output_path, mask)
    print(f"Thành công! Mask đã được lưu tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tạo Mask từ thư mục chứa N ảnh PS.")
    parser.add_argument("--input_dir", type=str, required=True, help="Thư mục chứa 96 ảnh của vật thể.")
    parser.add_argument("--output", type=str, default="mask.png", help="Đường dẫn lưu file mask.")
    parser.add_argument("--thresh", type=int, default=15, help="Ngưỡng cắt nền (0-255).")
    
    args = parser.parse_args()
    generate_robust_mask(args.input_dir, args.output, args.thresh)

if __name__ == "__main__":
    main()