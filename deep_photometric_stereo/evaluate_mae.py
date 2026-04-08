"""
Evaluate Predicted Normal Map (.npy) against Ground Truth.
Hỗ trợ đọc GT định dạng: .mat, .npy, .png, .tif
"""
import argparse
import numpy as np
import cv2
import os
from scipy.io import loadmat

def mean_angular_error(pred, gt, mask):
    """Tính sai số góc trung bình (Mean Angular Error)"""
    # Chuẩn hóa L2 cho chắc chắn
    pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
    pred = pred / (pred_norm + 1e-8)
    
    gt_norm = np.linalg.norm(gt, axis=-1, keepdims=True)
    gt = gt / (gt_norm + 1e-8)
    
    # Tính góc
    dot_product = np.sum(pred * gt, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angular_error = np.arccos(dot_product) * (180.0 / np.pi)
    
    if mask is not None:
        mae = np.sum(angular_error * mask) / (np.sum(mask) + 1e-8)
    else:
        mae = np.mean(angular_error)
        
    return mae, angular_error

def load_ground_truth(gt_path):
    print(f"Đang đọc Ground Truth từ: {gt_path}")
    if gt_path.endswith('.mat'):
        mat = loadmat(gt_path)
        # Bộ DiLiGenT thường lưu GT trong key 'Normal_gt'
        if 'Normal_gt' in mat:
            return mat['Normal_gt']
        # Nếu khác key, lấy ma trận đầu tiên tìm được
        for key in mat.keys():
            if not key.startswith('__'):
                return mat[key]
    elif gt_path.endswith('.npy'):
        return np.load(gt_path)
    else:
        # Đọc ảnh (.png, .tif)
        img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Không thể đọc ảnh GT: {gt_path}")
        
        # OpenCv đọc ảnh theo hệ BGR, Normal Map chuẩn là RGB -> Phải đảo lại
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = img.astype(np.float32)
        # Nếu dải màu là [0, 255] thì đưa về [-1, 1]
        if img.max() > 1.5:
            img = img / 255.0
            
        # Các file ảnh Normal Map thường lưu hệ [0, 1]. Cần ép về vector [-1, 1]
        if img.min() >= 0:
            img = img * 2.0 - 1.0
            
        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Đường dẫn file dự đoán (predicted_normal.npy)")
    parser.add_argument("--gt", type=str, required=True, help="Đường dẫn file Ground Truth (.mat, .png, .npy)")
    parser.add_argument("--mask", type=str, default=None, help="Đường dẫn Mask (Để chỉ chấm điểm vùng quả bóng)")
    args = parser.parse_args()

    # 1. Đọc dự đoán
    pred_normal = np.load(args.pred)
    H, W, _ = pred_normal.shape
    
    # 2. Đọc Ground Truth
    gt_normal = load_ground_truth(args.gt)
    
    # Đảm bảo cùng kích thước
    if gt_normal.shape[:2] != (H, W):
        print(f"Cảnh báo: Kích thước chênh lệch! Đang resize GT từ {gt_normal.shape[:2]} về {(H, W)}")
        gt_normal = cv2.resize(gt_normal, (W, H), interpolation=cv2.INTER_NEAREST)

    # 3. Đọc Mask
    if args.mask and os.path.exists(args.mask):
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_np = (mask_img > 127).astype(np.float32)
    else:
        # Tự suy mask từ GT (những chỗ có vector > 0)
        mask_np = (np.linalg.norm(gt_normal, axis=-1) > 0.1).astype(np.float32)

    # 4. Tính điểm
    mae, err_map = mean_angular_error(pred_normal, gt_normal, mask_np)
    
    print("========================================")
    print(f"🎯 KẾT QUẢ ĐÁNH GIÁ (MEAN ANGULAR ERROR)")
    print(f"   MAE = {mae:.2f} độ")
    print("========================================")
    
    # 5. Lưu ảnh Heatmap sai số (Tùy chọn)
    err_map = err_map * mask_np
    # Chuẩn hóa dải [0, 90 độ] sang [0, 255] để vẽ màu (Nóng = Sai nhiều)
    err_vis = np.clip(err_map / 90.0 * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(err_vis, cv2.COLORMAP_JET)
    heatmap[mask_np == 0] = [0, 0, 0] # Nền đen
    
    out_heat = os.path.join(os.path.dirname(args.pred), "error_heatmap.png")
    cv2.imwrite(out_heat, heatmap)
    print(f"Đã lưu bản đồ nhiệt thể hiện vùng bị sai tại: {out_heat}")

if __name__ == "__main__":
    main()