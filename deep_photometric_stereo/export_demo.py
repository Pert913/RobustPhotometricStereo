import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import json
import random

def get_optimal_lights(folder_path):
    clean_path = folder_path.rstrip('/\\')
    parent_dir = os.path.dirname(clean_path)
    config_path = os.path.join(parent_dir, "point_lights.config")
    
    lights = []
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and 'point_light_' in parts[0]:
                        idx = int(parts[0].split('_')[-1])
                        spatial_val = float(parts[2]) 
                        lights.append((idx, spatial_val))
        except: pass
        
    if len(lights) >= 3:
        lights.sort(key=lambda x: x[1])
        return [lights[0][0], lights[len(lights)//2][0], lights[-1][0]]
    return [4, 11, 18]

def read_light_means(folder_path):
    mean_ip, mean_ienv = 1.0, 1.0
    clean_path = folder_path.rstrip('/\\')
    parent_dir = os.path.dirname(clean_path)
    config_path = os.path.join(parent_dir, "light_means.config")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    parts = line.strip().split() 
                    if len(parts) >= 2:
                        if 'point_mean' in parts[0]: mean_ip = float(parts[1])
                        elif 'env_mean' in parts[0]: mean_ienv = float(parts[1])
        except: pass
    return mean_ip, mean_ienv

def load_exr(path):
    if not os.path.exists(path): return None
    try:
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None: return None
        if len(img.shape) == 2: img = cv2.merge([img, img, img])
        return img.astype(np.float32)
    except: return None

def create_alpha_series_demo(folder_path, out_path):
    mask = load_exr(os.path.join(folder_path, "binary_mask.exr"))
    if mask is None: return False
    if mask.ndim == 3: mask = mask[:,:,0]
    mask_bool = mask > 0.5

    p_indices = get_optimal_lights(folder_path)
    mean_ip, mean_ienv = read_light_means(folder_path)

    random.seed(42) 
    env_idx = random.randint(1, 10) # Cố định 1 map môi trường để dễ so sánh
    
    # Load trước 3 ảnh Point và 3 ảnh Env tương ứng
    img_ps, img_es = [], []
    for i in p_indices:
        p = load_exr(os.path.join(folder_path, f"point_light_{str(i).zfill(5)}.exr"))
        e = load_exr(os.path.join(folder_path, f"env_light_{str(env_idx).zfill(5)}.exr"))
        if p is None or e is None: return False
        
        img_ps.append(p / (mean_ip + 1e-8))
        img_es.append(e / (mean_ienv + 1e-8))

    # Danh sách các Alpha muốn test gửi cho thầy
    alphas = [1.0, 0.7, 0.4, 0.0]
    result_images = []

    for alpha in alphas:
        channels = []
        for i in range(3):
            blended = alpha * img_ps[i] + (1 - alpha) * img_es[i]
            gray = 0.2989*blended[:,:,2] + 0.5870*blended[:,:,1] + 0.1140*blended[:,:,0]
            channels.append(gray)

        input_rgb = np.stack(channels, axis=-1)
        
        # CHỐNG CHÁY SÁNG & CHUẨN HÓA
        if mask_bool.any():
            p99 = np.percentile(input_rgb[mask_bool], 99.0)
            input_rgb = np.clip(input_rgb, 0.0, p99)
            
            m = input_rgb[mask_bool].mean()
            s = input_rgb[mask_bool].std() + 1e-8
            input_rgb = (input_rgb - m) / s

        # CHUYỂN ĐỔI THÀNH LDR [0-255]
        vis_img = np.zeros_like(input_rgb)
        if mask_bool.any():
            min_val = input_rgb[mask_bool].min()
            max_val = input_rgb[mask_bool].max()
            vis_img[mask_bool] = (input_rgb[mask_bool] - min_val) / (max_val - min_val + 1e-8)
            
        vis_img[~mask_bool] = 0.0
        vis_img = (vis_img * 255).astype(np.uint8)
        bgr_final = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # Ghi chữ Alpha lên góc ảnh
        text = f"Alpha: {alpha}"
        if alpha == 1.0: text += " (Point only)"
        elif alpha == 0.0: text += " (Env only)"
        
        # Vẽ một hộp đen lót dưới chữ để dễ đọc
        cv2.rectangle(bgr_final, (5, 5), (280, 40), (0, 0, 0), -1)
        cv2.putText(bgr_final, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        result_images.append(bgr_final)

    # Ghép 4 bức ảnh lại với nhau theo chiều ngang
    final_grid = np.hstack(result_images)
    cv2.imwrite(out_path, final_grid)
    return True

def main():
    json_path = "valid_samples_ssd_clean.json"
    output_dir = "./demo_gvhd"
    num_samples = 3
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        all_folders = json.load(f)
        
    random.seed() 
    selected_folders = random.sample(all_folders, min(num_samples, len(all_folders)))
    
    print(f"🎨 Đang tạo dải test Alpha cho {len(selected_folders)} vật thể...")
    for folder in selected_folders:
        folder_name = os.path.basename(os.path.normpath(folder))
        out_name = f"alpha_test_{folder_name}.png"
        out_path = os.path.join(output_dir, out_name)
        
        success = create_alpha_series_demo(folder, out_path)
        if success:
            print(f"  ✅ Đã lưu: {out_path}")

if __name__ == "__main__":
    main()