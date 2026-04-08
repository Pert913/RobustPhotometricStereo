import os
# BẮT BUỘC: Đặt ở dòng đầu tiên để kích hoạt OpenEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import json
import random
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from PIL import Image

# ==============================================================================
# 1. DILIGENT DATASET (TRAIN & TEST)
# ==============================================================================

class DiLiGentDataset(Dataset):
    """
    DiLiGenT dataset cho training dựa trên Patch.
    Sử dụng Global Normalization để giữ tín hiệu Photometric.
    """
    def __init__(
        self, data_root: str, objects: list, num_images: int = 32,
        min_images: int = 8, patch_size: int = 128,
        patches_per_epoch: int = 2000, augment: bool = True
    ):
        self.data_root = data_root
        self.objects = objects
        self.num_images = num_images
        self.min_images = min_images
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment

        self.object_data = []
        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            data = self._load_object(obj_dir, obj_name)
            if data is not None:
                self.object_data.append(data)

        if not self.object_data:
            raise RuntimeError(f"No valid objects found in {data_root}")

        print(f"✅ DiLiGentDataset: {len(self.object_data)} objects, patch={patch_size}")

    def _load_object(self, obj_dir: str, obj_name: str):
        normal_path = os.path.join(obj_dir, "Normal_gt.npy")
        if not os.path.exists(normal_path): return None
        normal_gt = np.load(normal_path).astype(np.float32) # (H, W, 3)

        # Đảm bảo Unit Vector [-1, 1]
        norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
        normal_gt = normal_gt / norm

        mask_path = os.path.join(obj_dir, "mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)
        else:
            mask = np.ones(normal_gt.shape[:2], dtype=np.float32)

        images = self._load_images(obj_dir)
        if images is None: return None

        return {"name": obj_name, "images": images, "normal_gt": normal_gt, "mask": mask}

    @staticmethod
    def _load_single_image(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            img = np.load(path).astype(np.float32)
            if img.ndim == 2: img = np.stack((img,)*3, axis=-1)
            if img.max() > 1.5: img /= 255.0
        else:
            img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        return img

    @staticmethod
    def _load_images(obj_dir, max_images=None):
        image_pngs = sorted(glob.glob(os.path.join(obj_dir, "*.png")))
        image_pngs = [p for p in image_pngs if os.path.basename(p)[0].isdigit()]
        if not image_pngs: return None
        if max_images: image_pngs = image_pngs[:max_images]
        
        img_list = [DiLiGentDataset._load_single_image(p) for p in image_pngs]
        return np.stack(img_list, axis=0) if img_list else None

    def __len__(self): return self.patches_per_epoch

    def __getitem__(self, idx):
        obj = random.choice(self.object_data)
        images, normal_gt, mask = obj["images"], obj["normal_gt"], obj["mask"]
        H, W = images.shape[1:3]
        ps = self.patch_size

        y, x = random.randint(0, H - ps), random.randint(0, W - ps)
        n_select = random.randint(self.min_images, self.num_images)
        indices = random.sample(range(images.shape[0]), n_select)

        img_patches = images[indices, y:y+ps, x:x+ps, :]
        normal_patch = normal_gt[y:y+ps, x:x+ps, :]
        mask_patch = mask[y:y+ps, x:x+ps]

        # GLOBAL NORMALIZATION (Giữ tỉ lệ cường độ sáng)
        mask_bool = mask_patch > 0.5
        if mask_bool.any():
            m = img_patches[:, mask_bool, :].mean()
            s = img_patches[:, mask_bool, :].std() + 1e-8
            img_patches = (img_patches - m) / s

        if self.augment and random.random() > 0.5:
            img_patches = img_patches[:, :, ::-1, :].copy()
            normal_patch = normal_patch[:, ::-1, :].copy()
            normal_patch[:, :, 0] = -normal_patch[:, :, 0]
            mask_patch = mask_patch[:, ::-1].copy()

        return torch.from_numpy(img_patches).permute(0, 3, 1, 2), \
               torch.from_numpy(normal_patch).permute(2, 0, 1), \
               torch.from_numpy(mask_patch).unsqueeze(0)


class DiLiGentTestDataset(Dataset):
    """
    Dùng cho Evaluate trên bộ DiLiGenT cũ (full res).
    """
    def __init__(self, data_root: str, objects: list, num_images: int = 96):
        self.object_data = []
        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            normal_path = os.path.join(obj_dir, "Normal_gt.npy")
            if not os.path.exists(normal_path): continue
            
            normal_gt = np.load(normal_path).astype(np.float32)
            norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
            normal_gt = normal_gt / norm

            mask_path = os.path.join(obj_dir, "mask.npy")
            mask = np.load(mask_path).astype(np.float32) if os.path.exists(mask_path) else np.ones(normal_gt.shape[:2])
            
            images = DiLiGentDataset._load_images(obj_dir, max_images=num_images)
            if images is not None:
                # Global Normalization
                mask_bool = mask > 0.5
                if mask_bool.any():
                    m = images[:, mask_bool, :].mean()
                    s = images[:, mask_bool, :].std() + 1e-8
                    images = (images - m) / s
                
                self.object_data.append({"name": obj_name, "images": images, "normal_gt": normal_gt, "mask": mask})

    def __len__(self): return len(self.object_data)

    def __getitem__(self, idx):
        obj = self.object_data[idx]
        return torch.from_numpy(obj["images"]).permute(0, 3, 1, 2), \
               torch.from_numpy(obj["normal_gt"]).permute(2, 0, 1), \
               torch.from_numpy(obj["mask"]).unsqueeze(0), obj["name"]

# ==============================================================================
# 2. SYNTHETIC DATASET (300GB - 8-8-8 LED)
# ==============================================================================

class SyntheticDataset(Dataset):
    # ĐÃ THÊM LẠI augment=True ĐỂ TRAIN.PY KHÔNG BỊ BÁO LỖI
    def __init__(self, json_path, patch_size=128, alpha_min=0.35, augment=True): 
        with open(json_path, 'r') as f:
            self.folder_list = json.load(f)
        self.patch_size, self.alpha_min, self.augment = patch_size, alpha_min, augment
        
        # Tự động quét file config và tìm 3 góc chiếu tối ưu nhất
        self.p_indices = self._get_optimal_lights(self.folder_list)
        print(f"👉 [Train] Đã tự động chọn 3 góc đèn tối ưu: {self.p_indices}")

    def __len__(self): 
        return len(self.folder_list)

    def _get_optimal_lights(self, folder_list):
        if not folder_list: return [4, 11, 18]
        clean_path = folder_list[0].rstrip('/\\')
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
            idx_1 = lights[0][0]
            idx_2 = lights[len(lights)//2][0]
            idx_3 = lights[-1][0]
            return [idx_1, idx_2, idx_3]
            
        return [4, 11, 18]

    def load_exr(self, path):
        if not os.path.exists(path): return None
        try:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None: return None
            if len(img.shape) == 2:
                img = cv2.merge([img, img, img])
            return img.astype(np.float32)
        except:
            return None

    def read_light_means(self, folder_path):
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

    def __getitem__(self, idx):
            attempts = 0
            ps = self.patch_size
            
            # --- THIẾT LẬP SỐ LƯỢNG ẢNH NHAI MỖI BATCH ---
            # 32 là con số an toàn cho VRAM 16GB. Bạn có thể giảm xuống 16 
            # nếu bị báo lỗi CUDA Out of Memory, hoặc tăng lên 48 nếu còn dư VRAM.
            K_train = 32 

            while attempts < 20:
                folder_path = self.folder_list[idx]
                try:
                    normal_gt = self.load_exr(os.path.join(folder_path, "local_normal.exr"))
                    mask = self.load_exr(os.path.join(folder_path, "binary_mask.exr"))
                    if normal_gt is None or mask is None: raise FileNotFoundError

                    # --- 1. SỬA LỖI HỆ TỌA ĐỘ VÀ MÀU SẮC GROUND TRUTH ---
                    normal_gt = normal_gt[:, :, ::-1]       # Đổi BGR sang RGB
                    normal_gt = normal_gt * 2.0 - 1.0       # ĐƯA VỀ [-1, 1] ĐỂ AI TÍNH TOÁN ĐÚNG!

                    norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
                    normal_gt = normal_gt / norm
                    if mask.ndim == 3: mask = mask[:,:,0]

                    H, W = normal_gt.shape[:2]
                    y, x = random.randint(0, H-ps), random.randint(0, W-ps)
                    for _ in range(30):
                        ty, tx = random.randint(0, H-ps), random.randint(0, W-ps)
                        if mask[ty:ty+ps, tx:tx+ps].sum() > ps*ps*0.05:
                            y, x = ty, tx; break

                    mean_ip, mean_ienv = self.read_light_means(folder_path)

                    # --- 2. BỐC NGẪU NHIÊN K ẢNH THAY VÌ 3 ẢNH ---
                    # Quét nhanh thư mục để lấy tất cả các index đèn hợp lệ
                    all_files = os.listdir(folder_path)
                    all_indices = [int(f.split('_')[2].split('.')[0]) for f in all_files if f.startswith("point_light_") and f.endswith(".exr")]
                    
                    # Nếu số lượng ảnh trong folder nhiều hơn K_train, bốc random K ảnh
                    if len(all_indices) > K_train:
                        sampled_indices = random.sample(all_indices, K_train)
                    else:
                        sampled_indices = all_indices

                    channels = []
                    for i in sampled_indices:
                        img_p = self.load_exr(os.path.join(folder_path, f"point_light_{str(i).zfill(5)}.exr"))
                        img_e = self.load_exr(os.path.join(folder_path, f"env_light_{str(random.randint(1, 10)).zfill(5)}.exr"))
                        if img_p is None or img_e is None: raise FileNotFoundError
                        
                        patch_p = img_p[y:y+ps, x:x+ps]
                        patch_e = img_e[y:y+ps, x:x+ps]

                        patch_p = patch_p / (mean_ip + 1e-8)
                        patch_e = patch_e / (mean_ienv + 1e-8)
                        
                        alpha = random.uniform(max(0.35, self.alpha_min), 0.95)
                        blended = alpha * patch_p + (1.0 - alpha) * patch_e
                        
                        # Trộn thành ảnh xám Grayscale
                        gray = 0.2989*blended[:,:,2] + 0.5870*blended[:,:,1] + 0.1140*blended[:,:,0]
                        
                        # ÉP CHIỀU: Đổi từ (H, W) thành (1, H, W) cho mỗi ảnh
                        gray = np.expand_dims(gray, axis=0)
                        channels.append(gray)

                    # STACK THEO CHIỀU K: Kết quả input_patch có shape (K, 1, H, W)
                    input_patch = np.stack(channels, axis=0)
                    
                    # --- 3. CHỐNG CHÁY SÁNG MÔ PHỎNG ESP32 & CHUẨN HÓA Z-SCORE ---
                    mask_patch = mask[y:y+ps, x:x+ps]
                    mask_bool = mask_patch > 0.5
                    
                    if mask_bool.any():
                        # Trích xuất các pixel hợp lệ trên TOÀN BỘ K ảnh để tính toán
                        valid_pixels = input_patch[:, 0, mask_bool] 
                        
                        p99 = np.percentile(valid_pixels, 99.0)
                        input_patch = np.clip(input_patch, 0.0, p99)
                        
                        valid_pixels_clipped = input_patch[:, 0, mask_bool]
                        m = valid_pixels_clipped.mean()
                        s = valid_pixels_clipped.std() + 1e-8
                        
                        # Normalize đồng bộ cho tất cả các góc sáng
                        input_patch = (input_patch - m) / s

                    normal_patch = normal_gt[y:y+ps, x:x+ps, :]

                    # ĐẦU RA MỚI: input_patch ĐÃ CHUẨN FORM (K, 1, ps, ps) NÊN KHÔNG CẦN PERMUTE
                    return torch.from_numpy(input_patch.copy()), \
                        torch.from_numpy(normal_patch.copy()).permute(2, 0, 1), \
                        torch.from_numpy(mask_patch.copy()).unsqueeze(0)
                except:
                    attempts += 1
                    idx = random.randint(0, len(self.folder_list)-1)
            
            # Fallback an toàn nếu lỗi: Trả về tensor zero với đúng K dimension
            return torch.zeros((K_train, 1, ps, ps)), torch.zeros((3, ps, ps)), torch.zeros((1, ps, ps))
        
class SyntheticTestDataset(Dataset):
    def __init__(self, json_path, alpha_min=0.35):
        with open(json_path, 'r') as f:
            self.folder_list = json.load(f)
        self.alpha_min = alpha_min
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        
        self.p_indices = self._get_optimal_lights(self.folder_list)
        print(f"👉 [Test] Đã tự động chọn 3 góc đèn tối ưu: {self.p_indices}")

    def __len__(self): return len(self.folder_list)

    def _get_optimal_lights(self, folder_list):
        if not folder_list: return [4, 11, 18]
        clean_path = folder_list[0].rstrip('/\\')
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
            idx_1 = lights[0][0]
            idx_2 = lights[len(lights)//2][0]
            idx_3 = lights[-1][0]
            return [idx_1, idx_2, idx_3]
            
        return [4, 11, 18] 

    def load_exr(self, path):
        if not os.path.exists(path): return None
        try:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None: return None
            if len(img.shape) == 2: img = cv2.merge([img, img, img])
            return img.astype(np.float32)
        except: return None

    def read_light_means(self, folder_path):
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

    def __getitem__(self, idx):
        folder_path = self.folder_list[idx]
        normal_gt = self.load_exr(os.path.join(folder_path, "local_normal.exr"))
        mask = self.load_exr(os.path.join(folder_path, "binary_mask.exr"))
        
        if normal_gt is None or mask is None:
             # Trả về tensor rỗng đúng shape (N, 1, H, W) nếu lỗi (ví dụ N tạm để là 1)
             return torch.zeros((1, 1, 512, 512)), torch.zeros((3, 512, 512)), torch.zeros((1, 512, 512)), "error"

        # --- ĐỒNG BỘ HỆ TỌA ĐỘ VÀ MÀU SẮC ---
        normal_gt = normal_gt[:, :, ::-1]
        normal_gt = normal_gt * 2.0 - 1.0

        norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
        normal_gt /= norm
        if mask.ndim == 3: mask = mask[:,:,0]
        
        mean_ip, mean_ienv = self.read_light_means(folder_path)

        # Cố định random seed bằng idx để lần nào test cũng ra cùng 1 mức nhiễu
        random.seed(idx) 
        
        # --- LẤY TOÀN BỘ ẢNH TRONG FOLDER (KHÔNG GIỚI HẠN) ---
        all_files = os.listdir(folder_path)
        all_indices = [int(f.split('_')[2].split('.')[0]) for f in all_files if f.startswith("point_light_") and f.endswith(".exr")]
        all_indices.sort() # Sắp xếp theo thứ tự cho chuẩn bài

        channels = []
        for i in all_indices:
            img_p = self.load_exr(os.path.join(folder_path, f"point_light_{str(i).zfill(5)}.exr"))
            img_e = self.load_exr(os.path.join(folder_path, f"env_light_{str(random.randint(1, 10)).zfill(5)}.exr"))
            if img_p is None or img_e is None: continue
                
            img_p = img_p / (mean_ip + 1e-8)
            img_e = img_e / (mean_ienv + 1e-8)
            
            alpha = random.uniform(max(0.35, self.alpha_min), 0.95)
            blended = alpha * img_p + (1-alpha) * img_e
            
            gray = 0.2989*blended[:,:,2] + 0.5870*blended[:,:,1] + 0.1140*blended[:,:,0]
            gray = np.expand_dims(gray, axis=0) # Ép thành (1, H, W)
            channels.append(gray)

        # Gộp tất cả lại. input_patch có shape: (N_tổng, 1, H, W)
        input_patch = np.stack(channels, axis=0)
        
        mask_bool = mask > 0.5
        if mask_bool.any():
            # Trích xuất toàn bộ pixel hợp lệ trên mọi ảnh đèn
            valid_pixels = input_patch[:, 0, mask_bool]
            
            p99 = np.percentile(valid_pixels, 99.0)
            input_patch = np.clip(input_patch, 0.0, p99)
            
            valid_pixels_clipped = input_patch[:, 0, mask_bool]
            m = valid_pixels_clipped.mean()
            s = valid_pixels_clipped.std() + 1e-8
            
            input_patch = (input_patch - m) / s

        name = os.path.basename(os.path.dirname(folder_path)) + "_" + os.path.basename(folder_path)
        
        return torch.from_numpy(input_patch.copy()), \
               torch.from_numpy(normal_gt.copy()).permute(2, 0, 1), \
               torch.from_numpy(mask.copy()).unsqueeze(0), name
               

def collate_variable_n(batch):
    images_list, normals_list, masks_list = [], [], []
    counts = []
    for imgs, normal, mask in batch:
        images_list.append(imgs); normals_list.append(normal)
        masks_list.append(mask); counts.append(imgs.shape[0])
    max_n, B = max(counts), len(batch)
    _, C, H, W = images_list[0].shape
    padded_images = torch.zeros(B, max_n, C, H, W)
    for i, imgs in enumerate(images_list): padded_images[i, :counts[i]] = imgs
    return padded_images, torch.stack(normals_list), torch.stack(masks_list), torch.tensor(counts)