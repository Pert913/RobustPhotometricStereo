"""
Python client for TransUNetPS API.

Usage:
    from client import TransUNetPSClient

    client = TransUNetPSClient("http://localhost:8000")

    # Predict from a folder of images
    normal_map = client.predict_from_folder("./data/testing/dimooPNG")
    normal_map.save("output.png")

    # Predict from specific files
    normal_map = client.predict(["img1.png", "img2.png", "img3.png"])

    # Get raw numpy array
    normals = client.predict_numpy(["img1.png", "img2.png"])

or simple
    python client.py "./data/testing/dimooPNG"
"""
import os
import glob
import io
import zipfile
import requests
import numpy as np
from PIL import Image
import sys


class TransUNetPSClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self):
        """Check server health."""
        r = requests.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def info(self):
        """Get model info."""
        r = requests.get(f"{self.base_url}/info")
        r.raise_for_status()
        return r.json()

    def predict(self, image_paths, tile_size=128, max_images=96):
        """
        Predict normal map and return as PIL Image.

        Args:
            image_paths: list of image file paths
            tile_size: tile size for inference
            max_images: max images to use

        Returns:
            PIL.Image (RGB normal map)
        """
        files = [("images", (os.path.basename(p), open(p, "rb"), "image/png"))
                 for p in image_paths]
        r = requests.post(
            f"{self.base_url}/predict",
            files=files,
            params={"tile_size": tile_size, "max_images": max_images, "format": "png"},
        )
        for _, (_, f, _) in files:
            f.close()
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))

    def predict_numpy(self, image_paths, tile_size=128, max_images=96):
        """
        Predict normal map and return as numpy array (H, W, 3).
        """
        files = [("images", (os.path.basename(p), open(p, "rb"), "image/png"))
                 for p in image_paths]
        r = requests.post(
            f"{self.base_url}/predict",
            files=files,
            params={"tile_size": tile_size, "max_images": max_images, "format": "npy"},
        )
        for _, (_, f, _) in files:
            f.close()
        r.raise_for_status()
        return np.load(io.BytesIO(r.content))

    def predict_zip(self, image_paths, output_dir="./output", tile_size=128, max_images=96):
        """
        Predict and download all outputs as zip (normal map + components).
        """
        files = [("images", (os.path.basename(p), open(p, "rb"), "image/png"))
                 for p in image_paths]
        r = requests.post(
            f"{self.base_url}/predict",
            files=files,
            params={"tile_size": tile_size, "max_images": max_images, "format": "zip"},
        )
        for _, (_, f, _) in files:
            f.close()
        r.raise_for_status()

        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            zf.extractall(output_dir)
        print(f"Saved to {output_dir}/")
        return output_dir

    def predict_from_folder(self, folder_path, tile_size=128, max_images=96):
        """
        Predict from all images in a folder.
        Returns PIL Image (RGB normal map).
        """
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(folder_path, ext)))
        paths = sorted(paths)[:max_images]

        if not paths:
            raise FileNotFoundError(f"No images found in {folder_path}")

        print(f"Sending {len(paths)} images from {folder_path}...")
        return self.predict(paths, tile_size=tile_size, max_images=max_images)

    def predict_json(self, image_paths, tile_size=128, max_images=96):
        """Predict and return stats as JSON (no file download)."""
        files = [("images", (os.path.basename(p), open(p, "rb"), "image/png"))
                 for p in image_paths]
        r = requests.post(
            f"{self.base_url}/predict/json",
            files=files,
            params={"tile_size": tile_size, "max_images": max_images},
        )
        for _, (_, f, _) in files:
            f.close()
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    client = TransUNetPSClient()

    print("Health:", client.health())
    print("Info:", client.info())

    if len(sys.argv) > 1:
        folder = sys.argv[1]
        result = client.predict_from_folder(folder)
        result.save("predicted_normal.png")
        print("Saved predicted_normal.png")
