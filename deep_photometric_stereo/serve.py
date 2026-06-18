"""
REST API server for TransUNetPS normal map prediction.

Usage:
    # Start server
    pip install fastapi uvicorn python-multipart tifffile
    python3 serve.py --checkpoint checkpoints/best_trans.pt --port 8000

    # Test with curl
    curl -X POST http://localhost:8000/predict \
        -F "images=@image1.png" -F "images=@image2.png" \
        --output result.zip

    # Or use the Python client (see below)
"""
import argparse
import io
import os
import zipfile
from datetime import datetime
import numpy as np
import tifffile
import torch
from PIL import Image, ImageOps

from config import Config, ModelConfig
from model import get_model
from utils import load_checkpoint, detect_model_type, count_parameters, normal_to_rgb
from predict import predict_at_native_scale, _srgb_to_linear
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware


def _decode_image_bytes(data: bytes, filename: str) -> np.ndarray:
    """Load any uploaded image to float32 RGB in [0,1], in LINEAR space.

    Handles 16-bit linear TIFF (from the Android Camera2 RAW path) via tifffile
    so we keep full sensor precision; falls back to PIL for JPEG/PNG.

    JPEG/PNG are gamma-encoded (sRGB), so they are converted to linear to match
    the training distribution — the same sRGB->linear step the predict.py CLI
    applies. RAW TIFF is already linear and is left untouched.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(io.BytesIO(data))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0  # already linear
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return _srgb_to_linear(arr)  # gamma-encoded JPEG/PNG -> linear


_model = None
_device = None
_save_dir = None   # set at startup; None = no saving
_no_pred_mask = False  # set at startup; True = Otsu/dark-ref only, skip model seg head


def _save_request(raw_image_bytes_list: list[bytes], filenames: list[str], pred_normal: np.ndarray):
    """Persist input image(s) and predicted normal map to a timestamped subfolder."""
    if _save_dir is None:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # YYYYMMDD_HHMMSS_mmm
    folder = os.path.join(_save_dir, ts)
    os.makedirs(folder, exist_ok=True)

    # Save each raw input image with its original extension preserved
    for i, (data, fname) in enumerate(zip(raw_image_bytes_list, filenames)):
        ext = os.path.splitext(fname)[1].lower() or ".jpg"
        out_path = os.path.join(folder, f"input_{i:02d}{ext}")
        with open(out_path, "wb") as f:
            f.write(data)

    # Save normal map as PNG
    from utils import normal_to_rgb
    Image.fromarray(normal_to_rgb(pred_normal)).save(os.path.join(folder, "predicted_normal.png"))
    # Save raw numpy array for offline analysis
    np.save(os.path.join(folder, "predicted_normal.npy"), pred_normal)
    print(f"[save] {folder}")

def load_model(checkpoint_path, device_str="auto"):
    """Load model once at startup."""
    global _model, _device

    config = Config(device=device_str)
    _device = config.resolve_device()

    mt = detect_model_type(checkpoint_path)
    print(f"Model type: {mt}")
    _model = get_model(config.model, model_type=mt).to(_device)
    epoch, val_loss = load_checkpoint(checkpoint_path, _model)
    _model.eval()
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")
    print(f"Parameters: {count_parameters(_model):,}")
    print(f"Device: {_device}")


def predict_from_images(image_arrays, dark_array=None, tile_size=512, max_images=96,
                        resize_max=1024, no_pred_mask=None):
    # image_arrays: list of (H, W, 3) linear RGB. A single RGB upload becomes a
    # (1, H, W, 3) stack which predict_normal unpacks into 3 PS channels.
    # no_pred_mask defaults to the server-wide --mask choice set at startup.
    if no_pred_mask is None:
        no_pred_mask = _no_pred_mask
    images = np.stack(image_arrays, axis=0)
    pred = predict_at_native_scale(
        _model, images, _device,
        dark_frame=dark_array,
        tile_size=tile_size, no_pred_mask=no_pred_mask, resize_max=resize_max,
    )
    return pred


def create_app():
    app = FastAPI(
        title="TransUNetPS API",
        description="Predict surface normal maps from multiple images under unknown lighting",
        version="1.0.0",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000"],
        allow_credentials=True, 
        allow_methods=["*"],
        allow_headers=["*"]
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": _model is not None,
            "device": str(_device),
        }

    @app.get("/info")
    def info():
        if _model is None:
            return {"error": "Model not loaded"}
        return {
            "model_type": type(_model).__name__,
            "parameters": count_parameters(_model),
            "device": str(_device),
        }

    @app.post("/predict")
    async def predict(
        images: list[UploadFile] = File(..., description="Multiple grayscale images of the same object"),
        tile_size: int = Query(512, description="Tile size for inference"),
        max_images: int = Query(96, description="Max images to use"),
        format: str = Query("png", description="Output format: png, npy, or zip (all outputs)"),
    ):
        """
        Predict normal map from uploaded images.

        Send multiple images (different lighting) of the same object.
        Returns the predicted surface normal map.
        """
        if _model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded"})

        # if len(images) < 2:
        #     return JSONResponse(status_code=400, content={"error": "Need at least 2 images"})

        # Load images (keep raw bytes for saving).
        # Files whose name starts with "dark" (case-insensitive) are treated as the
        # all-LEDs-off reference frame used to build a robust foreground mask.
        raw_bytes, filenames, image_arrays = [], [], []
        dark_array = None
        for img_file in images:
            data = await img_file.read()
            fname = img_file.filename or "image.jpg"
            raw_bytes.append(data)
            filenames.append(fname)
            arr = _decode_image_bytes(data, fname)
            if os.path.basename(fname).lower().startswith("dark") and dark_array is None:
                dark_array = arr
            else:
                image_arrays.append(arr)

        # Check all images (lit + dark) are same size
        all_shapes = set(a.shape for a in image_arrays)
        if dark_array is not None:
            all_shapes.add(dark_array.shape)
        if len(all_shapes) > 1:
            return JSONResponse(status_code=400, content={
                "error": f"All images must be the same size. Got: {all_shapes}"
            })

        print(f"Received {len(image_arrays)} lit + {0 if dark_array is None else 1} dark, "
              f"shape={(image_arrays[0] if image_arrays else dark_array).shape}")

        # Predict
        pred_normal = predict_from_images(
            image_arrays, dark_array=dark_array,
            tile_size=tile_size, max_images=max_images,
        )
        H, W, _ = pred_normal.shape

        # Persist input + output to disk
        _save_request(raw_bytes, filenames, pred_normal)

        if format == "npy":
            # Return raw .npy
            buf = io.BytesIO()
            np.save(buf, pred_normal)
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/octet-stream",
                headers={"Content-Disposition": "attachment; filename=predicted_normal.npy"},
            )

        elif format == "png":
            # Return RGB normal map PNG
            rgb = normal_to_rgb(pred_normal)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="image/png",
                headers={"Content-Disposition": "attachment; filename=predicted_normal.png"},
            )

        elif format == "zip":
            # Return zip with all outputs
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # Normal map PNG
                rgb = normal_to_rgb(pred_normal)
                img_buf = io.BytesIO()
                Image.fromarray(rgb).save(img_buf, format="PNG")
                zf.writestr("predicted_normal.png", img_buf.getvalue())

                # Normal map NPY
                npy_buf = io.BytesIO()
                np.save(npy_buf, pred_normal)
                zf.writestr("predicted_normal.npy", npy_buf.getvalue())

                # Per-component PNGs
                for i, name in enumerate(["nx", "ny", "nz"]):
                    comp = pred_normal[:, :, i]
                    comp_vis = ((comp + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
                    comp_buf = io.BytesIO()
                    Image.fromarray(comp_vis).save(comp_buf, format="PNG")
                    zf.writestr(f"predicted_{name}.png", comp_buf.getvalue())

            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=prediction_result.zip"},
            )

        else:
            return JSONResponse(status_code=400, content={"error": f"Unknown format: {format}"})

    @app.post("/predict/json")
    async def predict_json(
        images: list[UploadFile] = File(...),
        tile_size: int = Query(512),
        max_images: int = Query(96),
    ):
        if _model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded"})

        image_arrays = []
        dark_array = None
        for img_file in images:
            data = await img_file.read()
            fname = img_file.filename or "image.jpg"
            arr = _decode_image_bytes(data, fname)
            if os.path.basename(fname).lower().startswith("dark") and dark_array is None:
                dark_array = arr
            else:
                image_arrays.append(arr)

        all_shapes = set(a.shape for a in image_arrays)
        if dark_array is not None:
            all_shapes.add(dark_array.shape)
        if len(all_shapes) > 1:
            return JSONResponse(status_code=400, content={"error": "All images must be same size"})

        pred_normal = predict_from_images(
            image_arrays, dark_array=dark_array,
            tile_size=tile_size, max_images=max_images,
        )
        H, W, _ = pred_normal.shape

        # Compute stats
        norms = np.linalg.norm(pred_normal, axis=-1)

        return {
            "shape": [H, W, 3],
            "num_images_used": len(image_arrays),
            "normal_mean": pred_normal.mean(axis=(0, 1)).tolist(),
            "normal_std": pred_normal.std(axis=(0, 1)).tolist(),
            "norm_mean": float(norms.mean()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
        }

    return app

def main():
    parser = argparse.ArgumentParser(description="Serve TransUNetPS as REST API")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/train/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="./captures",
                        help="Directory to save each request's input and output (default: ./captures). Pass empty string to disable.")
    parser.add_argument("--mask", type=str, default="model", choices=["model", "otsu"],
                        help="Foreground masking method. 'model' (default): Otsu/dark-ref seed "
                             "grown through the model's segmentation head. 'otsu': Otsu/dark-ref "
                             "brightness seed only, no segmentation head (use for legacy checkpoints "
                             "whose seg head is unreliable).")
    args = parser.parse_args()

    global _save_dir, _no_pred_mask
    _no_pred_mask = (args.mask == "otsu")
    print(f"Masking: {args.mask}"
          + (" (Otsu/dark-ref seed only)" if _no_pred_mask else " (Otsu/dark-ref seed + seg-head grow)"))
    if args.save_dir:
        _save_dir = args.save_dir
        os.makedirs(_save_dir, exist_ok=True)
        print(f"Saving requests to: {os.path.abspath(_save_dir)}")
    else:
        _save_dir = None

    # Load model
    load_model(args.checkpoint, device_str=args.device)

    # Start server
    import uvicorn
    app = create_app()
    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"API docs at http://localhost:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
