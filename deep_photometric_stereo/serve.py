"""
REST API server for TransUNetPS normal map prediction.

Usage:
    # Start server
    pip install fastapi uvicorn python-multipart
    python serve.py --checkpoint checkpoints/train/best.pt --port 8000

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
import numpy as np
import torch
from PIL import Image

from config import Config, ModelConfig
from model import get_model
from utils import load_checkpoint, detect_model_type, count_parameters, normal_to_rgb
from predict import predict_normal, normalize_images
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse


_model = None
_device = None


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


def predict_from_images(image_arrays, tile_size=128, max_images=96):
    images = np.stack(image_arrays, axis=0)
    images = normalize_images(images)
    pred = predict_normal(_model, images, _device, tile_size=tile_size, max_images=max_images)
    return pred


def create_app():
    app = FastAPI(
        title="TransUNetPS API",
        description="Predict surface normal maps from multiple images under unknown lighting",
        version="1.0.0",
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
        tile_size: int = Query(128, description="Tile size for inference"),
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

        if len(images) < 2:
            return JSONResponse(status_code=400, content={"error": "Need at least 2 images"})

        # Load images
        image_arrays = []
        for img_file in images:
            data = await img_file.read()
            img = Image.open(io.BytesIO(data)).convert("L")
            arr = np.array(img).astype(np.float32) / 255.0
            image_arrays.append(arr)

        # Check all images are same size
        shapes = set(a.shape for a in image_arrays)
        if len(shapes) > 1:
            return JSONResponse(status_code=400, content={
                "error": f"All images must be the same size. Got: {shapes}"
            })

        print(f"Received {len(image_arrays)} images, shape={image_arrays[0].shape}")

        # Predict
        pred_normal = predict_from_images(image_arrays, tile_size=tile_size, max_images=max_images)
        H, W, _ = pred_normal.shape

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
        tile_size: int = Query(128),
        max_images: int = Query(96),
    ):
        if _model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded"})

        image_arrays = []
        for img_file in images:
            data = await img_file.read()
            img = Image.open(io.BytesIO(data)).convert("L")
            arr = np.array(img).astype(np.float32) / 255.0
            image_arrays.append(arr)

        shapes = set(a.shape for a in image_arrays)
        if len(shapes) > 1:
            return JSONResponse(status_code=400, content={"error": "All images must be same size"})

        pred_normal = predict_from_images(image_arrays, tile_size=tile_size, max_images=max_images)
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
    args = parser.parse_args()

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
