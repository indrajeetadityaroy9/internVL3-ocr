"""
InternVL-3 OCR microservice
single, reusable torchvision transform
strict input-size guard & JSON safety
unified device / dtype placement (float16 on CUDA, float32 on CPU)
inference wrapped in `torch.no_grad()`
autograd & Flash-Attn friendly flags
16 MiB DoS-protection limit (base-64-inflated)
ready for gunicorn --preload (model fork-safe)
"""

from __future__ import annotations
import argparse
import base64
import binascii
import io
import logging
import os
from typing import List, Optional, Tuple, Dict, Any, Union
from PIL import features
import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError, ImageFile, TiffImagePlugin, PngImagePlugin, JpegImagePlugin
from flask import Flask, jsonify, request
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import json
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGE_SIZE: int = 448
MAX_UPLOAD_BYTES: int = 16 * 1024 * 1024  # 16 MiB (post-base64)
DEBUG_MODE: bool = os.environ.get("OCR_DEBUG", "0") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

torch.backends.cuda.matmul.allow_tf32 = True  # Ampere/Hopper speedup
torch.set_grad_enabled(False)  # global autograd off

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s ▶  %(message)s",
)
log = logging.getLogger("internvl3-ocr")

TRANSFORM: T.Compose = T.Compose(
    [
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def _dynamic_tiling(img: Image.Image, *, max_tiles: int = 12, base: int = IMAGE_SIZE):
    w, h = img.size
    aspect = w / h
    best_err, tiling = 1e9, (1, 1)

    for rows in range(1, max_tiles + 1):
        cols = max_tiles // rows
        cand_aspect = cols / rows
        err = abs(aspect - cand_aspect)
        if rows * cols <= max_tiles and err < best_err:
            best_err, tiling = err, (rows, cols)

    rows, cols = tiling
    tile_w, tile_h = w / cols, h / rows
    patches: List[Image.Image] = []

    for r in range(rows):
        for c in range(cols):
            l, u = int(c * tile_w), int(r * tile_h)
            rgt, dwn = int((c + 1) * tile_w), int((r + 1) * tile_h)
            patch = img.crop((l, u, rgt, dwn)).resize((base, base), Image.BICUBIC)
            patches.append(patch)

    return patches


def preprocess(img: Image.Image) -> torch.Tensor:
    log.info("Processing image: format=%s, size=%dx%d, mode=%s",
             getattr(img, 'format', 'Unknown'), img.width, img.height, img.mode)

    if img.width < IMAGE_SIZE * 2 and img.height < IMAGE_SIZE * 2:
        log.info("Small image detected - using direct resize")
        resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        tensor = TRANSFORM(resized).unsqueeze(0)
        return tensor.to(device=device, dtype=dtype)

    log.info("Using dynamic tiling for larger image")
    patches = _dynamic_tiling(img)
    log.info("Created %d image patches", len(patches))
    return torch.stack([TRANSFORM(p) for p in patches]).to(device=device, dtype=dtype)



model = tokenizer = None

def load_model(model_id: str = "OpenGVLab/InternVL3-1B") -> None:
    global model, tokenizer
    if model is not None:
        return

    log.info("Loading %s (device=%s dtype=%s)…", model_id, device, dtype)
    model = (
        AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    log.info("Model ready")


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

PROMPT = (
    "<image>\n"
    "Please perform OCR on this image. "
    "Extract every visible text region and output it verbatim."
)

GEN_ARGS = dict(
    max_new_tokens=1024,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    num_beams=1,
)


def _decode_base64_image(b64: str) -> Image.Image:

    if "," in b64:
        prefix, b64_clean = b64.split(",", 1)
        log.info("Detected data URL format with prefix: %s", prefix)
    else:
        b64_clean = b64
        log.info("Processing raw base64 data (no prefix)")

    b64_clean = re.sub(r"\s+", "", b64_clean)
    padding_needed = -len(b64_clean) % 4
    if padding_needed:
        log.info("Adding %d padding characters", padding_needed)
        b64_clean += "=" * padding_needed

    try:
        raw = base64.b64decode(b64_clean, validate=True)
        log.info("Successfully decoded base64 data: %d bytes", len(raw))
    except binascii.Error as exc:
        log.error("Base64 decode error: %s", str(exc))
        if len(b64_clean) > 100:
            log.error("Input starts with: %s", b64_clean[:100])
        raise ValueError(f"Invalid base64 data: {exc}") from None

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Image exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MiB limit")

    img_data = io.BytesIO(raw)

    try:
        # Attempt to determine the format based on magic bytes
        img_data.seek(0)
        header = img_data.read(16)
        img_data.seek(0)

        # Log the file signature for debugging
        log.info("Image header bytes: %s", header.hex()[:24])

        # Try to open the image
        img = Image.open(img_data)

        # Force loading the image data to catch potential errors early
        img.load()
        log.info("Successfully loaded image: format=%s, size=%dx%d, mode=%s",
                 getattr(img, "format", "Unknown"), img.width, img.height, img.mode)

        return img.convert("RGB")
    except Exception as e:
        log.error("Image decode error: %s", str(e))
        log.error("Image header: %s", header.hex()[:24])
        log.error("Image size: %d bytes", len(raw))

        # If in debug mode, save the problematic image for inspection
        if DEBUG_MODE:
            with open("debug_image_error.bin", "wb") as f:
                f.write(raw)
            log.info("Saved problematic image to debug_image_error.bin")

        raise ValueError(
            f"Could not decode image data ({str(e)}). "
            "Ensure the image is a valid PNG, JPEG, or TIFF."
        ) from None


def _get_image_from_request() -> Image.Image:
    log.info("Request content type: %s", request.content_type)

    # Process file upload
    if file := request.files.get("image"):
        log.info("Processing file upload: %s", file.filename)
        return Image.open(file.stream)

    # Process JSON with base64 data
    if request.is_json:
        log.info("Processing JSON request")
        if b64 := request.json.get("image"):
            log.info("Base64 image found in JSON (length: %d)", len(b64))
            return _decode_base64_image(b64)
        else:
            log.error("No 'image' field in JSON data")
            raise ValueError("JSON request missing 'image' field")

    if request.data and len(request.data) > 0:
        log.info("Attempting to process raw request body (length: %d)", len(request.data))
        try:
            data = request.data.decode('utf-8')
            if data.startswith("{"):
                try:
                    json_data = json.loads(data)
                    if "image" in json_data:
                        log.info("Found image in manually parsed JSON")
                        return _decode_base64_image(json_data["image"])
                except:
                    log.error("Failed to parse body as JSON")

            return _decode_base64_image(data)
        except Exception as e:
            log.error("Failed to process raw body: %s", str(e))

    log.error("No usable image found in request")
    raise ValueError("No image provided in request")


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    try:
        log.info("OCR request received")
        load_model()  # lazy load once

        img = _get_image_from_request()
        log.info("Image received: format=%s, size=%dx%d, mode=%s",
                 getattr(img, 'format', 'Unknown'), img.width, img.height, img.mode)

        tensor = preprocess(img)
        log.info("Image preprocessed: tensor shape=%s", tensor.shape)

        with torch.no_grad():
            log.info("Running OCR inference")
            text: str = model.chat(tokenizer, tensor, PROMPT, GEN_ARGS)
            log.info("OCR extraction complete: %d chars extracted", len(text))

        return jsonify(success=True, text=text)

    except Exception as exc:
        log.exception("OCR failure")
        return jsonify(success=False, error=str(exc)), 400


@app.route("/debug/image", methods=["POST"])
def debug_image():
    try:
        log.info("Debug image request received")
        img = _get_image_from_request()

        if DEBUG_MODE:
            debug_path = "debug_image.png"
            img.save(debug_path)
            log.info("Saved debug image to %s", debug_path)

        return jsonify({
            "success": True,
            "format": getattr(img, "format", "Unknown"),
            "size": {"width": img.width, "height": img.height},
            "mode": img.mode,
            "info": str(getattr(img, "info", {}))
        })
    except Exception as exc:
        log.exception("Debug image failure")
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/health")
def health():
    try:
        pil_version = Image.__version__
        model_status = "loaded" if model is not None else "not loaded"
        supported_formats = {
            "JPEG": features.check("jpg"),
            "PNG": features.check("png"),
            "TIFF": features.check("tiff"),
            "WEBP": features.check("webp"),
        }

        return jsonify(
            status="ok",
            device=str(device),
            dtype=str(dtype),
            pil_version=pil_version,
            model_status=model_status,
            supported_formats=supported_formats,
            debug_mode=DEBUG_MODE
        )
    except Exception as e:
        return jsonify(
            status="warning",
            error=str(e),
            device=str(device),
            dtype=str(dtype)
        )

def main(host: str, port: int, model_id: str, debug: bool = False):
    global DEBUG_MODE
    DEBUG_MODE = debug

    if debug:
        log.setLevel(logging.DEBUG)
        app.config['DEBUG'] = True
        log.info("Debug mode enabled")

    load_model(model_id)
    log.info("Starting OCR service on %s:%d", host, port)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("InternVL-3 OCR service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5001, type=int)
    parser.add_argument("--model", default="OpenGVLab/InternVL3-1B")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    main(args.host, args.port, args.model, args.debug)
