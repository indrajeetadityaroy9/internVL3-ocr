"""
InternVL-3 OCR service:
Optimized thread handling
Simplified resource management
Consolidated text output for multi-images
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
import atexit
import base64
import binascii
import concurrent.futures
import gc
import io
import json
import logging
import os
import signal
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torchvision.transforms as T
from flask import Flask, jsonify, request
from PIL import Image, ImageFile
from torchvision.transforms.functional import InterpolationMode

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
IMAGE_SIZE: int = 448
MAX_UPLOAD_BYTES: int = 16 * 1024 * 1024  # 16 MiB (post-base64)
DEBUG_MODE: bool = os.environ.get("OCR_DEBUG", "0") == "1"
MAX_THREADS: int = 4  # Maximum number of concurrent OCR tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(process)d  %(name)s ▶  %(message)s",
)
log = logging.getLogger("internvl3-ocr")

# Global variables
tasks = {}  # Task dictionary
image_store = tempfile.mkdtemp(prefix="ocr_images_")  # Temporary directory for images
task_lock = threading.Lock()  # Lock for task dictionary
shutdown_requested = False  # Shutdown flag

# Global model objects (always loaded)
model = None
tokenizer = None
model_lock = threading.Lock()  # Lock for model access during inference
thread_pool = None  # Thread pool for OCR tasks

class Task:
    """Task representation"""

    def __init__(self, task_id, image_path=None, status="pending"):
        self.task_id = task_id
        self.image_path = image_path
        self.status = status
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.updated_at = time.time()

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


def create_task(image_data: bytes) -> str:
    """Create a new OCR task and return its ID."""
    task_id = str(uuid.uuid4())

    # Save image to file
    image_path = os.path.join(image_store, f"{task_id}.bin")
    with open(image_path, "wb") as f:
        f.write(image_data)

    # Create task
    with task_lock:
        tasks[task_id] = Task(task_id, image_path)

    log.info(f"Created task {task_id}")
    return task_id


def get_task(task_id: str) -> Optional[Task]:
    with task_lock:
        return tasks.get(task_id)


def update_task(task_id: str, **kwargs) -> None:
    with task_lock:
        if task_id in tasks:
            task = tasks[task_id]
            # Update fields
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

            # Update timestamp
            task.updated_at = time.time()


def cleanup_old_tasks(max_age: float = 3600) -> None:
    current_time = time.time()
    task_ids_to_remove = []

    with task_lock:
        for task_id, task in list(tasks.items()):
            if (current_time - task.updated_at > max_age and
                    task.status in ["completed", "failed"]):
                task_ids_to_remove.append(task_id)

    # Remove tasks outside the lock to reduce lock contention
    for task_id in task_ids_to_remove:
        remove_task(task_id)


def remove_task(task_id: str) -> None:
    with task_lock:
        if task_id in tasks:
            task = tasks[task_id]

            # Remove image file
            if task.image_path and os.path.exists(task.image_path):
                try:
                    os.unlink(task.image_path)
                except:
                    pass

            # Remove task
            del tasks[task_id]

    log.info(f"Removed task {task_id}")


def init_thread_pool():
    """Initialize the thread pool for OCR tasks."""
    global thread_pool
    if thread_pool is None:
        thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=MAX_THREADS,
            thread_name_prefix="ocr_worker"
        )
        log.info(f"Initialized thread pool with {MAX_THREADS} workers")


def submit_task(task_id: str):
    global thread_pool

    # Ensure thread pool is initialized
    if thread_pool is None:
        init_thread_pool()

    # Submit task to thread pool
    future = thread_pool.submit(process_ocr_task, task_id)

    # Add callback for error handling
    future.add_done_callback(lambda f: handle_task_completion(f, task_id))

    log.info(f"Submitted task {task_id} to thread pool")
    return future


def handle_task_completion(future, task_id):
    """Handle task completion or error."""
    if future.exception():
        log.error(f"Error processing task {task_id}: {future.exception()}")
        update_task(task_id, status="failed", error=str(future.exception()))


def load_model():
    global model, tokenizer

    log.info("Loading model and tokenizer")
    start_time = time.time()

    # Import here to avoid loading unnecessarily
    from transformers import AutoModel, AutoTokenizer

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Disable multiprocessing in tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load model with optimized settings
    model = (
        AutoModel.from_pretrained(
            "OpenGVLab/InternVL3-1B",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=torch.cuda.is_available()
        )
        .to(device=device, dtype=dtype)
        .eval()
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL3-1B",
        trust_remote_code=True,
        use_fast=False
    )

    load_time = time.time() - start_time
    log.info(f"Model loaded successfully in {load_time:.2f} seconds")


def process_ocr_task(task_id: str) -> None:
    log.info(f"Processing task {task_id}")

    # Get task
    task = get_task(task_id)
    if not task:
        log.error(f"Task {task_id} not found")
        return

    # Mark task as processing
    update_task(task_id, status="processing")

    try:
        # Verify image path
        if not task.image_path or not os.path.exists(task.image_path):
            raise ValueError(f"Image file not found for task {task_id}")

        # Load image
        with open(task.image_path, "rb") as f:
            image_data = f.read()

        img_data = io.BytesIO(image_data)
        img = Image.open(img_data)
        img.load()

        # Run OCR
        text = run_ocr(img)

        # Update task
        update_task(task_id, status="completed", result=text)
        log.info(f"Completed task {task_id}")

    except Exception as e:
        log.exception(f"Failed to process task {task_id}")
        update_task(task_id, status="failed", error=str(e))


def run_ocr(img: Image.Image) -> str:
    global model, tokenizer

    # Ensure model is loaded
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    # Preprocess image
    tensor = preprocess_image(img)

    # Lock for thread safety during inference
    with model_lock:
        # Run inference
        prompt = (
            "<image>\n"
            "Please perform OCR on this image. "
            "Extract every visible text region and output it verbatim."
        )

        gen_args = dict(
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
        )

        # Run inference with torch.no_grad()
        with torch.no_grad():
            text = model.chat(tokenizer, tensor, prompt, gen_args)

    return text


def preprocess_image(img: Image.Image) -> torch.Tensor:
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Create transform
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Handle small vs. large images
    if img.width < IMAGE_SIZE * 2 and img.height < IMAGE_SIZE * 2:
        # Small image - use direct resize
        resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        tensor = transform(resized).unsqueeze(0)
        return tensor.to(device=device, dtype=dtype)
    else:
        # Larger image - use dynamic tiling
        patches = dynamic_tiling(img)
        return torch.stack([transform(p) for p in patches]).to(device=device, dtype=dtype)


def dynamic_tiling(img: Image.Image, max_tiles: int = 12, base: int = IMAGE_SIZE) -> List[Image.Image]:
    w, h = img.size
    aspect = w / h
    best_err, tiling = 1e9, (1, 1)

    # Find optimal tiling
    for rows in range(1, max_tiles + 1):
        cols = max_tiles // rows
        if cols == 0:
            continue
        cand_aspect = cols / rows
        err = abs(aspect - cand_aspect)
        if rows * cols <= max_tiles and err < best_err:
            best_err, tiling = err, (rows, cols)

    rows, cols = tiling
    tile_w, tile_h = w / cols, h / rows
    patches = []

    # Create patches
    for r in range(rows):
        for c in range(cols):
            l, u = int(c * tile_w), int(r * tile_h)
            rgt, dwn = int((c + 1) * tile_w), int((r + 1) * tile_h)
            patch = img.crop((l, u, rgt, dwn)).resize((base, base), Image.BICUBIC)
            patches.append(patch)

    return patches


def decode_base64_image(b64: str) -> bytes:
    import re

    # Handle both raw base64 and data URL formats
    if "," in b64:
        # Handle data URL format (e.g. "data:image/png;base64,...")
        prefix, b64_clean = b64.split(",", 1)
        log.info("Detected data URL format with prefix: %s", prefix)
    else:
        b64_clean = b64
        log.info("Processing raw base64 data (no prefix)")

    # Remove whitespace and auto-pad to ensure length is multiple of 4
    b64_clean = re.sub(r"\s+", "", b64_clean)
    padding_needed = -len(b64_clean) % 4
    if padding_needed:
        log.info("Adding %d padding characters", padding_needed)
        b64_clean += "=" * padding_needed

    # Decode base64 data
    try:
        raw = base64.b64decode(b64_clean, validate=True)
        log.info("Successfully decoded base64 data: %d bytes", len(raw))

        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Image exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MiB limit")

        return raw

    except binascii.Error as exc:
        log.error("Base64 decode error: %s", str(exc))
        if len(b64_clean) > 100:
            log.error("Input starts with: %s", b64_clean[:100])
        raise ValueError(f"Invalid base64 data: {exc}") from None


def get_image_from_request() -> Union[bytes, List[bytes]]:
    """Extract image(s) from request and return the raw bytes."""
    log.info("Request content type: %s", request.content_type)

    # Check for batch processing flag
    is_batch = False
    if request.is_json:
        is_batch = request.json.get("batch", False)

    # Process file upload (single or multiple)
    if 'image' in request.files:
        # Single file upload
        file = request.files['image']
        log.info("Processing file upload: %s", file.filename)
        return file.read()
    elif 'images[]' in request.files:
        # Multiple file upload
        files = request.files.getlist('images[]')
        log.info("Processing multiple file uploads: %d files", len(files))
        if is_batch:
            # Return raw bytes for each image
            return [file.read() for file in files]
        else:
            # Process first image only if not in batch mode
            log.warning("Multiple files uploaded but batch mode not enabled. Processing first file only.")
            return request.files.getlist('images[]')[0].read()

    # Process JSON with base64 data (single or batch)
    if request.is_json:
        log.info("Processing JSON request")
        if "image" in request.json:
            # Single image in JSON
            b64 = request.json["image"]
            log.info("Base64 image found in JSON (length: %d)", len(b64))
            return decode_base64_image(b64)
        elif "images" in request.json and isinstance(request.json["images"], list):
            # Multiple images in JSON
            images = request.json["images"]
            log.info("Found %d images in JSON batch", len(images))
            if is_batch:
                # Return raw base64 strings for batch processing
                return [decode_base64_image(img) for img in images]
            else:
                # Process first image only if not in batch mode
                log.warning("Multiple images in JSON but batch mode not enabled. Processing first image only.")
                return decode_base64_image(images[0])
        else:
            log.error("No 'image' or 'images' field in JSON data")
            raise ValueError("JSON request missing image data")

    # Fall back to raw body processing
    if request.data and len(request.data) > 0:
        log.info("Attempting to process raw request body (length: %d)", len(request.data))
        try:
            data = request.data.decode('utf-8')
            if data.startswith("{"):
                # Try parsing as JSON
                import json
                try:
                    json_data = json.loads(data)
                    if "image" in json_data:
                        log.info("Found image in manually parsed JSON")
                        return decode_base64_image(json_data["image"])
                except:
                    log.error("Failed to parse body as JSON")

            # Try treating whole body as base64
            return decode_base64_image(data)
        except Exception as e:
            log.error("Failed to process raw body: %s", str(e))

    log.error("No usable image found in request")
    raise ValueError("No image provided in request")


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# Background thread for maintenance
def maintenance_thread_func():
    log.info("Starting maintenance thread")

    while not shutdown_requested:
        try:
            # Clean up old tasks
            cleanup_old_tasks()

            # Sleep
            time.sleep(60.0)  # Run every minute
        except Exception as e:
            log.exception(f"Error in maintenance thread: {e}")
            time.sleep(300.0)  # Wait longer after error


# Initialize background thread
maintenance_thread_handle = None


@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    try:
        log.info("OCR request received")

        # Check if this is a batch request
        is_batch = False
        if request.is_json:
            is_batch = request.json.get("batch", False)

        # Process batch or single image
        if is_batch:
            return handle_batch_request()
        else:
            return handle_single_request()

    except Exception as exc:
        log.exception("OCR failure")
        return jsonify(success=False, error=str(exc)), 400


def handle_single_request():
    """Handle a request for a single image."""
    try:
        # Get image from request
        image_data = get_image_from_request()

        # Handle list of images (from batch processing)
        if isinstance(image_data, list):
            if len(image_data) > 0:
                image_data = image_data[0]
            else:
                raise ValueError("No images provided")

        # Create task
        task_id = create_task(image_data)

        # Start processing
        submit_task(task_id)

        # Return task ID for client to poll
        return jsonify(
            success=True,
            message="Processing image",
            task_id=task_id
        )

    except Exception as exc:
        log.exception("OCR single request failure")
        return jsonify(success=False, error=str(exc)), 400


def handle_batch_request():
    """Handle a batch request with multiple images."""
    try:
        # Get images from request
        images_data = get_image_from_request()

        if not isinstance(images_data, list):
            # Convert single image to list for batch processing
            images_data = [images_data]

        log.info("Batch request received with %d images", len(images_data))

        # Create tasks
        task_ids = []
        futures = []

        for image_data in images_data:
            task_id = create_task(image_data)
            task_ids.append(task_id)
            futures.append(submit_task(task_id))

        # Return task IDs for client to poll
        return jsonify(
            success=True,
            message=f"Processing {len(task_ids)} images",
            task_ids=task_ids
        )

    except Exception as exc:
        log.exception("Batch request failure")
        return jsonify(success=False, error=str(exc)), 400


@app.route("/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id):
    """Get the status of a specific task."""
    task = get_task(task_id)

    if not task:
        return jsonify(success=False, error="Task not found"), 404

    response = {
        "success": True,
        "task_id": task.task_id,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at
    }

    if task.status == "completed":
        response["result"] = task.result
    elif task.status == "failed":
        response["error"] = task.error

    return jsonify(response)


@app.route("/tasks/batch", methods=["POST"])
def get_batch_status():
    if not request.is_json:
        return jsonify(success=False, error="Expected JSON"), 400

    task_ids = request.json.get("task_ids", [])
    if not task_ids:
        return jsonify(success=False, error="No task IDs provided"), 400

    results = {}
    all_completed = True

    for task_id in task_ids:
        task = get_task(task_id)
        if not task:
            results[task_id] = {"status": "not_found"}
            all_completed = False
        else:
            task_result = {"status": task.status}
            if task.status == "completed":
                task_result["result"] = task.result
            elif task.status == "failed":
                task_result["error"] = task.error
            else:
                all_completed = False

            results[task_id] = task_result

    return jsonify(
        success=True,
        completed=all_completed,
        results=results
    )


@app.route("/consolidate", methods=["POST"])
def consolidate_tasks():
    if not request.is_json:
        return jsonify(success=False, error="Expected JSON"), 400

    task_ids = request.json.get("task_ids", [])
    if not task_ids:
        return jsonify(success=False, error="No task IDs provided"), 400

    # Check if all tasks are complete
    all_complete = True
    all_results = []

    for task_id in task_ids:
        task = get_task(task_id)
        if not task or task.status != "completed":
            all_complete = False
            break

        all_results.append({
            "task_id": task_id,
            "result": task.result,
            "created_at": task.created_at
        })

    if not all_complete:
        return jsonify(success=False, error="Some tasks are not yet complete"), 400

    # Sort results by creation time to maintain original order
    all_results.sort(key=lambda x: x["created_at"])

    # Combine results with clear separation
    consolidated_text = "\n\n----- NEXT SCREENSHOT -----\n\n".join(
        [item["result"] for item in all_results]
    )

    return jsonify(
        success=True,
        task_ids=task_ids,
        consolidated_text=consolidated_text
    )


@app.route("/health")
def health():
    # Get task stats
    task_count = len(tasks)
    task_status = {
        "pending": sum(1 for t in tasks.values() if t.status == "pending"),
        "processing": sum(1 for t in tasks.values() if t.status == "processing"),
        "completed": sum(1 for t in tasks.values() if t.status == "completed"),
        "failed": sum(1 for t in tasks.values() if t.status == "failed"),
    }

    # Get model status
    model_status = "not_loaded"
    if model is not None:
        model_status = "loaded"

    # Get memory stats
    memory_info = {}
    if torch.cuda.is_available():
        memory_info["cuda"] = {
            "allocated": f"{torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB",
            "cached": f"{torch.cuda.memory_reserved() / 1024 ** 2:.2f}MB",
        }

    # Get thread pool info
    thread_info = {}
    if thread_pool is not None:
        thread_info = {
            "max_workers": thread_pool._max_workers,
            "active_threads": len([t for t in threading.enumerate() if t.name.startswith("ocr_worker")]),
        }

    return jsonify(
        status="ok",
        time=datetime.now().isoformat(),
        model={
            "status": model_status,
            "always_loaded": True
        },
        tasks={
            "total": task_count,
            **task_status
        },
        memory=memory_info,
        threads=thread_info
    )

def cleanup_resources():
    """Clean up all resources."""
    global shutdown_requested, thread_pool

    if shutdown_requested:
        return

    shutdown_requested = True
    log.info("Cleaning up resources...")

    try:
        # Shutdown thread pool
        if thread_pool is not None:
            log.info("Shutting down thread pool")
            thread_pool.shutdown(wait=False)
            thread_pool = None

        # Clean up image store
        if os.path.exists(image_store):
            log.info(f"Cleaning up image store: {image_store}")
            for filename in os.listdir(image_store):
                file_path = os.path.join(image_store, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    log.error(f"Error removing file {file_path}: {e}")

            try:
                os.rmdir(image_store)
                log.info(f"Removed image store directory: {image_store}")
            except:
                pass

        log.info("Cleanup complete")

    except Exception as e:
        log.error(f"Error during cleanup: {e}")


atexit.register(cleanup_resources)


def graceful_shutdown(signum, frame):
    """Handle graceful shutdown."""
    log.info(f"Graceful shutdown initiated from signal {signum}")

    # Clean up resources
    cleanup_resources()

    # Wait a moment for cleanup to complete
    time.sleep(1.0)

    # Exit
    os._exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

def main(host: str, port: int, debug: bool = False, threads: int = MAX_THREADS):
    """Start the OCR service."""
    global DEBUG_MODE, MAX_THREADS, maintenance_thread_handle

    # Update global settings
    DEBUG_MODE = debug
    MAX_THREADS = threads

    if debug:
        log.setLevel(logging.DEBUG)
        log.info("Debug mode enabled")

    # Disable multiprocessing in transformers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create image store if it doesn't exist
    if not os.path.exists(image_store):
        os.makedirs(image_store)

    # Load model at startup and keep it loaded
    load_model()

    # Initialize thread pool
    init_thread_pool()

    # Start maintenance thread
    maintenance_thread_handle = threading.Thread(
        target=maintenance_thread_func,
        daemon=True,
        name="maintenance_thread"
    )
    maintenance_thread_handle.start()

    try:
        # Start Flask app
        log.info(f"Starting OCR service on {host}:{port}")
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        log.info("Received KeyboardInterrupt in main thread")
        graceful_shutdown(signal.SIGINT, None)
    except Exception as e:
        log.exception(f"Error in main thread: {e}")
        graceful_shutdown(signal.SIGTERM, None)
    finally:
        # Make sure cleanup is called
        cleanup_resources()


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser("Simplified OCR service with always-loaded model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5001, type=int)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--threads", type=int, default=MAX_THREADS, help="Number of worker threads")
    args = parser.parse_args()

    main(args.host, args.port, args.debug, args.threads)
