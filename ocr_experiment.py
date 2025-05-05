"""
Robust OCR service using Qwen2-VL-OCR-2B-Instruct:
- Follows official implementation patterns
- Uses qwen-vl-utils for proper image processing
- Includes proper exception handling and resource management

Installation requirements:
pip install transformers torch pillow flask
pip install qwen-vl-utils

Run (dev):
  python ocr_service.py --host 0.0.0.0 --port 5001

Run (prod):
  gunicorn -w 1 -k gevent --preload ocr_service:app --bind 0.0.0.0:5001
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
from flask import Flask, jsonify, request
from PIL import Image, ImageFile

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ────────────────────────────
# 1. Global configuration
# ────────────────────────────
MAX_UPLOAD_BYTES: int = 16 * 1024 * 1024  # 16 MiB (post-base64)
DEBUG_MODE: bool = os.environ.get("OCR_DEBUG", "0") == "1"
MAX_THREADS: int = 4  # Maximum number of concurrent OCR tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(process)d  %(name)s ▶  %(message)s",
)
log = logging.getLogger("qwen-ocr")

# Global variables
tasks = {}  # Task dictionary
image_store = tempfile.mkdtemp(prefix="ocr_images_")  # Temporary directory for images
task_lock = threading.Lock()  # Lock for task dictionary
shutdown_requested = False  # Shutdown flag

# Global model objects (always loaded)
model = None
processor = None
model_lock = threading.Lock()  # Lock for model access during inference
thread_pool = None  # Thread pool for OCR tasks


# ────────────────────────────
# 2. Task Management
# ────────────────────────────
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
    """Get a task by its ID."""
    with task_lock:
        return tasks.get(task_id)


def update_task(task_id: str, **kwargs) -> None:
    """Update task properties."""
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
    """Remove completed tasks older than max_age seconds."""
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
    """Remove a task and its associated data."""
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
    """Submit a task to the thread pool."""
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


# ────────────────────────────
# 3. Model Loading & OCR
# ────────────────────────────
def load_model():
    """Load the Qwen2-VL-OCR-2B-Instruct model following the official pattern."""
    global model, processor

    log.info("Loading Qwen2-VL-OCR-2B-Instruct model and processor")
    start_time = time.time()

    try:
        # Import required modules
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # Check if qwen_vl_utils is available
        try:
            import qwen_vl_utils
            log.info("qwen_vl_utils package found")
        except ImportError:
            log.warning("qwen_vl_utils not found. Please install with: pip install qwen-vl-utils")

        # Set device - use CPU for better memory management if needed
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8 * 1024 ** 3:
            device_map = "auto"  # Use CUDA when available with sufficient memory
            dtype = torch.float16
            log.info("Using CUDA with auto device mapping")
        else:
            device_map = "cpu"  # Force CPU for limited memory systems
            dtype = torch.float32
            log.info("Using CPU device mapping")

        # Load processor first
        processor = AutoProcessor.from_pretrained(
            "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
            use_fast=False
        )

        # Load model with appropriate settings
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

        # Set model to evaluation mode
        model.eval()

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        load_time = time.time() - start_time
        log.info(f"Model loaded successfully in {load_time:.2f} seconds")

    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise


def process_ocr_task(task_id: str) -> None:
    """Process a single OCR task."""
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
    """Run OCR on an image using the Qwen2-VL-OCR-2B-Instruct model."""
    global model, processor

    # Ensure model is loaded
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")

    # Import qwen_vl_utils for processing
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise RuntimeError(
            "Required package 'qwen_vl_utils' not found. Please install it with: pip install qwen-vl-utils")

    # Lock for thread safety during inference
    with model_lock:
        try:
            # Create a message with the image for OCR
            prompt = (
                "Extract text from this programming problem screenshot. "
                "Ignore images and only focus on text. "
                "Focus specifically on extracting: "
                "1. The problem statement/description "
                "2. Any example inputs and outputs "
                "3. The exact function signature or method declaration "
                "4. Any constraints or requirements. "
                "Output the raw text only without any commentary."
            )

            # Format message following Qwen's pattern
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img  # Pass PIL image directly
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Process the message following the official pattern
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process image data
            image_inputs, video_inputs = process_vision_info(messages)

            # Prepare inputs for the model
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to the same device as model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Run inference with torch.no_grad() to save memory
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1
                )

                # Trim the input ids to get only the generated part
                # Fix: Access input_ids as a dictionary key, not an attribute
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]

                # Decode the generated text
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True
                )[0]  # Get first result

            # Clean up the text
            output_text = output_text.strip()

            return output_text

        except Exception as e:
            log.exception(f"Error during OCR inference: {e}")
            raise
        finally:
            # Force garbage collection after each inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ────────────────────────────
# 4. Image Processing
# ────────────────────────────
def decode_base64_image(b64: str) -> bytes:
    """Decode base64-encoded image data to bytes."""
    import re

    # Handle both raw base64 and data URL formats
    if "," in b64:
        # Handle data URL format (e.g. "data:image/png;base64,...")
        prefix, b64_clean = b64.split(",", 1)
        log.info("Detected data URL format with prefix: %s", prefix)
    else:
        # Handle raw base64 without prefix (what the macOS app sends)
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


# ────────────────────────────
# 5. Flask App
# ────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# Background thread for maintenance
def maintenance_thread_func():
    """Background thread for cleanup and maintenance."""
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
    """Handle OCR requests."""
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
    """Get the status of multiple tasks."""
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
    """Consolidate text from multiple tasks into a single result."""
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
    """Health check endpoint with enhanced status information."""
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
            "name": "Qwen2-VL-OCR-2B-Instruct",
            "always_loaded": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        tasks={
            "total": task_count,
            **task_status
        },
        memory=memory_info,
        threads=thread_info
    )


# ────────────────────────────
# 6. Cleanup
# ────────────────────────────
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


# ────────────────────────────
# 7. Shutdown handler
# ────────────────────────────
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


# ────────────────────────────
# 8. Entrypoint
# ────────────────────────────
def main(host: str, port: int, debug: bool = False, threads: int = MAX_THREADS):
    """Start the OCR service."""
    global DEBUG_MODE, MAX_THREADS, maintenance_thread_handle

    # Update global settings
    DEBUG_MODE = debug
    MAX_THREADS = threads

    if debug:
        log.setLevel(logging.DEBUG)
        log.info("Debug mode enabled")

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
    parser = argparse.ArgumentParser("OCR service with Qwen2-VL-OCR-2B-Instruct")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--port", default=5001, type=int, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--threads", type=int, default=MAX_THREADS, help="Number of worker threads")
    args = parser.parse_args()

    main(args.host, args.port, args.debug, args.threads)
