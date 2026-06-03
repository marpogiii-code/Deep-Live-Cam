"""Gradio web UI for Deep-Live-Cam face swapping.

Designed for Hugging Face Spaces deployment (GPU).
Also works locally: `python app.py`
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr

# Set up environment before importing modules
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import modules.globals

# Run in headless mode (no PySide6 UI needed)
modules.globals.headless = True

import modules.face_analyser as face_analyser
from modules.processors.frame.face_swapper import (
    get_face_swapper,
    swap_face,
    pre_check as swapper_pre_check,
)

# Configure for CPU/GPU based on availability
import onnxruntime

available_providers = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
elif "CoreMLExecutionProvider" in available_providers:
    modules.globals.execution_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
else:
    modules.globals.execution_providers = ["CPUExecutionProvider"]

modules.globals.execution_threads = 2
modules.globals.many_faces = False


def download_models():
    """Download required models if not present."""
    swapper_pre_check()


def process_face_swap(source_image, target_image, swap_all_faces):
    """Perform face swap between source and target images.

    Args:
        source_image: numpy array of the face to use (source)
        target_image: numpy array of the target image/face to replace
        swap_all_faces: whether to swap all detected faces

    Returns:
        Result image with swapped face(s)
    """
    if source_image is None or target_image is None:
        return None

    # Ensure models are downloaded
    download_models()

    # Convert RGB (Gradio) to BGR (OpenCV)
    source_bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

    # Detect source face
    source_face = face_analyser.get_one_face(source_bgr)
    if source_face is None:
        raise gr.Error("No face detected in the source image. Please upload an image with a clear face.")

    # Set globals for processing
    modules.globals.many_faces = swap_all_faces

    # Get face swapper
    swapper = get_face_swapper()
    if swapper is None:
        raise gr.Error("Failed to load face swapper model. Please check model files.")

    result = target_bgr.copy()

    if swap_all_faces:
        # Swap all faces in target
        target_faces = face_analyser.get_many_faces(target_bgr)
        if not target_faces:
            raise gr.Error("No faces detected in the target image.")
        for target_face in target_faces:
            result = swap_face(source_face, target_face, result)
    else:
        # Swap only the most prominent face
        target_face = face_analyser.get_one_face(target_bgr)
        if target_face is None:
            raise gr.Error("No face detected in the target image. Please upload an image with a clear face.")
        result = swap_face(source_face, target_face, result)

    # Convert BGR back to RGB for Gradio display
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result_rgb


# Build Gradio UI
with gr.Blocks(
    title="Deep-Live-Cam",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .gradio-container { max-width: 1200px !important; }
    .header { text-align: center; margin-bottom: 1rem; }
    """
) as demo:
    gr.Markdown(
        """
        # Deep-Live-Cam
        ### Real-time face swap with a single image

        Upload a **source face** (the face you want to use) and a **target image**
        (the image where you want to replace the face). Click **Swap Face** to process.
        """
    )

    with gr.Row():
        with gr.Column():
            source_input = gr.Image(label="Source Face", type="numpy")
            target_input = gr.Image(label="Target Image", type="numpy")
            swap_all = gr.Checkbox(label="Swap all faces in target", value=False)
            swap_btn = gr.Button("Swap Face", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="Result", type="numpy")

    swap_btn.click(
        fn=process_face_swap,
        inputs=[source_input, target_input, swap_all],
        outputs=output_image,
    )

    gr.Markdown(
        """
        ---
        **Note:** This app uses AI models for face swapping. Please use responsibly
        and ethically. Do not use real people's faces without their consent.

        [GitHub Repository](https://github.com/marpogiii-code/Deep-Live-Cam) |
        Built with [Gradio](https://gradio.app)
        """
    )


if __name__ == "__main__":
    download_models()
    demo.launch(server_name="0.0.0.0", server_port=7860)
