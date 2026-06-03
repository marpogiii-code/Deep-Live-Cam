---
title: Deep-Live-Cam
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: agpl-3.0
hardware: t4-small
---

# Deep-Live-Cam

Real-time face swap and video deepfake with a single click and only a single image.

## How to Use

1. Upload a **source face** image (the face you want to use)
2. Upload a **target image** (the image where the face will be replaced)
3. Optionally check "Swap all faces" to replace every face in the target
4. Click **Swap Face**

## Deployment Instructions

To deploy this as a Hugging Face Space:

1. Create a new Space at https://huggingface.co/new-space
2. Choose **Gradio** as the SDK
3. Select **T4 small** (or higher) for GPU hardware
4. Link this GitHub repository, or copy the files
5. Rename this file to `README.md` (replacing the project README) in your Space repo

The Space needs these files from this repository:
- `app.py` — Gradio web interface
- `modules/` — Core processing modules
- `models/` — Will be auto-downloaded on first run

## Requirements

The Space will use `requirements.txt` from the repo root to install dependencies.
GPU hardware (T4 or better) is required for reasonable inference speed.
