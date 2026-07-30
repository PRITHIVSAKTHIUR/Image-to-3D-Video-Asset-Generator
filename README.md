# **Image-to-3D-Video-Asset-Generator**

Image-to-3D-Video-Asset-Generator is an all-in-one generative 3D pipeline that transitions smoothly from textual concepts or reference images into fully realized 3D mesh assets (`.glb`), dynamic camera movements in 5-second MP4 videos, and clean bundle exports (`.zip`).

Powered by **TRELLIS.2-4B** for high-fidelity 3D structural reconstruction and **Z-Image-Turbo** for rapid initial concept generation, the application includes custom PBR baking, mesh decimation, Rerun 3D visualization, and automated offscreen video rendering via PyRender (with a Matplotlib fallback).

### **Key Features**

* **Cascaded Generative Pipeline:** Unifies Text-to-Image (`Z-Image-Turbo`), Image-to-3D (`TRELLIS.2-4B`), 3D-to-Video camera moves, and single-click ZIP asset bundling.
* **Advanced 3D Mesh Processing:** Converts raw latents into decimation-targeted `.glb` models with custom PBR texture layouts and flex-GEMM autotuned acceleration.
* **Interactive Rerun 3D Viewer:** Renders spatial 3D views, bounding boxes, and orientation vectors directly within the web workspace.
* **Customizable Camera Video Rendering:** Offers 7 camera movement styles (*Orbit*, *Zoom In*, *Zoom Out*, *Turntable*, *Spiral*, *Top Sweep*, and *Cinematic*) with custom background and polygon material Overrides.
* **Session-Isolated Asset Export:** Packages the final background-removed image, GLB mesh, MP4 preview, and metadata into an organized ZIP archive.

---

### **Repository Structure**

```text
├── assets/
│   ├── app/
│   └── example_image/
├── example-images/
├── trellis2/
├── app.py
├── autotune_cache.json
├── LICENSE.txt
├── packages.txt
├── pre-requirements.txt
├── README.md
└── requirements.txt

```

### **Installation and Requirements**

To configure the Image-to-3D-Video-Asset-Generator suite locally, set up a **Python 3.12** environment with the exact CUDA 13.0 compiled dependencies listed below. A modern, high-performance CUDA-enabled GPU is strictly required to execute the models and compiled C++ extensions.

* **Python Version:** Python **3.12** is strictly required and recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for compatibility with pre-built wheel binaries.
* **CUDA Version:** **CUDA 13.0** is required (`--extra-index-url https://download.pytorch.org/whl/cu130`), matching the live Hugging Face Space environment.

#### **Standard PIP Installation**

**1. Install System Packages**
Install the necessary EGL and X11 rendering libraries specified in `packages.txt`:

```bash
sudo apt-get update && sudo apt-get install -y \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon0 \
    libdbus-1-3

```

**2. Upgrade Package Manager**
Ensure your local package manager is updated:

```bash
pip install "pip>=23.0.0"

```

**3. Install Core Dependencies**
Install PyTorch 2.11.0 + CUDA 13.0 wheels, custom compiled RTX Pro 6000 binaries (`cumesh`, `flex_gemm`, `nvdiffrast`, `nvdiffrec_render`, `o_voxel`), and secondary libraries from `requirements.txt`:

```bash
pip install -r requirements.txt

```

### **Core Requirements (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
-f https://whl.natten.org

torch==2.11.0
torchvision==0.26.0
triton==3.6.0

flash_attn @ https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator/resolve/main/flash_attention_2/flash_attn-2.8.3%2Bcu13torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
cumesh @ https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator/resolve/main/rtx_pro_6000_prebuild_wheels/cumesh-0.0.1%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
flex_gemm @ https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator/resolve/main/rtx_pro_6000_prebuild_wheels/flex_gemm-1.0.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
nvdiffrast @ https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator/resolve/main/rtx_pro_6000_prebuild_wheels/nvdiffrast-0.4.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
nvdiffrec_render @ https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator/resolve/main/rtx_pro_6000_prebuild_wheels/nvdiffrec_render-0.0.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl
o_voxel @ https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator/resolve/main/rtx_pro_6000_prebuild_wheels/o_voxel-0.0.1%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl

natten==0.21.6+torch2110cu130

pillow==12.0.0
imageio==2.37.2
imageio-ffmpeg==0.6.0
opencv-python-headless==4.12.0.88
trimesh==4.10.1
plyfile==1.1.3
pyrender

transformers==4.57.3
diffusers==0.37.1
accelerate==1.13.0
huggingface-hub
spaces
gradio==6.14.0
gradio_rerun

timm==1.0.22
kornia==0.8.2
einops==0.8.2
jaxtyping
monopriors

scipy
jax
zstandard==0.25.0
tqdm==4.67.1
easydict==1.13
omegaconf
termcolor
pyserde
braceexpand
rerun-sdk

git+https://github.com/microsoft/MoGe.git

```

---

### **Usage**

Once dependencies are verified, launch the application by running the main module script:

```bash
python app.py

```

The script will configure offscreen EGL bindings, autotune GEMM kernels, and load the pipeline weights before starting a local Gradio interface (typically at `http://127.0.0.1:7860/`).

1. **Step 1 — Concept Input:** Use the **Text-to-Image-3D** tab to generate a new 2D base image from a prompt (e.g., *"A Plane 3D model"*), or switch to **Image-to-3D** and upload a custom image file directly.
2. **Step 2 — 3D Reconstruction:** Configure resolution ($512$, $1024$, or $1536$), target face decimation count, and texture size parameters, then click **2. Generate 3D**. Inspect the generated GLB model using the embedded Rerun 3D viewer.
3. **Step 3 — Video Rendering:** Select camera movement style, rotation speed, background color, and output resolution, then click **4. Generate Video (5 sec)**.
4. **Step 4 — Asset Packaging:** Click **6. Generate ZIP File** to compress the preprocessed input image, `.glb` mesh, `.mp4` camera path, and metadata into a single downloadable package.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Image-to-3D-Video-Asset-Generator.git](https://github.com/PRITHIVSAKTHIUR/Image-to-3D-Video-Asset-Generator.git)
* **Hugging Face Live Demo:** [https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator](https://huggingface.co/spaces/prithivMLmods/Image-to-3D-Video-Asset-Generator)
* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Image-to-3D-Video-Asset-Generator/blob/main/LICENSE.txt)
