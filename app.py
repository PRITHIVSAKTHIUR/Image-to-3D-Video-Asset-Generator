import os
import shutil
import torch
import numpy as np
from PIL import Image
import tempfile
import uuid
import io
import json
import random
import string
import zipfile
from typing import *
from datetime import datetime
from pathlib import Path

# --- Environment Setup ---
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_2"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

# --- Video Rendering: Try pyrender with different backends ---
HAS_PYRENDER = False
PYRENDER_BACKEND = None

for _backend in ['egl', 'osmesa']:
    try:
        os.environ['PYOPENGL_PLATFORM'] = _backend
        import pyrender
        HAS_PYRENDER = True
        PYRENDER_BACKEND = _backend
        print(f"pyrender loaded with {_backend} backend for video rendering.")
        break
    except Exception as e:
        print(f"pyrender import failed with {_backend}: {e}")

if not HAS_PYRENDER:
    print("pyrender not available. Will use matplotlib for video rendering.")

import trimesh
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Theme and UI ---
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

import gradio as gr
from gradio_client import Client, handle_file
import spaces
from diffusers import DiffusionPipeline
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

# --- Rerun ---
import rerun as rr
try:
    import rerun.blueprint as rrb
except ImportError:
    rrb = None

from gradio_rerun import Rerun

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

# --- Theme Configuration ---
colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --- Model Loading ---
print("Initializing models...")

print("Loading Z-Image-Turbo...")
try:
    z_pipe = DiffusionPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_pipe.to(device)
    print("Z-Image-Turbo loaded.")
except Exception as e:
    print(f"Failed to load Z-Image-Turbo: {e}")
    z_pipe = None

print("Loading TRELLIS.2...")
try:
    trellis_pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    trellis_pipeline.rembg_model = None
    trellis_pipeline.low_vram = False
    trellis_pipeline.cuda()
    print("TRELLIS.2 loaded.")
except Exception as e:
    print(f"Failed to load TRELLIS.2: {e}")
    trellis_pipeline = None

rmbg_client = Client("briaai/BRIA-RMBG-2.0")

# --- Session Management ---
def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

# --- Background Removal & Preprocessing ---
def remove_background(input: Image.Image) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        input = input.convert('RGB')
        input.save(f.name)
        output = rmbg_client.predict(handle_file(f.name), api_name="/image")[0][0]
        output = Image.open(output)
        return output

def preprocess_image(input: Image.Image) -> Image.Image:
    """Preprocess the input image: remove bg, crop, resize."""
    if input is None:
        return None

    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
    if has_alpha:
        output = input
    else:
        output = remove_background(input)

    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if bbox.size == 0:
        return output
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

# --- Text-to-Image ---
@spaces.GPU(size="xlarge", duration=120)
def generate_txt2img(prompt, progress=gr.Progress(track_tqdm=True)):
    if z_pipe is None:
        raise gr.Error("Z-Image-Turbo model failed to load.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(42)

    progress(0.1, desc="Generating Text-to-Image...")
    try:
        result = z_pipe(
            prompt=prompt,
            negative_prompt=None,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        )
        return result.images[0]
    except Exception as e:
        raise gr.Error(f"Z-Image Generation failed: {str(e)}")

# --- 3D Generation ---
@spaces.GPU(size="xlarge", duration=120)
def generate_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    decimation_target: int,
    texture_size: int,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_guidance: float,
    shape_rescale: float,
    shape_steps: int,
    shape_rescale_t: float,
    tex_guidance: float,
    tex_rescale: float,
    tex_steps: int,
    tex_rescale_t: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str, str]:
    """Returns: rrd_path, glb_path, glb_path_for_state, image_path_for_state"""

    if image is None:
        raise gr.Error("Please provide an input image.")

    if trellis_pipeline is None:
        raise gr.Error("TRELLIS model is not loaded.")

    session_hash = req.session_hash if req is not None else "default_session"
    user_dir = os.path.join(TMP_DIR, session_hash)
    os.makedirs(user_dir, exist_ok=True)

    # Save input image for dataset export
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S")
    image_path = os.path.join(user_dir, f'input_image_{timestamp}.png')
    image.save(image_path)

    progress(0.1, desc="Generating 3D Geometry...")
    try:
        outputs, latents = trellis_pipeline.run(
            image,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_slat_sampler_params={
                "steps": shape_steps,
                "guidance_strength": shape_guidance,
                "guidance_rescale": shape_rescale,
                "rescale_t": shape_rescale_t,
            },
            tex_slat_sampler_params={
                "steps": tex_steps,
                "guidance_strength": tex_guidance,
                "guidance_rescale": tex_rescale,
                "rescale_t": tex_rescale_t,
            },
            pipeline_type={"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}[resolution],
            return_latent=True,
        )

        # Process Mesh
        progress(0.7, desc="Processing Mesh...")
        mesh = outputs[0]
        mesh.simplify(1000000)

        # Export to GLB
        progress(0.9, desc="Baking Texture & Exporting GLB...")

        grid_size = latents[2]

        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=trellis_pipeline.pbr_attr_layout,
                grid_size=grid_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=True,
            )
        except RuntimeError as e:
            print(f"Warning: Post-processing failed with remesh=True. Error: {e}")
            print("Retrying with remesh=False (Standard mesh generation)...")
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=trellis_pipeline.pbr_attr_layout,
                grid_size=grid_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=False,
                remesh_band=1,
                remesh_project=0,
                use_tqdm=True,
            )

        glb_path = os.path.join(user_dir, f'trellis_output_{timestamp}.glb')
        glb.export(glb_path, extension_webp=False)

        # --- Rerun Visualization ---
        progress(0.95, desc="Creating Rerun Visualization...")

        run_id = str(uuid.uuid4())

        rec = None
        if hasattr(rr, "new_recording"):
            rec = rr.new_recording(application_id="TRELLIS-3D-Viewer", recording_id=run_id)
        elif hasattr(rr, "RecordingStream"):
            rec = rr.RecordingStream(application_id="TRELLIS-3D-Viewer", recording_id=run_id)
        else:
            rr.init("TRELLIS-3D-Viewer", recording_id=run_id, spawn=False)
            rec = rr

        rec.log("world", rr.Clear(recursive=True), static=True)
        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        try:
            rec.log("world/axes/x", rr.Arrows3D(vectors=[[0.5, 0, 0]], colors=[[255, 0, 0]]), static=True)
            rec.log("world/axes/y", rr.Arrows3D(vectors=[[0, 0.5, 0]], colors=[[0, 255, 0]]), static=True)
            rec.log("world/axes/z", rr.Arrows3D(vectors=[[0, 0, 0.5]], colors=[[0, 0, 255]]), static=True)
        except Exception:
            pass

        rec.log("world/model", rr.Asset3D(path=glb_path), static=True)

        if rrb is not None:
            try:
                blueprint = rrb.Blueprint(
                    rrb.Spatial3DView(
                        origin="/world",
                        name="3D View",
                    ),
                    collapse_panels=True,
                )
                rec.send_blueprint(blueprint)
            except Exception as e:
                print(f"Blueprint creation failed (non-fatal): {e}")

        rrd_path = os.path.join(user_dir, f'trellis_output_{timestamp}.rrd')
        rec.save(rrd_path)

        torch.cuda.empty_cache()
        return rrd_path, glb_path, glb_path, image_path

    except Exception as e:
        torch.cuda.empty_cache()
        raise gr.Error(f"Generation failed: {str(e)}")


# ====================================================================
# --- Video Generation (GLB to MP4) ---
# ====================================================================

def hex_to_rgb_norm(hex_color: str) -> tuple:
    """Convert hex color string to normalized (0-1) RGB tuple."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

def get_polygon_color_norm(color_name: str) -> tuple:
    """Convert color name to normalized RGB tuple."""
    colors_map = {
        "Pink": (1.0, 0.75, 0.80),
        "Green": (0.0, 1.0, 0.0),
        "White": (1.0, 1.0, 1.0),
        "Orange": (1.0, 0.65, 0.0)
    }
    return colors_map.get(color_name, (1.0, 0.75, 0.80))

def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    """Compute a 4x4 camera look-at matrix."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up_vec = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up_vec
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def compute_camera_pose(style: str, t: float, rotation_speed: str = "Normal") -> np.ndarray:
    """
    Compute camera pose (4x4 matrix) for given style and normalized time t (0 to 1).
    The camera always looks at the origin where the model is centered.
    """
    speed_map = {"Slow": 0.5, "Normal": 1.0, "Fast": 2.0}
    speed = speed_map.get(rotation_speed, 1.0)

    angle = t * 2 * np.pi * speed
    base_distance = 3.0

    if style == "Orbit":
        x = base_distance * np.cos(angle)
        z = base_distance * np.sin(angle)
        y = 0.0
    elif style == "Zoom In":
        d = base_distance * (1.8 - t * 1.0)
        x = d * np.cos(angle)
        z = d * np.sin(angle)
        y = 0.0
    elif style == "Zoom Out":
        d = base_distance * (0.8 + t * 1.0)
        x = d * np.cos(angle)
        z = d * np.sin(angle)
        y = 0.0
    elif style == "Turntable":
        x = base_distance * np.cos(angle)
        z = base_distance * np.sin(angle)
        y = 0.3
    elif style == "Spiral":
        d = base_distance * (1.4 - t * 0.2)
        y = 2.0 * (1.0 - t)
        x = d * np.cos(angle * 1.5)
        z = d * np.sin(angle * 1.5)
    elif style == "Top Sweep":
        h = base_distance * np.sin(t * np.pi / 2.0)
        v = base_distance * np.cos(t * np.pi / 2.0) * 0.9
        x = h * np.cos(angle)
        z = h * np.sin(angle)
        y = v
    elif style == "Cinematic":
        d = base_distance * (1.0 + 0.3 * np.sin(t * 2 * np.pi))
        y = 0.5 * np.sin(t * 2 * np.pi)
        x = d * np.cos(angle)
        z = d * np.sin(angle)
    else:
        x = base_distance * np.cos(angle)
        z = base_distance * np.sin(angle)
        y = 0.0

    eye = np.array([x, y, z])
    target = np.array([0.0, 0.0, 0.0])
    return look_at_matrix(eye, target)


def compute_norm_transform(scene_or_mesh) -> np.ndarray:
    """Compute a 4x4 transform that centers and scales the object to fit in a unit sphere."""
    if isinstance(scene_or_mesh, trimesh.Scene):
        bounds = scene_or_mesh.bounds
    else:
        bounds = np.array([scene_or_mesh.bounds[0], scene_or_mesh.bounds[1]])

    center = (bounds[0] + bounds[1]) / 2.0
    extent = bounds[1] - bounds[0]
    scale = 2.0 / max(extent.max(), 1e-6)

    T = np.eye(4)
    T[:3, 3] = -center
    S = np.eye(4)
    S[:3, :3] *= scale
    return S @ T


def write_video_frames(frames: list, output_path: str, fps: int):
    """Write a list of RGB uint8 frames to an MP4 file using OpenCV."""
    if len(frames) == 0:
        raise ValueError("No frames to write.")

    height, width = frames[0].shape[:2]

    # Try H264/avc1 first, fall back to mp4v
    writer = None
    for codec in ['avc1', 'mp4v', 'X264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            break
        writer = None

    if writer is None:
        raise RuntimeError("Could not open video writer with any codec.")

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        writer.write(frame_bgr)

    writer.release()


def render_video_pyrender(
    glb_path: str,
    output_path: str,
    style: str,
    rotation_speed: str,
    bg_color_hex: str,
    polygon_color_name: str,
    resolution: tuple,
    fps: int = 30,
    duration: int = 5,
    progress: gr.Progress = None,
) -> str:
    """Render GLB to video using pyrender (high quality)."""
    tri_scene = trimesh.load(glb_path, force='scene')
    norm_transform = compute_norm_transform(tri_scene)

    bg_color = hex_to_rgb_norm(bg_color_hex)
    # pyrender expects RGBA for bg_color
    bg_color_rgba = (*bg_color, 1.0)

    pyr_scene = pyrender.Scene(bg_color=bg_color_rgba, ambient_light=[0.25, 0.25, 0.30])

    poly_color = get_polygon_color_norm(polygon_color_name)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[poly_color[0], poly_color[1], poly_color[2], 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.5
    )

    # Add meshes
    if isinstance(tri_scene, trimesh.Scene):
        for node_name in tri_scene.graph.nodes_geometry:
            transform, geometry_name = tri_scene.graph.get(node_name)
            geometry = tri_scene.geometry[geometry_name]
            final_transform = norm_transform @ transform
            pr_mesh = pyrender.Mesh.from_trimesh(geometry, material=material, smooth=True)
            pyr_scene.add(pr_mesh, pose=final_transform)
    else:
        pr_mesh = pyrender.Mesh.from_trimesh(tri_scene, material=material, smooth=True)
        pyr_scene.add(pr_mesh, pose=norm_transform)

    # Add lights
    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 0.95], intensity=3.5)
    fill_light = pyrender.DirectionalLight(color=[0.6, 0.65, 0.8], intensity=1.5)
    rim_light = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=2.0)

    pyr_scene.add(key_light, pose=look_at_matrix(np.array([3, 3, 3]), np.array([0, 0, 0])))
    pyr_scene.add(fill_light, pose=look_at_matrix(np.array([-3, 1, -2]), np.array([0, 0, 0])))
    pyr_scene.add(rim_light, pose=look_at_matrix(np.array([0, 2, -4]), np.array([0, 0, 0])))

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=resolution[0] / resolution[1])
    camera_node = pyr_scene.add(camera, pose=np.eye(4))

    # Renderer
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

    num_frames = fps * duration
    frames = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        cam_pose = compute_camera_pose(style, t, rotation_speed)
        pyr_scene.set_pose(camera_node, cam_pose)

        color, _ = renderer.render(pyr_scene)
        frames.append(color)

        if progress is not None:
            progress(0.1 + 0.85 * (i + 1) / num_frames, desc=f"Rendering frame {i+1}/{num_frames}...")

    renderer.delete()

    write_video_frames(frames, output_path, fps)
    return output_path


def render_video_matplotlib(
    glb_path: str,
    output_path: str,
    style: str,
    rotation_speed: str,
    bg_color_hex: str,
    polygon_color_name: str,
    resolution: tuple,
    fps: int = 30,
    duration: int = 5,
    progress: gr.Progress = None,
) -> str:
    """Render GLB to video using matplotlib (fallback, lower quality)."""
    tri_obj = trimesh.load(glb_path, force='mesh')

    # Simplify for performance
    if len(tri_obj.faces) > 8000:
        try:
            tri_obj = tri_obj.simplify_quadric_decimation(8000)
        except Exception:
            pass

    # Normalize
    vertices = tri_obj.vertices.copy().astype(np.float64)
    vertices -= vertices.mean(axis=0)
    max_extent = np.abs(vertices).max()
    if max_extent > 1e-6:
        vertices /= max_extent

    # Swap Y and Z axes to make the object upright in matplotlib's 3D coordinate system
    # Matplotlib uses Z as the vertical axis by default, while GLB typically uses Y.
    vertices = vertices[:, [0, 2, 1]]

    faces = tri_obj.faces

    # Get face colors
    poly_color = get_polygon_color_norm(polygon_color_name)
    face_colors = np.tile(np.array([poly_color]), (len(faces), 1))

    # Compute face normals for lighting
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_lens[norm_lens < 1e-8] = 1.0
    normals /= norm_lens

    light_dir = np.array([0.5, 0.8, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shading = np.maximum(0, normals @ light_dir)
    shading = 0.35 + 0.65 * shading

    shaded_colors = face_colors * shading[:, np.newaxis]
    shaded_colors = np.clip(shaded_colors, 0, 1)

    bg_rgb = hex_to_rgb_norm(bg_color_hex)
    num_frames = fps * duration

    speed_map = {"Slow": 0.5, "Normal": 1.0, "Fast": 2.0}
    speed = speed_map.get(rotation_speed, 1.0)

    dpi = 100
    fig_w = resolution[0] / dpi
    fig_h = resolution[1] / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    frames = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        angle = t * 360 * speed
        elev = 0  # Set to 0 for straight axis view

        zoom = 1.6
        if style == "Zoom In":
            zoom = 2.2 - t * 1.0
        elif style == "Zoom Out":
            zoom = 1.2 + t * 1.0
        elif style == "Spiral":
            elev = 45 - t * 45
            zoom = 1.8 - t * 0.3
        elif style == "Top Sweep":
            elev = 45 - t * 45
        elif style == "Turntable":
            elev = 0
        elif style == "Cinematic":
            elev = 10 + 10 * np.sin(t * 2 * np.pi)
            zoom = 1.6 + 0.2 * np.sin(t * 2 * np.pi)

        ax.clear()
        ax.set_facecolor(bg_rgb)
        fig.patch.set_facecolor(bg_rgb)

        poly = Poly3DCollection(vertices[faces], alpha=1.0)
        poly.set_facecolor(shaded_colors)
        poly.set_edgecolor('none')
        ax.add_collection3d(poly)

        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        ax.set_zlim(-zoom, zoom)
        ax.view_init(elev=elev, azim=angle)
        ax.set_axis_off()

        fig.canvas.draw()
        
        # FIX: Use buffer_rgba instead of tostring_rgb for newer matplotlib versions
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy()
        frames.append(frame)

        if progress is not None:
            progress(0.1 + 0.85 * (i + 1) / num_frames, desc=f"Rendering frame {i+1}/{num_frames} (matplotlib)...")

    plt.close(fig)

    write_video_frames(frames, output_path, fps)
    return output_path


def generate_video(
    glb_path: str,
    style: str,
    rotation_speed: str,
    bg_color: str,
    polygon_color: str,
    resolution_str: str,
    req: Optional[gr.Request] = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> str:
    """Generate a 5-second MP4 video from a GLB file."""
    if glb_path is None or not os.path.exists(glb_path):
        raise gr.Error("No GLB file found. Please generate a 3D model first.")

    session_hash = req.session_hash if req is not None else "default_video_session"
    user_dir = os.path.join(TMP_DIR, session_hash)
    os.makedirs(user_dir, exist_ok=True)

    resolution_map = {
        "512x512": (512, 512),
        "720p (1280x720)": (1280, 720),
        "1080p (1920x1080)": (1920, 1080),
    }
    resolution = resolution_map.get(resolution_str, (512, 512))

    fps = 30
    duration = 5  # Fixed at 5 seconds as requested
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    output_path = os.path.join(user_dir, f'video_{timestamp}.mp4')

    progress(0.05, desc="Loading 3D model...")

    try:
        if HAS_PYRENDER:
            progress(0.1, desc="Rendering with pyrender (high quality)...")
            try:
                return render_video_pyrender(
                    glb_path, output_path, style, rotation_speed, bg_color, polygon_color,
                    resolution, fps, duration, progress
                )
            except Exception as e:
                print(f"pyrender rendering failed: {e}. Falling back to matplotlib.")

        progress(0.1, desc="Rendering with matplotlib (fallback)...")
        return render_video_matplotlib(
            glb_path, output_path, style, rotation_speed, bg_color, polygon_color,
            resolution, fps, duration, progress
        )

    except Exception as e:
        raise gr.Error(f"Video generation failed: {str(e)}")


# ====================================================================
# --- ZIP Export ---
# ====================================================================

def random_name(length: int = 8) -> str:
    """Generate a random alphanumeric name."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def export_zip(
    image_path: str,
    glb_path: str,
    video_path: str,
    prompt: str,
    video_style: str,
    req: Optional[gr.Request] = None,
) -> str:
    """Export image, GLB, and video as a single zip file."""
    session_hash = req.session_hash if req is not None else "default_export_session"
    user_dir = os.path.join(TMP_DIR, session_hash)
    os.makedirs(user_dir, exist_ok=True)

    if (not image_path or not os.path.exists(image_path)) and \
       (not glb_path or not os.path.exists(glb_path)) and \
       (not video_path or not os.path.exists(video_path)):
        raise gr.Error("No files available to export. Please generate a 3D model and video first.")

    name = random_name()
    zip_path = os.path.join(user_dir, f'{name}.zip')

    metadata = {
        'prompt': prompt if prompt else '',
        'video_style': video_style if video_style else '',
        'timestamp': datetime.now().isoformat(),
    }

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            if image_path and os.path.exists(image_path):
                ext = os.path.splitext(image_path)[1]
                zf.write(image_path, arcname=f'image{ext}')
            if glb_path and os.path.exists(glb_path):
                zf.write(glb_path, arcname='model.glb')
            if video_path and os.path.exists(video_path):
                ext = os.path.splitext(video_path)[1]
                zf.write(video_path, arcname=f'video{ext}')
            
            # Write metadata
            zf.writestr('metadata.json', json.dumps(metadata, indent=4))
    except Exception as e:
        raise gr.Error(f"Failed to create ZIP file: {str(e)}")

    return zip_path


# ====================================================================
# --- Gradio UI ---
# ====================================================================

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.4em !important;}
#video-section, #zip-section {
    margin-top: 20px;
    padding: 16px;
    border-radius: 12px;
    border: 2px solid var(--block-border-color);
}
"""

if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    with gr.Blocks() as demo:
        gr.Markdown("# **Image-to-3D-Video-Asset-Generator**", elem_id="main-title")
        gr.Markdown("""
        Generate a complete 3D asset pipeline: Text-to-Image to 3D (GLB) to Video, then export everything as a ZIP file.
        Powered by [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B), [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), and custom rendering. [GitHub](https://github.com/PRITHIVSAKTHIUR/Image-to-3D-Video-Asset-Generator)
        """)

        # State variables for tracking file paths
        state_glb = gr.State(value=None)
        state_image = gr.State(value=None)
        state_video = gr.State(value=None)

        # --- Section 1: 3D Generation ---
        with gr.Row():
            with gr.Column(scale=1, min_width=360):

                with gr.Tabs():
                    with gr.Tab("Text-to-Image-3D"):
                        txt_prompt = gr.Textbox(label="Prompt", placeholder="eg. A Plane 3D model", lines=2)
                        btn_gen_img = gr.Button("1. Generate Image", variant="primary")
                    with gr.Tab("Image-to-3D"):
                        gr.Markdown("Upload an image directly if you have one.")

                image_prompt = gr.Image(label="Input Image", format="png", image_mode="RGBA", type="pil", height=350)

                with gr.Accordion(label="3D Settings", open=False):
                    resolution = gr.Radio(["512", "1024", "1536"], label="Generation Resolution", value="1024")
                    seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)

                decimation_target = gr.Slider(50000, 500000, label="Target Faces", value=150000, step=10000)
                texture_size = gr.Slider(512, 4096, label="Texture Size", value=1024, step=512)

                btn_gen_3d = gr.Button("2. Generate 3D", variant="primary", scale=2)

                with gr.Accordion(label="Advanced Sampler Settings", open=False):
                    gr.Markdown("**Stage 1: Sparse Structure**")
                    ss_guidance_strength = gr.Slider(1.0, 10.0, value=7.5, label="Guidance")
                    ss_guidance_rescale = gr.Slider(0.0, 1.0, value=0.7, label="Rescale")
                    ss_sampling_steps = gr.Slider(1, 50, value=12, label="Steps")
                    ss_rescale_t = gr.Slider(1.0, 6.0, value=5.0, label="Rescale T")

                    gr.Markdown("**Stage 2: Shape**")
                    shape_guidance = gr.Slider(1.0, 10.0, value=7.5, label="Guidance")
                    shape_rescale = gr.Slider(0.0, 1.0, value=0.5, label="Rescale")
                    shape_steps = gr.Slider(1, 50, value=12, label="Steps")
                    shape_rescale_t = gr.Slider(1.0, 6.0, value=3.0, label="Rescale T")

                    gr.Markdown("**Stage 3: Material**")
                    tex_guidance = gr.Slider(1.0, 10.0, value=1.0, label="Guidance")
                    tex_rescale = gr.Slider(0.0, 1.0, value=0.0, label="Rescale")
                    tex_steps = gr.Slider(1, 50, value=12, label="Steps")
                    tex_rescale_t = gr.Slider(1.0, 6.0, value=3.0, label="Rescale T")

            with gr.Column(scale=2):
                gr.Markdown("### 3D Output")

                rerun_output = Rerun(
                    label="Rerun 3D Viewer",
                    height=600
                )

                download_btn = gr.DownloadButton(label="3. Download GLB File", variant="primary")

                gr.Examples(
                    examples=[
                        ["example-images/A (1).webp"],
                        ["example-images/A (2).webp"],
                        ["example-images/A (3).webp"],
                        ["example-images/A (4).webp"],
                        ["example-images/A (5).webp"],
                        ["example-images/A (6).webp"],
                        ["example-images/A (7).webp"],
                        ["example-images/A (8).webp"],
                        ["example-images/A (9).webp"],
                        ["example-images/A (10).webp"],
                        ["example-images/A (11).webp"],
                        ["example-images/A (12).webp"],
                        ["example-images/A (13).webp"],
                        ["example-images/A (14).webp"],
                        ["example-images/A (15).webp"],
                        ["example-images/A (16).webp"],
                        ["example-images/A (17).webp"],
                        ["example-images/A (18).webp"],
                        ["example-images/A (19).webp"],
                        ["example-images/A (20).webp"],
                    ],
                    inputs=[image_prompt],
                    label="Image Examples [image-to-3d]"
                )

                gr.Examples(
                    examples=[
                        ["A Cat 3D model"],
                        ["A realistic Cat 3D model"],
                        ["A cartoon Cat 3D model"],
                        ["A low poly Cat 3D"],
                        ["A cyberpunk Cat 3D"],
                        ["A robotic Cat 3D"],
                        ["A fluffy Cat 3D"],
                        ["A fantasy Cat 3D creature"],
                        ["A stylized Cat 3D"],
                        ["A Cat 3D sculpture"],

                        ["A Plane 3D model"],
                        ["A commercial Plane 3D"],
                        ["A fighter jet Plane 3D"],
                        ["A low poly Plane 3D"],
                        ["A vintage Plane 3D"],
                        ["A futuristic Plane 3D"],
                        ["A cargo Plane 3D"],
                        ["A private jet Plane 3D"],
                        ["A toy Plane 3D"],
                        ["A realistic Plane 3D"],

                        ["A Car 3D model"],
                        ["A sports Car 3D"],
                        ["A luxury Car 3D"],
                        ["A low poly Car 3D"],
                        ["A racing Car 3D"],
                        ["A cyberpunk Car 3D"],
                        ["A vintage Car 3D"],
                        ["A futuristic Car 3D"],
                        ["A SUV Car 3D"],
                        ["A electric Car 3D"],

                        ["A Shoe 3D model"],
                        ["A sneaker Shoe 3D"],
                        ["A running Shoe 3D"],
                        ["A leather Shoe 3D"],
                        ["A high heel Shoe 3D"],
                        ["A boot Shoe 3D"],
                        ["A low poly Shoe 3D"],
                        ["A futuristic Shoe 3D"],
                        ["A sports Shoe 3D"],
                        ["A casual Shoe 3D"],

                        ["A Chair 3D model"],
                        ["A Table 3D model"],
                        ["A Sofa 3D model"],
                        ["A Lamp 3D model"],
                        ["A Watch 3D model"],
                        ["A Backpack 3D model"],
                        ["A Drone 3D model"],
                        ["A Robot 3D model"],
                        ["A Smartphone 3D model"],
                        ["A Headphones 3D model"],

                        ["A House 3D model"],
                        ["A Skyscraper 3D model"],
                        ["A Bridge 3D model"],
                        ["A Castle 3D model"],
                        ["A Spaceship 3D model"],
                        ["A Rocket 3D model"],
                        ["A Satellite 3D model"],
                        ["A Tank 3D model"],
                        ["A Motorcycle 3D model"],
                        ["A Bicycle 3D model"]
                    ],
                    inputs=[txt_prompt],
                    label="3D Prompt Examples [text-to-3d]"
                )

        # --- Section 2: 3D to Video Converter ---
        with gr.Column(elem_id="video-section"):
            gr.Markdown("## 3D (GLB) to Video Converter")
            gr.Markdown("Convert your generated 3D model into a 5-second MP4 video with customizable camera styles and effects.")

            with gr.Row():
                with gr.Column(scale=2):
                    video_style = gr.Dropdown(
                        choices=["Orbit", "Zoom In", "Zoom Out", "Turntable", "Spiral", "Top Sweep", "Cinematic"],
                        label="Video Style",
                        value="Zoom In",
                        info="Choose the camera movement style for the video."
                    )
                    rotation_speed = gr.Dropdown(
                        choices=["Slow", "Normal", "Fast"],
                        label="Rotation Speed",
                        value="Normal",
                        info="Controls how fast the camera rotates around the model."
                    )
                    polygon_color = gr.Dropdown(
                        choices=["Pink", "Green", "White", "Orange"],
                        label="Polygon Color",
                        value="Pink",
                        info="Override the 3D model's color in the video."
                    )
                with gr.Column(scale=2):
                    bg_color = gr.Dropdown(
                        choices=["#000000", "#FFFFFF", "#1A1A2E", "#2C3E50", "#0F0F0F", "#F0F0F0"],
                        label="Background Color",
                        value="#000000",
                        info="Background color for the video."
                    )
                    video_resolution = gr.Dropdown(
                        choices=["512x512", "720p (1280x720)", "1080p (1920x1080)"],
                        label="Video Resolution",
                        value="1080p (1920x1080)",
                        info="Higher resolutions take longer to render."
                    )

            with gr.Row():
                btn_gen_video = gr.Button("4. Generate Video (5 sec)", variant="primary")
                download_video_btn = gr.DownloadButton(label="5. Download Video", variant="secondary")

            video_output = gr.Video(label="Video Output", height=400)

        # --- Section 3: ZIP Export ---
        with gr.Column(elem_id="zip-section"):
            gr.Markdown("## ZIP Export")
            gr.Markdown("Export your generated assets (Image, 3D GLB, Video) as a single ZIP file.")

            with gr.Row():
                btn_export_zip = gr.Button("6. Generate ZIP File", variant="primary")
                download_zip_btn = gr.DownloadButton(label="7. Download ZIP File", variant="secondary")

        # --- Event Wiring ---
        demo.load(start_session)
        demo.unload(end_session)

        # Text-to-Image flow
        btn_gen_img.click(
            generate_txt2img,
            inputs=[txt_prompt],
            outputs=[image_prompt]
        ).then(
            preprocess_image,
            inputs=[image_prompt],
            outputs=[image_prompt]
        )

        image_prompt.upload(
            preprocess_image,
            inputs=[image_prompt],
            outputs=[image_prompt],
        )

        # 3D generation flow
        btn_gen_3d.click(
            get_seed,
            inputs=[randomize_seed, seed],
            outputs=[seed],
        ).then(
            generate_3d,
            inputs=[
                image_prompt, seed, resolution,
                decimation_target, texture_size,
                ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
                shape_guidance, shape_rescale, shape_steps, shape_rescale_t,
                tex_guidance, tex_rescale, tex_steps, tex_rescale_t,
            ],
            outputs=[rerun_output, download_btn, state_glb, state_image],
        )

        # Video generation flow
        def generate_video_wrapper(
            glb_path: str, 
            style: str, 
            rot_speed: str, 
            bg: str, 
            poly_color: str,
            res: str, 
            req: gr.Request, 
            progress: gr.Progress = gr.Progress(track_tqdm=True)
        ):
            video_path = generate_video(glb_path, style, rot_speed, bg, poly_color, res, req, progress)
            return video_path, video_path

        btn_gen_video.click(
            generate_video_wrapper,
            inputs=[state_glb, video_style, rotation_speed, bg_color, polygon_color, video_resolution],
            outputs=[video_output, state_video],
        ).then(
            lambda v: v,
            inputs=[state_video],
            outputs=[download_video_btn],
        )

        # ZIP export flow
        def export_zip_wrapper(
            image_path: str, 
            glb_path: str, 
            video_path: str, 
            prompt: str, 
            v_style: str, 
            req: gr.Request
        ):
            zip_path = export_zip(image_path, glb_path, video_path, prompt, v_style, req)
            return zip_path

        btn_export_zip.click(
            export_zip_wrapper,
            inputs=[state_image, state_glb, state_video, txt_prompt, video_style],
            outputs=[download_zip_btn],
        )

    demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)