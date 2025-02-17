import os
import json
import gradio as gr
import modules.scripts as scripts
from modules.processing import Processed
from modules.shared import opts, state
import cv2
import numpy as np
from PIL import Image

PRESETS_FILE = os.path.join(os.path.dirname(__file__), "amateur_filter_presets.json")

def load_presets():
    """Load presets from JSON file, or return empty dict if missing."""
    if os.path.exists(PRESETS_FILE):
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_presets(presets):
    """Save presets dict to JSON file."""
    with open(PRESETS_FILE, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=4)

DEFAULT_VALUES = {
    "dynamic_range_factor": 1.0,
    "hsv_hue_var": 2,
    "hsv_sat_var": 5,
    "hsv_val_var": 6,
    "rgb_noise_var": 0.007,
    "sharpen_intensity": 0.01,
    "warmth_factor": 0.03,
    "contrast_alpha": 1.0,
    "contrast_beta": 3.0,
    "desaturation_factor": 0.85,
    "grain_amount": 0.02,
    "jpeg_quality": 80
}

# --------------------------------------------------------------------------------
# Filter functions
# --------------------------------------------------------------------------------

def lower_dynamic_range(img, factor=1.0):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def add_hsv_noise(img, hue_var=2, sat_var=5, val_var=6):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    h_noise = np.random.randint(-hue_var, hue_var, h.shape, dtype=np.int16)
    s_noise = np.random.randint(-sat_var, sat_var, s.shape, dtype=np.int16)
    v_noise = np.random.randint(-val_var, val_var, v.shape, dtype=np.int16)
    h = np.clip(h + h_noise, 0, 179).astype(np.uint8)
    s = np.clip(s + s_noise, 0, 255).astype(np.uint8)
    v = np.clip(v + v_noise, 0, 255).astype(np.uint8)
    merged = cv2.merge([h, s, v])
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

def add_rgb_noise(img, var=0.007):
    noise_range = int(var * 255)
    noise = np.random.randint(-noise_range, noise_range, img.shape, dtype=np.int16)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def sharpen(img, intensity=0.15):
    kernel = np.array([
        [0, -intensity, 0],
        [-intensity, 1 + 4*intensity, -intensity],
        [0, -intensity, 0]
    ])
    return cv2.filter2D(img, -1, kernel)

def add_warm_tone(img, warmth_factor=0.03):
    b, g, r = cv2.split(img.astype(np.float32) / 255.0)
    g *= (1.0 - warmth_factor)
    b *= (1.0 - warmth_factor)
    merged = cv2.merge([b, g, r])
    return np.clip(merged * 255, 0, 255).astype(np.uint8)

def adjust_contrast(img, alpha=1.0, beta=0.0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def desaturate(img, factor=1.0):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] = (hsv_img[..., 1].astype(np.float32) * factor).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def add_grain(img, grain_amount=0.08):
    if grain_amount <= 0:
        return img
    std_dev = grain_amount * 255
    noise = np.random.normal(0, std_dev, img.shape).astype(np.int16)
    blended = np.clip(0.5 * img + 0.5 * (img + noise), 0, 255).astype(np.uint8)
    return blended

def apply_amateur_filter(
    pil_image,
    dynamic_range_factor=1.0,
    hsv_hue_var=2,
    hsv_sat_var=5,
    hsv_val_var=6,
    rgb_noise_var=0.007,
    sharpen_intensity=0.15,
    warmth_factor=0.03,
    contrast_alpha=1.0,
    contrast_beta=0.0,
    desaturation_factor=0.78,
    grain_amount=0.08,
    jpeg_quality=54
):
    original_metadata = getattr(pil_image, "info", {}).copy()
    image = np.array(pil_image.convert("RGB"))[:, :, ::-1]

    image = lower_dynamic_range(image, dynamic_range_factor)
    image = add_hsv_noise(image, hsv_hue_var, hsv_sat_var, hsv_val_var)
    image = add_rgb_noise(image, rgb_noise_var)
    image = sharpen(image, sharpen_intensity)
    image = add_warm_tone(image, warmth_factor)
    image = adjust_contrast(image, contrast_alpha, contrast_beta)
    image = desaturate(image, desaturation_factor)
    image = add_grain(image, grain_amount)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success, encimg = cv2.imencode('.jpg', image, encode_param)
    if success:
        image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    final_rgb = image[:, :, ::-1]
    filtered_pil = Image.fromarray(final_rgb)
    filtered_pil.info.update(original_metadata)

    return filtered_pil

# --------------------------------------------------------------------------------
#  Main Script
# --------------------------------------------------------------------------------

class Script(scripts.Script):
    def title(self):
        return "Amateur Filter"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        self.presets = load_presets()

        with gr.Accordion("Amateur Filter", open=False):
            style_html = """
            <style>
            .amateur-filter-presets .small-square > button {
                width: 2em !important;
                height: 2em !important;
                padding: 0 !important;
                margin: 0 4px 0 0 !important;
                min-width: 2em !important;
            }
            </style>
            """
            gr.HTML(style_html)

            enable_filter = gr.Checkbox(
                value=False,
                label="Enable"
            )

            with gr.Accordion("Basic", open=True):
                sharpen_intensity = gr.Slider(
                    minimum=0.0, maximum=1.0,
                    value=DEFAULT_VALUES["sharpen_intensity"],
                    step=0.01,
                    label="Sharpen Intensity"
                )
                desaturation_factor = gr.Slider(
                    minimum=0.0, maximum=1.0,
                    value=DEFAULT_VALUES["desaturation_factor"],
                    step=0.01,
                    label="Desaturation Factor"
                )
                grain_amount = gr.Slider(
                    minimum=0.0, maximum=0.3,
                    value=DEFAULT_VALUES["grain_amount"],
                    step=0.01,
                    label="Grain Amount"
                )
                jpeg_quality = gr.Slider(
                    minimum=10, maximum=100,
                    value=DEFAULT_VALUES["jpeg_quality"],
                    step=1,
                    label="JPEG Quality"
                )

            with gr.Accordion("Advanced", open=False):
                dynamic_range_factor = gr.Slider(
                    minimum=0.0, maximum=2.0,
                    value=DEFAULT_VALUES["dynamic_range_factor"],
                    step=0.01,
                    label="Dynamic Range Factor"
                )
                hsv_hue_var = gr.Slider(
                    minimum=0, maximum=30,
                    value=DEFAULT_VALUES["hsv_hue_var"],
                    step=1,
                    label="HSV Hue Variation"
                )
                hsv_sat_var = gr.Slider(
                    minimum=0, maximum=100,
                    value=DEFAULT_VALUES["hsv_sat_var"],
                    step=1,
                    label="HSV Saturation Variation"
                )
                hsv_val_var = gr.Slider(
                    minimum=0, maximum=100,
                    value=DEFAULT_VALUES["hsv_val_var"],
                    step=1,
                    label="HSV Value Variation"
                )
                rgb_noise_var = gr.Slider(
                    minimum=0.0, maximum=0.1,
                    value=DEFAULT_VALUES["rgb_noise_var"],
                    step=0.001,
                    label="RGB Noise Strength"
                )
                warmth_factor = gr.Slider(
                    minimum=0.0, maximum=0.3,
                    value=DEFAULT_VALUES["warmth_factor"],
                    step=0.01,
                    label="Warmth Factor"
                )
                contrast_alpha = gr.Slider(
                    minimum=0.5, maximum=2.0,
                    value=DEFAULT_VALUES["contrast_alpha"],
                    step=0.01,
                    label="Contrast Alpha"
                )
                contrast_beta = gr.Slider(
                    minimum=-50, maximum=50,
                    value=DEFAULT_VALUES["contrast_beta"],
                    step=1,
                    label="Contrast Beta"
                )

            with gr.Row(elem_classes="amateur-filter-presets"):
                preset_dropdown = gr.Dropdown(
                    label=None,
                    choices=sorted(self.presets.keys()),
                    value=None,
                    show_label=False,
                    interactive=True,
                    placeholder="Select a preset..."
                )
                new_preset_name = gr.Textbox(
                    label=None,
                    show_label=False,
                    placeholder="Preset name..."
                )
                add_preset_button = gr.Button(
                    "➕",
                    elem_classes=["small-square"]
                )
                delete_preset_button = gr.Button(
                    "🗑️",
                    elem_classes=["small-square"]
                )

            reset_button = gr.Button("Reset to defaults")

            # --------------- HELPERS ----------------

            def reset_values():
                """Reset only slider values (not enable_filter)."""
                return (
                    DEFAULT_VALUES["dynamic_range_factor"],
                    DEFAULT_VALUES["hsv_hue_var"],
                    DEFAULT_VALUES["hsv_sat_var"],
                    DEFAULT_VALUES["hsv_val_var"],
                    DEFAULT_VALUES["rgb_noise_var"],
                    DEFAULT_VALUES["sharpen_intensity"],
                    DEFAULT_VALUES["warmth_factor"],
                    DEFAULT_VALUES["contrast_alpha"],
                    DEFAULT_VALUES["contrast_beta"],
                    DEFAULT_VALUES["desaturation_factor"],
                    DEFAULT_VALUES["grain_amount"],
                    DEFAULT_VALUES["jpeg_quality"]
                )

            def load_preset(preset_name):
                """Load only the slider parameters from the preset."""
                if not preset_name or preset_name not in self.presets:
                    return reset_values()
                preset = self.presets[preset_name]
                return (
                    preset["dynamic_range_factor"],
                    preset["hsv_hue_var"],
                    preset["hsv_sat_var"],
                    preset["hsv_val_var"],
                    preset["rgb_noise_var"],
                    preset["sharpen_intensity"],
                    preset["warmth_factor"],
                    preset["contrast_alpha"],
                    preset["contrast_beta"],
                    preset["desaturation_factor"],
                    preset["grain_amount"],
                    preset["jpeg_quality"]
                )

            def add_preset(
                preset_name,
                drf, hue, sat, val, rgb, sharp,
                warm, c_alpha, c_beta, desat, grain, quality
            ):
                cur_presets = load_presets()
                if not preset_name.strip():
                    return gr.update(), gr.update(choices=sorted(cur_presets.keys()))

                cur_presets[preset_name.strip()] = {
                    "dynamic_range_factor": drf,
                    "hsv_hue_var": hue,
                    "hsv_sat_var": sat,
                    "hsv_val_var": val,
                    "rgb_noise_var": rgb,
                    "sharpen_intensity": sharp,
                    "warmth_factor": warm,
                    "contrast_alpha": c_alpha,
                    "contrast_beta": c_beta,
                    "desaturation_factor": desat,
                    "grain_amount": grain,
                    "jpeg_quality": quality
                }
                save_presets(cur_presets)
                self.presets = cur_presets

                return gr.update(value=preset_name.strip()), gr.update(choices=sorted(cur_presets.keys()))

            def delete_preset(selected):
                cur_presets = load_presets()
                if selected in cur_presets:
                    del cur_presets[selected]
                    save_presets(cur_presets)
                self.presets = cur_presets
                return gr.update(value=None), gr.update(choices=sorted(cur_presets.keys()))

            # --------------- BUTTONS ----------------

            reset_button.click(
                fn=reset_values,
                inputs=[],
                outputs=[
                    dynamic_range_factor,
                    hsv_hue_var,
                    hsv_sat_var,
                    hsv_val_var,
                    rgb_noise_var,
                    sharpen_intensity,
                    warmth_factor,
                    contrast_alpha,
                    contrast_beta,
                    desaturation_factor,
                    grain_amount,
                    jpeg_quality
                ],
                show_progress=False
            )

            preset_dropdown.change(
                fn=load_preset,
                inputs=[preset_dropdown],
                outputs=[
                    dynamic_range_factor,
                    hsv_hue_var,
                    hsv_sat_var,
                    hsv_val_var,
                    rgb_noise_var,
                    sharpen_intensity,
                    warmth_factor,
                    contrast_alpha,
                    contrast_beta,
                    desaturation_factor,
                    grain_amount,
                    jpeg_quality
                ]
            )

            add_preset_button.click(
                fn=add_preset,
                inputs=[
                    new_preset_name,
                    dynamic_range_factor,
                    hsv_hue_var,
                    hsv_sat_var,
                    hsv_val_var,
                    rgb_noise_var,
                    sharpen_intensity,
                    warmth_factor,
                    contrast_alpha,
                    contrast_beta,
                    desaturation_factor,
                    grain_amount,
                    jpeg_quality,
                ],
                outputs=[preset_dropdown, preset_dropdown]
            )

            delete_preset_button.click(
                fn=delete_preset,
                inputs=[preset_dropdown],
                outputs=[preset_dropdown, preset_dropdown]
            )

        return [
            enable_filter,
            dynamic_range_factor,
            hsv_hue_var,
            hsv_sat_var,
            hsv_val_var,
            rgb_noise_var,
            sharpen_intensity,
            warmth_factor,
            contrast_alpha,
            contrast_beta,
            desaturation_factor,
            grain_amount,
            jpeg_quality
        ]

    def postprocess(
        self,
        p,
        processed,
        enable_filter,
        dynamic_range_factor,
        hsv_hue_var,
        hsv_sat_var,
        hsv_val_var,
        rgb_noise_var,
        sharpen_intensity,
        warmth_factor,
        contrast_alpha,
        contrast_beta,
        desaturation_factor,
        grain_amount,
        jpeg_quality
    ):
        """Apply the amateur filter after generation, preserving metadata."""
        if not enable_filter:
            return

        for i in range(len(processed.images)):
            original_pil = processed.images[i]
            filtered_pil = apply_amateur_filter(
                original_pil,
                dynamic_range_factor=dynamic_range_factor,
                hsv_hue_var=hsv_hue_var,
                hsv_sat_var=hsv_sat_var,
                hsv_val_var=hsv_val_var,
                rgb_noise_var=rgb_noise_var,
                sharpen_intensity=sharpen_intensity,
                warmth_factor=warmth_factor,
                contrast_alpha=contrast_alpha,
                contrast_beta=contrast_beta,
                desaturation_factor=desaturation_factor,
                grain_amount=grain_amount,
                jpeg_quality=jpeg_quality
            )
            processed.images[i] = filtered_pil
