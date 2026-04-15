import io
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


#Config generale

st.set_page_config(
    page_title="UNet Segmentation App",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CSS custom (Interface)
CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 1220px;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120, 120, 120, 0.15);
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 0.45rem;
    letter-spacing: -0.02em;
}

.subtitle {
    font-size: 1.08rem;
    color: #7c8798;
    margin-bottom: 1.25rem;
}

.hero-card {
    border: 1px solid rgba(120,120,120,0.12);
    border-radius: 24px;
    padding: 1.25rem 1.25rem 1rem 1.25rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.78) 0%, rgba(250,250,252,0.94) 100%);
    box-shadow: 0 14px 32px rgba(20, 24, 40, 0.06);
    margin-bottom: 1.15rem;
}

.soft-card {
    border: 1px solid rgba(120,120,120,0.10);
    border-radius: 22px;
    padding: 1rem 1rem 0.8rem 1rem;
    background: rgba(255,255,255,0.72);
    box-shadow: 0 10px 24px rgba(20, 24, 40, 0.05);
    margin-bottom: 1rem;
}

.info-card {
    border: 1px solid rgba(120,120,120,0.10);
    border-radius: 18px;
    padding: 1rem 1rem 0.9rem 1rem;
    background: rgba(255,255,255,0.62);
    min-height: 110px;
}

.metric-card {
    border-radius: 18px;
    padding: 1rem 1rem;
    background: rgba(255,255,255,0.8);
    border: 1px solid rgba(120,120,120,0.10);
    box-shadow: 0 10px 24px rgba(20, 24, 40, 0.05);
    text-align: center;
}

.image-card-title {
    font-size: 1.02rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
}

.small-label {
    font-size: 0.84rem;
    color: #7c8798;
}

.big-value {
    font-size: 1.28rem;
    font-weight: 800;
}

.status-note {
    border-radius: 16px;
    padding: 0.9rem 1rem;
    background: rgba(248, 250, 252, 0.95);
    border: 1px solid rgba(120,120,120,0.10);
    margin-top: 0.7rem;
    margin-bottom: 0.7rem;
}

.section-title {
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 0.5rem;
    margin-bottom: 0.9rem;
}

hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
    border: none;
    height: 1px;
    background: rgba(120,120,120,0.12);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Paramètres

@dataclass
class AppConfig:
    mask_threshold: float = 0.5
    overlay_alpha: float = 0.45
    target_size: Optional[Tuple[int, int]] = None  # ex: (256, 256)
    color_map: Tuple[int, int, int] = (0, 255, 180)  # turquoise
    model_name: str = "UNet"
    version: str = "1.0"


CONFIG = AppConfig()


# Model, prétraitement et inférence


MODEL_PATH = "statapp/models/FINAL.keras"


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def iou_loss(y_true, y_pred):
    return 1.0 - iou_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def _extract_boundary(y_true, kernel_size=3):
    dilated = tf.nn.max_pool2d(y_true, ksize=kernel_size, strides=1, padding='SAME')
    eroded = 1.0 - tf.nn.max_pool2d(1.0 - y_true, ksize=kernel_size, strides=1, padding='SAME')
    boundary = dilated - eroded
    return tf.clip_by_value(boundary, 0.0, 1.0)


def make_boundary_weighted_loss(w0, kernel_size=5):
    def boundary_weighted_bce_dice(y_true, y_pred):
        boundary = _extract_boundary(y_true, kernel_size)
        weight_map = 1.0 + w0 * boundary
        bce_per_pixel = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(weight_map * bce_per_pixel)
        d_loss = dice_loss(y_true, y_pred)
        return weighted_bce + d_loss
    boundary_weighted_bce_dice.__name__ = f'boundary_w{int(w0)}_bce_dice'
    return boundary_weighted_bce_dice


boundary_w5_bce_dice = make_boundary_weighted_loss(w0=5)
boundary_w10_bce_dice = make_boundary_weighted_loss(w0=10)

CUSTOM_OBJECTS = {
    'dice_coefficient': dice_coefficient,
    'iou_coefficient': iou_coefficient,
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'iou_loss': iou_loss,
    'boundary_w5_bce_dice': boundary_w5_bce_dice,
    'boundary_w10_bce_dice': boundary_w10_bce_dice,
}


@st.cache_resource
def load_model():
    """
    Charge le modèle Keras sauvegardé une seule fois.
    Adapte MODEL_PATH si ton fichier final porte un autre nom.
    """
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    return model


def preprocess_image(image: Image.Image, model) -> Tuple[tf.Tensor, Tuple[int, int]]:
    """
    Prétraitement cohérent avec ton entraînement:
    - RGB
    - conversion float32 dans [0, 1]
    - resize bilinear vers la taille d'entrée du modèle
    - ajout de la dimension batch

    Retourne:
    - tensor de shape (1, H, W, 3)
    - taille originale PIL (width, height)
    """
    image = image.convert("RGB")
    original_size = image.size

    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)

    input_h, input_w = model.input_shape[1], model.input_shape[2]
    img_tensor = tf.image.resize(img_tensor, (input_h, input_w), method='bilinear')
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    return img_tensor, original_size


def predict_mask(model, preprocessed_input: tf.Tensor) -> np.ndarray:
    """
    Inférence Keras/TensorFlow.
    Sortie attendue du modèle: (1, H, W, 1) avec sigmoid.
    Retourne un masque 2D float32 entre 0 et 1.
    """
    pred = model.predict(preprocessed_input, verbose=0)

    if pred.ndim == 4 and pred.shape[-1] == 1:
        mask = pred[0, :, :, 0]
    elif pred.ndim == 3:
        mask = pred[0]
    else:
        raise ValueError(f"Forme de sortie inattendue pour le modèle: {pred.shape}")

    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    return mask


def postprocess_mask(mask: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Post-traitement du masque:
    - clamp entre 0 et 1
    - resize vers la taille originale de l'image

    output_size est au format PIL: (width, height)
    """
    mask = np.clip(mask, 0.0, 1.0)
    mask_tensor = tf.convert_to_tensor(mask[..., np.newaxis], dtype=tf.float32)
    target_wh = output_size
    target_hw = (target_wh[1], target_wh[0])
    mask_resized = tf.image.resize(mask_tensor, target_hw, method='bilinear')
    mask_resized = tf.squeeze(mask_resized, axis=-1)
    return mask_resized.numpy().astype(np.float32)


# Utilitaire d'affichage

def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(mask_uint8)


def create_colored_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 180),
    alpha: float = 0.45,
) -> Image.Image:
    base = np.array(image.convert("RGB")).astype(np.uint8)
    mask_bin = (mask > CONFIG.mask_threshold).astype(np.uint8)

    overlay = np.zeros_like(base, dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    blended = base.copy().astype(np.float32)
    idx = mask_bin.astype(bool)
    blended[idx] = (1 - alpha) * blended[idx] + alpha * overlay[idx]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def create_contour_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 80, 80),
) -> Image.Image:
    """
    Crée un contour simple du masque sans dépendance externe.
    """
    base = np.array(image.convert("RGB")).astype(np.uint8)
    mask_bin = (mask > CONFIG.mask_threshold).astype(np.uint8)

    up = np.roll(mask_bin, -1, axis=0)
    down = np.roll(mask_bin, 1, axis=0)
    left = np.roll(mask_bin, -1, axis=1)
    right = np.roll(mask_bin, 1, axis=1)

    edge = ((mask_bin != up) | (mask_bin != down) | (mask_bin != left) | (mask_bin != right)) & (mask_bin == 1)
    result = base.copy()
    result[edge] = np.array(color, dtype=np.uint8)

    return Image.fromarray(result)


def compute_mask_stats(mask: np.ndarray) -> dict:
    mask_bin = mask > CONFIG.mask_threshold
    area_pixels = int(mask_bin.sum())
    total_pixels = int(mask_bin.size)
    area_ratio = 100.0 * area_pixels / max(total_pixels, 1)

    return {
        "area_pixels": area_pixels,
        "total_pixels": total_pixels,
        "area_ratio": area_ratio,
    }

#UI

def render_header():
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Segmentation d\'image avec UNet</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Charge une image, lance l\'inférence et visualise le masque, la superposition et le contour.</div>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.title("Paramètres")
    st.sidebar.markdown("Réglages visuels et seuils d'affichage")

    CONFIG.mask_threshold = st.sidebar.slider(
        "Seuil de masque",
        min_value=0.0,
        max_value=1.0,
        value=float(CONFIG.mask_threshold),
        step=0.01,
    )

    CONFIG.overlay_alpha = st.sidebar.slider(
        "Transparence overlay",
        min_value=0.0,
        max_value=1.0,
        value=float(CONFIG.overlay_alpha),
        step=0.05,
    )

    color_choice = st.sidebar.selectbox(
        "Couleur du masque",
        options=["Turquoise", "Rouge", "Vert", "Jaune", "Magenta"],
        index=0,
    )

    color_map = {
        "Turquoise": (0, 255, 180),
        "Rouge": (255, 80, 80),
        "Vert": (80, 220, 120),
        "Jaune": (255, 210, 70),
        "Magenta": (220, 80, 255),
    }
    CONFIG.color_map = color_map[color_choice]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Modèle**: {CONFIG.model_name}")
    st.sidebar.markdown(f"**Version app**: {CONFIG.version}")


def render_intro_cards():
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            '<div class="info-card"><b>1. Upload</b><br><br>Ajoute une image PNG, JPG ou JPEG.</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-card"><b>2. Inférence</b><br><br>Le modèle prédit le masque de segmentation.</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-card"><b>3. Visualisation</b><br><br>Affichage du masque, overlay et contour.</div>',
            unsafe_allow_html=True,
        )


def main():
    render_header()
    render_sidebar()

    render_intro_cards()
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Impossible de charger le modèle: {e}")
        st.stop()

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image d\'entrée</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Dépose une image ici",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

    run_inference = st.button("Lancer la segmentation", type="primary", use_container_width=True)

    if uploaded_file is None:
        st.markdown(
            '<div class="status-note">Charge une image puis clique sur <b>Lancer la segmentation</b>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.markdown(
        '<div class="status-note">Image chargée. Clique sur <b>Lancer la segmentation</b> pour générer le résultat.</div>',
        unsafe_allow_html=True,
    )
    st.image(image, caption="Image originale", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not run_inference:
        return

    start_time = time.perf_counter()

    with st.spinner("Inférence en cours..."):
        preprocessed, original_size = preprocess_image(image, model)
        raw_mask = predict_mask(model, preprocessed)
        final_mask = postprocess_mask(raw_mask, output_size=original_size)
        overlay = create_colored_overlay(
            image=image,
            mask=final_mask,
            color=CONFIG.color_map,
            alpha=CONFIG.overlay_alpha,
        )
        contour = create_contour_overlay(image=image, mask=final_mask)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    stats = compute_mask_stats(final_mask)
    mask_pil = mask_to_pil(final_mask)

    st.markdown('<div class="section-title">Résultats</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="small-label">Temps d\'inférence</div><div class="big-value">{elapsed_ms:.1f} ms</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="small-label">Pixels segmentés</div><div class="big-value">{stats["area_pixels"]}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="small-label">Surface estimée</div><div class="big-value">{stats["area_ratio"]:.2f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    a, b, c = st.columns(3, gap="large")
    with a:
        st.markdown('<div class="image-card-title">Originale</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with b:
        st.markdown('<div class="image-card-title">Masque prédit</div>', unsafe_allow_html=True)
        st.image(mask_pil, use_container_width=True)
    with c:
        st.markdown('<div class="image-card-title">Superposition</div>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="image-card-title">Contour du masque</div>', unsafe_allow_html=True)
    st.image(contour, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Téléchargement")
    d1, d2, d3 = st.columns(3, gap="large")
    with d1:
        st.download_button(
            label="Télécharger le masque",
            data=image_to_bytes(mask_pil, fmt="PNG"),
            file_name="mask_prediction.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="Télécharger l'overlay",
            data=image_to_bytes(overlay, fmt="PNG"),
            file_name="overlay_prediction.png",
            mime="image/png",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            label="Télécharger le contour",
            data=image_to_bytes(contour, fmt="PNG"),
            file_name="contour_prediction.png",
            mime="image/png",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


    st.caption("Application Streamlit de démonstration pour segmentation d'image avec modèle UNet.")


if __name__ == "__main__":
    main()
