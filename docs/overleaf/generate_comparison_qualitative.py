#!/usr/bin/env python3
"""
Generate qualitative comparison figure:
K-means LAB / CNN Patch / U-Net Final on 4 representative test images.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import numpy as np
from PIL import Image as PILImage
from skimage import color
from sklearn.cluster import KMeans
from scipy.ndimage import label as ndimage_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import zipfile
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# ─── Paths ──────────────────────────────────────────────────────────────
BASE = '/Users/sfarmehdi/Stat_app/statapp'
TEST_IMG_DIR = os.path.join(BASE, 'dataset_ISIC/ISBI2016_ISIC_Part1_Test_Data')
TEST_GT_DIR = os.path.join(BASE, 'dataset_ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth')
PATCH_MODEL_PATH = os.path.join(BASE, 'models/patch_cnn_p32.keras')
UNET_MODEL_PATH = os.path.join(BASE, 'models/FINAL.keras')
OUTPUT_PATH = os.path.join(BASE, 'docs/overleaf/figures/fig_comparison_qualitative.pdf')

# ─── Selected images ────────────────────────────────────────────────────
SELECTED = [
    ('ISIC_0000003', 0.9715, 'Cas facile'),
    ('ISIC_0000036', 0.8688, 'Cas moyen'),
    ('ISIC_0000198', 0.7614, 'Cas difficile'),
    ('ISIC_0000125', 0.4712, u'Cas d\u2019\u00e9chec'),
]

# ─── Fix .keras files: strip quantization_config ────────────────────────
def strip_quantization_config(obj):
    """Recursively remove 'quantization_config' keys from nested dicts/lists."""
    if isinstance(obj, dict):
        return {k: strip_quantization_config(v) for k, v in obj.items()
                if k != 'quantization_config'}
    elif isinstance(obj, list):
        return [strip_quantization_config(item) for item in obj]
    return obj

def fix_keras_file(src_path):
    """
    Create a patched copy of a .keras file with quantization_config removed
    from config.json. Returns path to the patched file (temp).
    """
    tmpdir = tempfile.mkdtemp()
    patched_path = os.path.join(tmpdir, os.path.basename(src_path))

    with zipfile.ZipFile(src_path, 'r') as zin:
        with zipfile.ZipFile(patched_path, 'w') as zout:
            for item in zin.namelist():
                data = zin.read(item)
                if item == 'config.json':
                    config = json.loads(data)
                    config = strip_quantization_config(config)
                    data = json.dumps(config).encode('utf-8')
                zout.writestr(item, data)

    return patched_path, tmpdir

# ─── Custom objects for U-Net ────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f, y_pred_f = K.flatten(y_true), K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

custom_objects = {
    'dice_coefficient': dice_coefficient,
    'iou_coefficient': iou_coefficient,
    'dice_loss': dice_loss,
}

# ─── Helper: compute DSC ────────────────────────────────────────────────
def compute_dsc(pred_mask, gt_mask, smooth=1e-6):
    pred_f = pred_mask.flatten()
    gt_f = gt_mask.flatten()
    inter = np.sum(pred_f * gt_f)
    return (2.0 * inter + smooth) / (np.sum(pred_f) + np.sum(gt_f) + smooth)

# ─── K-means LAB segmentation ───────────────────────────────────────────
def kmeans_lab_segment(img_rgb_01, k=2):
    img_lab = color.rgb2lab(img_rgb_01)
    h, w, c = img_lab.shape
    pixels = img_lab.reshape(-1, c)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(pixels)
    cluster_L = [pixels[labels == i, 0].mean() for i in range(k)]
    lesion_cluster = np.argmin(cluster_L)
    mask = (labels == lesion_cluster).reshape(h, w).astype(np.float32)
    # keep largest connected component
    labeled, n = ndimage_label(mask)
    if n > 1:
        sizes = [np.sum(labeled == kk) for kk in range(1, n + 1)]
        mask = (labeled == np.argmax(sizes) + 1).astype(np.float32)
    return mask

# ─── CNN Patch segmentation ─────────────────────────────────────────────
def segment_cnn_patch(model, img_256, patch_size=32, stride=8):
    half = patch_size // 2
    ys = list(range(half, 256 - half, stride))
    xs = list(range(half, 256 - half, stride))
    patches = []
    for y in ys:
        for x in xs:
            patches.append(img_256[y - half:y + half, x - half:x + half])
    patches = np.array(patches)
    preds = model.predict(patches, batch_size=64, verbose=0).flatten()
    grid = preds.reshape(len(ys), len(xs))
    prob = np.array(
        PILImage.fromarray(grid.astype(np.float32)).resize(
            (256, 256), PILImage.BILINEAR
        )
    )
    mask = (prob > 0.5).astype(np.float32)
    labeled, n = ndimage_label(mask)
    if n > 1:
        sizes = [np.sum(labeled == kk) for kk in range(1, n + 1)]
        mask = (labeled == np.argmax(sizes) + 1).astype(np.float32)
    return mask

# ─── Error map (VP=green, FP=red, FN=blue) ──────────────────────────────
def make_error_map(pred_mask, gt_mask):
    h, w = pred_mask.shape
    error = np.ones((h, w, 3), dtype=np.float32)  # white background
    vp = (pred_mask > 0.5) & (gt_mask > 0.5)       # true positive
    fp = (pred_mask > 0.5) & (gt_mask < 0.5)       # false positive
    fn = (pred_mask < 0.5) & (gt_mask > 0.5)       # false negative
    error[vp] = [0.2, 0.8, 0.2]   # green
    error[fp] = [0.9, 0.2, 0.2]   # red
    error[fn] = [0.2, 0.4, 0.9]   # blue
    return error

# ─── Load models (with patched .keras files) ────────────────────────────
tmpdirs = []

print("Patching and loading CNN Patch model...")
patched_patch, tmpdir1 = fix_keras_file(PATCH_MODEL_PATH)
tmpdirs.append(tmpdir1)
patch_model = load_model(patched_patch)
print(f"  Input: {patch_model.input_shape}, Output: {patch_model.output_shape}")

print("Patching and loading U-Net FINAL model...")
patched_unet, tmpdir2 = fix_keras_file(UNET_MODEL_PATH)
tmpdirs.append(tmpdir2)
unet_model = load_model(patched_unet, custom_objects=custom_objects)
print(f"  Input: {unet_model.input_shape}, Output: {unet_model.output_shape}")
print("Models loaded.")

# ─── Load images and generate predictions ────────────────────────────────
rows_data = []
for img_id, unet_dsc_ref, label in SELECTED:
    print(f"\nProcessing {img_id} ({label})...")
    # Load original image
    img_path = os.path.join(TEST_IMG_DIR, img_id + '.jpg')
    img_pil = PILImage.open(img_path).convert('RGB')
    img_256 = img_pil.resize((256, 256), PILImage.BILINEAR)
    img_arr = np.array(img_256).astype(np.float32) / 255.0

    # Load ground truth mask
    gt_path = os.path.join(TEST_GT_DIR, img_id + '_Segmentation.png')
    gt_pil = PILImage.open(gt_path).convert('L')
    gt_256 = gt_pil.resize((256, 256), PILImage.NEAREST)
    gt_arr = (np.array(gt_256).astype(np.float32) / 255.0)
    gt_arr = (gt_arr > 0.5).astype(np.float32)

    # K-means LAB
    kmeans_mask = kmeans_lab_segment(img_arr, k=2)
    kmeans_dsc = compute_dsc(kmeans_mask, gt_arr)
    print(f"  K-means LAB DSC: {kmeans_dsc:.4f}")

    # CNN Patch
    cnn_mask = segment_cnn_patch(patch_model, img_arr, patch_size=32, stride=8)
    cnn_dsc = compute_dsc(cnn_mask, gt_arr)
    print(f"  CNN Patch DSC: {cnn_dsc:.4f}")

    # U-Net Final
    unet_pred = unet_model.predict(img_arr[np.newaxis, ...], verbose=0)[0, :, :, 0]
    unet_mask = (unet_pred > 0.5).astype(np.float32)
    unet_dsc = compute_dsc(unet_mask, gt_arr)
    print(f"  U-Net Final DSC: {unet_dsc:.4f}")

    # Error map for U-Net
    error_map = make_error_map(unet_mask, gt_arr)

    rows_data.append({
        'img_id': img_id,
        'label': label,
        'img': img_arr,
        'gt': gt_arr,
        'kmeans_mask': kmeans_mask,
        'kmeans_dsc': kmeans_dsc,
        'cnn_mask': cnn_mask,
        'cnn_dsc': cnn_dsc,
        'unet_mask': unet_mask,
        'unet_dsc': unet_dsc,
        'error_map': error_map,
    })

# Cleanup temp dirs
for td in tmpdirs:
    shutil.rmtree(td, ignore_errors=True)

# ─── Create figure ──────────────────────────────────────────────────────
print("\nCreating figure...")
fig, axes = plt.subplots(4, 6, figsize=(16, 11))

col_titles = [
    'Image originale',
    u'V\u00e9rit\u00e9 terrain',
    'K-means LAB',
    'CNN Patch (P=32)',
    u'U-Net Final',
    u'Carte d\u2019erreur U-Net',
]

for row_idx, rd in enumerate(rows_data):
    ax_row = axes[row_idx]

    # Col 0: Original image
    ax_row[0].imshow(rd['img'])
    ax_row[0].set_ylabel(rd['label'], fontsize=11, fontweight='bold',
                         rotation=90, labelpad=10)

    # Col 1: Ground truth
    ax_row[1].imshow(rd['gt'], cmap='gray', vmin=0, vmax=1)

    # Col 2: K-means LAB
    ax_row[2].imshow(rd['kmeans_mask'], cmap='gray', vmin=0, vmax=1)
    ax_row[2].set_xlabel(f"DSC = {rd['kmeans_dsc']:.3f}", fontsize=9, color='#333')

    # Col 3: CNN Patch
    ax_row[3].imshow(rd['cnn_mask'], cmap='gray', vmin=0, vmax=1)
    ax_row[3].set_xlabel(f"DSC = {rd['cnn_dsc']:.3f}", fontsize=9, color='#333')

    # Col 4: U-Net Final
    ax_row[4].imshow(rd['unet_mask'], cmap='gray', vmin=0, vmax=1)
    ax_row[4].set_xlabel(f"DSC = {rd['unet_dsc']:.3f}", fontsize=9, color='#333')

    # Col 5: Error map
    ax_row[5].imshow(rd['error_map'])

    # Remove ticks
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])

# Column titles
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold', pad=8)

# Add legend for error map in bottom right
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=[0.2, 0.8, 0.2], label='Vrai Positif (VP)'),
    Patch(facecolor=[0.9, 0.2, 0.2], label='Faux Positif (FP)'),
    Patch(facecolor=[0.2, 0.4, 0.9], label=u'Faux N\u00e9gatif (FN)'),
]
fig.legend(
    handles=legend_elements,
    loc='lower right',
    fontsize=9,
    framealpha=0.9,
    ncol=3,
    bbox_to_anchor=(0.98, 0.01),
)

fig.suptitle(
    u'Comparaison qualitative \u2014 K-means LAB / CNN Patch / U-Net Final (test set ISBI 2016)',
    fontsize=13,
    fontweight='bold',
    y=0.995,
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
print(f"\nFigure saved to: {OUTPUT_PATH}")
plt.close(fig)
