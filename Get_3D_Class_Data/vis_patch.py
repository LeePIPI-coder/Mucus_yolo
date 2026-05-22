#!/usr/bin/env python3
"""
Visualize 3D CT patches (.npy) from Class_3D_patch dataset.

Mode 1 - Interactive viewer (3 orthogonal views + slider):
    python vis_patch.py --file /data/Class_3D_patch/fold_0/000000_xxx_TP_j0.npy

    Keyboard shortcuts:
      Left/Right  prev/next file in directory
      Up/Down     change z-slice

Mode 2 - Batch HTML report (thumbnails grouped by patient):
    python vis_patch.py --fold 0
    python vis_patch.py --fold 0 --output my_report.html
"""

import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import argparse
import sys
import io
import base64
import pandas as pd
from tqdm import tqdm

HU_MIN, HU_MAX = -1000.0, 400.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata(fold_dir):
    csv_path = Path(fold_dir) / "metadata.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def make_thumbnail(patch, label, filename):
    """Generate 3-view thumbnail PNG bytes for a single patch."""
    nx, ny, nz = patch.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.8))
    axes[0].imshow(patch[:, :, cz].T, cmap="gray", origin="lower", aspect="auto")
    axes[0].set_title("Axial", fontsize=7)
    axes[0].axis("off")

    axes[1].imshow(patch[:, cy, :].T, cmap="gray", origin="lower", aspect="auto")
    axes[1].set_title("Coronal", fontsize=7)
    axes[1].axis("off")

    axes[2].imshow(patch[cx, :, :].T, cmap="gray", origin="lower", aspect="auto")
    axes[2].set_title("Sagittal", fontsize=7)
    axes[2].axis("off")

    border_color = "green" if label == 1 else "red"
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2)

    short_name = Path(filename).name
    if len(short_name) > 50:
        short_name = short_name[:47] + "..."
    fig.suptitle(short_name, fontsize=6, color=border_color)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _array_to_b64(arr):
    """Convert 2D uint8 numpy array to base64 PNG bytes."""
    from PIL import Image
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_slice_strips(patch):
    """Generate horizontal sprite-strip base64 PNGs for 3 orthogonal views.

    Each strip concatenates all slices along one axis.  The HTML viewer
    uses CSS overflow clipping to show one slice at a time.

    Returns dict: {view_name: {b64, n_slices, sw, sh, dim_label}}
      sw / sh = native pixel width / height of a single slice.
    """
    nx, ny, nz = patch.shape

    # Axial (xy plane, iterate z): slice shape (64, 64) → T → (64, 64)
    axial = np.hstack([patch[:, :, z].T for z in range(nz)])
    # Coronal (xz plane, iterate y): slice shape (64, 32) → T → (32, 64)
    coronal = np.hstack([patch[:, y, :].T for y in range(ny)])
    # Sagittal (yz plane, iterate x): slice shape (64, 32) → T → (32, 64)
    sagittal = np.hstack([patch[x, :, :].T for x in range(nx)])

    return {
        "axial":   {"b64": _array_to_b64(axial),   "n": nz, "sw": nx, "sh": ny, "dim": "z"},
        "coronal": {"b64": _array_to_b64(coronal), "n": ny, "sw": nx, "sh": nz, "dim": "y"},
        "sagittal": {"b64": _array_to_b64(sagittal), "n": nx, "sw": ny, "sh": nz, "dim": "x"},
    }


def make_interactive_html(filepath, output_path=None, scale=4):
    """Generate an interactive single-patch HTML viewer with sprite-strip sliders.

    Args:
        filepath: path to .npy patch file
        output_path: output .html path (default: <filepath>.html)
        scale: pixel scale factor for display (default 4x, so 64px→256px)
    """
    filepath = Path(filepath)
    patch = np.load(filepath)
    nx, ny, nz = patch.shape

    if output_path is None:
        output_path = filepath.with_suffix(".html")

    # Metadata
    fold_dir = filepath.parent
    meta_df = load_metadata(fold_dir)
    meta_row = None
    if meta_df is not None:
        matches = meta_df[meta_df["patch_file"] == filepath.name]
        if len(matches) > 0:
            meta_row = matches.iloc[0]

    # Try metadata first, fall back to filename parsing
    if "_FP" in filepath.stem:
        label_str = "FP"
    elif "_TP_" in filepath.name:
        label_str = "TP"
    else:
        label_str = "?"
    conf_str = "?"
    # patient_key is the second underscore-delimited token: 000000_PATIENT_TP_j0.npy
    parts = filepath.stem.split("_")
    patient_str = parts[1] if len(parts) >= 2 else "?"
    if meta_row is not None:
        label_str = "TP" if meta_row["label"] == 1 else "FP"
        conf_str = str(meta_row.get("confidence_level", "?"))
        patient_str = str(meta_row["patient_key"])

    strips = _make_slice_strips(patch)

    # Each view's display geometry at the chosen scale
    views_css = {}
    for name, s in strips.items():
        sw = s["sw"] * scale  # display slice width (px)
        sh = s["sh"] * scale  # display slice height (px)
        views_css[name] = {"sw": sw, "sh": sh}

    # Common slice width for sprite offset (all slices are 64 native → scale*64 px)
    slice_w_px = strips["axial"]["sw"] * scale  # 256 at 4x

    # ---- Build HTML ----
    border_color = "#27ae60" if label_str == "TP" else "#e74c3c"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Patch Viewer — {filepath.name}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 20px; min-height: 100vh; }}
  .header {{ text-align: center; margin-bottom: 20px; }}
  .header h1 {{ font-size: 1.1em; color: {border_color}; word-break: break-all; }}
  .header .meta {{ font-size: 0.85em; color: #999; margin-top: 4px; }}
  .viewer {{ display: flex; gap: 24px; flex-wrap: wrap; justify-content: center; }}
  .view {{ display: flex; flex-direction: column; align-items: center; background: #16213e; border-radius: 10px;
           padding: 12px; border: 2px solid #2a2a4a; }}
  .view:focus-within {{ border-color: #667eea; }}
  .view-label {{ font-size: 0.9em; font-weight: 600; margin-bottom: 8px; color: #ccc; }}
  .view-label span {{ color: #667eea; }}
  .viewport {{ overflow: hidden; position: relative; border: 1px solid #444; border-radius: 4px;
               background: #0a0a15; cursor: grab; }}
  .viewport:active {{ cursor: grabbing; }}
  .viewport img {{ position: absolute; left: 0; top: 0; image-rendering: pixelated;
                    pointer-events: none; user-select: none; }}
  .slider {{ width: 100%; margin: 8px 0 4px; accent-color: #667eea; cursor: pointer; }}
  .info {{ font-size: 0.78em; color: #888; }}
  .hint {{ margin-top: 20px; font-size: 0.78em; color: #555; text-align: center; }}
  .hint kbd {{ background: #2a2a4a; padding: 2px 7px; border-radius: 4px; border: 1px solid #444; }}
</style>
</head>
<body>

<div class="header">
  <h1>{filepath.name}</h1>
  <div class="meta">
    Patient: {patient_str} &ensp;|&ensp; Label: <b style="color:{border_color}">{label_str}</b>
    &ensp;|&ensp; Confidence: {conf_str}
    &ensp;|&ensp; Shape: {patch.shape} &ensp;|&ensp; dtype: {patch.dtype}
    &ensp;|&ensp; Range: [{patch.min()}, {patch.max()}]
  </div>
</div>

<div class="viewer">
"""

    for name, s in strips.items():
        v = views_css[name]
        dim = s["dim"]
        mid = s["n"] // 2
        html += f"""
  <div class="view" id="view-{name}">
    <div class="view-label">{name.title()} (<span>{dim}</span>-axis, {s['n']} slices)</div>
    <div class="viewport" id="vp-{name}"
         style="width:{v['sw']}px; height:{v['sh']}px;"
         data-name="{name}" data-n="{s['n']}" data-swpx="{slice_w_px}">
      <img src="data:image/png;base64,{s['b64']}"
           id="img-{name}"
           style="height:{v['sh']}px; width:auto;"
           draggable="false">
    </div>
    <input type="range" class="slider" id="slider-{name}"
           min="0" max="{s['n'] - 1}" value="{mid}"
           data-name="{name}">
    <div class="info" id="info-{name}">{dim}={mid}/{s['n'] - 1}</div>
  </div>"""

    html += """
</div>

<div class="hint">
  <kbd>Drag</kbd> slider &nbsp;|&nbsp;
  <kbd>Scroll</kbd> over viewport &nbsp;|&nbsp;
  <kbd>&larr;&rarr;</kbd> keys when viewport focused
</div>

<script>
(function() {
  // ---- State ----
  const views = ['axial', 'coronal', 'sagittal'];
  const state = {};
  views.forEach(name => {
    state[name] = {
      idx: parseInt(document.getElementById('slider-' + name).value),
      n: parseInt(document.getElementById('vp-' + name).dataset.n),
      swpx: parseInt(document.getElementById('vp-' + name).dataset.swpx),
      img: document.getElementById('img-' + name),
      slider: document.getElementById('slider-' + name),
      info: document.getElementById('info-' + name),
      dim: document.getElementById('vp-' + name).dataset.name === 'axial' ? 'z' :
           document.getElementById('vp-' + name).dataset.name === 'coronal' ? 'y' : 'x',
    };
  });

  function updateView(name) {
    const s = state[name];
    s.img.style.left = -(s.idx * s.swpx) + 'px';
    s.slider.value = s.idx;
    s.info.textContent = s.dim + '=' + s.idx + '/' + (s.n - 1);
  }

  // ---- Slider events ----
  views.forEach(name => {
    const s = state[name];
    s.slider.addEventListener('input', function() {
      s.idx = parseInt(this.value);
      updateView(name);
    });

    // ---- Scroll wheel on viewport ----
    const vp = document.getElementById('vp-' + name);
    vp.addEventListener('wheel', function(e) {
      e.preventDefault();
      if (e.deltaY > 0) s.idx = Math.min(s.idx + 1, s.n - 1);
      else s.idx = Math.max(s.idx - 1, 0);
      updateView(name);
    }, {{ passive: false }});

    // ---- Keyboard on viewport ----
    vp.addEventListener('keydown', function(e) {
      if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
        e.preventDefault();
        s.idx = Math.min(s.idx + 1, s.n - 1);
        updateView(name);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
        e.preventDefault();
        s.idx = Math.max(s.idx - 1, 0);
        updateView(name);
      }
    });
    vp.setAttribute('tabindex', '0');  // make focusable
  });

  // ---- Initialize ----
  views.forEach(name => updateView(name));
})();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Interactive HTML viewer saved to: {output_path}")
    print(f"  Patch: {filepath.name}")
    print(f"  Shape: {patch.shape}")
    print(f"  Label: {label_str}  |  Patient: {patient_str}")
    return str(output_path)


# ---------------------------------------------------------------------------
# Mode 4: XY Montage (stitch all z-slices of each patch into one grid image)
# ---------------------------------------------------------------------------

def make_xy_montage(patch, cols=8):
    """Arrange all z-slices of a 3D patch into a 2D grid (PIL Image, mode L).

    Args:
        patch: 3D numpy array (nx, ny, nz)
        cols: grid columns (default 8 → 4 rows for 32 slices)

    Returns:
        PIL Image (grayscale), size = (cols*nx, rows*ny)
    """
    from PIL import Image
    nx, ny, nz = patch.shape
    rows = (nz + cols - 1) // cols

    montage = Image.new("L", (cols * nx, rows * ny))
    for z in range(nz):
        col = z % cols
        row = z // cols
        sl = Image.fromarray(patch[:, :, z].T, mode="L")
        montage.paste(sl, (col * nx, row * ny))
    return montage


def stitch_patch(npy_path, output_path=None, cols=8):
    """Generate XY montage for a single .npy patch."""
    npy_path = Path(npy_path)
    patch = np.load(npy_path)
    img = make_xy_montage(patch, cols=cols)
    if output_path is None:
        output_path = npy_path.with_suffix(".png")
    img.save(output_path)
    print(f"Montage saved: {output_path}  ({img.size[0]}x{img.size[1]}, {patch.shape[2]} slices)")
    return str(output_path)


def stitch_fold(fold_dir, output_dir=None, cols=8):
    """Generate XY montages for all .npy files in a directory."""
    fold_dir = Path(fold_dir)
    if output_dir is None:
        output_dir = fold_dir / "xy_montages"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(fold_dir.glob("*.npy"))
    for fp in tqdm(npy_files, desc="Stitching XY montages"):
        patch = np.load(fp)
        img = make_xy_montage(patch, cols=cols)
        img.save(output_dir / f"{fp.stem}.png")

    print(f"Saved {len(npy_files)} montages to {output_dir}/")
    return str(output_dir)


# ---------------------------------------------------------------------------
# Mode 1: Interactive Viewer
# ---------------------------------------------------------------------------

class PatchViewer:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.fold_dir = self.filepath.parent
        self.meta_df = load_metadata(self.fold_dir)
        self.all_files = sorted(self.fold_dir.glob("*.npy"))
        if self.filepath in self.all_files:
            self.current_idx = self.all_files.index(self.filepath)
        else:
            self.current_idx = 0

        self.patch = None
        self.meta_row = None
        self._load_current()

        self.slice_z = self.patch.shape[2] // 2
        self.cx = self.patch.shape[0] // 2
        self.cy = self.patch.shape[1] // 2

        self._setup_ui()

    def _load_current(self):
        fp = self.all_files[self.current_idx]
        self.patch = np.load(fp)
        self.meta_row = None
        if self.meta_df is not None:
            matches = self.meta_df[self.meta_df["patch_file"] == fp.name]
            if len(matches) > 0:
                self.meta_row = matches.iloc[0]

    def _make_title(self):
        fp = self.all_files[self.current_idx]
        if self.meta_row is not None:
            label_str = "TP" if self.meta_row["label"] == 1 else "FP"
            conf = self.meta_row.get("confidence_level", "?")
            return (f"{fp.name} | Patient: {self.meta_row['patient_key']} | "
                    f"{label_str} | conf={conf} | Shape: {self.patch.shape}")
        return f"{fp.name} | Shape: {self.patch.shape} | dtype: {self.patch.dtype}"

    def _setup_ui(self):
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.canvas.manager.set_window_title(
            f"3D Patch Viewer — {self.all_files[self.current_idx].name}"
        )

        gs = self.fig.add_gridspec(2, 3, height_ratios=[20, 1], hspace=0.35, wspace=0.3)
        self.ax_axial = self.fig.add_subplot(gs[0, 0])
        self.ax_coronal = self.fig.add_subplot(gs[0, 1])
        self.ax_sagittal = self.fig.add_subplot(gs[0, 2])
        self.ax_slider = self.fig.add_subplot(gs[1, :])

        patch = self.patch
        nx, ny, nz = patch.shape
        sz = self.slice_z

        self.im_axial = self.ax_axial.imshow(
            patch[:, :, sz].T, cmap="gray", origin="lower", aspect="auto"
        )
        self.ax_axial.set_title(f"Axial (xy plane, z={sz}/{nz - 1})")
        self.ax_axial.set_xlabel("x")
        self.ax_axial.set_ylabel("y")

        self.im_coronal = self.ax_coronal.imshow(
            patch[:, self.cy, :].T, cmap="gray", origin="lower", aspect="auto"
        )
        self.line_coronal = self.ax_coronal.axhline(
            y=sz, color="r", linewidth=1, linestyle="--", alpha=0.7
        )
        self.ax_coronal.set_title(f"Coronal (xz plane, y={self.cy})")
        self.ax_coronal.set_xlabel("x")
        self.ax_coronal.set_ylabel("z")

        self.im_sagittal = self.ax_sagittal.imshow(
            patch[self.cx, :, :].T, cmap="gray", origin="lower", aspect="auto"
        )
        self.line_sagittal = self.ax_sagittal.axhline(
            y=sz, color="r", linewidth=1, linestyle="--", alpha=0.7
        )
        self.ax_sagittal.set_title(f"Sagittal (yz plane, x={self.cx})")
        self.ax_sagittal.set_xlabel("y")
        self.ax_sagittal.set_ylabel("z")

        self.fig.suptitle(self._make_title(), fontsize=10, fontweight="bold")

        self.slider = Slider(self.ax_slider, "Z slice", 0, nz - 1, valinit=sz, valfmt="%d")
        self.slider.on_changed(self._on_slider)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        print(f"Viewing: {self.all_files[self.current_idx].name}")
        print(f"Files in directory: {len(self.all_files)}")
        print("Controls: Left/Right = prev/next file | Up/Down = change z-slice")

    def _on_slider(self, val):
        self.slice_z = int(val)
        nz = self.patch.shape[2]
        self.im_axial.set_data(self.patch[:, :, self.slice_z].T)
        self.ax_axial.set_title(f"Axial (z={self.slice_z}/{nz - 1})")
        self.line_coronal.set_ydata([self.slice_z, self.slice_z])
        self.line_sagittal.set_ydata([self.slice_z, self.slice_z])
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == "right":
            self.current_idx = (self.current_idx + 1) % len(self.all_files)
            self._reload()
        elif event.key == "left":
            self.current_idx = (self.current_idx - 1) % len(self.all_files)
            self._reload()
        elif event.key == "up":
            new_z = min(self.slice_z + 1, self.patch.shape[2] - 1)
            self.slider.set_val(new_z)
        elif event.key == "down":
            new_z = max(self.slice_z - 1, 0)
            self.slider.set_val(new_z)

    def _reload(self):
        self._load_current()
        patch = self.patch
        nx, ny, nz = patch.shape
        self.cx, self.cy = nx // 2, ny // 2
        self.slice_z = min(self.slice_z, nz - 1)

        self.slider.valmin = 0
        self.slider.valmax = nz - 1
        self.slider.set_val(self.slice_z)

        self.im_axial.set_data(patch[:, :, self.slice_z].T)
        self.im_coronal.set_data(patch[:, self.cy, :].T)
        self.im_sagittal.set_data(patch[self.cx, :, :].T)

        self.line_coronal.set_ydata([self.slice_z, self.slice_z])
        self.line_sagittal.set_ydata([self.slice_z, self.slice_z])

        self.ax_axial.set_title(f"Axial (z={self.slice_z}/{nz - 1})")
        self.ax_coronal.set_title(f"Coronal (y={self.cy})")
        self.ax_sagittal.set_title(f"Sagittal (x={self.cx})")
        self.ax_axial.set_xlim(0, nx)
        self.ax_axial.set_ylim(0, ny)
        self.ax_coronal.set_xlim(0, nx)
        self.ax_coronal.set_ylim(0, nz)
        self.ax_sagittal.set_xlim(0, ny)
        self.ax_sagittal.set_ylim(0, nz)

        self.fig.suptitle(self._make_title(), fontsize=10, fontweight="bold")
        self.fig.canvas.manager.set_window_title(
            f"3D Patch Viewer — {self.all_files[self.current_idx].name}"
        )
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# Mode 2: HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(fold_dir, output_path):
    fold_dir = Path(fold_dir)
    output_path = Path(output_path)
    thumb_dir = output_path.parent / f"{output_path.stem}_thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    meta_df = load_metadata(fold_dir)
    npy_files = sorted(fold_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {fold_dir}")

    patients = {}

    for fp in tqdm(npy_files, desc="Generating thumbnails"):
        patch = np.load(fp)

        label = None
        meta_row = None
        if meta_df is not None:
            matches = meta_df[meta_df["patch_file"] == fp.name]
            if len(matches) > 0:
                meta_row = matches.iloc[0]
                label = int(meta_row["label"])
        if label is None:
            label = 0 if "_FP" in fp.stem else 1

        patient_key = (
            meta_row["patient_key"]
            if meta_row is not None
            else fp.stem.split("_", 1)[-1].rsplit("_", 2)[0]
        )

        png_bytes = make_thumbnail(patch, label, fp.name)
        png_name = fp.stem + ".png"
        with open(thumb_dir / png_name, "wb") as f:
            f.write(png_bytes)

        if patient_key not in patients:
            patients[patient_key] = {"TP": [], "FP": []}
        cat = "TP" if label == 1 else "FP"
        patients[patient_key][cat].append({
            "filename": fp.name,
            "png": png_name,
            "label": label,
            "confidence": meta_row["confidence_level"] if meta_row is not None else "?",
            "jitter": int(meta_row["jitter_index"]) if meta_row is not None else "?",
            "min": int(patch.min()),
            "max": int(patch.max()),
            "mean": float(patch.mean()),
        })

    tp_total = sum(len(v["TP"]) for v in patients.values())
    fp_total = sum(len(v["FP"]) for v in patients.values())

    html = _build_html(fold_dir.name, patients, npy_files, thumb_dir.name, tp_total, fp_total)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReport: {output_path}")
    print(f"Thumbnails: {thumb_dir}/ ({len(npy_files)} files)")
    print(f"TP: {tp_total}, FP: {fp_total}, Patients: {len(patients)}")


def _build_html(fold_name, patients, npy_files, thumb_dir_name, tp_total, fp_total):
    parts = [f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D Patch Report — {fold_name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 0; padding: 20px; background: #f0f2f5; color: #333; }}
  h1 {{ border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
  .nav {{ position: sticky; top: 10px; background: white; padding: 12px 20px;
          border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
          margin-bottom: 25px; z-index: 100; display: flex; gap: 20px;
          align-items: center; flex-wrap: wrap; }}
  .nav a {{ color: #667eea; text-decoration: none; font-weight: 500; }}
  .nav a:hover {{ text-decoration: underline; }}
  .nav strong {{ color: #555; }}
  .summary {{ display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; }}
  .stat {{ background: white; border-radius: 10px; padding: 16px 24px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; min-width: 120px; }}
  .stat .num {{ font-size: 2em; font-weight: bold; }}
  .stat.tp .num {{ color: #27ae60; }}
  .stat.fp .num {{ color: #e74c3c; }}
  .stat .lbl {{ color: #999; font-size: 0.85em; margin-top: 4px; }}
  .patient {{ background: white; border-radius: 10px; padding: 15px 20px;
              margin: 12px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
  .patient summary {{ cursor: pointer; font-weight: 600; font-size: 1.05em; }}
  .patient summary:hover {{ color: #667eea; }}
  .patch-grid {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .patch-card {{ border: 2px solid #e0e0e0; border-radius: 6px; padding: 3px;
                 background: #fafafa; }}
  .patch-card.tp {{ border-color: #27ae60; }}
  .patch-card.fp {{ border-color: #e74c3c; }}
  .patch-card img {{ display: block; max-width: 280px; height: auto; }}
  .patch-info {{ font-size: 0.68em; color: #aaa; margin-top: 2px; text-align: center;
                 line-height: 1.4; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.78em; color: white; margin-left: 6px; }}
  .badge.tp {{ background: #27ae60; }}
  .badge.fp {{ background: #e74c3c; }}
  h2 {{ margin-top: 35px; }}
  .toc {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 15px 0; }}
  .toc a {{ background: white; padding: 4px 12px; border-radius: 15px; color: #667eea;
            text-decoration: none; font-size: 0.85em; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
  .toc a:hover {{ background: #667eea; color: white; }}
</style>
</head>
<body>
<h1>3D CT Patch Report &mdash; {fold_name}</h1>

<div class="nav">
  <a href="#summary">Summary</a>
  <a href="#tp">TP Patches</a>
  <a href="#fp">FP Patches</a>
  <strong>Total: {len(npy_files)} patches | {len(patients)} patients</strong>
</div>

<div class="summary" id="summary">
  <div class="stat tp"><div class="num">{tp_total}</div><div class="lbl">TP patches</div></div>
  <div class="stat fp"><div class="num">{fp_total}</div><div class="lbl">FP patches</div></div>
  <div class="stat"><div class="num">{len(patients)}</div><div class="lbl">Patients</div></div>
  <div class="stat"><div class="num">{len(npy_files)}</div><div class="lbl">Total files</div></div>
</div>
"""]

    # ---- TP Section ----
    parts.append('<h2 id="tp">TP Patches <span class="badge tp">TP</span></h2>')
    parts.append('<div class="toc">')
    for pk in sorted(patients):
        if patients[pk]["TP"]:
            parts.append(f'<a href="#tp-{pk}">{pk} ({len(patients[pk]["TP"])})</a>')
    parts.append("</div>")

    for pk in sorted(patients):
        patches = patients[pk]["TP"]
        if not patches:
            continue
        parts.append(
            f'<div class="patient"><details>'
            f'<summary id="tp-{pk}">{pk} &mdash; {len(patches)} TP patches</summary>'
        )
        parts.append('<div class="patch-grid">')
        for p in patches:
            parts.append(f"""
  <div class="patch-card tp">
    <img src="{thumb_dir_name}/{p['png']}" alt="{p['filename']}" loading="lazy">
    <div class="patch-info">{p['filename']}<br>
      jitter={p['jitter']} | [{p['min']},{p['max']}] | &mu;={p['mean']:.0f}</div>
  </div>""")
        parts.append("</div></details></div>")

    # ---- FP Section ----
    parts.append('<h2 id="fp">FP Patches <span class="badge fp">FP</span></h2>')
    parts.append('<div class="toc">')
    for pk in sorted(patients):
        if patients[pk]["FP"]:
            parts.append(f'<a href="#fp-{pk}">{pk} ({len(patients[pk]["FP"])})</a>')
    parts.append("</div>")

    for pk in sorted(patients):
        patches = patients[pk]["FP"]
        if not patches:
            continue
        parts.append(
            f'<div class="patient"><details>'
            f'<summary id="fp-{pk}">{pk} &mdash; {len(patches)} FP patches</summary>'
        )
        parts.append('<div class="patch-grid">')
        for p in patches:
            parts.append(f"""
  <div class="patch-card fp">
    <img src="{thumb_dir_name}/{p['png']}" alt="{p['filename']}" loading="lazy">
    <div class="patch-info">{p['filename']}<br>
      conf={p['confidence']} | [{p['min']},{p['max']}] | &mu;={p['mean']:.0f}</div>
  </div>""")
        parts.append("</div></details></div>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D CT patches from Class_3D_patch dataset"
    )
    parser.add_argument(
        "--file", type=str,
        help="Path to a single .npy file (Mode 1: interactive viewer)"
    )
    parser.add_argument(
        "--html", type=str, metavar="FILE",
        help="Path to a single .npy file (Mode 3: interactive HTML viewer, works without display)"
    )
    parser.add_argument(
        "--fold", type=int,
        help="Fold index 0-4 (Mode 2: batch HTML report)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/Class_3D_patch",
        help="Root directory of Class_3D_patch"
    )
    parser.add_argument(
        "--stitch", type=str, metavar="PATH", default='/data/Class_3D_patch/jitter_10/fold_0',
        help="Generate XY-slice montage for a .npy file or all files in a directory"
    )
    parser.add_argument(
        "--stitch-out", type=str, default=None,
        help="Output directory for --stitch (default: <file>.png or <dir>/xy_montages/)"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output HTML path (Mode 2/3, default: auto-generated)"
    )

    args = parser.parse_args()

    # ---- Mode 3: Explicit HTML viewer ----
    if args.html:
        filepath = Path(args.html)
        if not filepath.exists():
            print(f"ERROR: file not found: {args.html}")
            sys.exit(1)
        plt.switch_backend("Agg")
        make_interactive_html(args.html, args.output)
        return

    # ---- Mode 4: XY Montage ----
    if args.stitch:
        stitch_path = Path(args.stitch)
        if not stitch_path.exists():
            print(f"ERROR: path not found: {args.stitch}")
            sys.exit(1)
        plt.switch_backend("Agg")
        if stitch_path.is_file():
            stitch_patch(args.stitch, output_path=args.stitch_out)
        else:
            stitch_fold(args.stitch, output_dir=args.stitch_out)
        return

    # ---- Mode 1: Interactive Viewer (with display) ----
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"ERROR: file not found: {args.file}")
            sys.exit(1)
        try:
            plt.switch_backend("TkAgg")
            viewer = PatchViewer(args.file)
            viewer.run()
        except Exception:
            print("No display available, falling back to interactive HTML viewer ...")
            plt.switch_backend("Agg")
            output_path = args.output or str(filepath.with_suffix(".html"))
            make_interactive_html(args.file, output_path=output_path)

    elif args.fold is not None:
        # ---- Mode 2: Batch HTML Report ----
        plt.switch_backend("Agg")
        fold_dir = Path(args.data_dir) / f"fold_{args.fold}"
        if not fold_dir.exists():
            print(f"ERROR: fold directory not found: {fold_dir}")
            sys.exit(1)
        output_path = args.output or str(
            Path(args.data_dir) / f"fold_{args.fold}_report.html"
        )
        generate_html_report(fold_dir, output_path)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Mode 1: Interactive viewer (needs display)")
        print("  python vis_patch.py --file /data/Class_3D_patch/fold_0/000000_xxx_TP_j0.npy")
        print("  # Mode 2: Batch HTML report")
        print("  python vis_patch.py --fold 0")
        print("  # Mode 3: Interactive HTML viewer (works headless)")
        print("  python vis_patch.py --html /data/Class_3D_patch/fold_0/000000_xxx_TP_j0.npy")
        print("  # Mode 4: XY-slice montage (32 slices → one 512x256 PNG)")
        print("  python vis_patch.py --stitch /data/Class_3D_patch/fold_0/000000_xxx_TP_j0.npy")
        print("  python vis_patch.py --stitch /data/Class_3D_patch/fold_0  # batch entire fold")


if __name__ == "__main__":
    main()
