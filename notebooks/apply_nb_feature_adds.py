"""
apply_nb_feature_adds.py
------------------------
Second patch for jp_void_spatial_features.ipynb.
Adds the missing features requested in Wanda's review:
  Fix 1  — center_frac / edge_frac (concentric void mass split)
  Fix 2  — largest_void_centroid_r, largest_void_edge_dist, largest_void_circularity
  Fix 3  — radial_var, radial_skew (radial second moments)
  Fix 4  — pad_in_frame_frac + per-pad gating at PAD_IN_FRAME_THRESHOLD
  Fix 5  — electrical_min_*, electrical_max_*, electrical_asym_* (additive; keep mean)
  Fix 6  — thread valid[i] through process_led_type -> led_feature_vector -> pad_features

Run once, then delete:
    python apply_nb_feature_adds.py

Does NOT run the notebook. Writes backup to .ipynb.bak2.
"""

import json
import shutil
import sys
from pathlib import Path

NB_PATH = Path(
    r"C:\Users\JPagtalu\Documents\Joanna\MEng AI\ai221\ai221_project\notebooks"
    r"\jp_void_spatial_features_improved.ipynb"
)
BAK_PATH = NB_PATH.with_suffix(".ipynb.bak2")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def source_str(cell):
    """Join source list to a single string (as stored in .ipynb)."""
    return "".join(cell["source"])


def source_list(text):
    """Split a string back to the per-line list format ipynb expects."""
    lines = text.splitlines(keepends=True)
    return lines


def find_cell(nb, snippet):
    """Return the first code cell whose source contains *snippet*."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and snippet in source_str(cell):
            return cell
    return None


def apply_fix(label, cell, old, new):
    """Replace old with new in cell source; print [OK] or [FAIL]."""
    src = source_str(cell)
    if old not in src:
        print(f"  [FAIL] {label}: old_string not found in cell")
        return False
    patched = src.replace(old, new, 1)
    cell["source"] = source_list(patched)
    print(f"  [OK]   {label}")
    return True


# ---------------------------------------------------------------------------
# load + validate
# ---------------------------------------------------------------------------

print("Loading notebook…")
try:
    raw = NB_PATH.read_bytes()
    nb = json.loads(raw)
except Exception as exc:
    print(f"[FAIL] Cannot parse notebook JSON: {exc}")
    sys.exit(1)

print("[OK]   JSON valid — proceeding")

# backup
shutil.copy2(NB_PATH, BAK_PATH)
print(f"[OK]   Backup written to {BAK_PATH.name}")

# ---------------------------------------------------------------------------
# Fix 6a — add `scipy.stats` import to the imports cell
# ---------------------------------------------------------------------------
print("\n--- Fix 6a: add scipy.stats import ---")
imports_cell = find_cell(nb, "from scipy import ndimage as ndi")
if imports_cell is None:
    print("  [FAIL] imports cell not found")
else:
    apply_fix(
        "add scipy.stats import",
        imports_cell,
        "from scipy import ndimage as ndi\n",
        "from scipy import ndimage as ndi\nfrom scipy.stats import skew as _scipy_skew\n",
    )

# ---------------------------------------------------------------------------
# Fix 1–5: rewrite pad_features cell
# ---------------------------------------------------------------------------
print("\n--- Fix 1-5: rewrite pad_features ---")

PAD_FEATURES_ANCHOR = "def pad_features(void_mask, pad_bbox, per_pad_mask=None):"

pad_cell = find_cell(nb, PAD_FEATURES_ANCHOR)
if pad_cell is None:
    print("  [FAIL] pad_features cell not found — aborting remaining fixes")
    sys.exit(1)

OLD_PAD_FEATURES_FULL = '''\
def pad_features(void_mask, pad_bbox, per_pad_mask=None):
    """Compute spatial void features for one pad, in normalized pad coordinates."""
    x0, y0, x1, y1, role, _area_px = pad_bbox
    pad_crop = void_mask[y0:y1, x0:x1]
    pad_h, pad_w = pad_crop.shape
    pad_area = pad_h * pad_w

    feats = {"void_area_frac": float(pad_crop.sum()) / pad_area}

    if pad_crop.sum() == 0:
        feats.update({
            "centroid_x": 0.0, "centroid_y": 0.0,
            "q_NW": 0.0, "q_NE": 0.0, "q_SW": 0.0, "q_SE": 0.0,
            "cluster_count": 0, "mean_circularity": np.nan,
            "largest_void_frac": 0.0, "edge_proximity_min": np.nan,
        })
        return feats

    # --- Pad centroid + principal-axis frame ---
    # Compute pad centroid and orientation via regionprops on the pad mask
    # (per_pad_mask cropped to bbox, or a full-bbox rectangle if None).
    if per_pad_mask is not None:
        pad_region_mask = (per_pad_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    else:
        pad_region_mask = np.ones((pad_h, pad_w), dtype=np.uint8)

    _pad_lbl = label(pad_region_mask, connectivity=2)
    _pad_props = regionprops(_pad_lbl)
    if _pad_props:
        pc_y, pc_x = _pad_props[0].centroid      # (row, col) in bbox-local coords
        theta = _pad_props[0].orientation         # radians, CCW from col-axis
    else:
        pc_y, pc_x = pad_h / 2.0, pad_w / 2.0
        theta = 0.0

    # Void pixel coords in bbox-local frame, centred on pad centroid.
    ys_v, xs_v = np.nonzero(pad_crop)
    dy = ys_v - pc_y
    dx = xs_v - pc_x

    # Rotate void coords into pad principal-axis frame.
    # skimage orientation: angle of major axis from positive col-axis (radians).
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx_r =  dx * cos_t + dy * sin_t  # principal-axis x (along major axis)
    dy_r = -dx * sin_t + dy * cos_t  # principal-axis y (along minor axis)

    # Void centroid in normalised pad coordinates.
    half_w = pad_w / 2.0 if pad_w > 0 else 1.0
    half_h = pad_h / 2.0 if pad_h > 0 else 1.0
    if ys_v.size > 0:
        feats["centroid_x"] = float(dx_r.mean()) / half_w
        feats["centroid_y"] = float(dy_r.mean()) / half_h
    else:
        feats["centroid_x"] = 0.0
        feats["centroid_y"] = 0.0

    # Quadrant counts in principal-axis frame.
    # Denominator = per-pad-mask pixel count (or bbox area if mask unavailable).
    pad_area_denom = float(pad_region_mask.sum()) if pad_region_mask.sum() > 0 else pad_area
    if ys_v.size > 0:
        nw = int(((dx_r < 0) & (dy_r < 0)).sum())
        ne = int(((dx_r >= 0) & (dy_r < 0)).sum())
        sw = int(((dx_r < 0) & (dy_r >= 0)).sum())
        se = int(((dx_r >= 0) & (dy_r >= 0)).sum())
        tot = nw + ne + sw + se
        feats.update({"q_NW": nw / tot, "q_NE": ne / tot,
                      "q_SW": sw / tot, "q_SE": se / tot})
    else:
        feats.update({"q_NW": 0.0, "q_NE": 0.0, "q_SW": 0.0, "q_SE": 0.0})

    lbl = label(pad_crop, connectivity=2)
    regions = regionprops(lbl)
    feats["cluster_count"] = len(regions)
    areas  = np.array([r.area for r in regions], dtype=float)
    perims = np.array([r.perimeter if r.perimeter > 0 else 1.0 for r in regions])
    circs  = 4 * np.pi * areas / (perims ** 2)
    # Area-weighted mean: larger voids contribute proportionally more.
    feats["mean_circularity"]  = float(np.average(circs, weights=areas))
    feats["largest_void_frac"] = float(areas.max()) / pad_area_denom

    # edge_proximity_min: distance from each void pixel to the nearest
    # pad-boundary pixel, normalised by the pad's equivalent radius so
    # the feature is dimensionless and comparable across pad sizes.
    if per_pad_mask is not None:
        # EDT on the full per-pad mask gives true distance-from-pad-edge.
        # EDT value = 0 outside the pad, positive inside => min > 0 means
        # the closest void pixel is that many px from the pad boundary.
        pad_edt = ndi.distance_transform_edt(per_pad_mask > 0)
        pad_area_true = float((per_pad_mask > 0).sum())
        equiv_radius = np.sqrt(pad_area_true / np.pi)
        # Void pixels inside this pad (full-image coordinates).
        ys_full, xs_full = np.nonzero(
            (void_mask > 0) & (per_pad_mask > 0)
        )
        if ys_full.size > 0:
            feats["edge_proximity_min"] = (
                float(pad_edt[ys_full, xs_full].min()) / max(equiv_radius, 1.0)
            )
        else:
            feats["edge_proximity_min"] = np.nan
    else:
        # Fallback: bbox-interior approximation (original behaviour).
        interior = np.ones_like(pad_crop, dtype=np.uint8)
        interior[0, :]  = 0; interior[-1, :] = 0
        interior[:, 0]  = 0; interior[:, -1] = 0
        dist = ndi.distance_transform_edt(interior)
        normalizer = min(pad_h, pad_w) / 2
        feats["edge_proximity_min"] = float(dist[pad_crop > 0].min()) / normalizer
    return feats'''

NEW_PAD_FEATURES_FULL = '''\
# --- Named constants for pad feature extraction ---
PAD_IN_FRAME_THRESHOLD = 0.9   # pads with less coverage than this are gated (all features -> NaN)
INNER_RADIUS_FRAC = 0.5        # inner concentric region boundary: 0.5 * r_eq
ASYM_EPS = 1e-9                # denominator guard for electrical asymmetry


def pad_features(void_mask, pad_bbox, per_pad_mask=None, valid_mask=None):
    """Compute spatial void features for one pad, in normalized pad coordinates.

    Parameters
    ----------
    void_mask : ndarray (H, W) uint8
        Binary void mask (1 = void pixel) for the full image.
    pad_bbox : tuple
        (x0, y0, x1, y1, role, area_px) — bounding box and role of this pad.
    per_pad_mask : ndarray (H, W) uint8 or None
        Full-image binary mask of this pad's exact shape.  If None, the bbox
        rectangle is used as a fallback.
    valid_mask : ndarray (H, W) uint8 or None
        Full-image mask: 1 where the registered image has valid pixel data.
        Used to compute pad_in_frame_frac and gate unreliable pads.

    Returns
    -------
    dict
        Per-pad feature dictionary.  When pad_in_frame_frac < PAD_IN_FRAME_THRESHOLD
        all void-dependent features are set to NaN and a warning is printed.
    """
    x0, y0, x1, y1, role, _area_px = pad_bbox
    pad_crop = void_mask[y0:y1, x0:x1]
    pad_h, pad_w = pad_crop.shape
    pad_area = pad_h * pad_w

    # ------------------------------------------------------------------
    # Fix 4 — pad_in_frame_frac: fraction of this pad inside the valid
    # registered-image region.  Gate the pad if coverage is too low.
    # ------------------------------------------------------------------
    if per_pad_mask is not None and valid_mask is not None:
        pad_pixels = float((per_pad_mask > 0).sum())
        in_frame   = float(((per_pad_mask > 0) & (valid_mask > 0)).sum())
        pad_in_frame_frac = (in_frame / pad_pixels) if pad_pixels > 0 else 0.0
    elif valid_mask is not None:
        # fallback: use bbox
        bbox_pixels = float(pad_area)
        in_frame = float(valid_mask[y0:y1, x0:x1].sum())
        pad_in_frame_frac = (in_frame / bbox_pixels) if bbox_pixels > 0 else 0.0
    else:
        pad_in_frame_frac = 1.0   # no valid mask supplied: assume fully in frame

    _NAN_FEATS = {
        "void_area_frac": np.nan,
        "centroid_x": np.nan, "centroid_y": np.nan,
        "q_NW": np.nan, "q_NE": np.nan, "q_SW": np.nan, "q_SE": np.nan,
        "cluster_count": np.nan, "mean_circularity": np.nan,
        "largest_void_frac": np.nan, "edge_proximity_min": np.nan,
        "center_frac": np.nan, "edge_frac": np.nan,
        "largest_void_centroid_r": np.nan,
        "largest_void_edge_dist": np.nan,
        "largest_void_circularity": np.nan,
        "radial_var": np.nan, "radial_skew": np.nan,
        "pad_in_frame_frac": pad_in_frame_frac,
    }

    if pad_in_frame_frac < PAD_IN_FRAME_THRESHOLD:
        import warnings as _w
        _w.warn(
            f"pad_features: pad ({role}, bbox=({x0},{y0})-({x1},{y1})) "
            f"pad_in_frame_frac={pad_in_frame_frac:.3f} < {PAD_IN_FRAME_THRESHOLD} "
            "— all features set to NaN.",
            RuntimeWarning, stacklevel=2,
        )
        return _NAN_FEATS

    feats = {
        "void_area_frac": float(pad_crop.sum()) / pad_area,
        "pad_in_frame_frac": pad_in_frame_frac,
    }

    # --- zero-void fast path ---
    if pad_crop.sum() == 0:
        feats.update({
            "centroid_x": 0.0, "centroid_y": 0.0,
            "q_NW": 0.0, "q_NE": 0.0, "q_SW": 0.0, "q_SE": 0.0,
            "cluster_count": 0, "mean_circularity": np.nan,
            "largest_void_frac": 0.0, "edge_proximity_min": np.nan,
            # Fix 1
            "center_frac": np.nan, "edge_frac": np.nan,
            # Fix 2
            "largest_void_centroid_r": np.nan,
            "largest_void_edge_dist": np.nan,
            "largest_void_circularity": np.nan,
            # Fix 3
            "radial_var": np.nan, "radial_skew": np.nan,
        })
        return feats

    # --- Pad mask & geometry ---
    if per_pad_mask is not None:
        pad_region_mask = (per_pad_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    else:
        pad_region_mask = np.ones((pad_h, pad_w), dtype=np.uint8)

    _pad_lbl = label(pad_region_mask, connectivity=2)
    _pad_props = regionprops(_pad_lbl)
    if _pad_props:
        pc_y, pc_x = _pad_props[0].centroid      # (row, col) in bbox-local coords
        theta = _pad_props[0].orientation         # radians, CCW from col-axis
    else:
        pc_y, pc_x = pad_h / 2.0, pad_w / 2.0
        theta = 0.0

    # Denominator = per-pad-mask pixel count (or bbox area if mask unavailable).
    pad_area_denom = float(pad_region_mask.sum()) if pad_region_mask.sum() > 0 else float(pad_area)

    # Equivalent radius of this pad (dimensionless normaliser).
    equiv_radius = np.sqrt(pad_area_denom / np.pi)
    r_inner = INNER_RADIUS_FRAC * equiv_radius   # Fix 1 boundary

    # Void pixel coords in bbox-local frame, centred on pad centroid.
    ys_v, xs_v = np.nonzero(pad_crop)
    dy = ys_v - pc_y
    dx = xs_v - pc_x

    # Radial distances of all void pixels from pad centroid (unnormalised, px).
    radii_px = np.hypot(dx, dy)
    radii_norm = radii_px / max(equiv_radius, 1.0)   # dimensionless

    # Rotate void coords into pad principal-axis frame.
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx_r =  dx * cos_t + dy * sin_t
    dy_r = -dx * sin_t + dy * cos_t

    # Void centroid in normalised pad coordinates.
    half_w = pad_w / 2.0 if pad_w > 0 else 1.0
    half_h = pad_h / 2.0 if pad_h > 0 else 1.0
    feats["centroid_x"] = float(dx_r.mean()) / half_w
    feats["centroid_y"] = float(dy_r.mean()) / half_h

    # Quadrant counts in principal-axis frame.
    nw = int(((dx_r < 0) & (dy_r < 0)).sum())
    ne = int(((dx_r >= 0) & (dy_r < 0)).sum())
    sw = int(((dx_r < 0) & (dy_r >= 0)).sum())
    se = int(((dx_r >= 0) & (dy_r >= 0)).sum())
    tot = nw + ne + sw + se
    feats.update({"q_NW": nw / tot, "q_NE": ne / tot,
                  "q_SW": sw / tot, "q_SE": se / tot})

    # --- Fix 1: center_frac / edge_frac ---
    # void mass split at INNER_RADIUS_FRAC * r_eq from pad centroid.
    n_total_void = float(ys_v.size)
    n_inner = float((radii_px < r_inner).sum())
    feats["center_frac"] = n_inner / n_total_void
    feats["edge_frac"]   = 1.0 - feats["center_frac"]
    # Verification property: center_frac + edge_frac == 1.0 when voids exist.

    # --- Fix 3: radial second moments ---
    feats["radial_var"]  = float(np.var(radii_norm))
    if radii_norm.size >= 2:
        feats["radial_skew"] = float(_scipy_skew(radii_norm, nan_policy="omit"))
    else:
        feats["radial_skew"] = np.nan

    # --- Void cluster regionprops ---
    lbl = label(pad_crop, connectivity=2)
    regions = regionprops(lbl)
    feats["cluster_count"] = len(regions)
    areas  = np.array([r.area for r in regions], dtype=float)
    perims = np.array([r.perimeter if r.perimeter > 0 else 1.0 for r in regions])
    circs  = 4 * np.pi * areas / (perims ** 2)
    feats["mean_circularity"]  = float(np.average(circs, weights=areas))
    feats["largest_void_frac"] = float(areas.max()) / pad_area_denom

    # Largest void region (by area).
    _lv_idx = int(np.argmax(areas))
    _lv_region = regions[_lv_idx]

    # --- Fix 2: largest-void geometry ---
    # (a) centroid_r: radial distance of largest-void centroid from pad centroid,
    #     normalised by equiv_radius.
    lv_cy_local, lv_cx_local = _lv_region.centroid   # (row, col) in bbox-local
    lv_dy = lv_cy_local - pc_y
    lv_dx = lv_cx_local - pc_x
    lv_r_px = np.hypot(lv_dx, lv_dy)
    feats["largest_void_centroid_r"] = float(lv_r_px) / max(equiv_radius, 1.0)

    # (b) edge_dist: min EDT distance of largest-void pixels from pad boundary,
    #     normalised by equiv_radius.  Uses per_pad_mask EDT when available.
    if per_pad_mask is not None:
        pad_edt = ndi.distance_transform_edt(per_pad_mask > 0)
        # Largest void pixels in full-image coordinates.
        lv_rows = _lv_region.coords[:, 0] + y0
        lv_cols = _lv_region.coords[:, 1] + x0
        # Clip to avoid out-of-bounds if bbox slightly outside image.
        lv_rows = np.clip(lv_rows, 0, pad_edt.shape[0] - 1)
        lv_cols = np.clip(lv_cols, 0, pad_edt.shape[1] - 1)
        feats["largest_void_edge_dist"] = (
            float(pad_edt[lv_rows, lv_cols].min()) / max(equiv_radius, 1.0)
        )
    else:
        # Fallback: bbox-interior EDT at the largest-void pixels.
        interior = np.ones_like(pad_crop, dtype=np.uint8)
        interior[0, :] = 0; interior[-1, :] = 0
        interior[:, 0] = 0; interior[:, -1] = 0
        interior_edt = ndi.distance_transform_edt(interior)
        lv_ys = _lv_region.coords[:, 0]
        lv_xs = _lv_region.coords[:, 1]
        norm = min(pad_h, pad_w) / 2.0
        feats["largest_void_edge_dist"] = float(interior_edt[lv_ys, lv_xs].min()) / max(norm, 1.0)

    # (c) circularity of the single largest void.
    # Prefer perimeter_crofton (more stable for digitized blobs); fall back to .perimeter.
    _lv_perim = getattr(_lv_region, "perimeter_crofton", None)
    if _lv_perim is None or _lv_perim == 0:
        _lv_perim = _lv_region.perimeter if _lv_region.perimeter > 0 else 1.0
    feats["largest_void_circularity"] = float(4 * np.pi * _lv_region.area / (_lv_perim ** 2))

    # --- edge_proximity_min (original; kept for backwards compat) ---
    # pad_edt and equiv_radius were both computed in the Fix 2 / geometry blocks above.
    if per_pad_mask is not None:
        ys_full, xs_full = np.nonzero(
            (void_mask > 0) & (per_pad_mask > 0)
        )
        if ys_full.size > 0:
            feats["edge_proximity_min"] = (
                float(pad_edt[ys_full, xs_full].min()) / max(equiv_radius, 1.0)
            )
        else:
            feats["edge_proximity_min"] = np.nan
    else:
        interior = np.ones_like(pad_crop, dtype=np.uint8)
        interior[0, :]  = 0; interior[-1, :] = 0
        interior[:, 0]  = 0; interior[:, -1] = 0
        dist = ndi.distance_transform_edt(interior)
        normalizer = min(pad_h, pad_w) / 2
        feats["edge_proximity_min"] = float(dist[pad_crop > 0].min()) / normalizer
    return feats'''

ok = apply_fix(
    "rewrite pad_features with Fixes 1-5 (center_frac, edge_frac, lv geometry, radial moments, in-frame gate)",
    pad_cell,
    OLD_PAD_FEATURES_FULL,
    NEW_PAD_FEATURES_FULL,
)
if not ok:
    print("  [FAIL] pad_features replacement failed — writing notebook unchanged to avoid corruption")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fix 5b + 6b — rewrite led_feature_vector cell (_FEATURE_KEYS + aggregation)
# ---------------------------------------------------------------------------
print("\n--- Fix 5b + 6b: rewrite led_feature_vector ---")

LFV_CELL_ANCHOR = "_FEATURE_KEYS = ["
lfv_cell = find_cell(nb, LFV_CELL_ANCHOR)
if lfv_cell is None:
    print("  [FAIL] led_feature_vector cell not found")
else:
    OLD_LFV = '''\
_FEATURE_KEYS = [
    "void_area_frac", "centroid_x", "centroid_y",
    "q_NW", "q_NE", "q_SW", "q_SE",
    "cluster_count", "mean_circularity",
    "largest_void_frac", "edge_proximity_min",
]


def led_feature_vector(void_mask, pads_bbox, meta_row, pad_masks=None):
    thermal_f, electrical_f = None, []
    for _idx_pad, bbox in enumerate(pads_bbox):
        role = bbox[4]
        _pmask = pad_masks[_idx_pad] if (pad_masks is not None and _idx_pad < len(pad_masks)) else None
        f = pad_features(void_mask, bbox, per_pad_mask=_pmask)
        if role == "thermal":
            thermal_f = f
        else:
            electrical_f.append(f)

    row = {
        "unit_id":   meta_row["unit_id"],
        "led_type":  meta_row["led_type"],
        "paste":     meta_row["paste"],
        "panel":     meta_row["panel"],
        "led_num":   meta_row["led_num"],
        "has_thermal_pad": int(thermal_f is not None),
    }
    for k in _FEATURE_KEYS:
        row[f"thermal_{k}"] = thermal_f[k] if thermal_f else np.nan
    for k in _FEATURE_KEYS:
        vals = [p[k] for p in electrical_f
                if not (isinstance(p[k], float) and np.isnan(p[k]))]
        row[f"electrical_mean_{k}"] = float(np.mean(vals)) if vals else np.nan
    return row'''

    NEW_LFV = '''\
# All features returned by pad_features that should be aggregated per LED.
# pad_in_frame_frac is listed here so it is surfaced as thermal_pad_in_frame_frac
# and electrical_mean_pad_in_frame_frac (and electrical_min/max/asym variants).
_FEATURE_KEYS = [
    "void_area_frac", "centroid_x", "centroid_y",
    "q_NW", "q_NE", "q_SW", "q_SE",
    "cluster_count", "mean_circularity",
    "largest_void_frac", "edge_proximity_min",
    # Fix 1 — concentric void mass split
    "center_frac", "edge_frac",
    # Fix 2 — largest-void geometry
    "largest_void_centroid_r", "largest_void_edge_dist", "largest_void_circularity",
    # Fix 3 — radial second moments
    "radial_var", "radial_skew",
    # Fix 4 — in-frame fraction
    "pad_in_frame_frac",
]


def led_feature_vector(void_mask, pads_bbox, meta_row, pad_masks=None, valid_mask=None):
    """Build the per-LED feature row.

    Parameters
    ----------
    void_mask : ndarray
        Binary void mask for this DUT image.
    pads_bbox : list of tuples
        Output of detect_pads: [(x0,y0,x1,y1,role,area_px), ...].
    meta_row : dict
        Row from the metadata table (unit_id, led_type, paste, panel, led_num).
    pad_masks : list of ndarray or None
        Per-pad full-image binary masks; aligned with pads_bbox.
    valid_mask : ndarray or None
        Full-image mask: 1 where registered-image pixels are valid (Fix 4 / Fix 6).
    """
    thermal_f, electrical_f = None, []
    for _idx_pad, bbox in enumerate(pads_bbox):
        role = bbox[4]
        _pmask = pad_masks[_idx_pad] if (pad_masks is not None and _idx_pad < len(pad_masks)) else None
        f = pad_features(void_mask, bbox, per_pad_mask=_pmask, valid_mask=valid_mask)
        if role == "thermal":
            thermal_f = f
        else:
            electrical_f.append(f)

    row = {
        "unit_id":   meta_row["unit_id"],
        "led_type":  meta_row["led_type"],
        "paste":     meta_row["paste"],
        "panel":     meta_row["panel"],
        "led_num":   meta_row["led_num"],
        "has_thermal_pad": int(thermal_f is not None),
    }

    # Thermal pad: direct copy.
    for k in _FEATURE_KEYS:
        row[f"thermal_{k}"] = thermal_f[k] if thermal_f else np.nan

    # Electrical pads: mean (kept), plus min / max / asymmetry (Fix 5).
    for k in _FEATURE_KEYS:
        raw_vals = [p[k] for p in electrical_f]
        # _is_nan: safe NaN check that works for int, float, and np.nan.
        def _is_nan(v):
            try:
                return np.isnan(v)
            except (TypeError, ValueError):
                return False
        # Exclude NaN from mean (match original nan-drop behaviour).
        finite_vals = [v for v in raw_vals if not _is_nan(v)]
        row[f"electrical_mean_{k}"] = float(np.mean(finite_vals)) if finite_vals else np.nan

        if len(electrical_f) >= 2:
            # Fix 5 — per-electrical-pad min / max / asymmetry.
            has_finite = any(not _is_nan(v) for v in raw_vals)
            e_min = float(np.nanmin([float(v) for v in raw_vals])) if has_finite else np.nan
            e_max = float(np.nanmax([float(v) for v in raw_vals])) if has_finite else np.nan
            if (not has_finite) or np.isnan(e_min) or np.isnan(e_max):
                e_asym = np.nan
            else:
                e_asym = (e_max - e_min) / (e_max + e_min + ASYM_EPS)
            row[f"electrical_min_{k}"]  = e_min
            row[f"electrical_max_{k}"]  = e_max
            row[f"electrical_asym_{k}"] = e_asym

    return row'''

    apply_fix(
        "rewrite _FEATURE_KEYS + led_feature_vector (Fixes 5b, 6b)",
        lfv_cell,
        OLD_LFV,
        NEW_LFV,
    )

# ---------------------------------------------------------------------------
# Fix 6c — thread valid[i] through process_led_type
# ---------------------------------------------------------------------------
print("\n--- Fix 6c: thread valid_mask through process_led_type ---")

PLT_CELL_ANCHOR = "def process_led_type(led_type, verbose=False):"
plt_cell = find_cell(nb, PLT_CELL_ANCHOR)
if plt_cell is None:
    print("  [FAIL] process_led_type cell not found")
else:
    OLD_PLT_CALL = (
        "        row = led_feature_vector(mask, c[\"pads\"], mrow, pad_masks=c[\"pad_masks\"])\n"
    )
    NEW_PLT_CALL = (
        "        row = led_feature_vector(mask, c[\"pads\"], mrow, pad_masks=c[\"pad_masks\"],\n"
        "                                 valid_mask=c[\"valid\"][i])\n"
    )
    apply_fix(
        "pass valid_mask=c['valid'][i] into led_feature_vector",
        plt_cell,
        OLD_PLT_CALL,
        NEW_PLT_CALL,
    )

    # Update the nan-gating prefix tuple so the new electrical_min/max/asym columns
    # are also nulled out when the image is a placeholder.
    OLD_PREFIX_CHECK = (
        "                if k.startswith((\"thermal_\", \"electrical_mean_\")):\n"
    )
    NEW_PREFIX_CHECK = (
        "                if k.startswith((\"thermal_\", \"electrical_mean_\",\n"
        "                                  \"electrical_min_\", \"electrical_max_\",\n"
        "                                  \"electrical_asym_\")):\n"
    )
    apply_fix(
        "extend placeholder nan-gating to new electrical_* prefixes",
        plt_cell,
        OLD_PREFIX_CHECK,
        NEW_PREFIX_CHECK,
    )

# ---------------------------------------------------------------------------
# validate JSON and write
# ---------------------------------------------------------------------------
print("\nValidating patched notebook JSON…")
try:
    out_bytes = json.dumps(nb, indent=1, ensure_ascii=False).encode("utf-8")
    json.loads(out_bytes)   # re-parse to confirm round-trip validity
    print("[OK]   JSON round-trip valid")
except Exception as exc:
    print(f"[FAIL] JSON validation after patching: {exc}")
    print("       Notebook NOT written — original is intact, backup is at", BAK_PATH)
    sys.exit(1)

NB_PATH.write_bytes(out_bytes)
print(f"[OK]   Notebook written to {NB_PATH}")
print("\nDone. Run the notebook cell-by-cell from Section 5 onwards to verify.")
print("Summary of new feature columns added:")
print("  thermal_center_frac                   thermal_edge_frac")
print("  thermal_largest_void_centroid_r        thermal_largest_void_edge_dist")
print("  thermal_largest_void_circularity")
print("  thermal_radial_var                     thermal_radial_skew")
print("  thermal_pad_in_frame_frac")
print("  electrical_mean_center_frac            electrical_mean_edge_frac  (etc.)")
print("  electrical_min_*  electrical_max_*  electrical_asym_*  (all _FEATURE_KEYS)")
