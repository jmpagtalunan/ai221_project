"""
apply_nb_fixes.py
=================
One-shot script that applies the 7 quick-win fixes from Wanda's review to
jp_void_spatial_features.ipynb.

Run from any directory:
    python apply_nb_fixes.py

A timestamped backup (.ipynb.bak) is written before any change is made.
If any fix fails the script aborts, the backup is preserved, and the original
notebook is left unmodified (the write only happens if ALL fixes succeed).
"""

import copy
import json
import re
import shutil
import sys
from pathlib import Path

NB_PATH = Path(__file__).parent / "jp_void_spatial_features_improved.ipynb"
BAK_PATH = NB_PATH.with_suffix(".ipynb.bak")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def source_str(cell) -> str:
    """Return the source lines of a cell joined into one string."""
    return "".join(cell["source"])


def set_source(cell, text: str) -> None:
    """Replace the cell source with *text*, split into notebook-style lines."""
    lines = text.splitlines(keepends=True)
    # Last line in a notebook source list conventionally has no trailing newline.
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    cell["source"] = lines


def find_cell_by_pattern(cells, pattern, cell_type="code"):
    """Return the first cell of the given type whose source matches *pattern*."""
    rx = re.compile(pattern, re.DOTALL)
    for cell in cells:
        if cell.get("cell_type") != cell_type:
            continue
        if rx.search(source_str(cell)):
            return cell
    return None


def find_cell_index_by_pattern(cells, pattern, cell_type="code"):
    rx = re.compile(pattern, re.DOTALL)
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != cell_type:
            continue
        if rx.search(source_str(cell)):
            return i
    return -1


def apply_fix(description, fn, cells):
    fn(cells)
    print(f"  [OK]  {description}")


# ---------------------------------------------------------------------------
# Fix 1 — _REG_CACHE key includes led_type; register_led_type gains led_type param
# ---------------------------------------------------------------------------


def fix1_reg_cache_key(cells):
    """
    In the register_led_type cell:
      - Add `led_type=None` parameter.
      - Replace the cache key with `(led_type, images[0].shape, len(images))`.
    In build_void_context:
      - Pass led_type to register_led_type.
    """
    # --- Part A: register_led_type signature + cache key ---
    cell = find_cell_by_pattern(cells, r"def register_led_type\(images")
    if cell is None:
        raise RuntimeError("register_led_type cell not found")

    src = source_str(cell)
    assert "def register_led_type(images, n_pass=3):" in src, \
        "register_led_type signature not found"
    assert "key = (len(images), images[0].shape, int(images[0].sum()), int(images[-1].sum()))" in src, \
        "register_led_type cache key not found"

    src = src.replace(
        "def register_led_type(images, n_pass=3):",
        "def register_led_type(images, n_pass=3, led_type=None):",
    )
    src = src.replace(
        "key = (len(images), images[0].shape, int(images[0].sum()), int(images[-1].sum()))",
        "key = (led_type, images[0].shape, len(images))",
    )
    set_source(cell, src)

    # --- Part B: build_void_context — pass led_type to register_led_type ---
    cell2 = find_cell_by_pattern(cells, r"def build_void_context\(led_type")
    if cell2 is None:
        raise RuntimeError("build_void_context cell not found")

    src2 = source_str(cell2)
    assert "reg = register_led_type(imgs)" in src2, \
        "register_led_type(imgs) call not found in build_void_context"
    src2 = src2.replace(
        "reg = register_led_type(imgs)",
        "reg = register_led_type(imgs, led_type=led_type)",
    )
    set_source(cell2, src2)


# ---------------------------------------------------------------------------
# Fix 2 — edge_proximity_min: EDT on per-pad mask, normalized by equiv. radius
# ---------------------------------------------------------------------------


def fix2_edge_proximity(cells):
    """
    Replace the bbox-interior EDT with EDT on the actual per-pad mask.
    pad_features gains an optional `per_pad_mask` parameter.
    """
    cell = find_cell_by_pattern(cells, r"def pad_features\(void_mask, pad_bbox\)")
    if cell is None:
        raise RuntimeError("pad_features cell not found")

    src = source_str(cell)

    # --- Signature update (also done by fix 4; we keep both idempotent) ---
    if "def pad_features(void_mask, pad_bbox):" in src:
        src = src.replace(
            "def pad_features(void_mask, pad_bbox):",
            "def pad_features(void_mask, pad_bbox, per_pad_mask=None):",
        )

    # --- Replace the bbox-interior EDT block ---
    old_edt_block = (
        "    interior = np.ones_like(pad_crop, dtype=np.uint8)\n"
        "    interior[0, :]  = 0; interior[-1, :] = 0\n"
        "    interior[:, 0]  = 0; interior[:, -1] = 0\n"
        "    dist = ndi.distance_transform_edt(interior)\n"
        "    normalizer = min(pad_h, pad_w) / 2\n"
        "    feats[\"edge_proximity_min\"] = float(dist[pad_crop > 0].min()) / normalizer\n"
        "    return feats"
    )

    new_edt_block = (
        "    # edge_proximity_min: distance from each void pixel to the nearest\n"
        "    # pad-boundary pixel, normalised by the pad's equivalent radius so\n"
        "    # the feature is dimensionless and comparable across pad sizes.\n"
        "    if per_pad_mask is not None:\n"
        "        # EDT on the full per-pad mask gives true distance-from-pad-edge.\n"
        "        # EDT value = 0 outside the pad, positive inside => min > 0 means\n"
        "        # the closest void pixel is that many px from the pad boundary.\n"
        "        pad_edt = ndi.distance_transform_edt(per_pad_mask > 0)\n"
        "        pad_area_true = float((per_pad_mask > 0).sum())\n"
        "        equiv_radius = np.sqrt(pad_area_true / np.pi)\n"
        "        # Void pixels inside this pad (full-image coordinates).\n"
        "        ys_full, xs_full = np.nonzero(\n"
        "            (void_mask > 0) & (per_pad_mask > 0)\n"
        "        )\n"
        "        if ys_full.size > 0:\n"
        "            feats[\"edge_proximity_min\"] = (\n"
        "                float(pad_edt[ys_full, xs_full].min()) / max(equiv_radius, 1.0)\n"
        "            )\n"
        "        else:\n"
        "            feats[\"edge_proximity_min\"] = np.nan\n"
        "    else:\n"
        "        # Fallback: bbox-interior approximation (original behaviour).\n"
        "        interior = np.ones_like(pad_crop, dtype=np.uint8)\n"
        "        interior[0, :]  = 0; interior[-1, :] = 0\n"
        "        interior[:, 0]  = 0; interior[:, -1] = 0\n"
        "        dist = ndi.distance_transform_edt(interior)\n"
        "        normalizer = min(pad_h, pad_w) / 2\n"
        "        feats[\"edge_proximity_min\"] = float(dist[pad_crop > 0].min()) / normalizer\n"
        "    return feats"
    )

    assert old_edt_block in src, \
        "Original EDT block not found in pad_features — check indentation or source drift"

    src = src.replace(old_edt_block, new_edt_block)
    set_source(cell, src)


# ---------------------------------------------------------------------------
# Fix 3 — CSV dedup: keep='first' + logging.warning
# ---------------------------------------------------------------------------


def fix3_csv_dedup(cells):
    cell = find_cell_by_pattern(
        cells,
        r"drop_duplicates\(_KEY, keep=False\)",
        cell_type="code",
    )
    if cell is None:
        raise RuntimeError("drop_duplicates(keep=False) cell not found")

    src = source_str(cell)

    # Locate the original line regardless of trailing comment variation.
    # We replace the whole logical line.
    old_line = (
        "gt_df = gt_df.drop_duplicates(_KEY, keep=False)"
        "          # FC-SP4 / SAC105 / panel 2 lists LED 7 twice -> ambiguous, dropped"
    )
    old_line_bare = "gt_df = gt_df.drop_duplicates(_KEY, keep=False)"

    if old_line in src:
        needle = old_line
    elif old_line_bare in src:
        needle = old_line_bare
    else:
        raise RuntimeError("drop_duplicates old line not found — check source drift")

    new_block = (
        "_dup_mask = gt_df.duplicated(_KEY, keep=False)\n"
        "if _dup_mask.any():\n"
        "    import logging as _logging\n"
        "    _logging.warning(\n"
        "        \"CSV dedup: dropping %d duplicate-key rows (keeping first occurrence). \"\n"
        "        \"Affected keys:\\n%s\",\n"
        "        _dup_mask.sum(),\n"
        "        gt_df[_dup_mask][_KEY].drop_duplicates().to_string(),\n"
        "    )\n"
        "gt_df = gt_df.drop_duplicates(_KEY, keep='first')"
        "          # FC-SP4 / SAC105 / panel 2 lists LED 7 twice -> keep first, log dropped"
    )

    src = src.replace(needle, new_block)
    set_source(cell, src)


# ---------------------------------------------------------------------------
# Fix 4 — Quadrant / centroid in pad principal-axis frame
# ---------------------------------------------------------------------------


def fix4_principal_axis_quadrants(cells):
    """
    Replace axis-aligned centroid + quadrant calculation with principal-axis frame.
    Uses skimage regionprops on the pad mask (per_pad_mask crop or bbox-sized ones).
    """
    cell = find_cell_by_pattern(cells, r"def pad_features\(void_mask, pad_bbox")
    if cell is None:
        raise RuntimeError("pad_features cell not found (fix 4)")

    src = source_str(cell)

    # --- Signature update (idempotent with fix 2) ---
    if "def pad_features(void_mask, pad_bbox):" in src:
        src = src.replace(
            "def pad_features(void_mask, pad_bbox):",
            "def pad_features(void_mask, pad_bbox, per_pad_mask=None):",
        )

    # --- Replace the axis-aligned centroid + quadrant block ---
    old_centroid_quad = (
        "    ys, xs = np.nonzero(pad_crop)\n"
        "    cy, cx = ys.mean(), xs.mean()\n"
        "    feats[\"centroid_x\"] = (cx - pad_w / 2) / (pad_w / 2)\n"
        "    feats[\"centroid_y\"] = (cy - pad_h / 2) / (pad_h / 2)\n"
        "\n"
        "    mid_x, mid_y = pad_w / 2, pad_h / 2\n"
        "    nw = ((xs <  mid_x) & (ys <  mid_y)).sum()\n"
        "    ne = ((xs >= mid_x) & (ys <  mid_y)).sum()\n"
        "    sw = ((xs <  mid_x) & (ys >= mid_y)).sum()\n"
        "    se = ((xs >= mid_x) & (ys >= mid_y)).sum()\n"
        "    tot = nw + ne + sw + se\n"
        "    feats.update({\"q_NW\": nw/tot, \"q_NE\": ne/tot, \"q_SW\": sw/tot, \"q_SE\": se/tot})"
    )

    new_centroid_quad = (
        "    # --- Pad centroid + principal-axis frame ---\n"
        "    # Compute pad centroid and orientation via regionprops on the pad mask\n"
        "    # (per_pad_mask cropped to bbox, or a full-bbox rectangle if None).\n"
        "    if per_pad_mask is not None:\n"
        "        pad_region_mask = (per_pad_mask[y0:y1, x0:x1] > 0).astype(np.uint8)\n"
        "    else:\n"
        "        pad_region_mask = np.ones((pad_h, pad_w), dtype=np.uint8)\n"
        "\n"
        "    _pad_lbl = label(pad_region_mask, connectivity=2)\n"
        "    _pad_props = regionprops(_pad_lbl)\n"
        "    if _pad_props:\n"
        "        pc_y, pc_x = _pad_props[0].centroid      # (row, col) in bbox-local coords\n"
        "        theta = _pad_props[0].orientation         # radians, CCW from col-axis\n"
        "    else:\n"
        "        pc_y, pc_x = pad_h / 2.0, pad_w / 2.0\n"
        "        theta = 0.0\n"
        "\n"
        "    # Void pixel coords in bbox-local frame, centred on pad centroid.\n"
        "    ys_v, xs_v = np.nonzero(pad_crop)\n"
        "    dy = ys_v - pc_y\n"
        "    dx = xs_v - pc_x\n"
        "\n"
        "    # Rotate void coords into pad principal-axis frame.\n"
        "    # skimage orientation: angle of major axis from positive col-axis (radians).\n"
        "    cos_t, sin_t = np.cos(theta), np.sin(theta)\n"
        "    dx_r =  dx * cos_t + dy * sin_t  # principal-axis x (along major axis)\n"
        "    dy_r = -dx * sin_t + dy * cos_t  # principal-axis y (along minor axis)\n"
        "\n"
        "    # Void centroid in normalised pad coordinates.\n"
        "    half_w = pad_w / 2.0 if pad_w > 0 else 1.0\n"
        "    half_h = pad_h / 2.0 if pad_h > 0 else 1.0\n"
        "    if ys_v.size > 0:\n"
        "        feats[\"centroid_x\"] = float(dx_r.mean()) / half_w\n"
        "        feats[\"centroid_y\"] = float(dy_r.mean()) / half_h\n"
        "    else:\n"
        "        feats[\"centroid_x\"] = 0.0\n"
        "        feats[\"centroid_y\"] = 0.0\n"
        "\n"
        "    # Quadrant counts in principal-axis frame.\n"
        "    # Denominator = per-pad-mask pixel count (or bbox area if mask unavailable).\n"
        "    pad_area_denom = float(pad_region_mask.sum()) if pad_region_mask.sum() > 0 else pad_area\n"
        "    if ys_v.size > 0:\n"
        "        nw = int(((dx_r < 0) & (dy_r < 0)).sum())\n"
        "        ne = int(((dx_r >= 0) & (dy_r < 0)).sum())\n"
        "        sw = int(((dx_r < 0) & (dy_r >= 0)).sum())\n"
        "        se = int(((dx_r >= 0) & (dy_r >= 0)).sum())\n"
        "        tot = nw + ne + sw + se\n"
        "        feats.update({\"q_NW\": nw / tot, \"q_NE\": ne / tot,\n"
        "                      \"q_SW\": sw / tot, \"q_SE\": se / tot})\n"
        "    else:\n"
        "        feats.update({\"q_NW\": 0.0, \"q_NE\": 0.0, \"q_SW\": 0.0, \"q_SE\": 0.0})"
    )

    assert old_centroid_quad in src, \
        "Original centroid+quadrant block not found in pad_features — check source drift"

    src = src.replace(old_centroid_quad, new_centroid_quad)

    # --- Update largest_void_frac to use pad_area_denom ---
    # pad_area_denom is now defined by the quadrant block above.
    assert 'feats["largest_void_frac"] = float(areas.max()) / pad_area' in src, \
        "largest_void_frac line not found — check source drift"
    src = src.replace(
        'feats["largest_void_frac"] = float(areas.max()) / pad_area',
        'feats["largest_void_frac"] = float(areas.max()) / pad_area_denom',
    )

    set_source(cell, src)


# ---------------------------------------------------------------------------
# Fix 5 — Area-weighted mean circularity + explicit connectivity=2
# ---------------------------------------------------------------------------


def fix5_weighted_circularity(cells):
    # --- connectivity=2 and area-weighted circularity in pad_features ---
    cell = find_cell_by_pattern(cells, r"def pad_features\(void_mask, pad_bbox")
    if cell is None:
        raise RuntimeError("pad_features cell not found (fix 5)")

    src = source_str(cell)

    # Explicit connectivity=2 on the void-label call
    assert "    lbl = label(pad_crop)" in src, \
        "lbl = label(pad_crop) line not found — check fix order"
    src = src.replace(
        "    lbl = label(pad_crop)\n",
        "    lbl = label(pad_crop, connectivity=2)\n",
    )

    # Area-weighted mean circularity
    old_circ = (
        '    circs  = 4 * np.pi * areas / (perims ** 2)\n'
        '    feats["mean_circularity"]  = float(np.mean(circs))'
    )
    new_circ = (
        '    circs  = 4 * np.pi * areas / (perims ** 2)\n'
        '    # Area-weighted mean: larger voids contribute proportionally more.\n'
        '    feats["mean_circularity"]  = float(np.average(circs, weights=areas))'
    )
    assert old_circ in src, "mean_circularity line not found — check source drift"
    src = src.replace(old_circ, new_circ)

    set_source(cell, src)

    # --- connectivity=2 in extract_void_mask ---
    cell_evm = find_cell_by_pattern(cells, r"def extract_void_mask\(ctx, i")
    if cell_evm is None:
        raise RuntimeError("extract_void_mask cell not found (fix 5)")

    src_evm = source_str(cell_evm)

    old_lbl_line = (
        "    lbl = label(apply_hysteresis_threshold("
        "np.where(region > 0, R, -99), hyst * T, T))"
    )
    new_lbl_line = (
        "    lbl = label(apply_hysteresis_threshold("
        "np.where(region > 0, R, -99), hyst * T, T), connectivity=2)"
    )
    assert old_lbl_line in src_evm, \
        "extract_void_mask label call not found — check source drift"
    src_evm = src_evm.replace(old_lbl_line, new_lbl_line)

    set_source(cell_evm, src_evm)


# ---------------------------------------------------------------------------
# Fix 6 — _global_gain degeneracy guards
# ---------------------------------------------------------------------------


def fix6_global_gain_guards(cells):
    cell = find_cell_by_pattern(cells, r"def _global_gain\(img, ref, mask\)")
    if cell is None:
        raise RuntimeError("_global_gain cell not found")

    src = source_str(cell)

    # --- A: _global_gain body ---
    old_global_gain_body = (
        "def _global_gain(img, ref, mask):\n"
        "    \"\"\"Robust slope/offset of img against ref = contrast of this shot relative to the reference.\"\"\"\n"
        "    x, y = ref[mask > 0], img[mask > 0]\n"
        "    keep = np.ones_like(x, bool)\n"
        "    for _ in range(3):\n"
        "        a, b = np.polyfit(x[keep], y[keep], 1)\n"
        "        res = y - (a * x + b)\n"
        "        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6\n"
        "        keep = np.abs(res) < 2.5 * s\n"
        "    return a, b"
    )

    new_global_gain_body = (
        "MIN_POLYFIT_POINTS = 32   # minimum mask pixels required for a reliable polyfit\n"
        "\n"
        "\n"
        "def _global_gain(img, ref, mask):\n"
        "    \"\"\"Robust slope/offset of img against ref = contrast of this shot relative to the reference.\n"
        "\n"
        "    Returns (a, b) where img ~= a * ref + b inside the mask.\n"
        "    Falls back to (1.0, 0.0) if the mask contains fewer than MIN_POLYFIT_POINTS pixels.\n"
        "    \"\"\"\n"
        "    import warnings as _warnings\n"
        "    x, y = ref[mask > 0], img[mask > 0]\n"
        "    if x.size < MIN_POLYFIT_POINTS:\n"
        "        _warnings.warn(\n"
        "            f\"_global_gain: only {x.size} mask pixels \"\n"
        "            f\"(< MIN_POLYFIT_POINTS={MIN_POLYFIT_POINTS}); \"\n"
        "            \"falling back to gain=1.0, offset=0.0.\",\n"
        "            RuntimeWarning, stacklevel=2,\n"
        "        )\n"
        "        return 1.0, 0.0\n"
        "    keep = np.ones_like(x, bool)\n"
        "    for _ in range(3):\n"
        "        a, b = np.polyfit(x[keep], y[keep], 1)\n"
        "        res = y - (a * x + b)\n"
        "        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6\n"
        "        keep = np.abs(res) < 2.5 * s\n"
        "    return a, b"
    )

    assert old_global_gain_body in src, \
        "_global_gain body not found — check source drift"
    src = src.replace(old_global_gain_body, new_global_gain_body)

    # --- B: void_residual normalization guard ---
    old_norm = (
        "    a0, _ = _global_gain(img, ctx[\"reference\"], ctx[\"fit_region\"] & ctx[\"valid\"][i])\n"
        "    return _local_residual(img, ctx[\"X\"], fit, a0) / max(a0, 0.05)"
    )

    new_norm = (
        "    import warnings as _warnings\n"
        "    a0, _ = _global_gain(img, ctx[\"reference\"], ctx[\"fit_region\"] & ctx[\"valid\"][i])\n"
        "    resid = _local_residual(img, ctx[\"X\"], fit, a0)\n"
        "    if a0 <= 0.05:\n"
        "        _warnings.warn(\n"
        "            f\"void_residual (i={i}): global gain a0={a0:.4f} <= 0.05 \"\n"
        "            \"(inverted or degenerate contrast); skipping normalisation, \"\n"
        "            \"returning raw residual.\",\n"
        "            RuntimeWarning, stacklevel=2,\n"
        "        )\n"
        "        return resid\n"
        "    return resid / a0"
    )

    assert old_norm in src, \
        "void_residual normalization line not found — check source drift"
    src = src.replace(old_norm, new_norm)

    set_source(cell, src)


# ---------------------------------------------------------------------------
# Fix 7 — Disclaimer markdown cell after calibrate_threshold
# ---------------------------------------------------------------------------


def fix7_disclaimer_cell(cells):
    idx = find_cell_index_by_pattern(
        cells, r"cal = calibrate_threshold\(ctx\)", cell_type="code"
    )
    if idx < 0:
        raise RuntimeError("calibrate_threshold cell not found")

    # Check it hasn't already been inserted (idempotency guard).
    if idx + 1 < len(cells):
        next_src = source_str(cells[idx + 1])
        if "VOID_T is calibrated on the same CSV" in next_src:
            print("    (disclaimer cell already present — skipping insertion)")
            return

    disclaimer_source = [
        "> **Calibration caveat:** `VOID_T` is calibrated on the **same CSV** that is used\n",
        "> for the threshold sweep — the Pearson *r* and MAE values reported by\n",
        "> `calibrate_threshold` are a **training fit**, not held-out performance.\n",
        "> A panel-split cross-validation (e.g. leave-one-panel-out) is needed\n",
        "> before claiming generalisation to unseen panels or LED types.\n",
        "> **Do not report these numbers as test-set metrics.**",
    ]

    new_md_cell = {
        "cell_type": "markdown",
        "id": "calib-caveat-disclaimer",
        "metadata": {},
        "source": disclaimer_source,
    }

    cells.insert(idx + 1, new_md_cell)


# ---------------------------------------------------------------------------
# Call-site updates — pass per_pad_mask to pad_features everywhere
# ---------------------------------------------------------------------------


def fix_call_sites(cells):
    """
    Update all pad_features call sites to pass per_pad_mask:
      - Demo loop in the pad_features cell.
      - led_feature_vector signature + internal loop.
      - Demo call in led_feature_vector cell.
      - process_led_type call site.
    """
    # --- Demo loop (in the same cell as pad_features) ---
    cell_pf = find_cell_by_pattern(cells, r"def pad_features\(void_mask, pad_bbox")
    if cell_pf is None:
        raise RuntimeError("pad_features cell not found (call-site fix)")

    src_pf = source_str(cell_pf)

    # The demo loop immediately after the function definition
    old_demo = (
        "demo_mask = extract_void_mask(ctx, 0)\n"
        "for bbox in pads:\n"
        "    f = pad_features(demo_mask, bbox)"
    )
    new_demo = (
        "demo_mask = extract_void_mask(ctx, 0)\n"
        "for bbox, _pmask in zip(pads, pad_masks):\n"
        "    f = pad_features(demo_mask, bbox, per_pad_mask=_pmask)"
    )

    if old_demo in src_pf:
        src_pf = src_pf.replace(old_demo, new_demo)
        set_source(cell_pf, src_pf)
    else:
        print("    (WARNING: demo loop not found in pad_features cell — skipping)")

    # --- led_feature_vector signature + internal call ---
    cell_lfv = find_cell_by_pattern(
        cells, r"def led_feature_vector\(void_mask, pads_bbox, meta_row\)"
    )
    if cell_lfv is None:
        raise RuntimeError("led_feature_vector cell not found")

    src_lfv = source_str(cell_lfv)

    # Signature
    if "def led_feature_vector(void_mask, pads_bbox, meta_row):" in src_lfv:
        src_lfv = src_lfv.replace(
            "def led_feature_vector(void_mask, pads_bbox, meta_row):",
            "def led_feature_vector(void_mask, pads_bbox, meta_row, pad_masks=None):",
        )

    # Internal loop
    old_loop = (
        "    for bbox in pads_bbox:\n"
        "        role = bbox[4]\n"
        "        f = pad_features(void_mask, bbox)"
    )
    new_loop = (
        "    for _idx_pad, bbox in enumerate(pads_bbox):\n"
        "        role = bbox[4]\n"
        "        _pmask = pad_masks[_idx_pad] if (pad_masks is not None and _idx_pad < len(pad_masks)) else None\n"
        "        f = pad_features(void_mask, bbox, per_pad_mask=_pmask)"
    )
    if old_loop in src_lfv:
        src_lfv = src_lfv.replace(old_loop, new_loop)

    # Demo call in the same cell
    if "row = led_feature_vector(demo_mask, pads, meta[0])" in src_lfv:
        src_lfv = src_lfv.replace(
            "row = led_feature_vector(demo_mask, pads, meta[0])",
            "row = led_feature_vector(demo_mask, pads, meta[0], pad_masks=pad_masks)",
        )

    set_source(cell_lfv, src_lfv)

    # --- process_led_type ---
    cell_proc = find_cell_by_pattern(cells, r"def process_led_type\(led_type")
    if cell_proc is None:
        raise RuntimeError("process_led_type cell not found")

    src_proc = source_str(cell_proc)
    old_proc_call = '        row = led_feature_vector(mask, c["pads"], mrow)'
    new_proc_call = '        row = led_feature_vector(mask, c["pads"], mrow, pad_masks=c["pad_masks"])'

    if old_proc_call in src_proc:
        src_proc = src_proc.replace(old_proc_call, new_proc_call)
        set_source(cell_proc, src_proc)
    else:
        print("    (WARNING: process_led_type call site not found — skipping)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    if not NB_PATH.exists():
        sys.exit(f"ERROR: Notebook not found: {NB_PATH}")

    print(f"Reading: {NB_PATH}")
    with NB_PATH.open("r", encoding="utf-8") as fh:
        nb = json.load(fh)

    cells = nb["cells"]

    # Validate JSON is parseable before we start.
    _ = json.dumps(nb)
    print("Input JSON: valid")

    # Backup before any change.
    shutil.copy2(NB_PATH, BAK_PATH)
    print(f"Backup:   {BAK_PATH}")
    print()
    print("Applying fixes...")

    # Apply all fixes; collect any that fail and re-raise at the end.
    failures = []
    fixlist = [
        ("Fix 1  — _REG_CACHE key includes led_type", fix1_reg_cache_key),
        ("Fix 2  — edge_proximity_min uses EDT on per-pad mask", fix2_edge_proximity),
        ("Fix 3  — CSV dedup keep='first' + logging.warning", fix3_csv_dedup),
        ("Fix 4  — quadrant/centroid in principal-axis frame", fix4_principal_axis_quadrants),
        ("Fix 5  — area-weighted mean circularity + connectivity=2", fix5_weighted_circularity),
        ("Fix 6  — _global_gain degeneracy guards", fix6_global_gain_guards),
        ("Fix 2+4 — pass per_pad_mask to pad_features call sites", fix_call_sites),
        ("Fix 7  — calibration caveat disclaimer markdown cell", fix7_disclaimer_cell),
    ]

    for desc, fn in fixlist:
        try:
            apply_fix(desc, fn, cells)
        except Exception as exc:
            print(f"  [FAIL] {desc}: {exc}", file=sys.stderr)
            failures.append((desc, exc))

    if failures:
        print(f"\n{len(failures)} fix(es) FAILED — notebook NOT written. See errors above.")
        sys.exit(1)

    print()
    print("All fixes applied. Writing notebook...")
    output_text = json.dumps(nb, ensure_ascii=False, indent=1)

    # Final JSON validity check before writing.
    json.loads(output_text)
    print("Output JSON: valid")

    with NB_PATH.open("w", encoding="utf-8") as fh:
        fh.write(output_text)

    print(f"Written:  {NB_PATH}")
    print()
    print("Cell outputs are preserved. Re-run the notebook from top to regenerate them.")


if __name__ == "__main__":
    main()
