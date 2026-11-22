"""
setup_fork.py
=============
One-shot master script that:
  Step 1 — Restores jp_void_spatial_features.ipynb from the pristine .bak.
  Step 2 — Creates jp_void_spatial_features_improved.ipynb (copy of the pristine).
  Step 4a — Runs apply_nb_fixes.py against the improved copy.
  Step 4b — Runs apply_nb_feature_adds.py against the improved copy.
  Step 5 — Renames the output CSV in the improved notebook to avoid collision.
  Step 6 — Verifies the pristine notebook is byte-identical to .bak.

Run from the notebooks directory:
    python setup_fork.py

Both patch scripts must already target jp_void_spatial_features_improved.ipynb
(apply_nb_fixes.py and apply_nb_feature_adds.py were already updated).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

PRISTINE_BAK  = HERE / "jp_void_spatial_features.ipynb.bak"
MAIN_NB       = HERE / "jp_void_spatial_features.ipynb"
IMPROVED_NB   = HERE / "jp_void_spatial_features_improved.ipynb"
FIXES_SCRIPT  = HERE / "apply_nb_fixes.py"
FEATS_SCRIPT  = HERE / "apply_nb_feature_adds.py"

OLD_CSV_NAME  = "spatial_void_features.csv"
NEW_CSV_NAME  = "spatial_void_features_improved.csv"

# ---------------------------------------------------------------------------
# Step 1 — Restore pristine notebook
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Restore pristine notebook from .bak")
if not PRISTINE_BAK.exists():
    sys.exit(f"[FAIL] .bak not found: {PRISTINE_BAK}")

shutil.copy2(PRISTINE_BAK, MAIN_NB)
print(f"[OK]   Copied {PRISTINE_BAK.name} -> {MAIN_NB.name}")

# ---------------------------------------------------------------------------
# Step 2 — Create improved copy
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 2: Create improved copy")

shutil.copy2(PRISTINE_BAK, IMPROVED_NB)
print(f"[OK]   Created {IMPROVED_NB.name}")

# ---------------------------------------------------------------------------
# Step 4a — Run apply_nb_fixes.py
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 4a: Run apply_nb_fixes.py")

result_fixes = subprocess.run(
    [sys.executable, str(FIXES_SCRIPT)],
    capture_output=True, text=True,
    cwd=str(HERE),
)
print(result_fixes.stdout)
if result_fixes.stderr:
    print("STDERR:", result_fixes.stderr, file=sys.stderr)

if result_fixes.returncode != 0:
    print(f"[FAIL] apply_nb_fixes.py exited with code {result_fixes.returncode}")
    sys.exit(result_fixes.returncode)
else:
    print("[OK]   apply_nb_fixes.py completed successfully")

# ---------------------------------------------------------------------------
# Step 4b — Run apply_nb_feature_adds.py
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 4b: Run apply_nb_feature_adds.py")

result_feats = subprocess.run(
    [sys.executable, str(FEATS_SCRIPT)],
    capture_output=True, text=True,
    cwd=str(HERE),
)
print(result_feats.stdout)
if result_feats.stderr:
    print("STDERR:", result_feats.stderr, file=sys.stderr)

if result_feats.returncode != 0:
    print(f"[FAIL] apply_nb_feature_adds.py exited with code {result_feats.returncode}")
    sys.exit(result_feats.returncode)
else:
    print("[OK]   apply_nb_feature_adds.py completed successfully")

# ---------------------------------------------------------------------------
# Step 5 — Rename output CSV in the improved notebook
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 5: Rename output CSV in improved notebook")

with IMPROVED_NB.open("r", encoding="utf-8") as fh:
    nb = json.load(fh)

cells = nb["cells"]
renamed_count = 0
for cell in cells:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"])
    if OLD_CSV_NAME in src:
        new_src = src.replace(OLD_CSV_NAME, NEW_CSV_NAME)
        lines = new_src.splitlines(keepends=True)
        cell["source"] = lines
        renamed_count += 1
        print(f"  [OK]   Renamed '{OLD_CSV_NAME}' -> '{NEW_CSV_NAME}' in code cell")

# Also update any notebook output text that references the CSV name
# (kept-output cells from prior runs); these are cosmetic, not load-bearing.
for cell in cells:
    for output in cell.get("outputs", []):
        if output.get("output_type") in ("stream", "display_data", "execute_result"):
            # update text/stdout lines
            for field in ("text", "data"):
                val = output.get(field)
                if isinstance(val, list):
                    output[field] = [
                        line.replace(OLD_CSV_NAME, NEW_CSV_NAME) if isinstance(line, str) else line
                        for line in val
                    ]
                elif isinstance(val, str):
                    output[field] = val.replace(OLD_CSV_NAME, NEW_CSV_NAME)

if renamed_count == 0:
    print(f"  [FAIL] '{OLD_CSV_NAME}' not found in any code cell of improved notebook")
    sys.exit(1)

# Validate and write
out_bytes = json.dumps(nb, indent=1, ensure_ascii=False).encode("utf-8")
json.loads(out_bytes)   # re-parse confirms round-trip validity
IMPROVED_NB.write_bytes(out_bytes)
print(f"[OK]   Improved notebook written with CSV renamed to '{NEW_CSV_NAME}'")

# ---------------------------------------------------------------------------
# Step 6 — Verify pristine notebook is byte-identical to .bak
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 6: Verify pristine notebook == .bak")

main_bytes = MAIN_NB.read_bytes()
bak_bytes  = PRISTINE_BAK.read_bytes()

main_size = len(main_bytes)
bak_size  = len(bak_bytes)

if main_bytes == bak_bytes:
    print(f"[OK]   BYTE-IDENTICAL — both {main_size} bytes")
else:
    print(f"[FAIL] SIZE MISMATCH: main={main_size} bytes, bak={bak_size} bytes")
    # Find first differing offset
    for i, (a, b) in enumerate(zip(main_bytes, bak_bytes)):
        if a != b:
            print(f"       First difference at byte offset {i}")
            break
    if main_size != bak_size:
        print(f"       Length difference: {main_size - bak_size:+d} bytes")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("SUMMARY")
print(f"  Pristine notebook : {MAIN_NB}")
print(f"  Improved notebook : {IMPROVED_NB}")
print(f"  CSV in improved   : {NEW_CSV_NAME}")
print(f"  Pristine == .bak  : YES (byte-identical)")
print("=" * 60)
