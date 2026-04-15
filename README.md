# UM-ParkSight

Parking lot occupancy detection using image alignment (LoFTR) and object detection (Roboflow, YOLO11, or Grounding DINO).

---

## Project Structure

| File / Folder | Purpose |
|---|---|
| `parksight.ipynb` | Main notebook — configure, run, and visualize the full pipeline |
| `annotate_reference.py` | One-time tool to draw hexagon spot annotations on the reference image |
| `align_with_loftr.py` | Aligns a test image to the reference using LoFTR + RANSAC homography |
| `translate_spots.py` | Interactive tool to fine-tune spot positions on an aligned image |
| `crop_spots.py` | Crops per-spot segments from an aligned image |
| `spots.json` | Hexagon spot geometry annotated on the reference image |
| `reference_images/` | Reference parking lot image(s) |
| `test_images/` | New images to run inference on |
| `aligned_images/` | Homography-warped outputs from alignment |
| `translated_spots/` | Per-image adjusted spot JSON files |
| `spot_segments/` | Cropped per-spot images used for detection |
| `parksight_output/` | Final occupancy JSON and overlay images |

---

## Setup

Install dependencies:

```bash
pip install torch kornia opencv-python numpy matplotlib
pip install inference-sdk          # Roboflow backend
pip install ultralytics            # YOLO11 backend
pip install transformers           # Grounding DINO backend
```

---

## Usage

### 1. (One-time) Annotate the reference image

Run this once to draw hexagon regions for each parking spot on the reference image:

```bash
python annotate_reference.py --image reference_images/ref_image.jpeg --edit spots.json
```

- Left-click to place 6 vertices per spot (clockwise order), then press `n` to finalize.
- Press `s` to save `spots.json` and `spots_overlay.png`, then `q` to quit.

### 2. Run the notebook

Open `parksight.ipynb` and edit **Cell 1** (the config cell) to set your inputs:

```python
TEST_IMAGE_PATH = "./test_images/YOUR_IMAGE.jpeg"
DETECTOR_BACKEND = "roboflow"   # or "yolo11" or "grounding_dino"
```

Then run **all cells** sequentially. The pipeline will:

1. **Align** your test image to the reference using LoFTR homography warping.
2. **Translate spots** — a window opens to let you drag the spot overlay into place. Press `s` to save, then `q`.
3. **Crop** each parking spot into individual segment images.
4. **Detect** vehicles in each segment using the configured backend.
5. **Resolve occupancy** via rule-based conflict resolution (center/area gating + neighbor suppression).
6. **Save** results to `parksight_output/resolved_occupancy.json` and an annotated overlay image.

### 3. Detector backends

| Backend | Config key | Notes |
|---|---|---|
| Roboflow | `"roboflow"` | Requires `ROBOFLOW_API_KEY` and `ROBOFLOW_MODEL_ID` in Cell 1 |
| YOLO11 | `"yolo11"` | Uses local `yolo11n.pt`; adjust `YOLO_CONF_THRESHOLD` as needed |
| Grounding DINO | `"grounding_dino"` | Downloads model from Hugging Face on first run |

### 4. Scripts (standalone use)

```bash
# Align a single test image to the reference
python align_with_loftr.py --input test_images/IMG_3375.jpeg --output aligned_images/IMG_3375.jpeg

# Interactively adjust spot positions over an aligned image
python translate_spots.py --image aligned_images/IMG_3375.jpeg --spots spots.json --output translated_spots/IMG_3375.json

# Crop per-spot segments
python crop_spots.py --image aligned_images/IMG_3375.jpeg --spots translated_spots/IMG_3375.json --output-dir spot_segments/IMG_3375
```

---

## Output

- `parksight_output/resolved_occupancy.json` — per-spot occupancy decisions with confidence scores and reasoning.
- `parksight_output/resolved_occupancy_overlay.png` — full lot image annotated with occupied/empty labels.
- `parksight_output/segment_detection.png` — tiled view of per-spot detection results.

