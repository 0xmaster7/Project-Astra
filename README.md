# Galaxy Image Processing Pipeline (DIP Project)

This project performs a complete Digital Image Processing workflow on galaxy images:
- image enhancement
- image segmentation
- an ML step (K-means clustering)
- visual outputs and analysis plots
- an optional local web UI for drag-and-drop usage

The default sample input is:
- `Hubble_ultra_deep_field_high_rez_edit1.jpg`

---

## 1) Full Pipeline Explanation

### 1.1 Input Stage
The pipeline reads a color image from disk using OpenCV.

What happens:
- The image is loaded in BGR format.
- If the image is large, it is resized so the longest side equals `max_size` (default: 1600).
- Resizing is done to speed up processing and keep memory usage stable.

Why this matters:
- Space images are often high resolution.
- Fast processing is useful for demos and repeated experimentation.

---

### 1.2 Enhancement Stage
Enhancement is applied to make faint structures (galaxies/stars) more visible.

Operations used:
- Convert BGR to LAB color space.
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel.
- Convert back to BGR.
- Apply light sharpening using unsharp masking logic (`addWeighted` with blurred image).

Why this helps:
- CLAHE improves local contrast in dark astronomical regions.
- Sharpening makes edges/bright spots clearer.
- Faint galaxy features become easier to segment.

---

### 1.3 Segmentation Stage
Segmentation isolates likely foreground celestial objects from dark background.

Operations used:
- Convert enhanced image to grayscale.
- Gaussian blur to reduce noise.
- Otsu thresholding (`THRESH_BINARY + THRESH_OTSU`) for automatic threshold selection.
- Morphological opening and closing to clean small noise and fill tiny gaps.
- Connected components filtering to remove tiny blobs (`min_area = 20`).
- Contour drawing on enhanced image for visual inspection.

Why this helps:
- Otsu threshold adapts automatically to image intensity distribution.
- Morphology and area filtering improve mask quality.
- Overlay lets you quickly verify whether segmentation looks reasonable.

---

### 1.4 ML Stage (Model Used)
The ML model used is **K-means clustering** (unsupervised learning).

Model details:
- Algorithm: OpenCV `cv2.kmeans`
- Type: Unsupervised clustering (not deep learning)
- Number of clusters: controlled by `clusters` argument (default: 5)

Feature vector per pixel:
- LAB color: `L, a, b`
- Spatial features: normalized `x, y` coordinates (weakly weighted)

Why include spatial features:
- Pure color clustering may produce fragmented noise.
- Adding weak position features encourages spatially coherent regions.

ML outputs:
- A pseudo-color cluster map (each cluster shown with a distinct color).
- Bright-cluster highlight image:
  - compute mean L (brightness) per cluster center
  - pick cluster with highest L
  - overlay this cluster in red tint on original enhanced image

Why this is useful in your project:
- Demonstrates an actual ML method integrated into DIP.
- Produces clearly visible output for presentation.
- Helps identify bright celestial structures automatically.

---

### 1.5 Analysis Plot Stage (3 Separate Graphs)
The project generates 3 separate matplotlib graphs.

1. `07_input_intensity_histogram.png`
- Histogram of grayscale intensities from input image.
- X-axis: intensity 0..255
- Y-axis: pixel count
- Typical galaxy image behavior: many dark pixels, smaller bright tail.

2. `08_enhanced_intensity_histogram.png`
- Histogram after enhancement.
- Shows how contrast redistribution changed intensity spread.
- Useful for explaining the effect of CLAHE + sharpening.

3. `09_segmentation_area_percentage.png`
- Bar chart of Background % vs Segmented %.
- Segmented % = non-zero pixels in final segmentation mask.
- Gives one compact quantitative segmentation metric.

---

## 2) What Happens When You Run the Code

When you run `main.py`, the sequence is:
1. parse CLI arguments
2. locate/read input image
3. resize if needed
4. apply enhancement
5. perform segmentation
6. run K-means clustering ML step
7. save all output images
8. generate and save the 3 analysis plots
9. print output file paths in terminal

---

## 3) CLI Options (Arguments)

`main.py` supports:

- `--input`
  - type: string path
  - default: `Hubble_ultra_deep_field_high_rez_edit1.jpg`
  - meaning: image file to process

- `--output-dir`
  - type: directory path
  - default: `outputs`
  - meaning: folder where results are saved

- `--max-size`
  - type: int
  - default: `1600`
  - meaning: longest side after resize
  - lower value: faster, less detail
  - higher value: slower, more detail

- `--clusters`
  - type: int
  - default: `5`
  - meaning: number of K-means clusters
  - lower: simpler grouping
  - higher: finer grouping, can become noisy

Example:
```bash
./dip/bin/python main.py --input Hubble_ultra_deep_field_high_rez_edit1.jpg --output-dir outputs --max-size 1600 --clusters 5
```

---

## 4) Output Files and Their Meaning

The pipeline generates these files in output directory:

- `01_input_resized.jpg`
  - input image after resize stage

- `02_enhanced.jpg`
  - CLAHE + sharpening result

- `03_segmentation_mask.png`
  - binary mask of segmented foreground objects

- `04_segmentation_overlay.jpg`
  - contours from mask overlaid on enhanced image

- `05_kmeans_cluster_map.jpg`
  - colorized cluster assignment map (ML output)

- `06_ml_bright_cluster_highlight.jpg`
  - brightest K-means cluster highlighted in red

- `07_input_intensity_histogram.png`
  - histogram of input grayscale intensities

- `08_enhanced_intensity_histogram.png`
  - histogram after enhancement

- `09_segmentation_area_percentage.png`
  - bar chart of background vs segmented area

---

## 5) Web UI Explanation

`web_ui.py` provides a local browser-based interface.

Features:
- drag-and-drop or choose image file
- set `Max side for processing`
- set `K-means clusters`
- run full pipeline and view all output images in browser

Field meanings:
- Max side for processing:
  - same as `--max-size`
  - controls resize before processing
- K-means clusters:
  - same as `--clusters`
  - controls clustering granularity

Run artifacts are stored under:
- `web_runs/uploads/<run_id>/...`
- `web_runs/results/<run_id>/...`

---

## 6) Code Structure

Main files:
- `main.py`
  - complete processing pipeline and CLI
  - contains `run_pipeline(...)` reusable function

- `web_ui.py`
  - local HTTP server UI
  - uploads image and calls `run_pipeline(...)`

- `requirements.txt`
  - Python dependency list

---

## 7) How to Use the Repo (From Scratch)

### 7.1 Clone repository
```bash
git clone <YOUR_REPO_URL>
cd DIP-Project
```

### 7.2 Create and activate virtual environment
macOS/Linux:
```bash
python3 -m venv dip
source dip/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv dip
.\dip\Scripts\Activate.ps1
```

### 7.3 Install requirements
```bash
pip install -r requirements.txt
```

### 7.4 Run CLI pipeline
```bash
python main.py
```

### 7.5 Run web UI
```bash
python web_ui.py
```
Open in browser:
- `http://127.0.0.1:8000`
