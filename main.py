import argparse
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".mplconfig")))
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Galaxy image pipeline: enhancement, segmentation, and ML-based clustering."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Hubble_ultra_deep_field_high_rez_edit1.jpg",
        help="Path to input image (.jpg/.png).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where output images will be saved.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1600,
        help="Resize longest side to this value for faster processing.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of K-means clusters for the ML step.",
    )
    return parser.parse_args()


def ensure_input_image(path: Path) -> Path:
    if path.exists():
        return path

    jpgs = sorted(list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.jpeg")))
    if jpgs:
        print(f"Input not found. Using available image: {jpgs[0]}")
        return jpgs[0]

    raise FileNotFoundError(
        f"Input image not found: {path}. Put a .jpg/.jpeg file in this folder or pass --input."
    )


def resize_for_processing(img_bgr: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_size:
        return img_bgr

    scale = max_size / float(longest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"Resized image from {w}x{h} to {new_w}x{new_h} for processing speed.")
    return resized


def enhance_image(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.6, blurred, -0.6, 0)

    return sharpened


def segment_galaxies(enhanced_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = 20
    filtered_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == i] = 255

    overlay = enhanced_bgr.copy()
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

    return filtered_mask, overlay


def kmeans_clustering_visual(img_bgr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]

    yy, xx = np.indices((h, w))
    x_norm = (xx.astype(np.float32) / max(1, w - 1)) * 20.0
    y_norm = (yy.astype(np.float32) / max(1, h - 1)) * 20.0

    features = np.concatenate(
        [
            lab.reshape(-1, 3).astype(np.float32),
            x_norm.reshape(-1, 1),
            y_norm.reshape(-1, 1),
        ],
        axis=1,
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    compactness, labels, centers = cv2.kmeans(
        features,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    label_map = labels.reshape(h, w)

    rng = np.random.default_rng(42)
    palette = rng.integers(40, 255, size=(k, 3), dtype=np.uint8)
    cluster_map = palette[label_map]

    centers_l = centers[:, 0]
    bright_cluster = int(np.argmax(centers_l))

    highlight = img_bgr.copy()
    mask = (label_map == bright_cluster).astype(np.uint8) * 255
    heat = np.zeros_like(img_bgr)
    heat[:, :] = (0, 0, 255)
    highlight = np.where(
        mask[..., None] > 0,
        cv2.addWeighted(highlight, 0.5, heat, 0.5, 0),
        highlight,
    )

    print(f"K-means compactness: {compactness:.2f}; brightest cluster id: {bright_cluster}")
    return cluster_map, highlight


def save_image(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"Failed to save image: {path}")


def generate_matplotlib_reports(
    output_dir: Path,
    input_bgr: np.ndarray,
    enhanced_bgr: np.ndarray,
    seg_mask: np.ndarray,
) -> list[Path]:
    report_paths: list[Path] = []

    gray_input = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(gray_input.ravel(), bins=64, color="steelblue", alpha=0.85)
    ax1.set_title("Input Intensity Histogram")
    ax1.set_xlabel("Pixel Value")
    ax1.set_ylabel("Count")
    fig1.tight_layout()
    report1 = output_dir / "07_input_intensity_histogram.png"
    fig1.savefig(report1, dpi=180)
    plt.close(fig1)
    report_paths.append(report1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(gray_enhanced.ravel(), bins=64, color="seagreen", alpha=0.85)
    ax2.set_title("Enhanced Intensity Histogram")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Count")
    fig2.tight_layout()
    report2 = output_dir / "08_enhanced_intensity_histogram.png"
    fig2.savefig(report2, dpi=180)
    plt.close(fig2)
    report_paths.append(report2)

    mask_ratio = (np.count_nonzero(seg_mask) / seg_mask.size) * 100.0
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.bar(
        ["Background", "Segmented"],
        [100 - mask_ratio, mask_ratio],
        color=["#777", "#f28e2b"],
    )
    ax3.set_title("Segmentation Area (%)")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Percentage")
    fig3.tight_layout()
    report3 = output_dir / "09_segmentation_area_percentage.png"
    fig3.savefig(report3, dpi=180)
    plt.close(fig3)
    report_paths.append(report3)

    return report_paths


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    max_size: int = 1600,
    clusters: int = 5,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {input_path}")

    img_bgr = resize_for_processing(img_bgr, max_size)

    enhanced = enhance_image(img_bgr)
    seg_mask, seg_overlay = segment_galaxies(enhanced)
    cluster_map, ml_highlight = kmeans_clustering_visual(enhanced, k=max(2, clusters))

    outputs = {
        "input": output_dir / "01_input_resized.jpg",
        "enhanced": output_dir / "02_enhanced.jpg",
        "seg_mask": output_dir / "03_segmentation_mask.png",
        "seg_overlay": output_dir / "04_segmentation_overlay.jpg",
        "cluster_map": output_dir / "05_kmeans_cluster_map.jpg",
        "ml_highlight": output_dir / "06_ml_bright_cluster_highlight.jpg",
    }

    save_image(outputs["input"], img_bgr)
    save_image(outputs["enhanced"], enhanced)
    save_image(outputs["seg_mask"], seg_mask)
    save_image(outputs["seg_overlay"], seg_overlay)
    save_image(outputs["cluster_map"], cluster_map)
    save_image(outputs["ml_highlight"], ml_highlight)

    reports = generate_matplotlib_reports(
        output_dir,
        img_bgr,
        enhanced,
        seg_mask,
    )
    outputs["input_histogram"] = reports[0]
    outputs["enhanced_histogram"] = reports[1]
    outputs["segmentation_area_chart"] = reports[2]

    return outputs


def main() -> None:
    args = parse_args()

    input_path = ensure_input_image(Path(args.input))
    output_dir = Path(args.output_dir)

    outputs = run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        max_size=args.max_size,
        clusters=args.clusters,
    )

    print("\nPipeline completed successfully. Output files:")
    for _, path in outputs.items():
        print(f"- {path}")


if __name__ == "__main__":
    main()
