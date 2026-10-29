"""Convert binary semantic masks to single-class YOLO instance polygons.

Separate connected sky regions become separate rows. YOLO polygon labels cannot
represent interior holes; every conversion therefore reports a raster round-trip
IoU rather than silently claiming lossless conversion.
"""

from __future__ import annotations

import cv2
import numpy as np

from .common import mask_iou


def polygons_to_mask(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    raster = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.rint(polygon * [width, height]).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        # Fill separately: fillPoly on all contours at once uses an even-odd rule.
        cv2.fillPoly(raster, [points], 1)
    return raster.astype(bool)


def mask_to_polygons(mask: np.ndarray, min_area: int = 64, epsilon: float = 1.0):
    if mask.ndim != 2 or min_area < 0 or epsilon < 0:
        raise ValueError("Expected a 2D mask, min_area >= 0 and epsilon >= 0")
    height, width = mask.shape
    binary = np.asarray(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    holes = 0
    dropped = 0
    if hierarchy is not None:
        for contour, info in zip(contours, hierarchy[0]):
            if info[3] >= 0:
                holes += 1
                continue
            if cv2.contourArea(contour) < min_area:
                dropped += 1
                continue
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
            if len(simplified) < 3 or cv2.contourArea(simplified) == 0:
                simplified = contour
            points = simplified.reshape(-1, 2)
            if len(points) < 3 or cv2.contourArea(points) == 0:
                dropped += 1
                continue
            # Ultralytics uses x / image_width, y / image_height, not (width - 1).
            polygons.append(np.round(points.astype(np.float64) / [width, height], 8))
    reconstructed = polygons_to_mask(polygons, mask.shape)
    return polygons, {
        "polygon_iou": mask_iou(mask.astype(bool), reconstructed),
        "holes": holes,
        "dropped_components": dropped,
        "instances": len(polygons),
        "vertices": sum(len(p) for p in polygons),
    }


def labels_text(polygons: list[np.ndarray]) -> str:
    return "".join("0 " + " ".join(f"{v:.8f}" for v in polygon.ravel()) + "\n" for polygon in polygons)
