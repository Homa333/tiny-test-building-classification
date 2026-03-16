from typing import Any, Dict, List, Tuple

from PIL import Image
import numpy as np
from transformers import pipeline


class ImageSegmentation:
    def __init__(self):
        self.segmenter = pipeline("image-segmentation",
                                  model="nvidia/segformer-b2-finetuned-ade-512-512")

    def segment_image(self, image: Image):
        results = self.segmenter(image)
        return results

    def create_object_mask(self, results: List[Dict[str, Any]], labels: list = None) -> Image.Image:
        """
        This function should create a mask for the given object
        """
        final_mask = None

        for r in results:
            label = r["label"]

            if labels is not None and label not in labels:
                continue
            mask = np.array(r["mask"])

            if final_mask is None:
                final_mask = mask
            else:
                final_mask = np.logical_or(final_mask, mask)

        return final_mask

    def crop_image_with_mask(self, image: Image, mask: np.ndarray) -> Image.Image:
        """
        This function should crop the image using the mask
        """
        image_np = np.array(image)
        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add some padding
        pad = 10
        y_min = max(0, y_min - pad)
        x_min = max(0, x_min - pad)
        y_max = min(image_np.shape[0], y_max + pad)
        x_max = min(image_np.shape[1], x_max + pad)

        cropped_image = image_np[y_min:y_max, x_min:x_max]
        return cropped_image

    def visualize_mask(self, image: Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.4) -> Image.Image:
        """
        Overlay the mask on the original image for visualization.
        """

        image_np = np.array(image).copy()

        # Ensure mask is boolean
        mask = mask.astype(bool)

        overlay = image_np.copy()
        overlay[mask] = color

        # Blend overlay with original image
        blended = (
            image_np * (1 - alpha) +
            overlay * alpha
        ).astype(np.uint8)

        return Image.fromarray(blended)

    def get_object_ratio(self, image: Image.Image, results: list, labels: list) -> float:
        """
        Calculate the pixel ratio of specified object labels in the segmentation output.
        """

        object_pixels = 0

        for r in results:

            if r["label"] not in labels:
                continue

            mask = np.array(r["mask"])

            object_pixels += (mask > 0).sum()

        width, height = image.size
        total_pixels = width * height

        return object_pixels / total_pixels
