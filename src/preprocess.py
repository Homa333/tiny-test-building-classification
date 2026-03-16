import cv2
import numpy as np
from PIL import Image


class Preprocessor:

    def __init__(self):
        pass

    def resize_image(self, image: Image.Image, size=(640, 640)) -> Image.Image:
        """Resize image to fixed resolution"""
        return image.resize(size)

    def convert_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to numpy array"""
        return np.array(image)

    def convert_to_rgb(self, image_np: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB if using OpenCV"""
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    def convert_to_grayscale(self, image_np: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    def gaussian_blur(self, image_np: np.ndarray, kernel=(5, 5)) -> np.ndarray:
        """Reduce image noise"""
        return cv2.GaussianBlur(image_np, kernel, 0)

    def sharpen_image(self, image_np: np.ndarray) -> np.ndarray:
        """Sharpen image to enhance edges"""
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(image_np, -1, kernel)

    def enhance_contrast(self, image_np: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        return enhanced

    def normalize_image(self, image_np: np.ndarray) -> np.ndarray:
        """Normalize brightness across images"""
        return cv2.normalize(
            image_np,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

    def numpy_to_pil(self, image_np: np.ndarray) -> Image.Image:
        """Convert numpy array back to PIL"""
        return Image.fromarray(image_np)
