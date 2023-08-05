import cv2
import numpy as np


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def select_best_image(images):
    # Convert images to grayscale
    gray_images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        for img in images
    ]

    # Compute Laplacian variances
    laplacian_variances = [variance_of_laplacian(img) for img in gray_images]

    # Select the best image index
    best_image_idx = np.argmax(laplacian_variances)

    return images[best_image_idx]
