import numpy as np


def median_blend(segmented_images):
    """Blend segmented images using a median filter.

    Args:
    segmented_images (list of np.array): List of segmented images.

    Returns:
    blended_image (np.array): Blended image.
    """
    print(">> Applying median blending...")

    # Stack images along the third axis
    stacked_images = np.stack(segmented_images, axis=-1)

    # Apply median filter
    blended_image = np.median(stacked_images, axis=-1)

    print(">> Median blending completed.")

    return blended_image


def median_blend1(images):
    # Stack images together along a new dimension
    image_stack = np.stack(images, axis=-1)

    # Compute the median along the new dimension
    static_image = np.median(image_stack, axis=-1)

    return static_image.astype(np.uint8)
