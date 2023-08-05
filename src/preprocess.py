import cv2


def preprocess_images(images):
    """Preprocess images.

    Args:
    images (list of np.array): List of loaded images.

    Returns:
    images (list of np.array): List of preprocessed images.
    """
    # Initialize an empty list to store the preprocessed images
    print(">> Preprocessing images...")
    preprocessed_images = []

    # Loop over all images
    for img in images:
        # Resize image if it's larger than Full HD while maintaining aspect ratio
        max_res = (1080, 1920)  # Full HD resolution (height, width)

        # Find the aspect ratio of the image
        aspect_ratio = img.shape[1] / img.shape[0]

        if img.shape[0] > max_res[0]:
            new_height = max_res[0]
            new_width = int(new_height * aspect_ratio)
            img = cv2.resize(img, (new_width, new_height))

        elif img.shape[1] > max_res[1]:
            new_width = max_res[1]
            new_height = int(new_width / aspect_ratio)
            img = cv2.resize(img, (new_width, new_height))

        # Append the preprocessed image to the list
        preprocessed_images.append(img)

    print(">> Preprocessing images completed.")
    return preprocessed_images
