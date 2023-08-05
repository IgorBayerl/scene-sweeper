import cv2
import os


def load_images(input_dir):
    """Load images.

    Args:
    input_dir (str): Directory containing the input images.

    Returns:
    images (list of np.array): List of loaded images.
    """
    # Get a list of all image files in input_dir
    print(">> Loading images...")
    image_filenames = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]

    # Sort the image filenames
    image_filenames.sort()

    # Initialize an empty list to store the images
    images = []

    # Loop over all image files
    for filename in image_filenames:
        # Construct the full image path
        image_path = os.path.join(input_dir, filename)

        # Load the image using OpenCV
        img = cv2.imread(image_path)

        # Append the loaded image to the list
        images.append(img)

    print(">> Loading images completed.")

    return images
