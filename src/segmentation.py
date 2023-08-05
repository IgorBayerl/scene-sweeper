import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path="models/deeplabv3_1_default_1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors information from the model file
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def segment_images(images):
    """Applies segmentation to a list of images.

    Args:
    images (list of np.array): List of images.

    Returns:
    segmented_images (list of np.array): List of segmented images.
    masks (list of np.array): List of binary masks.
    """
    print(">> Applying image segmentation...")
    segmented_images = []
    masks = []

    for image in images:
        # Keep the original image size for later
        original_image_size = (image.shape[1], image.shape[0])

        # Ensure image is RGB
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Resize to model input size
        resized_image = cv2.resize(image, (257, 257))
        input_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Feed the image to the model
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        # Retrieve the output from the model
        output_data = interpreter.get_tensor(output_details[0]["index"])

        # The output is a one-hot encoded tensor, we remove the batch dimension and convert it to a segmentation mask
        segmentation_mask = np.argmax(output_data[0], axis=-1)

        # Resize the segmentation mask back to the original image size
        segmentation_mask = cv2.resize(
            segmentation_mask,
            original_image_size,
            interpolation=cv2.INTER_NEAREST,
        )

        # Create a binary mask
        binary_mask = np.where(segmentation_mask == 0, 0, 255).astype(np.uint8)
        masks.append(binary_mask)

        segmented_images.append(segmentation_mask)

    print(">> Image segmentation completed.")
    return segmented_images, masks
