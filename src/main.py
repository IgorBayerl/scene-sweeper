import os
import cv2
from blend import median_blend
from load import load_images
from preprocess import preprocess_images
from align import align_images
from select_best import select_best_image
from scipy.signal import wiener
import numpy as np

INPUT_DIR = "data/input/"
# INPUT_DIR = "data/input1/"
# INPUT_DIR = "data/input2/"
OUTPUT_DIR = "data/output/"
ALIGNED_IMAGES_DIR = "data/aligned_images/"
MASKS_IMAGES_DIR = "data/masks/"


def save_image(image, filename, output_dir=OUTPUT_DIR):
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Image saved to {filepath}")


def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Gaussian smoothing
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)

    # Add the residual (The original image - the blurred image) back to the original image:
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    return sharpened


def deblur_image(image, mysize=(5, 5)):
    print(">> Deblurring image...")
    # Check if image is grayscale or color
    if len(image.shape) == 2:
        # Image is grayscale
        deblurred = wiener(image, mysize)
    else:
        # Image is color, apply filter to each channel
        deblurred = np.zeros_like(image)
        for i in range(3):  # Assuming image is in BGR format
            deblurred[..., i] = wiener(image[..., i], mysize)

    # Convert deblurred image back to uint8
    deblurred = cv2.convertScaleAbs(deblurred)

    print(">> Deblurring image completed.")
    return deblurred


def crop_black_border(image):
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Threshold the image to get binary image, black as background
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the maximum contour, this is expected to be the frame of the actual image
    max_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle for the maximum contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image to this bounding rectangle and return it
    return image[y : y + h, x : x + w]


def create_difference_mask(reference, image, threshold=20):
    """Create a mask based on the difference between the reference and the input image."""
    difference = cv2.absdiff(reference, image)
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask.astype(np.uint8)


def selective_stack(images, best_image):
    """Stack images selectively, only including consistent parts."""
    height, width = best_image.shape[:2]
    stacked = np.zeros(
        (height, width), dtype=np.uint8
    )  # Only 2 dimensions as it's grayscale

    for idx, img in enumerate(images):
        # Make all the images black and white
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create a mask of the differences
        mask = create_difference_mask(best_image, img)

        processed_mask = process_mask(mask)

        # save mask with index of image
        save_image(processed_mask, f"mask_{idx}.jpg", MASKS_IMAGES_DIR)

        # Ensure the mask is of the right datatype
        mask = mask.astype(np.uint8)

        # Print shapes for debugging
        print("Image shape:", img.shape)
        print("Mask shape:", mask.shape)

        # Apply the mask
        masked_image = cv2.bitwise_and(img, img, mask=mask)

        # Add to the stacked image
        stacked = cv2.add(stacked, masked_image)

    # Average the stacked images
    stacked = (stacked / len(images)).astype(np.uint8)

    return stacked


def create_mask(original, processed, threshold=30):
    print(">> Creating mask...")
    # Calculate absolute difference between images
    diff = cv2.absdiff(original, processed)

    # Convert difference to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where difference is below the threshold
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY_INV)

    print(">> Creating mask completed.")
    return mask


def apply_mask(original, processed, mask):
    # Normalize mask to range 0.0 - 1.0
    mask = mask.astype(float) / 255

    # Expand dimensions of the mask to match the number of color channels in the image
    mask = np.expand_dims(mask, axis=-1)

    # Convert image data types for proper computation
    original = original.astype(float)
    processed = processed.astype(float)

    # Create a weighted image by multiplying with the mask and its inverse
    orig_masked = original * mask
    proc_masked = processed * (1.0 - mask)

    # Combine the masked images
    combined = orig_masked + proc_masked

    # Make sure all pixel intensities are within the range 0-255
    combined = np.clip(combined, 0, 255).astype("uint8")

    return combined


def process_mask(mask, kernel_size=6, blur_size=9, dilation_size=7):
    # Create the kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Remove small black spots (erosion followed by dilation: closing)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small white spots (dilation followed by erosion: opening)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Invert the mask to expand the black regions
    inverted_mask = cv2.bitwise_not(mask)

    # Increase size of the black objects
    dilation_kernel = np.ones((dilation_size, dilation_size), np.uint8)
    inverted_mask = cv2.dilate(inverted_mask, dilation_kernel, iterations=1)

    # Invert the mask back to its original form
    mask = cv2.bitwise_not(inverted_mask)

    # Blur the edges
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask


def main():
    # Load images
    images = load_images(INPUT_DIR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_loaded_images.jpg"), images[0])

    # Preprocess images
    preprocessed_images = preprocess_images(images)

    # Determine the best image (least blurry) to use as the main image
    best_image = select_best_image(preprocessed_images)
    save_image(best_image, "3_best_image.jpg")

    # Create new list of images with the main image repeated some times and the rest of the images (should not be more than half of the total images)
    BEST_IMAGE_REPETITIONS = 1
    new_images = [best_image] * BEST_IMAGE_REPETITIONS + preprocessed_images

    # Align images
    aligned_images = align_images(new_images)

    # Save all aligned images
    for i, img in enumerate(aligned_images):
        save_image(img, f"aligned_image_{i}.jpg", ALIGNED_IMAGES_DIR)

    # test_image = selective_stack(aligned_images, best_image)
    # save_image(test_image, "test_image.jpg")

    # Blend images
    blended_image = median_blend(aligned_images)
    save_image(blended_image, "4_blended_image.jpg")

    deblured_image = deblur_image(blended_image)
    save_image(deblured_image, "5_deblured_image.jpg")

    sharped_image = unsharp_mask(deblured_image)
    save_image(sharped_image, "6_sharped_image.jpg")

    # Crop the image
    cropped_image = crop_black_border(sharped_image)
    save_image(cropped_image, "7_croped_image.jpg")

    # Align images
    aligned_for_mask = align_images([best_image, cropped_image])

    MASK_THRESHOLD = 5  # maybe 5 - 30 are good values
    # Create mask using the original best image and the sharpened image
    mask = create_mask(
        aligned_for_mask[0], aligned_for_mask[1], MASK_THRESHOLD
    )
    save_image(mask, "8_mask.jpg")

    processed_mask = process_mask(mask)
    save_image(processed_mask, "9_processed_mask.jpg")

    # Apply the mask to the images
    final_image = apply_mask(
        aligned_for_mask[0], aligned_for_mask[1], processed_mask
    )

    # Write the final output image
    save_image(final_image, "10_final_image.jpg")


if __name__ == "__main__":
    main()
