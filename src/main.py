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
OUTPUT_DIR = "data/output/"
ALIGNED_IMAGES_DIR = "data/aligned_images/"


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
    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Bitwise-and the images with the mask and its inverse
    orig_masked = cv2.bitwise_and(original, original, mask=mask)
    proc_masked = cv2.bitwise_and(processed, processed, mask=mask_inv)

    # Combine the masked images
    combined = cv2.add(orig_masked, proc_masked)

    return combined


def main():
    # Load images
    images = load_images(INPUT_DIR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_loaded_images.jpg"), images[0])

    # Preprocess images
    preprocessed_images = preprocess_images(images)

    # Determine the best image (least blurry) to use as the main image
    main_image = select_best_image(preprocessed_images)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "3_main_image.jpg"), main_image)

    # Create new list of images with the main image repeated 5 times and the rest of the images
    new_images = [main_image] * 5 + [
        image
        for image in preprocessed_images
        if not np.array_equal(image, main_image)
    ]

    # Align images
    aligned_images = align_images(preprocessed_images)

    # Print shapes of aligned images
    for i, img in enumerate(aligned_images):
        print(f"Shape of aligned image {i}: {img.shape}")

    # Save all aligned images
    if not os.path.exists(ALIGNED_IMAGES_DIR):
        os.makedirs(ALIGNED_IMAGES_DIR)
    for i, img in enumerate(aligned_images):
        cv2.imwrite(
            os.path.join(ALIGNED_IMAGES_DIR, f"aligned_image_{i}.jpg"), img
        )

    # Blend images
    blended_image = median_blend(aligned_images)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "4_blended_image.jpg"), blended_image)

    deblured_image = deblur_image(blended_image)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "5_deblured_image.jpg"), deblured_image
    )

    sharped_image = unsharp_mask(deblured_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "6_sharped_image.jpg"), sharped_image)

    # Crop the image
    cropped_image = crop_black_border(sharped_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "7_croped_image.jpg"), cropped_image)

    # Align images
    aligned_for_mask = align_images([main_image, cropped_image])

    MASK_THRESHOLD = 20
    # Create mask using the original best image and the sharpened image
    mask = create_mask(
        aligned_for_mask[0], aligned_for_mask[1], MASK_THRESHOLD
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "8_mask.jpg"), mask)

    # Apply the mask to the images
    final_image = apply_mask(aligned_for_mask[0], aligned_for_mask[1], mask)

    # Write the final output image
    cv2.imwrite(os.path.join(OUTPUT_DIR, "9_final_image.jpg"), final_image)


if __name__ == "__main__":
    main()
