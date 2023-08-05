import cv2


def inpaint(image, mask, inpaint_radius=3, inpaint_method=cv2.INPAINT_TELEA):
    """Inpaint image.

    Args:
    image (np.array): Image to be inpainted.
    mask (np.array): Inpainting mask.
    inpaint_radius (int, optional): Radius of a circular neighborhood of each point inpainted that is considered by the algorithm.
    inpaint_method (cv2.inpaintTypes, optional): Inpainting method (either cv2.INPAINT_NS or cv2.INPAINT_TELEA)

    Returns:
    inpainted_image (np.array): Inpainted image.
    """
    # Inpaint image
    print(">> Inpainting image...")
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, inpaint_method)

    print(">> Inpainting image completed.")
    return inpainted_image
