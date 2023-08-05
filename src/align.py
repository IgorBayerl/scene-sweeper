import cv2
import numpy as np


def align_images2(images):
    print(">> Aligning images...")
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Store aligned images
    aligned_images = []

    # Use the first image as reference
    ref_image = images[0]
    aligned_images.append(ref_image)

    # Convert the reference image to grayscale if it's not already
    ref_gray = (
        ref_image
        if len(ref_image.shape) == 2
        else cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    )

    # Detect ORB features and compute descriptors.
    ref_kps, ref_des = orb.detectAndCompute(ref_gray, None)

    # Create a BFMatcher object.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for image in images[1:]:
        # Convert the image to grayscale if it's not already
        gray = (
            image
            if len(image.shape) == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        # Detect ORB features and compute descriptors.
        kps, des = orb.detectAndCompute(gray, None)

        # Match descriptors between the reference image and the current image
        matches = matcher.match(ref_des, des)

        # Sort the features in order of distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Use the top matches to align the images
        aligned = cv2.warpPerspective(
            image,
            get_homography(
                ref_kps, kps, matches[:50]
            ),  # Updated to use top 50 matches
            image.shape[1::-1],
        )

        # Append the aligned image to the list of aligned images
        aligned_images.append(aligned)

    print(">> Aligning images completed.")
    return aligned_images


def align_images3(images):
    print(">> Aligning images...")
    # Initialize the SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Store aligned images
    aligned_images = []

    # Use the first image as reference
    ref_image = images[0]
    aligned_images.append(ref_image)

    # Convert the reference image to grayscale if it's not already
    ref_gray = (
        ref_image
        if len(ref_image.shape) == 2
        else cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    )

    # Detect SIFT features and compute descriptors.
    ref_kps, ref_des = sift.detectAndCompute(ref_gray, None)

    # Create a FLANN matcher object.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    for image in images[1:]:
        # Convert the image to grayscale if it's not already
        gray = (
            image
            if len(image.shape) == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        # Detect SIFT features and compute descriptors.
        kps, des = sift.detectAndCompute(gray, None)

        # Match descriptors between the reference image and the current image
        matches = matcher.knnMatch(ref_des, des, k=2)

        # Apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # Use the top matches to align the images
        aligned = cv2.warpPerspective(
            image,
            get_homography(ref_kps, kps, good_matches),
            image.shape[1::-1],
        )

        # Append the aligned image to the list of aligned images
        aligned_images.append(aligned)

    print(">> Aligning images completed.")
    return aligned_images


def align_images4(images):
    print(">> Aligning images...")
    # Initialize the SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Store aligned images
    aligned_images = []

    # Use the first image as reference
    ref_image = images[0]
    aligned_images.append(ref_image)

    # Convert the reference image to grayscale if it's not already
    ref_gray = (
        ref_image
        if len(ref_image.shape) == 2
        else cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    )

    # Detect SIFT features and compute descriptors.
    ref_kps, ref_des = sift.detectAndCompute(ref_gray, None)

    # Create a FLANN matcher object.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    for image in images[1:]:
        # Convert the image to grayscale if it's not already
        gray = (
            image
            if len(image.shape) == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        # Detect SIFT features and compute descriptors.
        kps, des = sift.detectAndCompute(gray, None)

        # Match descriptors between the reference image and the current image
        matches = matcher.knnMatch(ref_des, des, k=2)

        # Apply Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        # Use the top matches to align the images
        aligned = cv2.warpAffine(
            image,
            get_affine_transform(ref_kps, kps, good_matches),
            image.shape[1::-1],
        )

        # Append the aligned image to the list of aligned images
        aligned_images.append(aligned)

    print(">> Aligning images completed.")
    return aligned_images


def align_images(images):
    # Ensure at least 2 images
    if len(images) < 2:
        print("Need at least two images to align.")
        return

    print("\n>> Aligning images...")

    MAX_FEATURES = 15000
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints = []
    descriptors = []
    for img in images_gray:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Use first image as reference
    ref_kp = keypoints[0]
    ref_des = descriptors[0]

    aligned_images = [
        images[0]
    ]  # First image is reference, so already "aligned"
    for i in range(1, len(images)):
        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        )
        matches = matcher.match(ref_des, descriptors[i])

        # Convert tuple to list (if necessary) and sort matches by score
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove worst matches
        num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:num_good_matches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for j, match in enumerate(matches):
            points1[j, :] = ref_kp[match.queryIdx].pt
            points2[j, :] = keypoints[i][match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography to warp image
        height, width, channels = images[0].shape
        aligned = cv2.warpPerspective(images[i], h, (width, height))
        aligned_images.append(aligned)

    return aligned_images


def align_images_with_refinement(images):
    # Ensure at least 2 images
    if len(images) < 2:
        print("Need at least two images to align.")
        return

    print("\n>> Aligning images...")

    MAX_FEATURES = 15000
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints = []
    descriptors = []
    for img in images_gray:
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Use first image as reference
    ref_kp = keypoints[0]
    ref_des = descriptors[0]

    # First image is reference, so already "aligned"
    # Convert first image to grayscale and refine alignment with ECC
    ref_image_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    ref_image = refine_alignment_with_ecc(ref_image_gray, ref_image_gray)
    aligned_images = [ref_image]
    for i in range(1, len(images)):
        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        )
        matches = matcher.match(ref_des, descriptors[i])

        # Convert tuple to list (if necessary) and sort matches by score
        matches = list(matches)
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove worst matches
        num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:num_good_matches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for j, match in enumerate(matches):
            points1[j, :] = ref_kp[match.queryIdx].pt
            points2[j, :] = keypoints[i][match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography to warp image
        height, width, channels = images[0].shape
        aligned = cv2.warpPerspective(images[i], h, (width, height))

        # Convert aligned to grayscale
        aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        # Refine the alignment with ECC method
        aligned = refine_alignment_with_ecc(images_gray[0], aligned_gray)

        aligned_images.append(aligned)

    print(">> Aligning images completed.")
    return aligned_images


def refine_alignment_with_ecc(image1, image2):
    print(">> Refining alignment with ECC...")
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN

    # Define 2x3 or 3x3 warp matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 500

    # Specify the epsilon error.
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    # Run the ECC algorithm
    cc, warp_matrix = cv2.findTransformECC(
        image1, image2, warp_matrix, warp_mode, criteria
    )

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        image2_aligned = cv2.warpPerspective(
            image2,
            warp_matrix,
            (image1.shape[1], image1.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        image2_aligned = cv2.warpAffine(
            image2,
            warp_matrix,
            (image1.shape[1], image1.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

    return image2_aligned


def get_affine_transform(kpsA, kpsB, matches):
    print(">> Computing affine transform...")
    # Convert keypoints to an array of points
    ptsA = np.float32([kp.pt for kp in kpsA])
    ptsB = np.float32([kp.pt for kp in kpsB])

    # Extract location of good matches
    ptsA = np.float32([ptsA[m.queryIdx] for m in matches])
    ptsB = np.float32([ptsB[m.trainIdx] for m in matches])

    # Compute the affine matrix
    M, _ = cv2.estimateAffinePartial2D(ptsA, ptsB)

    print(">> Computing affine transform completed.")
    return M


def get_homography(kpsA, kpsB, matches):
    print(">> Computing homography...")
    # Convert keypoints to an array of points
    ptsA = np.float32([kp.pt for kp in kpsA])
    ptsB = np.float32([kp.pt for kp in kpsB])

    # Extract location of good matches
    ptsA = np.float32([ptsA[m.queryIdx] for m in matches])
    ptsB = np.float32([ptsB[m.trainIdx] for m in matches])

    # Compute the homography matrix
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 1000)

    print(">> Computing homography completed.")
    return H
