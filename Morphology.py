# Morphology for obstacle mask

import numpy as np
import cv2

def improve_obstacle_mask(fg_mask, min_contour_area = 100):
    """
    Applying morphology operations to clean up the obstacle detection
    1. Opening - removes small noise spots (erosion -> dilation)
    2. Closing - fills in holes in obstacles (dilation -> erosion)
    3. Dilation - Adds safety margin around obstacles
    4. Contour filtering - Removes artifacts below minimum area
    References: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html,
    https://www.geeksforgeeks.org/python/image-segmentation-using-morphological-operation/
    https://www.geeksforgeeks.org/python/find-and-draw-contours-using-opencv-python/
    """
    # Create the kernel
    kernel = np.ones((5,5), np.uint8)

    # 1. Opening: Erosion makes small noise disappear, dilation keeps real obstacles
    mask_open = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # 2. Closing: Dilation to fill holes, erosion to maintain obstacle shape
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 3. Dilation: make obstacles slightly bigger so robot maintains collision-free path
    mask_safe = cv2.dilate(mask_close, kernel, iterations=2)
    # 4. Contour filtering: removes small artifacts
    contours, _ = cv2.findContours(mask_safe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Empty mask and loop to maintain contours above minimum threshold
    mask_clean = np.zeros_like(mask_safe)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area:
            cv2.drawContours(mask_clean, [contour], -1, 255, thickness=-1)
    return mask_clean