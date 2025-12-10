"""
Robot Kinematics and Helper Functions to integrate with obstacle_detection.py and rrt.py

"""

import numpy as np
import cv2

# Robot model
class Robot3R:
    def __init__(self, link_lengths, base_position):
        self.L1 = link_lengths[0]
        self.L2 = link_lengths[1]
        self.L3 = link_lengths[2]
        self.base_x, self.base_y = base_position

        print(f" Robot initialized:")
        print(f" Link 1: {self.L1} pixels")
        print(f" Link 2: {self.L2} pixels")
        print(f" Link 3: {self.L3} pixels")
        print(f" Base: ({self.base_x}, {self.base_y})")
    # FK
    def forward_kinematics(self, theta1, theta2, theta3):
        # Base position
        x0, y0 = self.base_x, self.base_y
        # Joint 1 position
        x1 = x0 + self.L1 * np.cos(theta1)
        y1 = y0 + self.L1 * np.sin(theta1)
        # Joint 2 position
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        # Joint 3 position (EE)
        x3 = x2 + self.L3 * np.cos(theta1 + theta2 + theta3)
        y3 = y2 + self.L3 * np.sin(theta1 + theta2 + theta3)

        return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    # IK, used Lectures 11 and 12 from class notes as reference for equation derivations
    def inverse_kinematics(self, target_x, target_y, elbow = 'righty'):
        dx = target_x - self.base_x
        dy = target_y - self.base_y
        # Distance to target
        d = np.sqrt(dx**2 + dy**2)
        # Angle to target
        gamma = np.arctan2(dy, dx)
        # 2-link arm (L2 + L3)
        L2_3 = self.L2 + self.L3
        # Checking reachability of joint 2 (wrist)
        if d > (self.L1 + L2_3) or d < abs(self.L1 - L2_3):
            return None # unreachable target
        # Solving for alpha using Law of Cosines (angle opposite d)
        cos_alpha = (self.L1**2 + d**2 - L2_3**2) / (2 * self.L1 * d)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        # Solving for beta using Law of Cosines (angle at base)
        cos_beta = (self.L1**2 + L2_3**2 - d**2) / (2 * self.L1 * L2_3)
        cos_beta = np.clip(cos_beta, -1, 1)
        beta = np.arccos(cos_beta)
        # Loop to check if elbow-up or elbow-down
        if elbow == 'righty':
            theta1 = gamma - alpha
            theta2 = np.pi - beta
        else: # lefty
            theta1 = gamma + alpha
            theta2 = beta - np.pi
        # Joint 2 location
        x1 = self.base_x + self.L1 * np.cos(theta1)
        y1 = self.base_y + self.L1 * np.sin(theta1)
        x2 = x1 + self.L2 * np.cos(theta1 + theta2)
        y2 = y1 + self.L2 * np.sin(theta1 + theta2)
        # Angle from joint 2 to target
        target_angle = np.arctan2(target_y - y2, target_x - x2)
        # Solving for theta3
        theta3 = target_angle - (theta1 + theta2)
        return theta1, theta2, theta3
# Morphology for obstacle mask
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

# Collision detection
def check_link_collisions(robot, theta1, theta2, theta3, obstacle_mask, samples_per_link=20):
    """
    Checking all 3 links, samples_per_link is number of points to check along each link
    """
    # joint positions
    positions = robot.forward_kinematics(theta1,theta2,theta3)
    # image dimensions
    image_height,image_width = obstacle_mask.shape
