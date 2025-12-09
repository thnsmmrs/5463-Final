
import numpy as np
import cv2

# Robot model
class Robot3R:
    def initialize(self, link_lengths, base_position):
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

        return [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
    # IK
    def inverse_kinematics(self, target_x, target_y, phi = 0):
        # 3R robot has 3 DOF but only 2D position constraint (x,y) so phi is desired orientation of EE
        dx = target_x - self.base_x
        dy = target_y - self.base_y
        # Position of joint 2 (wrist)
        wx = dx - self.L3 * np.cos(phi)
        wy = dy - self.L3 * np.sin(phi)
        # Distance to joint 2
        d = np.sqrt(wx**2 + wy**2)
        # Checking reachability of joint 2 (wrist)
        if d > self.L1 + self.L2 or d < abs(self.L1 - self.L2):
            return None
        # Solving for first two joints
        cos_theta2 = (d**2 - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        # Elbow-up
        theta2 = np.arccos(cos_theta2)
        # Solving for theta1
        k1 = self.L1 + self.L2 * np.cos(theta2)
        k2 = self.L2 * np.sin(theta2)
        theta1 = np.arctan2(wy,wx) - np.arctan2(k2,k1)
        # Third joint
        theta3 = phi - theta1 - theta2
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

