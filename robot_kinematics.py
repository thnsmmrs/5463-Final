# Robot kinematics file written by Krish, contains the class Robot3R used in the main file as well as the corresponding forward and inverse kinematic functions
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





