import cv2
import numpy as np
from RRT import rrt
from robot_kinematics import improve_obstacle_mask

goal = None
path = None
#global vars needed for scope

def mouse_callback(event, x, y, flags, param): 
    global goal
    if event == cv2.EVENT_LBUTTONDOWN:
        goal = (x, y)
#function and RRT courtesy of https://github.com/nimRobotics/RRT/blob/master/rrt.py
cv2.namedWindow("Overlay")
cv2.setMouseCallback("Overlay", mouse_callback)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#Locks resolution at 720p (from logi c270 - stays same for all cameras)

start = (50,50)
#define home position (can be in terms of h,w for ratios)

Threshold = 30  #Threshold identifies what is an object and what is not, 20-30 suggested range from doc but will be testable
overlayStrength = 0.3 #Strength of overlay color on detected obstacles, 0-1 range

while True:
    ret, frame = cap.read()
    
    #Reads next frames continously
    if not ret:
        break
    blur = cv2.GaussianBlur(frame, (5,5), 0)
    color_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(color_hsv, (0,0,50), (180,40,255))
    fg_mask = cv2.bitwise_not(white)

    #implementing improve obstacle mask function to clean up mask using morphology
    obstacle_mask = fg_mask.copy()
    obstacle_mask = improve_obstacle_mask(obstacle_mask)
    obstacle_mask = cv2.bitwise_not(fg_mask)
    overlay = frame.copy()
    overlay[obstacle_mask == 255] = (255, 255, 255)  # Highlight obstacles in red (can change color)
    vis = cv2.addWeighted(frame, overlayStrength, overlay, 1 - overlayStrength, 0)
    cv2.circle(vis, start, 3, (0,0,255), -1) #params as defined in circle function
    if goal != None:
        cv2.circle(vis, goal, 3, (0,0,255), -1)
    #drawing start point and goal point
    #Overlay outputs the detected obstacles on the original input frame
        path = rrt(obstacle_mask, start, goal, 20)
        for (x, y) in path:
            cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)
    cv2.imshow("Overlay", vis)
    #cv2.imshow("Obstacle Mask", obstacle_mask) #improved qual

    
    if cv2.waitKey(1) & 0xFF == 27: #ESC to end program
        break

cap.release()
cv2.destroyAllWindows()
